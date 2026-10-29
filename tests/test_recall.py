"""Memory search by meaning (voice_agent/recall.py) against a stub /v1/embeddings server: indexing, search, the
timeout and a server that is down (the turn goes on without it), retention and erasure, the file kept across starts,
and a server without MRL cuts."""

import asyncio
import json
import time
import zlib

import httpx
import numpy as np
import pytest

from voice_agent import recall
from voice_agent.memory import with_context

DIM = 64


def vector(text: str, dims: int) -> list[float]:
    """A bag of words: each word's hash bucket, so texts sharing words point the same way (enough for a stub)."""

    v = np.zeros(dims, np.float32)
    body = text.split("query: ")[-1].split("text: ")[-1].lower()
    for w in "".join(c if c.isalnum() else " " for c in body).split():
        if len(w) > 3 and w not in {"what", "when", "about", "should", "will"}:
            v[zlib.crc32(w.encode()) % dims] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v + 1 / np.sqrt(dims)).tolist()


class Stub:
    def __init__(self, delay: float = 0, down: bool = False, mrl: bool = True) -> None:
        self.delay, self.down, self.mrl, self.calls = delay, down, mrl, []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content)
        self.calls.append(body)
        if "dimensions" in body and not self.mrl:
            return httpx.Response(400, json={"error": {"message": "dimensions is not supported by this model"}})
        if self.delay and any(t.startswith("task:") for t in body["input"]):
            await asyncio.sleep(self.delay)
        dims = body.get("dimensions", DIM)
        return httpx.Response(200, json={"object": "list", "data": [
            {"object": "embedding", "index": i, "embedding": vector(t, dims)} for i, t in enumerate(body["input"])]})


def make(tmp_path, stub: Stub, **spec) -> recall.Recall:
    client = httpx.AsyncClient(transport=httpx.MockTransport(stub))
    return recall.Recall({"url": "http://stub/v1", "dimensions": 32, "path": str(tmp_path / "recall.jsonl"), **spec}, client)


def test_turns_about_the_past_are_recognised():
    for q in ["when did I last water the plants?", "what did we say about the trip?", "what was I doing earlier?",
              "do you remember the red mug?", "where did I leave my keys?", "did I tell you about the dentist?"]:
        assert recall.asks(q), q
    for q in ["what is the weather now?", "tell me a joke", "make a video of a lighthouse"]:
        assert not recall.asks(q), q


def test_a_caption_and_a_line_are_found_by_meaning_in_time_order(tmp_path):
    async def go():
        r = make(tmp_path, Stub())
        now = time.time()
        r.add("seen", "A person waters the green plants by the window.", now - 3600)
        r.add("seen", "A grey cat sleeps on the keyboard.", now - 1800)
        r.add("you said", "We should book the trip to Lisbon in May.", now - 600)
        r.add("Boo said", "I will remember the Lisbon trip in May.", now - 590)
        await r.flush()
        plants = await r.search("when did I last water the plants?")
        trip = await r.search("what did we say about the Lisbon trip?")
        return plants, trip, recall.lines(trip, now)

    plants, trip, text = asyncio.run(go())
    assert max(plants, key=lambda h: h["score"])["text"].startswith("A person waters")
    top = sorted(trip, key=lambda h: -h["score"])[:2]
    assert sorted(h["kind"] for h in top) == ["Boo said", "you said"]  # both lines of the conversation score highest
    assert [h["ts"] for h in trip] == sorted(h["ts"] for h in trip)  # handed to the brain oldest first
    assert "Lisbon" in text and text.startswith("(It is ")
    assert "[Earlier, found by meaning" in with_context("what did we say?", None, None, recalled=text)


def test_a_slow_server_never_holds_the_turn_past_the_timeout(tmp_path):
    async def go():
        r = make(tmp_path, Stub(delay=0.5), timeout_ms=40)
        r.add("seen", "A person waters the plants.")
        await r.flush()
        t = time.perf_counter()
        hits = await r.search("when did I water the plants?")
        return hits, time.perf_counter() - t

    hits, took = asyncio.run(go())
    assert hits is None and took < 0.2


def test_a_server_that_is_down_is_skipped(tmp_path):
    async def go():
        stub = Stub()
        r = make(tmp_path, stub)
        r.add("seen", "A person waters the plants.")
        await r.flush()
        stub.down = True
        r.add("seen", "This caption is lost while the server is down.")
        await r.flush()
        return r, await r.search("when did I water the plants?")

    r, hits = asyncio.run(go())
    assert hits is None and len(r.items) == 1


def test_retention_erases_old_entries_from_memory_and_the_file(tmp_path):
    async def go():
        r = make(tmp_path, Stub(), days=7)
        now = time.time()
        r.add("seen", "An old scene from five days ago.", now - 5 * 86400)
        r.add("seen", "A scene from this morning.", now - 3600)
        await r.flush()
        r.keep(2)
        return r

    r = asyncio.run(go())
    assert [it["text"] for it in r.items] == ["A scene from this morning."]
    on_disk = (tmp_path / "recall.jsonl").read_text()
    assert "five days ago" not in on_disk and "this morning" in on_disk


def test_the_index_is_kept_across_starts_and_old_lines_dropped_on_load(tmp_path):
    async def go():
        r = make(tmp_path, Stub(), days=7)
        r.add("seen", "A scene from three days ago.", time.time() - 3 * 86400)
        r.add("you said", "Remind me to call the plumber.", time.time() - 60)
        await r.flush()

    asyncio.run(go())
    again = make(tmp_path, Stub(), days=2)  # a shorter retention on the next start
    assert [it["kind"] for it in again.items] == ["you said"]
    assert again.vecs.shape == (1, 32)
    assert "three days ago" not in (tmp_path / "recall.jsonl").read_text()


def test_a_server_without_mrl_gets_full_vectors_cut_here(tmp_path):
    async def go():
        stub = Stub(mrl=False)
        r = make(tmp_path, stub)
        r.add("seen", "A person waters the plants.")
        await r.flush()
        return r, stub, await r.search("when did I water the plants?")

    r, stub, hits = asyncio.run(go())
    assert hits and r.vecs.shape == (1, 32) and not r.server_cuts
    assert "dimensions" not in stub.calls[-1]


def test_nothing_indexed_means_no_search(tmp_path):
    stub = Stub()
    r = make(tmp_path, stub)
    assert asyncio.run(r.search("what did we say?")) is None and stub.calls == []


def test_a_caption_logged_to_kai_is_also_indexed(tmp_path):
    from voice_agent.memory import Memory

    async def kai(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/vision-caption":
            return httpx.Response(200, json={"caption": "A person waters the plants by the window."})
        return httpx.Response(200, json={"ok": True})

    async def go():
        m = Memory({"url": "http://kai", "interval_s": 1})
        m.client = httpx.AsyncClient(transport=httpx.MockTransport(kai))
        m.recall = make(tmp_path, Stub())
        m.look("ZmFrZQ==")
        for _ in range(50):
            await asyncio.sleep(0.02)
            if not m.busy:
                break
        await m.recall.flush()
        return m.recall

    r = asyncio.run(go())
    assert [(it["kind"], it["text"]) for it in r.items] == [("seen", "A person waters the plants by the window.")]
