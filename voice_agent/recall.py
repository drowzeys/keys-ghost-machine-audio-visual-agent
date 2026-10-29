"""Memory search by meaning (optional): what the camera saw (Eyes of Kai's captions) and what was said in conversation,
embedded by any OpenAI-compatible ``/v1/embeddings`` server and searched by cosine when a turn asks about the past.
EmbeddingGemma-2 (Google DeepMind) on TensorFold is the reference model; any embeddings server works.

Kept on this machine only (``path``), with the captions' retention: 1 to 7 days, older entries erased from memory and
from the file. A question's embedding waits at most ``timeout_ms`` (40 ms by default); past that, or with the server
down, the turn goes on without it and Boo answers from the caption log's time window as before."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np

from .memory import PAST

# questions about something said or seen before, beyond the caption log's "what was I doing ... ago"
ASKS = re.compile(r"\b(when did (?:i|we|you)|what did (?:we|i|you) (?:say|talk|discuss|decide|agree|plan)|"
                  r"did (?:i|we) (?:tell|mention|say|talk)|have (?:i|we) (?:told|mentioned|talked)|remind me|"
                  r"last time (?:i|we|you)|where did i|what was (?:i|that|the)|do you remember|you said)\b", re.I)
QUERY = "task: search result | query: {}"
DOCUMENT = "title: none | text: {}"


def asks(text: str) -> bool:
    """A turn that asks about the past: the caption log's own pattern or a question about what was said or done."""

    return bool(PAST.search(text) or ASKS.search(text))


class Recall:
    def __init__(self, spec: dict, client: httpx.AsyncClient | None = None) -> None:
        self.url = spec["url"].rstrip("/") + "/embeddings"
        self.model = spec.get("model")
        self.dimensions = int(spec.get("dimensions", 256))  # an MRL cut: smaller vectors, the same ranking here
        self.top_k = int(spec.get("top_k", 6))
        self.min_score = float(spec.get("min_score", 0.25))
        self.timeout_s = float(spec.get("timeout_ms", 40)) / 1000
        self.days = min(7, max(1, int(spec.get("days", 2))))
        self.path = Path(os.path.expanduser(spec.get("path", "~/.keys-ghost-machine/recall.jsonl")))
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(10, connect=2))
        self.server_cuts = True  # the server truncates and renormalises (`dimensions`); false after it refuses that
        self.items: list[dict] = []  # {"ts", "kind", "text"}
        self.vecs = np.zeros((0, self.dimensions), np.float32)
        self.pending: list[dict] = []
        self.flushing: asyncio.Task | None = None
        self.lock: asyncio.Lock | None = None  # made on first use, inside the running loop
        self.load()

    # ------------------------------------------------------------------ the index
    def load(self) -> None:
        """The kept entries younger than the retention; older lines are dropped and the file rewritten without them."""

        if not self.path.exists():
            return
        cutoff = time.time() - self.days * 86400
        items, vecs, stale = [], [], False
        for line in self.path.read_text().splitlines():
            try:
                o = json.loads(line)
                v = np.frombuffer(base64.b64decode(o["vec"]), np.float16).astype(np.float32)
            except (ValueError, KeyError):
                stale = True
                continue
            if o["ts"] < cutoff or v.shape[0] != self.dimensions:
                stale = True
                continue
            items.append({"ts": o["ts"], "kind": o["kind"], "text": o["text"]})
            vecs.append(v)
        self.items = items
        self.vecs = np.stack(vecs) if vecs else np.zeros((0, self.dimensions), np.float32)
        if stale:
            self.rewrite()

    def rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for it, v in zip(self.items, self.vecs):
                f.write(self.line(it, v))
        os.replace(tmp, self.path)

    @staticmethod
    def line(it: dict, v: np.ndarray) -> str:
        return json.dumps({**it, "vec": base64.b64encode(v.astype(np.float16).tobytes()).decode()}) + "\n"

    def add(self, kind: str, text: str, ts: float | None = None) -> None:
        """Index one caption or line in the background (batched, a fifth of a second after the first arrives)."""

        text = (text or "").strip()
        if not text:
            return
        self.pending.append({"ts": ts if ts is not None else time.time(), "kind": kind, "text": text[:500]})
        if self.flushing is None or self.flushing.done():
            self.flushing = asyncio.create_task(self.flush(0.2))

    async def flush(self, after: float = 0) -> None:
        """Embed and keep everything pending; one flush at a time, so a caller's flush returns only once every
        batch already taken by a background flush is kept too (and the file's lines stay in order)."""

        if after:
            await asyncio.sleep(after)
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            while self.pending:
                batch, self.pending = self.pending[:32], self.pending[32:]
                try:
                    vecs = await self.embed([DOCUMENT.format(it["text"]) for it in batch])
                except (httpx.HTTPError, ValueError, KeyError):
                    return  # the server is down: these are lost, the caption log still has the captions
                self.items.extend(batch)
                self.vecs = np.concatenate([self.vecs, vecs]) if len(self.vecs) else vecs
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    for it, v in zip(batch, vecs):
                        f.write(self.line(it, v))

    async def embed(self, texts: list[str]) -> np.ndarray:
        body = {"input": texts}
        if self.server_cuts:
            body["dimensions"] = self.dimensions
        if self.model:
            body["model"] = self.model
        r = await self.client.post(self.url, json=body)
        if r.status_code == 400 and self.server_cuts:  # a server without MRL: full vectors, cut here
            self.server_cuts = False
            return await self.embed(texts)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        v = np.asarray([d["embedding"] for d in data], np.float32)[:, : self.dimensions]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def keep(self, days: int) -> None:
        """The retention the page sets for the captions (1 to 7 days): older entries are erased now."""

        self.days = min(7, max(1, int(days)))
        self.prune()

    def prune(self, now: float | None = None) -> None:
        cutoff = (now or time.time()) - self.days * 86400
        keep = [i for i, it in enumerate(self.items) if it["ts"] >= cutoff]
        if len(keep) == len(self.items):
            return
        self.items = [self.items[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else np.zeros((0, self.dimensions), np.float32)
        self.rewrite()

    # ------------------------------------------------------------------ the search
    async def search(self, question: str) -> list[dict] | None:
        """The entries nearest the question, oldest first, or None when nothing is indexed, the server is down or
        slower than ``timeout_ms``: the turn never waits longer than that for this."""

        self.prune()
        if not self.items:
            return None
        try:
            q = await asyncio.wait_for(self.embed([QUERY.format(question)]), self.timeout_s)
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError, KeyError):
            return None
        scores = self.vecs @ q[0]
        best = np.argsort(-scores)[: self.top_k]
        hits = [{**self.items[i], "score": float(scores[i])} for i in best if scores[i] >= self.min_score]
        return sorted(hits, key=lambda h: h["ts"]) or None


def lines(hits: list[dict], now: float | None = None) -> str:
    """The hits as dated lines for the brain: "Tue 14:03, seen: ...", with the time it is now first."""

    today = datetime.fromtimestamp(now or time.time()).date()
    out = [f"(It is {datetime.fromtimestamp(now or time.time()).strftime('%a %H:%M')} now.)"]
    for h in hits:
        t = datetime.fromtimestamp(h["ts"])
        when = t.strftime("%H:%M") if t.date() == today else t.strftime("%a %H:%M")
        out.append(f"{when}, {h['kind']}: {h['text']}")
    return "\n".join(out)
