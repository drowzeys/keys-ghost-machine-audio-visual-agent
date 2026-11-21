"""d1-2B assist (voice_agent/d1.py) against a stub /v1/systemone: routing, low confidence, a slow or down server,
permission replies."""

import asyncio
import json

import httpx
import pytest

from voice_agent import d1
from voice_agent.memory import Memory


def stub(answer, delay=0.0, fail=None, seen=None):
    async def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if fail:
            raise fail
        await asyncio.sleep(delay)
        q = body["questions"]["q"]
        choice, confidence = answer(body["state"], q)
        probs = {k: (confidence if k == choice else (1 - confidence) / (len(q["criteria"]) - 1)) for k in q["criteria"]}
        return httpx.Response(200, json={"answers": {"q": {"type": "choice", "choice": choice, "confidence": confidence,
                                                             "probabilities": probs}}})
    return httpx.MockTransport(handle)


def make(answer=lambda s, q: ("chat", 0.99), **kw):
    spec = {"url": "http://d1.test", "threshold": 0.8, "timeout_ms": kw.pop("timeout_ms", 60)}
    return d1.D1(spec, transport=stub(answer, **kw))


ALL = {"camera", "side", "fingers", "fact", "task", "memory", "chat"}


def run(coro):
    return asyncio.run(coro)


def test_a_confident_route_is_taken():
    seen = []
    assist = make(lambda s, q: ("camera", 0.95), seen=seen)
    assert run(assist.route("how do I look today?", ALL)) == "camera"
    assert seen[0]["questions"]["q"]["type"] == "choice" and seen[0]["state"] == "how do I look today?"
    assert assist.last_ms is not None


def test_only_the_routes_this_boo_can_act_on_are_offered():
    seen = []
    assist = make(lambda s, q: ("memory", 0.9), seen=seen)
    assert run(assist.route("what was on my desk an hour back?", {"memory", "fact", "chat"})) == "memory"
    assert set(seen[0]["questions"]["q"]["criteria"]) == {"memory", "fact", "chat"}


def test_chat_and_low_confidence_leave_the_turn_as_it_was():
    assert run(make(lambda s, q: ("chat", 0.99)).route("tell me a joke", ALL)) is None
    assert run(make(lambda s, q: ("task", 0.6)).route("could you sort that out", ALL)) is None


def test_a_slow_server_gives_up_at_the_deadline():
    assist = make(lambda s, q: ("camera", 0.99), delay=0.3, timeout_ms=60)
    loop_start = asyncio.run(_timed(assist.route("how do I look?", ALL)))
    got, seconds = loop_start
    assert got is None and seconds < 0.2 and assist.last_ms is None


async def _timed(coro):
    t = asyncio.get_running_loop().time()
    got = await coro
    return got, asyncio.get_running_loop().time() - t


def test_a_down_server_changes_nothing():
    assist = make(fail=httpx.ConnectError("refused"))
    assert run(assist.route("how do I look?", ALL)) is None
    assert run(assist.permission("sure thing")) is None


def test_a_bad_answer_changes_nothing():
    async def handle(request):
        return httpx.Response(200, json={"answers": {"q": {"type": "choice"}}})
    assist = d1.D1({"url": "http://d1.test"}, transport=httpx.MockTransport(handle))
    assert run(assist.route("how do I look?", ALL)) is None


def test_permission_replies():
    table = {"go on then": ("allow_once", 0.93), "yes and stop asking me": ("allow_session", 0.9),
             "hold off on that": ("deny", 0.88), "what's the weather?": ("other", 0.97)}
    assist = make(lambda s, q: table[s.split('said: "')[1].rstrip('"')])
    assert run(assist.permission("go on then")) == "allow_once"
    assert run(assist.permission("yes and stop asking me")) == "allow_session"
    assert run(assist.permission("hold off on that")) == "deny"
    assert run(assist.permission("what's the weather?")) is None


def test_switches_turn_parts_off():
    spec = {"url": "http://d1.test", "route": False, "permission": False}
    assist = d1.D1(spec, transport=stub(lambda s, q: ("camera", 0.99)))
    assert run(assist.route("how do I look?", ALL)) is None and run(assist.permission("go on")) is None


def test_memory_answers_a_forced_question_about_the_past():
    got = []

    async def kai(request):
        got.append(request.url.params["from"])
        return httpx.Response(200, json={"ok": True, "observations": [{"ts": 1_000_000, "caption": "a mug on a desk"}]})
    m = Memory({"url": "http://kai.test"})
    m.client = httpx.AsyncClient(transport=httpx.MockTransport(kai))
    assert run(m.past("where's my blue mug got to")) is None  # the pattern alone does not read it as the past
    assert "a mug on a desk" in run(m.past("where's my blue mug got to", force=True))
    assert got


def test_a_task_needs_more_confidence_than_other_routes():
    assert run(make(lambda s, q: ("task", 0.82)).route("how do I make a video in an editor", ALL)) is None
    assert run(make(lambda s, q: ("task", 0.9)).route("send the report to the team channel", ALL)) == "task"
    assert run(make(lambda s, q: ("camera", 0.82)).route("what's behind me?", ALL)) == "camera"
    spec = {"url": "http://d1.test", "thresholds": {"task": 0.95}}
    assert run(d1.D1(spec, transport=stub(lambda s, q: ("task", 0.9))).route("send it", ALL)) is None
