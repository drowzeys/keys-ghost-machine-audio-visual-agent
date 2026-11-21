"""The brain lanes (gma/lanes.py): voice turns first, the Director yields at barge-in and asks again once quiet;
and a spoken refusal in a real conversation turn (voice_agent/server.py) with fake engines."""

import asyncio
import json
from pathlib import Path

import pytest

from gm_core.engine import Held
from gma.credits import CREDITS
from gma.lanes import Lanes, Yielded

ROOT = Path(__file__).resolve().parent.parent


def test_director_runs_when_quiet():
    lanes = Lanes(quiet_s=0)

    async def go():
        async def call():
            return "plan"
        return await lanes.director(call)

    assert asyncio.run(go()) == "plan"


def test_director_yields_at_barge_in_and_asks_again_once_quiet():
    lanes = Lanes(quiet_s=0.05)
    calls = []

    async def call():
        calls.append(len(calls))
        await asyncio.sleep(0.3 if len(calls) == 1 else 0.01)  # the first is cut midway
        return f"answer {len(calls)}"

    async def go():
        director = asyncio.create_task(lanes.director(call))
        await asyncio.sleep(0.1)
        assert lanes.status()["director_in_flight"] == 1
        async with lanes.voice():  # someone speaks: the director's request is cancelled at once
            assert lanes.preempted == 1
            await asyncio.sleep(0.1)
            assert not director.done()  # and it waits while the voice turn runs
        return await director

    assert asyncio.run(go()) == "answer 2" and calls == [0, 1]


def test_barge_in_preempts_and_the_busy_conversation_holds_the_director():
    talking = [True]
    lanes = Lanes(busy=lambda: talking[0], quiet_s=0)

    async def go():
        async def call():
            return "late"
        task = asyncio.create_task(lanes.director(call))
        await asyncio.sleep(0.1)
        assert not task.done()  # busy: never starts
        talking[0] = False
        return await task

    assert asyncio.run(go()) == "late"
    assert lanes.preempt("barge-in") == 0 and lanes.last_why == "barge-in"


def test_director_gives_up_after_too_many_yields_and_a_cancelled_caller_stops_it():
    lanes = Lanes(quiet_s=0)

    async def cut_every_time():
        async def call():
            await asyncio.sleep(1)

        task = asyncio.create_task(lanes.director(call, retries=2))
        for _ in range(3):
            await asyncio.sleep(0.05)
            lanes.preempt()
        with pytest.raises(Yielded):
            await task

    asyncio.run(cut_every_time())

    async def caller_cancelled():
        started = asyncio.Event()

        async def call():
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(lanes.director(call))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lanes.director_tasks

    asyncio.run(caller_cancelled())


def test_video_planner_asks_on_the_director_lane():
    from voice_agent import video

    lanes = Lanes(quiet_s=0)
    planner = video.Planner({"url": "http://127.0.0.1:9/v1", "model": "m"}, lanes=lanes)
    seen = []

    async def post(prompt, max_tokens):
        seen.append(lanes.status()["director_in_flight"])
        return "A fox runs through deep snow at dusk, the camera tracking low beside it."

    planner.post = post
    found = asyncio.run(planner.first("a fox", 5))
    assert seen == [1] and found["prompt"].startswith("A fox")


# ---------- a refusal in a real turn is spoken shortly ----------

class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def send_bytes(self, data):
        self.sent.append(data)


class HeldBrain:
    async def stream(self, messages, image_png=None):
        raise Held("Nemotron 3.5 Bolt is unloaded while a render (sharp) has the memory",
                   "My brain is paused while I render. Give me a moment.")
        yield ""  # noqa: unreachable - an async generator

    async def warm(self, messages):
        return None


class FakeVoice:
    sample_rate = 24000
    keep_warm = False

    def __init__(self):
        self.said = []

    async def stream(self, text, voice=None):
        self.said.append(text)
        yield b"\x00\x01" * 240


def test_a_held_brain_turn_speaks_one_short_line(monkeypatch):
    from voice_agent import server

    monkeypatch.setattr(server, "Vad", lambda path: None)
    voice = FakeVoice()

    class FakeEngines:
        profile = {"brain": {"system": "You are Boo."}, "voice": {"voice": None}, "turn": {"keep_warm": False}}
        vad_path = ""
        video = None
        memory = None
        eyes = None
        lookup = None
        agents = {}
        brain = HeldBrain()
        voices = {"full": voice}
        lanes = Lanes()
        pocket = recall = d1 = None  # Boo's optional parts, not in this profile

    async def go():
        talk = server.Conversation(FakeSocket(), FakeEngines())
        await talk.start_turn(text="tell me a joke")
        await talk.reply
        return talk

    talk = asyncio.run(go())
    events = [e for e in talk.ws.sent if isinstance(e, dict)]
    assert any(e["type"] == "error" and e.get("refused") for e in events)
    assert voice.said == ["My brain is paused while I render.", "Give me a moment."] or \
        " ".join(voice.said) == "My brain is paused while I render. Give me a moment."
    assert any(isinstance(e, bytes) for e in talk.ws.sent)  # spoken, not only written
    assert [m["role"] for m in talk.history] == ["system"]  # neither the question nor the refusal is kept


# ---------- credits: every name in the page's list is in CREDITS.md ----------

def test_every_credit_is_in_credits_md_and_the_page_shows_them():
    text = (ROOT / "CREDITS.md").read_text()
    missing = [c["name"] for c in CREDITS if c["name"] not in text]
    assert not missing, missing
    page = (ROOT / "voice_agent" / "static" / "index.html").read_text()
    assert 'id="credits"' in page and "/api/credits" in page
