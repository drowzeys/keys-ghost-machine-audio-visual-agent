"""Phase 4's voice loop around an AutoPilot film (voice_agent/video.py, voice_agent/server.py, gma/ap/service.py): a
spoken request becomes a film with the brain's seed, the film is steered by voice in code (and everyday sentences with
the same verbs are left alone), the spoken updates, and the Director's place behind voice turns. No model, no GPU."""

import asyncio
import json
import time

import pytest

from voice_agent import video
from voice_agent.server import news_line


class FakeService:
    """The video panel's service as the video agent sees it, an AutoPilot film at work."""

    def __init__(self, state="running"):
        self.streams = False
        self.autopilot = object()
        self.run = {"id": "r1", "mode": "autopilot", "state": state, "idea": "a lighthouse keeper lights the lamp as a storm rolls in",
                    "autopilot": {"seed": {"world": "A rocky coast", "characters": [{"name": "Elias"}]},
                                  "segments": [{"beat": "Elias climbs the lighthouse stairs"}], "settings": {"seconds": 10}}}
        self.calls, self.started = [], []

    async def autopilot_voice(self, action, text="", pieces=1):
        self.calls.append((action, text, pieces))
        return {}

    def check_autopilot(self, idea, seed=None):
        if self.run and self.run.get("state") not in video.ENDED + ("stopped",):
            raise video.Refused("A video is already being made. Say stop the video first.")

    async def start(self, idea, mode="single", count=None, seconds=None, seed=None, words=None):
        self.started.append((idea, mode, words))
        return {}

    def snapshot(self):
        return {"run": None, "clips": [], "state": "idle"}

    async def stop(self):
        self.calls.append(("stop-all", "", 1))
        return {"state": "stopped"}


def handle(agent, text):
    return asyncio.run(agent.handle(text, lambda kind, value: None))


@pytest.mark.parametrize("said,action,text", [
    ("Make it rain.", "steer", "make it rain"),
    ("Okay, now make it rain over the sea", "steer", "make it rain over the sea"),
    ("add a dog running along the beach", "steer", "add a dog running along the beach"),
    ("make the keeper wave at the camera", "steer", None),
    ("have Elias light the lantern", "steer", None),
    ("Keep going.", "extend", ""),
    ("twenty seconds more", "extend", ""),
    ("one more piece please", "extend", ""),
    ("Redo that.", "redo", ""),
    ("can you do that again", "redo", ""),
    ("go sharper", "sharp", ""),
    ("New scene: the harbour at dawn.", "scene", "the harbour at dawn"),
    ("cut to the village below", "scene", "the village below"),
    ("faster, more action", "pace", None),
    ("Stop.", "pause", ""),
    ("stop the film", "pause", ""),
    ("stop now", "cancel", ""),
    ("cancel the render", "cancel", ""),
])
def test_film_steers_are_mapped_in_code(said, action, text):
    agent = video.VideoAgent("video", {}, FakeService())
    if said.startswith("add a dog"):
        agent.video_at = time.monotonic()  # no film or picture word: it is the film's because the film was just talked about
    assert agent.wants(said), said
    got = agent.film_action(said.lower())
    assert got[0] == action and (text is None or got[1] == text), got


@pytest.mark.parametrize("said", [
    "add milk to my shopping list",
    "add a reminder for 5 pm",
    "make it a double espresso",
    "make it quick, I'm late",
    "make it six o'clock instead",
    "bring in the laundry",
    "can you talk faster",
    "make it louder",
    "give him a call tomorrow",
    "turn it off at nine pm",
    "What's a good name for a cat?",
    "change the thermostat to twenty",
])
def test_everyday_sentences_are_not_the_films(said):
    agent = video.VideoAgent("video", {}, FakeService())
    agent.video_at = 0.0  # nobody spoke about the film lately
    assert agent.film_action(said.lower()) is None and not agent.wants(said), said


def test_a_bare_steer_is_the_films_right_after_talk_about_it():
    agent = video.VideoAgent("video", {}, FakeService())
    agent.video_at = 0.0
    assert agent.film_action("add a second boat") is None  # no film word, no picture word, not the film's words
    agent.video_at = time.monotonic()  # the person just spoke about the film
    assert agent.film_action("add a second boat")[0] == "steer"
    assert agent.film_action("add milk to my list") is None  # an errand stays an errand


def test_a_bare_stop_right_after_cutting_boo_off_is_for_boo():
    agent = video.VideoAgent("video", {}, FakeService())
    agent.barged_at = time.monotonic()
    assert agent.film_action("stop") is None and agent.film_action("stop the film")[0] == "pause"


def test_no_film_no_steering():
    svc = FakeService()
    svc.run = None
    agent = video.VideoAgent("video", {}, svc)
    assert agent.film_action("make it rain") is None and agent.film_action("stop") is None


def test_a_steer_is_handed_to_autopilot_with_one_short_line():
    svc = FakeService()
    agent = video.VideoAgent("video", {}, svc)
    said, detail = handle(agent, "Make it rain.")
    assert said == "Okay, the next part will show that." and svc.calls == [("steer", "make it rain", 1)]
    said, _ = handle(agent, "twenty seconds more")
    assert said == "Okay, 20 more seconds." and svc.calls[-1] == ("extend", "", 2)
    assert handle(agent, "stop now")[0] == "Stopped the film." and svc.calls[-1][0] == "cancel"


def test_a_spoken_request_starts_an_autopilot_film_in_the_background(monkeypatch):
    monkeypatch.setattr(video, "MERGE_S", 0.2)
    svc = FakeService(state="done")
    agent = video.VideoAgent("video", {}, svc)
    agent.video_at = 0.0

    async def go():
        said = await agent.handle("Make a video of a lighthouse keeper lighting the lamp as a storm rolls in.", lambda k, v: None)
        assert not svc.started  # held a moment: more words may follow
        await asyncio.sleep(0.4)
        return said
    said, detail = asyncio.run(go())
    assert said.startswith("Okay, I'm making that film") and "lighthouse keeper" in detail
    assert svc.started and svc.started[0][1] == "autopilot" and "storm rolls in" in svc.started[0][2]


def test_words_that_follow_a_request_at_once_are_the_rest_of_it(monkeypatch):
    monkeypatch.setattr(video, "MERGE_S", 0.3)
    svc = FakeService(state="done")
    agent = video.VideoAgent("video", {}, svc)

    async def go():
        first = await agent.handle("Make a video of a lighthouse keeper.", lambda k, v: None)
        assert agent.wants("lighting the lamp as a storm rolls in.")
        second = await agent.handle("lighting the lamp as a storm rolls in.", lambda k, v: None)
        await asyncio.sleep(0.6)
        return first, second
    first, second = asyncio.run(go())
    assert first[0].startswith("Okay") and second[0] == ""  # the rest of the request: nothing said over the person
    assert len(svc.started) == 1 and svc.started[0][0] == "a lighthouse keeper lighting the lamp as a storm rolls in"
    assert "storm rolls in" in svc.started[0][2] and not agent.wants("lighting the lamp")  # the window has closed


def test_a_request_while_a_film_runs_is_refused_in_one_line():
    svc = FakeService(state="running")
    agent = video.VideoAgent("video", {}, svc)
    said, _ = handle(agent, "make a video of a red kite")
    assert "already" in said.lower() or "stop" in said.lower()


def test_the_length_is_read_from_the_words():
    assert video.length_in("make a thirty second video of a kite") == {"segments": 3}
    assert video.length_in("make a 2 minute film") == {"segments": 12}
    assert video.length_in("make an endless video of the sea") == {"infinity": True}
    assert video.length_in("make a video of a kite") == {}


def test_spoken_updates_of_a_film():
    run = {"mode": "autopilot", "count": 6}
    assert news_line("clip", run, 1, 3) == "The first part of the film is ready."
    assert news_line("clip", run, 2, 3) is None and news_line("clip", run, 3, 3) == "Part 3 of the film is ready."
    assert news_line("clip", run, 6, 3) is None  # the last part: "done" says it
    assert news_line("retake", run, 2, 3).startswith("That part didn't hold up")
    assert news_line("done", run, 6, 3) == "The film is finished, 6 parts."
    assert news_line("retake", {"mode": "single"}, 1, 3) is None


# ---------- the seed from the words (gma/ap/service.py) ----------

def test_the_brain_writes_the_seed_and_the_words_give_the_length(tmp_path):
    from gma.ap import autopilot
    from gma.ap.service import AutoPilotService

    class Svc:
        root = tmp_path

    asked = []
    service = AutoPilotService.__new__(AutoPilotService)
    service.spec, service.brain = {"settings": {"segments": 3}}, object()

    def director(system, prompt, think):
        asked.append((system, prompt, think))
        return json.dumps({"title": "The Lamp", "world": "A rocky North Sea coast, 1890s", "situation": "Elias climbs to light the lamp",
                           "characters": [{"name": "Elias", "look": "a weathered man of sixty", "wardrobe": "a navy wool coat and oilskin cap"}],
                           "style": "35 mm film, storm light", "pace": "calm", "opening_keyframe": "Elias at the foot of the lighthouse, dusk"})
    service.director = director
    made = service.seed_from_words("a lighthouse keeper lights the lamp", "Make a video of a lighthouse keeper lighting the lamp.")
    assert asked[0][2] == 0  # no reasoning: a short answer
    assert made["seed"]["characters"][0]["wardrobe"] == "a navy wool coat and oilskin cap"
    assert made["seed"]["opening_keyframe"].startswith("Elias at the foot") and made["settings"]["style"].startswith("35 mm")
    assert made["name"] == "The Lamp"
    service.director = lambda s, p, t: "no json here"
    assert service.seed_from_words("x", "make a video of x") is None
    begun = []
    autopilot.attach(tmp_path / "video")
    real = autopilot.begin
    autopilot.begin = lambda run: begun.append(run) or run
    try:
        service.director = director
        run = service.create("a lighthouse keeper lights the lamp", words="Make an endless video of a lighthouse keeper lighting the lamp.")
    finally:
        autopilot.begin = real
    assert run["settings"]["infinity"] is True and run["seed"]["characters"][0]["name"] == "Elias" and run["name"] == "The Lamp"
    assert run["seed_made"]["by"] == "brain"


def test_the_director_yields_to_a_voice_turn_and_is_asked_again():
    from gma.ap.service import AutoPilotService
    from gma.lanes import Lanes

    calls = []

    class Brain:
        async def complete(self, messages, **settings):
            calls.append(settings)
            await asyncio.sleep(0.3 if len(calls) == 1 else 0.01)
            return "the plan"

    loop = asyncio.new_event_loop()
    import threading
    threading.Thread(target=loop.run_forever, daemon=True).start()
    lanes = Lanes(quiet_s=0.05)
    service = AutoPilotService.__new__(AutoPilotService)
    service.brain, service.lanes, service.loop = Brain(), lanes, loop

    def barge():
        time.sleep(0.1)

        async def turn():
            async with lanes.voice():  # a voice turn: the Director's request in flight is cut
                await asyncio.sleep(0.05)
        asyncio.run_coroutine_threadsafe(turn(), loop).result(5)
    threading.Thread(target=barge).start()
    assert service.director("system", "prompt", 0) == "the plan"
    assert len(calls) == 2 and lanes.preempted == 1
    loop.call_soon_threadsafe(loop.stop)


def test_a_voice_turn_while_the_brain_is_held_is_answered_by_the_fallback():
    from gm_core.engine import Held
    from gma.brain import HeldFallback

    class Main:
        model = "bolt"
        held = True

        async def stream(self, messages, image_png=None):
            if self.held:
                raise Held("Bolt is unloaded while the DiT pass has the memory", "My brain is paused while I render.")
            yield "from bolt"

        async def complete(self, messages, **settings):
            return "plan"

    class Small:
        async def stream(self, messages, image_png=None):
            yield "from the small one"

    brain = HeldFallback(Main(), Small())

    async def said():
        return [p async for p in brain.stream([])]
    assert asyncio.run(said()) == ["from the small one"] and brain.answered_by_fallback == 1 and brain.model == "bolt"
    brain.main.held = False
    assert asyncio.run(said()) == ["from bolt"]
    brain.away = lambda: True  # out for a render (released, not loaded): never loaded for one voice turn
    assert asyncio.run(said()) == ["from the small one"] and brain.answered_by_fallback == 2
    assert asyncio.run(brain.complete([])) == "plan"


def test_a_retake_is_told_once_when_it_is_sent_back():
    one = {"autopilot": {"segments": [{"status": "retake", "takes": [{"passed": False}]}]}}
    two = {"autopilot": {"segments": [{"status": "retake", "takes": [{"passed": False}, {"passed": False}]}]}}
    kept = {"autopilot": {"segments": [{"status": "done", "takes": [{"passed": False}, {"passed": False}, {"passed": False}]}]}}
    passed = {"autopilot": {"segments": [{"status": "done", "takes": [{"passed": False}, {"passed": True}]}]}}
    assert video.retakes_of(one) == 1 and video.retakes_of(two) == 2
    assert video.retakes_of(kept) == 2  # the third take joined the film: no "making it again" for it
    assert video.retakes_of(passed) == 1


def test_the_fallback_brain_reads_only_the_latest_turns():
    from gma.brain import FALLBACK_TURNS, HeldFallback

    seen = []

    class Small:
        async def stream(self, messages, image_png=None):
            seen.append(messages)
            yield "ok"

    class Main:
        async def stream(self, messages, image_png=None):
            yield "never"

    brain = HeldFallback(Main(), Small(), away=lambda: True)
    history = [{"role": "system", "content": "be brief"}] + [{"role": "user", "content": str(i)} for i in range(40)]

    async def go():
        return [p async for p in brain.stream(history)]
    assert asyncio.run(go()) == ["ok"]
    assert seen[0][0]["content"] == "be brief" and len(seen[0]) == 1 + FALLBACK_TURNS and seen[0][-1]["content"] == "39"


def test_a_cut_turn_ends_its_brain_request_at_once():
    """A turn cut mid-reply closes the brain's stream (aclosing): its request is not left "in flight", which kept Bolt
    resident beside the DiT pass on the Spark latency test."""
    import contextlib

    from gma.brain import HeldFallback

    flight = [0]

    class Main:
        async def stream(self, messages, image_png=None):
            flight[0] += 1
            try:
                for i in range(100):
                    yield f"word{i} "
                    await asyncio.sleep(0)
            finally:
                flight[0] -= 1

    brain = HeldFallback(Main(), Main(), away=lambda: False)

    async def cut():
        async with contextlib.aclosing(brain.stream([])) as stream:
            async for piece in stream:
                break  # the person barged in
        return flight[0]
    assert asyncio.run(cut()) == 0


def test_a_turn_cut_while_its_brain_loads_leaves_nothing_in_flight():
    import threading as _threading

    from gm_core.engine import Engine
    from gma.registry import Registry

    class Slow(Engine):
        def __init__(self):
            super().__init__("bolt", {"role": "brain", "argv": ["true"]})
            self.gate = _threading.Event()

        def try_fast(self):
            return False

        def acquire(self, stopped=lambda: False, wait=True):
            self.gate.wait(5)  # loading
            with self.mutex:
                self.in_flight += 1
            return "http://127.0.0.1:1"

    registry = Registry({"engines": {}})
    one = Slow()
    registry.engines["bolt"] = one

    async def go():
        async def turn():
            async with registry.use("bolt", wait=False):
                await asyncio.sleep(10)
        task = asyncio.create_task(turn())
        await asyncio.sleep(0.1)
        task.cancel()  # the person spoke again while the brain loaded
        try:
            await task
        except asyncio.CancelledError:
            pass
        one.gate.set()
        await asyncio.sleep(0.3)
        return one.in_flight
    assert asyncio.run(go()) == 0
