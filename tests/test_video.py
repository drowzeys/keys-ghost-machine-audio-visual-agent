"""The video panel's job service (voice_agent/video.py) against a fake render host: no GPU, no network.

    python -m pytest tests/test_video.py -q
"""

import asyncio
import json
from pathlib import Path

import pytest

from voice_agent import video
from voice_agent.backstage import claims


class FakeRenderer(video.Renderer):
    """A render host for the service: each clip takes ``steps`` polls to finish; ``refuse`` is its room's answer."""

    name = "fake"

    def __init__(self, steps=3, refuse=None):
        self.jobs, self.steps, self.refuse, self.posted, self.cancelled = {}, steps, refuse, [], []

    async def room(self, clip):
        return self.refuse

    async def submit(self, clip):
        handle = f"j{len(self.jobs)}"
        self.jobs[handle] = {"polls": 0}
        self.posted.append(dict(clip))
        return handle

    async def poll(self, handle):
        job = self.jobs[handle]
        job["polls"] += 1
        if job["polls"] > self.steps:
            return video.Progress("done", output=f"/renders/{handle}.mp4")
        return video.Progress("rendering", at=job["polls"], of=self.steps, stage="denoising")

    async def fetch(self, output, dest):
        Path(dest).write_bytes(b"fake-mp4-" + output.encode())

    async def cancel(self, handle):
        self.cancelled.append(handle)


def service(tmp_path, renderer, **spec):
    svc = video.VideoService({"dir": str(tmp_path), "poll_s": 0, "ffmpeg": "/nonexistent", "max_clips": 3, **spec},
                             brain=None, renderer=renderer)
    svc.ffmpeg = None
    return svc


def test_single_clip_is_rendered_fetched_and_checkpointed(tmp_path):
    host = FakeRenderer()

    async def go():
        svc = service(tmp_path, host)
        seen = []
        svc.on_news.append(lambda kind, run: seen.append(kind))
        state = await svc.start("a lighthouse in a storm at dusk", "single")
        assert state["state"] == "planning"
        await svc.task
        return svc, seen

    svc, seen = asyncio.run(go())
    snap = svc.snapshot()
    assert snap["state"] == "done" and len(snap["clips"]) == 1
    assert snap["film"] and "/renders" not in json.dumps(snap)  # the renderer's own paths never reach the page
    assert seen == ["clip", "done"]
    run = json.loads((tmp_path / snap["run"] / "run.json").read_text())
    assert run["clips"][0]["state"] == "done" and run["clips"][0]["job"] == "j0"
    assert (tmp_path / snap["run"] / "clip000.mp4").read_bytes() == b"fake-mp4-/renders/j0.mp4"
    clip = host.posted[0]
    assert clip["seconds"] == 5 and "opens_on" not in clip


def test_continue_opens_each_clip_on_the_last(tmp_path):
    host = FakeRenderer(steps=1)

    async def go():
        svc = service(tmp_path, host)
        await svc.start("waves on a black sand beach", "continue", count=3)
        await svc.task
        return svc

    svc = asyncio.run(go())
    assert [c["index"] for c in svc.snapshot()["clips"]] == [0, 1, 2]
    froms = [p.get("opens_on") for p in host.posted]
    assert froms == [None, "/renders/j0.mp4", "/renders/j1.mp4"]
    seeds = [p["seed"] for p in host.posted]
    assert len(set(seeds)) == 3


def test_restart_resumes_the_job_in_flight_without_sending_it_again(tmp_path):
    host = FakeRenderer(steps=50)

    async def first():
        svc = service(tmp_path, host)
        await svc.start("a paper boat on a rain gutter", "single")
        while not (svc.run["clips"] and svc.run["clips"][0].get("job")):
            await asyncio.sleep(0)
        svc.task.cancel()  # the agent goes away mid-render
        try:
            await svc.task
        except asyncio.CancelledError:
            pass

    asyncio.run(first())
    assert len(host.posted) == 1
    host.steps = 1

    async def second():
        svc = service(tmp_path, host)
        await svc.resume()
        await svc.task
        return svc

    svc = asyncio.run(second())
    assert svc.snapshot()["state"] == "done"
    assert len(host.posted) == 1  # followed, not queued again


def test_stop_cancels_the_render(tmp_path):
    host = FakeRenderer(steps=1000)

    async def go():
        svc = service(tmp_path, host)
        await svc.start("a red kite over hills", "continue")
        while not (svc.run["clips"] and svc.run["clips"][0].get("job")):
            await asyncio.sleep(0)
        await svc.stop()
        return svc

    svc = asyncio.run(go())
    assert svc.snapshot()["state"] == "stopped" and host.cancelled == ["j0"]


def test_room_refusal_is_said(tmp_path):
    svc = service(tmp_path, FakeRenderer(refuse=video.Refused("No room for the clip now: it needs 70 GB", video.BUSY)))
    with pytest.raises(video.Refused, match="70 GB"):
        asyncio.run(svc.start("a fox in snow"))
    assert svc.run is None and "70 GB" in svc.snapshot()["message"]


def test_the_service_needs_this_apps_renderer(tmp_path):
    with pytest.raises(ValueError, match="FP4 clip worker"):
        video.VideoService({"dir": str(tmp_path), "renderer": "studio", "url": "http://127.0.0.1:7870"})


def test_voice_requests(tmp_path):
    host = FakeRenderer(steps=1)
    svc = service(tmp_path, host)
    agent = video.VideoAgent("video", {}, svc)
    other = type("Other", (), {"name": "hermes", "label": "Hermes", "kind": "acp"})()
    agents = {"hermes": other, "video": agent}
    for said in ("Boo, make a video of a lighthouse in a storm", "stop the video", "show me the last clip",
                 "make me a short film about a robot", "create an endless video of waves"):
        assert claims(said, agents, other) is agent, said
    for said in ("do you like this movie", "the end of the movie was sad", "what's the weather", "I made a video yesterday"):
        assert claims(said, agents, other) is None, said

    steps = []

    async def go():
        said = await agent.run("make an endless video of waves on a reef\n\n[Do it now ...]", lambda k, v: steps.append((k, v)))
        assert "until you say stop" in said and svc.run["mode"] == "continue" and svc.run["idea"] == "waves on a reef"
        assert "step" in video.said_status(svc.snapshot()) or svc.snapshot()["state"] in ("planning", "rendering")
        assert await agent.run("stop the video", steps.append) == "Stopped the video."
        assert await agent.run("show me the last clip", lambda k, v: steps.append((k, v)))

    asyncio.run(go())


def test_film_of_one_clip_is_the_clip(tmp_path):
    host = FakeRenderer(steps=1)

    async def go():
        svc = service(tmp_path, host)
        await svc.start("a lantern on a pier")
        await svc.task
        return svc, await svc.film()

    svc, path = asyncio.run(go())
    assert Path(path).name == "clip000.mp4"


# ----- asking for a video by voice: found in code, one short reply a turn (synthetic phrasings only) -----

def full_studio():
    """A render host with no room (the governor's refusal, whole for the panel, one short line for the voice)."""
    return FakeRenderer(refuse=video.Refused("No room for the clip now: this clip needs about 58 GB at its peak (give or "
                                             "take 4); this machine has 39 GB for it and keeps 12", video.BUSY))


def agents_with(svc):
    agent = video.VideoAgent("video", {}, svc)
    other = type("Other", (), {"name": "hermes", "label": "Hermes", "kind": "acp"})()
    return agent, {"hermes": other, "video": agent}


@pytest.mark.parametrize("said, idea", [
    ("generate a video of a dragon over a castle", "a dragon over a castle"),
    ("Can you make me a video about a lighthouse in a storm?", "a lighthouse in a storm"),
    ("create a clip of rain on a tin roof", "rain on a tin roof"),
    ("render a video of a paper boat on a river please", "a paper boat on a river"),
    ("show me a video of a red kite over green hills", "a red kite over green hills"),
    ("Boo, make a short film about a robot learning to paint", "a robot learning to paint"),
    ("make an endless video of waves on a reef", "waves on a reef"),
    ("could you generate me a ten second clip of a fox in the snow", "a fox in the snow"),
    ("I want a video of a hot air balloon at dawn", "a hot air balloon at dawn"),
])
def test_requests_route_to_the_video_agent_with_their_idea(tmp_path, said, idea):
    svc = service(tmp_path, FakeRenderer(steps=50))
    agent, agents = agents_with(svc)
    assert claims(said, agents, agents["hermes"]) is agent and claims(said, agents, None) is agent

    async def go():
        spoken, _ = await agent.handle(said, lambda k, v: None)
        await svc.stop()
        return spoken

    assert asyncio.run(go()).startswith("Starting the video")
    assert svc.run["idea"] == idea
    assert svc.run["mode"] == ("continue" if "endless" in said else "single")


@pytest.mark.parametrize("said", ["I watched a video yesterday", "I made a video yesterday", "do you like this movie",
                                  "the end of the movie was sad", "what's the weather", "make a video call to my sister",
                                  "how do I make a video in an editor", "I'm going to make a video for school",
                                  "that film was great", "can you show me how to do this"])
def test_talk_about_videos_is_not_a_request(tmp_path, said):
    svc = service(tmp_path, FakeRenderer())
    agent, agents = agents_with(svc)
    assert claims(said, agents, agents["hermes"]) is None and claims(said, agents, None) is None


def test_no_subject_asks_once_and_takes_the_next_words_as_the_idea(tmp_path):
    svc = service(tmp_path, FakeRenderer(steps=50))
    agent, agents = agents_with(svc)

    async def go():
        for said in ("make a video", "Can you make a video?", "generate a video for me please"):
            assert claims(said, agents, None) is agent, said
            assert await agent.handle(said, lambda k, v: None) == ("What should the video show?",) * 2
            assert svc.run is None
        # the answer: not a request on its own, taken because the agent asked
        assert claims("um, a red kite over green hills", agents, None) is agent
        spoken, _ = await agent.handle("um, a red kite over green hills", lambda k, v: None)
        assert spoken.startswith("Starting the video") and svc.run["idea"] == "a red kite over green hills"
        # answered: the next words are the front brain's again
        assert claims("tell me a joke", agents, None) is None
        await svc.stop()

    asyncio.run(go())


def test_no_subject_then_never_mind_or_a_question(tmp_path):
    svc = service(tmp_path, FakeRenderer())
    agent, agents = agents_with(svc)

    async def go():
        await agent.handle("make a video", lambda k, v: None)
        assert claims("what time is it", agents, None) is None  # a question of their own: the front brain's
        assert claims("never mind", agents, None) is agent
        assert await agent.handle("never mind", lambda k, v: None) == ("Okay, no video then.",) * 2
        assert svc.run is None and claims("a fox in the snow", agents, None) is None

    asyncio.run(go())


def test_refusals_are_short_to_say_and_whole_in_the_panel(tmp_path):
    svc = service(tmp_path, full_studio())
    agent, _ = agents_with(svc)
    spoken, detail = asyncio.run(agent.handle("make a video of a red kite over green hills", lambda k, v: None))
    assert spoken == "The video machine is busy right now, so I can't render yet."
    assert "58 GB" in detail and "58" not in spoken
    assert "58 GB" in svc.snapshot()["message"]  # the panel's status line has the reason


def test_a_video_turn_gets_exactly_one_reply_and_the_brain_stays_quiet(tmp_path):
    server = pytest.importorskip("voice_agent.server")
    svc = service(tmp_path, full_studio())
    agent, _ = agents_with(svc)

    class Brain:
        calls = 0

        async def stream(self, messages, image_png=None):
            Brain.calls += 1
            yield "[happy] I can't create media directly."

    class Voice:
        sample_rate = 24000

        async def stream(self, sentence, name=None):
            yield b"\0\0"

    engines = type("E", (), {})()
    engines.profile = {"voice": {}, "brain": {"system": "You are a friendly voice assistant."}}
    engines.brain, engines.eyes, engines.memory, engines.lookup = Brain(), None, None, None
    engines.agents, engines.voices, engines.video = {"video": agent}, {"full": Voice()}, svc
    talk = server.Conversation.__new__(server.Conversation)
    talk.e, talk.engine, talk.voice_name, talk.name, talk.agent = engines, "full", None, "Boo", None
    talk.history = [{"role": "system", "content": "You are a friendly voice assistant."}]
    talk.held, talk.asking, talk.frame, talk.frame_at, talk.visual, talk.hands = None, None, None, 0.0, None, []
    talk.speaking, talk.heard = False, {"yes": True}
    events = []

    async def record(event):
        events.append(event)

    talk.emit = talk.send = record
    asyncio.run(talk.turn(None, "make a video of a red kite over green hills"))
    replies = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "reply"]
    assert Brain.calls == 0
    assert " ".join(replies) == "The video machine is busy right now, so I can't render yet.", events  # one reply
    assert talk.history[-1] == {"role": "assistant", "content": " ".join(replies)}
    assert any(e.get("type") == "task" and "58 GB" in e.get("text", "") for e in events if isinstance(e, dict))
