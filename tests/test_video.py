"""The video panel's job service (voice_agent/video.py) against a fake Studio: no GPU, no network.

    python -m pytest tests/test_video.py -q
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from voice_agent import video
from voice_agent.backstage import claims


class FakeStudio:
    """The few Studio routes the renderer uses; each job takes ``steps`` polls to finish."""

    def __init__(self, steps=3, home="/studio"):
        self.jobs, self.steps, self.home, self.posted, self.cancelled = {}, steps, home, [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/state":
            return httpx.Response(200, json={"home": self.home})
        if path == "/api/estimate":
            return httpx.Response(200, json={"seconds": 60, "memory": {"fits": True}})
        if path == "/api/jobs" and request.method == "POST":
            body = json.loads(request.content)
            self.posted.append(body)
            job_id = f"j{len(self.jobs)}"
            self.jobs[job_id] = {"id": job_id, "polls": 0, "kind": body["kind"], "params": body["params"]}
            return httpx.Response(200, json={"id": job_id, "status": "queued"})
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            self.cancelled.append(path.split("/")[3])
            return httpx.Response(200, json={})
        if path.startswith("/api/jobs/"):
            job = self.jobs[path.split("/")[3]]
            job["polls"] += 1
            done = job["polls"] > self.steps
            ext = "png" if job["kind"] == "scout" else "mp4"
            return httpx.Response(200, json={
                "id": job["id"], "status": "done" if done else "running", "estimate": 30, "elapsed": job["polls"],
                "progress": None if done else {"stage": "picture", "at": job["polls"], "of": self.steps},
                "outputs": [f"{self.home}/renders/{job['id']}.{ext}"] if done else []})
        if path.startswith("/files/renders/"):
            return httpx.Response(200, content=b"fake-mp4-" + path.encode())
        return httpx.Response(404, json={"error": "no route " + path})


def service(tmp_path, studio, **spec):
    renderer = video.StudioRenderer({"url": "http://studio.test"}, transport=httpx.MockTransport(studio))
    svc = video.VideoService({"dir": str(tmp_path), "render_host": "remote", "poll_s": 0, "ffmpeg": "/nonexistent",
                              "max_clips": 3, **spec}, brain=None, renderer=renderer)
    svc.ffmpeg = None
    return svc


def test_single_clip_is_rendered_fetched_and_checkpointed(tmp_path):
    studio = FakeStudio()

    async def go():
        svc = service(tmp_path, studio)
        seen = []
        svc.on_news.append(lambda kind, run: seen.append(kind))
        state = await svc.start("a lighthouse in a storm at dusk", "single")
        assert state["state"] == "planning"
        await svc.task
        return svc, seen

    svc, seen = asyncio.run(go())
    snap = svc.snapshot()
    assert snap["state"] == "done" and len(snap["clips"]) == 1
    assert snap["film"] and "/studio" not in json.dumps(snap)  # the Studio's own paths never reach the page
    assert seen == ["clip", "done"]
    run = json.loads((tmp_path / snap["run"] / "run.json").read_text())
    assert run["clips"][0]["state"] == "done" and run["clips"][0]["job"] == "j0"
    assert (tmp_path / snap["run"] / "clip000.mp4").read_bytes().startswith(b"fake-mp4-/files/renders/j0.mp4")
    params = studio.posted[0]["params"]
    assert params["preset"] == "small" and params["seconds"] == 5 and "first_frame_from" not in params


def test_continue_opens_each_clip_on_the_last(tmp_path):
    studio = FakeStudio(steps=1)

    async def go():
        svc = service(tmp_path, studio)
        await svc.start("waves on a black sand beach", "continue", count=3)
        await svc.task
        return svc

    svc = asyncio.run(go())
    assert [c["index"] for c in svc.snapshot()["clips"]] == [0, 1, 2]
    froms = [p["params"].get("first_frame_from") for p in studio.posted]
    assert froms == [None, "/studio/renders/j0.mp4", "/studio/renders/j1.mp4"]
    seeds = [p["params"]["seed"] for p in studio.posted]
    assert len(set(seeds)) == 3


def test_restart_resumes_the_job_in_flight_without_sending_it_again(tmp_path):
    studio = FakeStudio(steps=50)

    async def first():
        svc = service(tmp_path, studio)
        await svc.start("a paper boat on a rain gutter", "single")
        while not (svc.run["clips"] and svc.run["clips"][0].get("job")):
            await asyncio.sleep(0)
        svc.task.cancel()  # the agent goes away mid-render
        try:
            await svc.task
        except asyncio.CancelledError:
            pass

    asyncio.run(first())
    assert len(studio.posted) == 1
    studio.steps = 1

    async def second():
        svc = service(tmp_path, studio)
        await svc.resume()
        await svc.task
        return svc

    svc = asyncio.run(second())
    assert svc.snapshot()["state"] == "done"
    assert len(studio.posted) == 1  # followed, not queued again


def test_stop_cancels_the_render(tmp_path):
    studio = FakeStudio(steps=1000)

    async def go():
        svc = service(tmp_path, studio)
        await svc.start("a red kite over hills", "continue")
        while not (svc.run["clips"] and svc.run["clips"][0].get("job")):
            await asyncio.sleep(0)
        await svc.stop()
        return svc

    svc = asyncio.run(go())
    assert svc.snapshot()["state"] == "stopped" and studio.cancelled == ["j0"]


def test_local_render_host_is_refused_without_room(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "free_gb", lambda: 20.0)
    svc = service(tmp_path, FakeStudio(), render_host="local", local_min_free_gb=64)
    with pytest.raises(video.Refused, match="voice models"):
        asyncio.run(svc.start("a fox in snow"))
    # a loopback URL with no render_host is taken as this machine too
    svc = service(tmp_path, FakeStudio(), render_host=None, url="http://127.0.0.1:7870")
    with pytest.raises(video.Refused):
        asyncio.run(svc.start("a fox in snow"))


def test_studio_memory_refusal_is_said(tmp_path):
    def full(request):
        if request.url.path == "/api/estimate":
            return httpx.Response(200, json={"memory": {"fits": False, "reason": "needs 70 GB, 40 GB free"}})
        return FakeStudio()(request)

    svc = service(tmp_path, full)
    with pytest.raises(video.Refused, match="70 GB"):
        asyncio.run(svc.start("a fox in snow"))


def test_hyperflow_stub_refuses(tmp_path):
    svc = video.VideoService({"dir": str(tmp_path), "renderer": "hyperflow", "render_host": "remote"})
    with pytest.raises(video.Refused, match="isn't built yet"):
        asyncio.run(svc.start("anything"))


def test_voice_requests(tmp_path):
    studio = FakeStudio(steps=1)
    svc = service(tmp_path, studio)
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
    studio = FakeStudio(steps=1)

    async def go():
        svc = service(tmp_path, studio)
        await svc.start("a lantern on a pier")
        await svc.task
        return svc, await svc.film()

    svc, path = asyncio.run(go())
    assert Path(path).name == "clip000.mp4"
