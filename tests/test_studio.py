"""Ghost Studio's render side (gma/studio.py, gma/clips.py) against a fake headless Studio and a fake FP4 clip worker:
the engine that starts the Studio, the client, the governed renderer (keyframes, edits beside references, clips from
the worker, the DiT pass from its stage, the governor's calls, refusals, cancels, a voice turn first) and the film's
sound pass (gma/ap/sound.py). No Studio, no model, no GPU."""

import asyncio
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from gm_core.engine import Refusal
from gm_core.memstat import MemReader
from gma import clips as C
from gma import studio as S
from gma.governor import Governor
from gma.registry import Registry
from voice_agent import video

ROOT = Path(__file__).resolve().parent.parent


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeStudio:
    """The routes Ghost Studio uses (keyframes and the film's sound). A job runs ``steps`` looks (GET /api/jobs/<id>),
    then writes its file into the home."""

    def __init__(self, home: Path, steps: int = 4, models=("flux3k-fast", "flux3k-full"), fail=()):
        self.home, self.steps, self.models, self.fail = Path(home), steps, list(models), set(fail)
        self.home.mkdir(parents=True, exist_ok=True)
        self.jobs, self.bodies, self.calls, self.cancelled = {}, [], [], []
        self.down = False

    def __call__(self, request, timeout=None):
        if self.down:
            raise urllib.error.URLError("refused")
        if isinstance(request, str):
            method, url, body = "GET", request, None
        else:
            method, url = request.get_method(), request.full_url
            body = json.loads(request.data) if request.data and request.headers.get("Content-type") == "application/json" else None
        path = url.split("127.0.0.1:7890", 1)[-1]
        self.calls.append((method, path))
        if path.startswith("/files/"):
            return Answer((self.home / urllib.request.unquote(path[7:])).read_bytes())
        return Answer(json.dumps(self.route(method, path, body)).encode())

    def error(self, code, said):
        raise urllib.error.HTTPError("http://x", code, said, {}, io.BytesIO(json.dumps({"error": said}).encode()))

    def route(self, method, path, body):
        if path == "/api/jobs" and method == "GET":
            return list(self.jobs.values())
        if path == "/api/state":
            return {"home": str(self.home), "image_models": [{"id": m} for m in self.models]}
        if path in ("/api/jobs", "/api/voice/takes", "/api/music/takes", "/api/music/soundtrack") and method == "POST":
            kind = {"/api/voice/takes": "voice", "/api/music/takes": "music", "/api/music/soundtrack": "soundtrack"}.get(path, body.get("kind"))
            params = body.get("params") if path == "/api/jobs" else body
            self.bodies.append({"kind": kind, "params": params, "title": body.get("title")})
            job_id = f"job{len(self.jobs) + 1}"
            self.jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued", "params": params, "looks": 0, "outputs": [],
                                 "progress": None, "error": None, "log": "[tensorfold] starting ComfyUI\n" if len(self.jobs) == 0 else ""}
            return {k: v for k, v in self.jobs[job_id].items() if k != "looks"}
        found = re.fullmatch(r"/api/jobs/([\w-]+)(/cancel)?", path)
        if found:
            job = self.jobs[found.group(1)]
            if found.group(2):
                self.cancelled.append(job["id"])
                job["status"] = "cancelled"
            elif job["status"] in ("queued", "running"):
                self.advance(job)
            return {k: v for k, v in job.items() if k != "looks"}
        self.error(404, "no such route")

    def advance(self, job):
        job["looks"] += 1
        job["status"] = "running"
        n = job["looks"]
        if n <= self.steps:
            return
        if job["kind"] in self.fail:
            job.update(status="failed", error="CUDA out of memory")
            return
        out = {"scout": f"scouts/{job['id']}/scout_s1.png",
               "edit": f"scouts/{job['id']}/edit.png", "voice": f"voice/{job['id']}.wav", "music": f"music/{job['id']}.wav",
               "soundtrack": f"exports/{job['id']}.mp4"}[job["kind"]]
        target = self.home / out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"{job['kind'].upper()}-{job['id']}".encode())
        job["outputs"] = [out] + (["scouts/%s/sheet.jpg" % job["id"]] if job["kind"] == "scout" else [])
        job.update(status="done", elapsed=12.5)


class FakeWorker:
    """The FP4 clip worker's API (gma/clips.py ClipWorker): a clip runs ``steps`` looks through its stages, then its
    file is written where it was asked; ``fail`` fails every clip with the worker's reason."""

    def __init__(self, steps=4, fail=False):
        self.steps, self.fail, self.bodies, self.clips = steps, fail, [], {}

    def health(self):
        return {"loaded": True, "pid": None, "busy": any(c["status"] == "running" for c in self.clips.values())}

    def submit(self, body):
        if any(c["status"] == "running" for c in self.clips.values()):
            raise C.ClipError("the FP4 clip worker refused: a clip is rendering already")
        self.bodies.append(dict(body))
        handle = f"c{len(self.clips) + 1}"
        self.clips[handle] = {"id": handle, "status": "running", "stage": "conditioning", "at": 0, "of": 8, "looks": 0,
                              "body": body, "timings": {}, "error": None}
        return {"id": handle, "status": "queued"}

    def clip(self, handle):
        c = self.clips[handle]
        if c["status"] == "running":
            c["looks"] += 1
            n = c["looks"]
            c["stage"] = "conditioning" if n == 1 else "denoising" if n < self.steps else "decoding the video"
            c["at"] = min(8, max(0, n - 1))
            if n > self.steps:
                if self.fail:
                    c.update(status="failed", stage="failed", error="RuntimeError: CUDA out of memory")
                else:
                    Path(c["body"]["output"]).write_bytes(b"CLIP-" + handle.encode())
                    c.update(status="done", stage="done", timings={"render_s": 139.2, "dit_s": 107.5, "decode_s": 30.3,
                                                                  "denoising_stage": 107.47})
        return {k: v for k, v in c.items() if k not in ("looks", "body")}


class FakeGovernor:
    schedules = {"live": {}, "sharp": {}, "keyframe": {}, "audio": {}}

    def __init__(self, fits=True, enter_refuses=False):
        self.fits, self.enter_refuses, self.calls, self.dit_evict = fits, enter_refuses, [], []

    def plan(self, name=None, frames=None):
        return {"fits": self.fits, "reason": "this render needs about 61 GB at its peak; this machine has 40 GB for it",
                "adopted_blockers": [], "available_mb": 40_000}

    async def enter(self, name=None, frames=None):
        self.calls.append(("enter", name))
        if self.enter_refuses:
            raise Refusal("a render with schedule live runs already", "I'm already rendering. Let me finish this one first.")
        return {"schedule": name}

    async def dit(self, starting):
        self.calls.append(("dit", starting))

    async def leave(self, why=""):
        self.calls.append(("leave", why))


class Reader:
    platform = "spark"

    def total_mb(self):
        return 124_546

    def available_mb(self):
        return 60_000

    def available_gb(self):
        return 60_000 / 1024

    def settle(self, *a, **k):
        return None


def headless(tmp_path, studio=None, governor=None, worker=None, **spec):
    studio = studio or FakeStudio(tmp_path / "home")
    client = S.StudioClient("http://127.0.0.1:7890", opener=studio)
    renderer = S.HeadlessRenderer("spark", {"dir": str(studio.home / "ghost" / "renders"), "home": str(studio.home),
                                            "keyframe_wait_s": 5, "poll_s": 0.001, **spec},
                                  governor=governor if governor is not None else FakeGovernor(), reader=Reader(), client=client,
                                  worker=worker or FakeWorker())
    renderer.held_mb = lambda: 0  # never this machine's own ComfyUI (a test may run beside a live one)
    return renderer, studio


def service(tmp_path, renderer):
    svc = video.VideoService({"dir": str(renderer.client.home() / "ghost" / "video"), "poll_s": 0.002, "keyframe": True,
                              "ffmpeg": "/nonexistent"}, brain=None, renderer=renderer)
    svc.ffmpeg = None
    return svc


async def settle(renderer):
    await asyncio.gather(*(j["task"] for j in renderer.jobs.values() if j.get("task")), return_exceptions=True)


# ---------- the headless Studio as an engine ----------

def test_the_studio_engine_runs_the_unmodified_studio_on_its_own_port_home_and_comfyui(tmp_path):
    spec = S.studio_engine({"src": str(tmp_path / "src"), "home": str(tmp_path / "home"), "prefix": str(tmp_path / "pfx")})
    env = spec["env"]
    assert spec["role"] == "renderer" and spec["port"] == 7890 and spec["idle_s"] == "never"
    assert spec["argv"] == [str(tmp_path / "pfx/venv/bin/python"), str(tmp_path / "src/app/server.py")]
    assert env["HOST"] == "127.0.0.1" and env["PORT"] == "7890" and env["STUDIO_HOME"] == str(tmp_path / "home")
    assert env["SPARK_COMFY_PORT"] == "8290" and env["STUDIO_PIDFILE"].endswith("app.pid")  # no updater, no captioner
    assert "--output-directory " + str(tmp_path / "home/.comfy/output") in env["SPARK_COMFY_ARGS"]  # not into the install
    assert spec["stop_argv"][-1] == "stop" and spec["counts_in_render"] is False
    for port in ("7870", "8190"):  # the worker's ports are never this one's
        assert port not in (env["PORT"], env["SPARK_COMFY_PORT"])


def test_the_supervisor_adds_the_studio_engine_and_the_headless_renderer():
    from gma.supervisor import Supervisor

    profile = json.loads((ROOT / "profiles" / "spark.example.json").read_text())
    sup = Supervisor(profile, reader=MemReader("spark", meminfo=lambda: "MemTotal: 127535104 kB\nMemAvailable: 90000000 kB\n"))
    assert isinstance(sup.renderer, S.HeadlessRenderer) and sup.renderer.engine() is sup.registry["studio"]
    assert sup.governor.renderer is sup.renderer and sup.registry["studio"].role == "renderer"
    assert "comfy" not in sup.registry.engines  # this app runs no ComfyUI of its own any more


def test_a_stop_argv_gets_the_engines_environment(tmp_path):
    from gm_core.engine import Engine

    out = tmp_path / "said"
    one = Engine("studio", {"role": "renderer", "port": 1, "env": {"SPARK_COMFY_PORT": "8290"},
                            "stop_argv": ["sh", "-c", f"echo $SPARK_COMFY_PORT > {out}"]}, log_dir=tmp_path)
    one.run_stop_argv()
    assert out.read_text().strip() == "8290"


# ---------- the client ----------

def test_the_client_names_files_by_the_studio_home_and_brings_outside_ones_in(tmp_path):
    studio = FakeStudio(tmp_path / "home")
    client = S.StudioClient("http://127.0.0.1:7890", opener=studio)
    inside = studio.home / "ghost" / "a.png"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"A")
    outside = tmp_path / "b.png"
    outside.write_bytes(b"B")
    assert client.inside(inside) == "ghost/a.png" and client.inside(outside) is None
    assert client.bring(inside) == "ghost/a.png"
    brought = client.bring(outside)
    assert brought.startswith("ghost-inputs/") and (studio.home / brought).read_bytes() == b"B"


def test_a_studio_refusal_is_its_own_words(tmp_path):
    client = S.StudioClient("http://127.0.0.1:7890", opener=FakeStudio(tmp_path / "home"))
    with pytest.raises(S.StudioError) as error:
        client.call("GET", "/api/nothing")
    assert error.value.status == 404 and "no such route" in str(error.value)
    down = FakeStudio(tmp_path / "home")
    down.down = True
    assert not S.StudioClient("http://127.0.0.1:7890", opener=down).up()


# ---------- the governor's view ----------

def test_a_clips_need_is_the_workers_measured_peak_less_what_it_holds(tmp_path):
    renderer, _ = headless(tmp_path)
    found = renderer.memory_request({"frames": 243})
    assert found["need_mb"] == C.need_mb(864, 480, 243) and found["error_mb"] == C.ERROR_MB and found["route"] == "fp4"
    assert renderer.memory_request({"width": 1376, "height": 768, "frames": 243})["need_mb"] == C.need_mb(1376, 768, 243)
    assert renderer.memory_request({"need_gb": 31, "error_gb": 4})["source"] == "profile"  # a keyframe: stated
    assert renderer.floor_mb() is None  # the governor's own floor (a tenth of the machine, or the profile's)

    class Held:  # the worker kept from the take before: only the rest of its peak is asked for
        def held_mb(self):
            return 40_000
    renderer.clip_engine = lambda: Held()
    assert renderer.memory_request({"frames": 243})["need_mb"] == C.need_mb(864, 480, 243) - 40_000


def test_need_mb_takes_the_nearest_measured_length():
    assert C.need_mb(864, 480, 192) == C.need_mb(864, 480, 124) == C.NEED_MB[(864, 480, 243)]  # a longer clip's peak
    assert C.need_mb(640, 360, 124) == max(C.NEED_MB.values())  # a size never measured: the largest


# ---------- keyframes and clips ----------

def test_a_keyframe_from_the_studio_then_a_clip_from_the_fp4_worker(tmp_path):
    async def go():
        governor, worker = FakeGovernor(), FakeWorker()
        renderer, studio = headless(tmp_path, governor=governor, worker=worker, schedule="live")
        svc = service(tmp_path, renderer)
        await svc.start("a red kite over green hills")
        await asyncio.wait_for(svc.task, 10)
        await settle(renderer)
        return svc, renderer, studio, governor, worker

    svc, renderer, studio, governor, worker = asyncio.run(go())
    assert svc.run["state"] == "done", svc.run.get("message")
    assert [b["kind"] for b in studio.bodies] == ["scout"]  # the Studio draws the keyframe and makes no clip
    scout = studio.bodies[0]
    assert scout["params"]["image_model"] == "flux3k-fast" and scout["params"]["preset"] == "small"  # no Qwen-Image here
    body = worker.bodies[0]
    assert (body["width"], body["height"], body["frames"]) == (864, 480, 124)
    assert body["image"] == str(studio.home / "scouts/job1/scout_s1.png")  # the keyframe the Studio made
    assert body["prompt"].startswith("For the target video, at 0.00 seconds")  # MiniMax H3's first-frame lead
    assert body["output"].startswith(str(renderer.root)) and body["output"].endswith("/clip.mp4")
    assert svc.clip_file(svc.run["id"], 0, "mp4").read_bytes().startswith(b"CLIP-")
    # one session for the keyframe and its clip: entered, the DiT pass said at its start and end, left at the end
    assert [c for c in governor.calls if c[0] == "dit"] == [("dit", True), ("dit", False)]
    assert governor.calls[0][0] == "enter" and governor.calls[-1][0] == "leave"
    job = next(j for j in renderer.jobs.values() if j["kind"] == "video")
    assert {"render_s", "dit_s", "decode_s", "clip_s", "min_available_gb"} <= set(job["timings"])
    assert job["timings"]["renderer"] == "fp4" and "denoising_stage" not in job["timings"]
    assert svc.snapshot()["previews"] == "stages"


def test_a_keyframe_beside_references_is_an_edit_job_and_a_sharp_clip_its_own_schedule(tmp_path):
    async def go():
        governor, worker = FakeGovernor(), FakeWorker()
        renderer, studio = headless(tmp_path, governor=governor, worker=worker, schedule="live")
        place = studio.home / "ghost" / "place.png"
        place.parent.mkdir(parents=True, exist_ok=True)
        place.write_bytes(b"PLACE")
        face = tmp_path / "face.png"  # outside the home: brought in
        face.write_bytes(b"FACE")
        handle = await renderer.keyframe("the keeper at the lamp", 7, references=[str(place), str(face)], size=(864, 480))
        await renderer.jobs[handle]["task"]
        clip = await renderer.submit({"prompt": "x", "composed": True, "frames": 243, "seed": 1, "schedule": "sharp",
                                      "width": 1376, "height": 768})
        await renderer.jobs[clip]["task"]
        return renderer, studio, governor, worker, handle

    renderer, studio, governor, worker, handle = asyncio.run(go())
    edit = studio.bodies[0]
    assert edit["kind"] == "edit" and edit["params"]["image"] == "ghost/place.png"
    assert edit["params"]["images"][0].startswith("ghost-inputs/") and edit["params"]["image_model"] == "flux3k-fast"
    assert (edit["params"]["width"], edit["params"]["height"]) == (864, 480)
    assert renderer.jobs[handle]["output"].endswith("edit.png")
    assert (worker.bodies[0]["width"], worker.bodies[0]["height"], worker.bodies[0]["frames"]) == (1376, 768, 243)
    assert worker.bodies[0]["image"] is None and worker.bodies[0]["prompt"] == "x"  # composed: sent as it is
    enters = [c for c in governor.calls if c[0] == "enter"]
    assert enters == [("enter", "keyframe"), ("enter", "sharp")]  # a clip never runs inside a keyframe's session


def test_a_clip_that_continues_opens_on_the_last_frame_of_the_clip_before(tmp_path, monkeypatch):
    made = []

    def last_frame(video_path, dest, ffmpeg="ffmpeg"):
        made.append(video_path)
        dest.write_bytes(b"PNG")
        return dest
    monkeypatch.setattr(C, "last_frame", last_frame)

    async def go():
        worker = FakeWorker(steps=1)
        renderer, studio = headless(tmp_path, worker=worker)
        clip = await renderer.submit({"prompt": "x", "composed": True, "frames": 124, "seed": 1, "opens_on": "/before.mp4"})
        await renderer.jobs[clip]["task"]
        return renderer.jobs[clip], worker

    job, worker = asyncio.run(go())
    assert job["state"] == "done" and made == ["/before.mp4"]
    assert worker.bodies[0]["image"].endswith("/first.png")


def test_no_room_is_refused_before_anything_is_asked(tmp_path):
    async def go():
        worker = FakeWorker()
        renderer, studio = headless(tmp_path, governor=FakeGovernor(fits=False), worker=worker, room_wait_s=0)
        svc = service(tmp_path, renderer)
        with pytest.raises(video.Refused) as refused:
            await svc.start("a red kite")
        return refused.value, studio, worker

    error, studio, worker = asyncio.run(go())
    assert error.said == "There isn't enough memory to render that right now." and "61 GB" in str(error)
    assert not studio.bodies and not worker.bodies


def test_no_clip_worker_is_said_and_nothing_else_renders_the_clip(tmp_path):
    renderer, studio = headless(tmp_path)
    renderer.worker = None  # a profile without the FP4 clip worker: there is no other route
    refused = asyncio.run(renderer.room({"prompt": "x", "frames": 124}))
    assert refused is not None and "FP4 clip worker" in str(refused) and not studio.bodies


def test_a_failed_clip_fails_with_the_workers_reason_and_still_leaves_and_ends_the_dit(tmp_path):
    async def go():
        governor = FakeGovernor()
        renderer, studio = headless(tmp_path, governor=governor, worker=FakeWorker(fail=True))
        svc = service(tmp_path, renderer)
        await svc.start("a red kite")
        await asyncio.wait_for(svc.task, 10)
        await settle(renderer)
        return svc, renderer, governor, studio

    svc, renderer, governor, studio = asyncio.run(go())
    assert svc.run["state"] == "failed" and "CUDA out of memory" in svc.run["message"]
    assert "FP4 clip worker" in svc.run["message"] and [b["kind"] for b in studio.bodies] == ["scout"]  # no fallback
    assert [c for c in governor.calls if c[0] == "dit"] == [("dit", True), ("dit", False)]
    assert governor.calls[-1][0] == "leave" and renderer.session is None


def test_a_cancelled_take_ends_its_session_and_the_next_waits_for_the_worker(tmp_path):
    async def go():
        worker = FakeWorker(steps=10_000)
        renderer, studio = headless(tmp_path, worker=worker, busy_wait_s=5)
        renderer.spec["poll_s"] = 0.01
        clip = await renderer.submit({"prompt": "x", "composed": True, "frames": 124, "seed": 1})
        for _ in range(200):
            await asyncio.sleep(0.01)
            if worker.bodies:
                break
        await renderer.cancel(clip)
        await settle(renderer)
        first = renderer.jobs[clip]
        worker.steps = 1
        for c in worker.clips.values():  # the worker finishes the cancelled take's clip
            c["looks"] = 10_000
            worker.clip(c["id"])
        second = await renderer.submit({"prompt": "y", "composed": True, "frames": 124, "seed": 2})
        await renderer.jobs[second]["task"]
        return first, renderer.jobs[second], renderer

    first, second, renderer = asyncio.run(go())
    assert first["state"] == "cancelled" and second["state"] == "done" and renderer.session is None


# ---------- the real governor: the clip worker kept, the brain released for the clip when it does not fit ----------

class Machine:
    def __init__(self, available_mb):
        self.available = available_mb

    def reader(self):
        return MemReader("spark", meminfo=lambda: f"MemTotal: {124546 * 1024} kB\nMemAvailable: {self.available * 1024} kB\n")


def test_with_the_real_governor_the_brain_goes_for_the_clip_and_the_worker_stays(tmp_path):
    from tests.test_governor import FakeEngine

    machine = Machine(118_700 - 2048)
    reader = machine.reader()
    reader.settle = lambda *a, **k: None
    registry = Registry({"engines": {}}, reader)
    for name, role, need, resident in (("qwen", "brain", 30, 23.5), ("kokoro", "voice", 5, 3.5), ("cosmos", "vision", 6, 3.4),
                                       ("d1", "judge", 8, 5.5), ("fp4", "clips", 0, 40.0)):
        one = FakeEngine(name, {"role": role, "mode": "managed", "port": 1, "need_gb": need, "resident_gb": resident}, reader, machine)
        registry.engines[name] = one
        one.bring_up()
    profile = json.loads((ROOT / "profiles" / "spark.example.json").read_text())
    renderer, studio = headless(tmp_path, governor=None, schedule="live")
    renderer.registry, renderer.reader = registry, reader
    governor = Governor(profile["governor"], registry, reader, platform="spark", renderer=renderer)
    renderer.governor = governor
    registry.set_admit(governor.admit)
    machine.available -= 21_000  # boo-transcribe beside it: the clip cannot sit beside the brain, the explicit fallback

    async def go():
        states = []
        original = governor.dit

        async def dit(starting):
            await original(starting)
            states.append((starting, registry["qwen"].loaded(), registry["d1"].loaded(), registry["fp4"].loaded()))

        governor.dit = dit
        registry["fp4"].acquire = lambda *a, **k: "http://127.0.0.1:8420"
        registry["fp4"].release_request = lambda: None
        clip = await renderer.submit({"prompt": "x", "composed": True, "frames": 243, "seed": 1})
        await renderer.jobs[clip]["task"]
        return renderer.jobs[clip], states

    job, states = asyncio.run(go())
    assert job["state"] == "done", job.get("error")
    assert states[0] == (True, False, False, True)  # the judge evicted, the brain released for the clip, the worker kept
    assert governor.phase == "talking" and not registry["qwen"].hold and registry["fp4"].loaded()


# ---------- a voice turn first ----------

def test_a_voice_turn_stops_the_clip_worker_until_its_first_audio(tmp_path):
    renderer, _ = headless(tmp_path, voice_hold_s=0.5)
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    renderer.worker_pid = proc.pid
    try:
        assert not renderer.hold_for_voice(True)  # nothing of this app's renders: nothing stopped
        renderer.jobs["v1"] = {"state": "rendering", "kind": "video"}
        assert renderer.hold_for_voice(True)
        time.sleep(0.1)
        assert open(f"/proc/{proc.pid}/stat").read().rsplit(")", 1)[1].split()[0] == "T"
        assert not renderer.hold_for_voice(True)  # one hold at a time
        assert renderer.hold_for_voice(False) and len(renderer.voice_pauses) == 1
        assert renderer.hold_for_voice(True)
        time.sleep(0.9)  # nobody released it: it ends by itself
        assert renderer.voice_held is None and len(renderer.voice_pauses) == 2
        assert open(f"/proc/{proc.pid}/stat").read().rsplit(")", 1)[1].split()[0] != "T"
    finally:
        os.killpg(proc.pid, 9)
        proc.wait()


def test_a_load_measured_beside_another_counts_its_resident_size(tmp_path):
    from gm_core import engine as eng

    class Flat:
        platform = "spark"

        def available_gb(self):
            return 80.0

        def available_mb(self):
            return 80 * 1024

    one = eng.Engine("bolt", {"role": "brain", "resident_gb": 28, "health": "/", "guard_gb": 1, "load_timeout_s": 30,
                              "argv": [sys.executable, "-m", "http.server", "{port}", "--bind", "127.0.0.1"]},
                     reader=Flat(), log_dir=tmp_path)
    try:
        one.acquire()
        one.release_request()
        assert one.held_gb == 28 and one.held_mb() == 28 * 1024
    finally:
        one.stop("the test ended")


# ---------- the film's sound: Drama 3 narration and a MiniMax Music 3 score, by the Studio ----------

def test_the_film_gets_its_narration_and_score_from_the_studio(tmp_path, monkeypatch):
    from gma.ap import sound

    studio = FakeStudio(tmp_path / "home", steps=0)  # each job done at its first look
    client = S.StudioClient("http://127.0.0.1:7890", opener=studio)
    film = studio.home / "ghost" / "video" / "r1" / "film.mp4"
    film.parent.mkdir(parents=True)
    film.write_bytes(b"FILM")
    laid = []
    monkeypatch.setattr(sound, "lay_narration", lambda f, v, t: laid.append((f, v)) or t.write_bytes(b"NARRATED"))
    monkeypatch.setattr(S.StudioClient, "wait", lambda self, job_id, **k: self.job(job_id))
    run = {"name": "Lamp", "idea": "a keeper lights the lamp", "film": {"path": "r1/film.mp4", "seconds": 29.9},
           "settings": {"narration": True, "score": True, "style": "35mm film, storm light"},
           "segments": [{"index": 1, "status": "done", "beat": "climbs the stair"}]}
    said = []
    director = lambda system, prompt, think: "The keeper climbed. The storm came. He lit the lamp."
    record = sound.finish_film(run, client, director, lambda run, line, **k: said.append(line), studio.home / "ghost" / "video")
    kinds = [b["kind"] for b in studio.bodies]
    assert kinds == ["voice", "music", "soundtrack"]
    assert studio.bodies[0]["params"]["text"].startswith("The keeper climbed")
    assert studio.bodies[1]["params"]["seconds"] == 31 and "Instrumental" in studio.bodies[1]["params"]["caption"]
    assert studio.bodies[2]["params"]["video"] == "ghost/video/r1/film-narrated.mp4"
    assert record["path"] == "r1/film-final.mp4" and (film.parent / "film-final.mp4").read_bytes().startswith(b"SOUNDTRACK")
    assert laid and laid[0][1].name == "narration.wav"


def test_a_sound_part_that_fails_is_left_out_and_the_film_stands(tmp_path):
    from gma.ap import sound

    studio = FakeStudio(tmp_path / "home", steps=0, fail={"music"})
    client = S.StudioClient("http://127.0.0.1:7890", opener=studio)
    film = studio.home / "r1" / "film.mp4"
    film.parent.mkdir(parents=True)
    film.write_bytes(b"FILM")
    client.wait = lambda job_id, **k: client.job(job_id)
    run = {"film": {"path": "r1/film.mp4", "seconds": 12}, "settings": {"score": True}, "segments": []}
    said = []
    assert sound.finish_film(run, client, None, lambda run, line, **k: said.append(line), studio.home) is None
    assert any("score was left out" in line for line in said) and film.read_bytes() == b"FILM"


def test_a_clip_queued_the_moment_one_fails_waits_for_its_session_to_end(tmp_path):

    class Strict(FakeGovernor):  # refuses a second enter before the first session has left, like the real one
        def __init__(self):
            super().__init__()
            self.phase = "talking"

        async def enter(self, name=None, frames=None):
            if self.phase == "rendering":
                raise Refusal("a render with schedule live runs already", "busy")
            self.phase = "rendering"
            await asyncio.sleep(0)
            return await super().enter(name, frames)

        async def leave(self, why=""):
            await asyncio.sleep(0.05)  # leaving frees the Studio's ComfyUI: it takes a moment
            self.phase = "talking"
            await super().leave(why)

    async def go():
        governor = Strict()
        worker = FakeWorker(steps=0, fail=True)
        renderer, studio = headless(tmp_path, governor=governor, worker=worker)
        first = await renderer.submit({"prompt": "x", "composed": True, "frames": 124, "seed": 1})
        while renderer.jobs[first]["state"] not in ("failed", "done"):
            await asyncio.sleep(0.001)
        assert await renderer.room({"prompt": "x", "frames": 124}) is None  # the ending job is not one at work
        worker.fail = False
        second = await renderer.submit({"prompt": "x", "composed": True, "frames": 124, "seed": 2})
        await renderer.jobs[second]["task"]
        return renderer.jobs[first], renderer.jobs[second]

    first, second = asyncio.run(go())
    assert first["state"] == "failed" and second["state"] == "done", second.get("error")


def test_a_clip_that_does_not_fit_yet_is_asked_again_while_the_memory_comes_back(tmp_path):
    class Freeing(FakeGovernor):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def plan(self, name=None, frames=None):
            self.asked += 1
            return {**super().plan(name, frames), "fits": self.asked >= 2}

    async def go():
        governor = Freeing()
        renderer, _ = headless(tmp_path, governor=governor, room_wait_s=10)
        return await renderer.room({"prompt": "x", "frames": 243}), governor.asked

    refused, asked = asyncio.run(go())
    assert refused is None and asked == 2


def test_what_the_studios_comfyui_holds_counts_as_the_renders_own_room(tmp_path):
    from tests.test_governor import spark

    machine, registry, governor = spark(available_mb=70_000, loaded=("kokoro", "cosmos"))
    renderer, _ = headless(tmp_path, governor=governor)
    governor.renderer = renderer
    before = governor.plan("live", 243)
    renderer.held_mb = lambda: 25_500  # FLUX 3 Klein's leftovers after its /free (DGX Spark, 2026-11-07)
    after = governor.plan("live", 243)
    assert after["renderer_held_mb"] == 25_500 and after["budget"]["left_before_fallbacks_mb"] - before["budget"]["left_before_fallbacks_mb"] == 25_500
