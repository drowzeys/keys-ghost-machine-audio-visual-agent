"""AutoPilot inside the app (gma/ap/service.py, eyes.py, ears.py and the video panel): a kill -9 mid-render resumes
from the checkpoint without rendering a finished piece again; a confirmed major finding is retaken at the next seed and
a minor one becomes a Director note; the eyes swap between Gemma 4 12B and Cosmos3-Edge by the profile; and on a
DGX Spark's schedule the eyes are never resident beside a render. A fake renderer (ffmpeg test patterns), a scripted
Director and stand-in eyes: no GPU, no model, no network beyond loopback."""

import asyncio
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from gm_core.engine import Held
from gma.ap import autopilot, eyes, judge
from gma.ap.service import AutoPilotService, status
from gma.governor import Governor
from gma.registry import Registry
from test_autopilot import HOME, SEED, Director, Watcher, go, loop, new_run  # noqa: F401
from test_autopilot_checks import Server, answers_for, bolt_like  # noqa: F401
from test_governor import FakeEngine, FakeRenderer, Machine
from voice_agent.video import Progress, VideoService

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ("submit", "job_of", "cancel", "director", "eyes", "ears", "on_change", "shell", "ask_text", "ask_vision", "gate", "ears_ready")
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


class Killed(BaseException):
    """kill -9: the thread ends where it is, nothing saved after it."""


class FakeLocal:
    """The local renderer's interface (gma/renderer.py) on tiny ffmpeg patterns; ``hang(n)`` keeps the n-th clip
    rendering forever (the render a kill -9 lands in)."""

    name, previews, preview_kind, governed = "local", True, "engine", True

    def __init__(self, root: Path, hang=lambda n: False):
        self.root, self.hang = root, hang
        self.jobs, self.calls, self.ended = {}, [], []
        root.mkdir(parents=True, exist_ok=True)

    async def room(self, clip):
        return None

    async def keyframe(self, prompt, seed, references=None, size=None):
        handle = f"k{len(self.jobs) + 1}-{time.time_ns()}"
        out = self.root / f"{handle}.png"
        ffmpeg("-f", "lavfi", "-i", "color=c=teal:size=160x96", "-frames:v", "1", str(out))
        self.jobs[handle] = {"state": "done", "output": str(out)}
        self.calls.append(("keyframe", prompt[:40], seed))
        return handle

    async def submit(self, clip):
        handle = f"v{len(self.jobs) + 1}-{time.time_ns()}"
        videos = sum(1 for c in self.calls if c[0] == "video") + 1
        self.calls.append(("video", clip["prompt"][:40], clip["seed"], clip.get("image_file")))
        out = self.root / f"{handle}.mp4"
        if self.hang(videos):
            self.jobs[handle] = {"state": "rendering", "output": None}
            return handle
        ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
               "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out))
        self.jobs[handle] = {"state": "done", "output": str(out)}
        return handle

    async def poll(self, handle):
        job = self.jobs[handle]
        if job["state"] == "rendering":
            return Progress("rendering", 3, 8, "denoising", preview=handle + "#1")
        return Progress(job["state"], output=job["output"], timings={"render_s": 1.0})

    async def picture(self, ref):
        path = Path(ref.split("#")[0]) if "#" not in ref else None
        return path.read_bytes() if path and path.is_file() else b"\xff\xd8preview"

    async def fetch(self, output, dest):
        shutil.copyfile(output, dest)

    async def end(self, why):
        self.ended.append(why)

    async def cancel(self, handle):
        self.jobs[handle]["state"] = "cancelled"


@pytest.fixture
def app_loop(monkeypatch):
    """An event loop on a thread of its own (the app's), the AutoPilot hooks put back afterwards."""

    saved = {k: getattr(autopilot, k) for k in HOOKS}
    saved_home = autopilot.HOME
    loop_ = asyncio.new_event_loop()
    threading.Thread(target=loop_.run_forever, daemon=True).start()
    yield loop_
    loop_.call_soon_threadsafe(loop_.stop)
    for k, v in saved.items():
        setattr(autopilot, k, v)
    autopilot.HOME = saved_home


def on(loop_, coro, timeout=60):
    return asyncio.run_coroutine_threadsafe(coro, loop_).result(timeout)


def wire(tmp_path, loop_, renderer, director, watcher):
    svc = VideoService({"renderer": "local", "dir": str(tmp_path / "video")}, renderer=renderer)
    ap = AutoPilotService(svc, {"settings": {"seconds": 1, "continuity": {"film_pass": False}, "arc": False, "location_anchor": False}}, loop=loop_)
    svc.autopilot = ap
    autopilot.ask_text, autopilot.ask_vision = director, watcher
    autopilot.gate = lambda clip, prompt, sheet=None, brain=None: {"passed": True, "audio": {"result": "pass", "heard": ""}, "identity": {"result": "skipped"}}
    autopilot.ears_ready = lambda: {"runs": True, "transcript": True}
    return svc, ap


def wait_for(check, seconds=120):
    ends = time.time() + seconds
    while time.time() < ends:
        if check():
            return True
        time.sleep(0.1)
    return False


# ---------- checkpoint and resume: a kill -9 mid-render of piece 2 ----------

def test_a_kill_mid_render_resumes_from_the_checkpoint_without_redoing_a_finished_piece(tmp_path, app_loop):
    first = FakeLocal(tmp_path / "renders1", hang=lambda n: n == 2)
    svc, ap = wire(tmp_path, app_loop, first, Director(), Watcher())
    real = ap.job_of

    def job_of(job_id):
        if first.jobs.get(job_id, {}).get("state") == "rendering":
            raise Killed()  # the process dies while piece 2 renders
        return real(job_id)
    autopilot.job_of = job_of
    snap = on(app_loop, svc.start("a lighthouse keeper's evening", "autopilot", 3, 1))
    assert snap["mode"] == "autopilot" and snap["autopilot"]["count"] == 3
    run_id = snap["run"]
    assert wait_for(lambda: not autopilot.alive(run_id))
    folder = svc.root / run_id
    shell = json.loads((folder / "run.json").read_text())
    run = shell["autopilot"]
    assert shell["mode"] == "autopilot" and run["state"] == "running"  # nobody saved a stop: a crash
    assert [s["status"] for s in run["segments"]] == ["done", "queued"] and len(shell["clips"]) == 1
    point = json.loads((folder / "checkpoint.json").read_text())
    assert point["segments"] == 1 and point["next"]["index"] == 2 and (folder / "checkpoints" / "S0001.json").is_file()
    piece1 = folder / run["segments"][0]["output"].split("/", 1)[1]
    made = piece1.stat().st_mtime_ns
    assert (folder / "film.mp4").is_file()  # joined after the piece

    # the app starts again: a new process (new renderer, no jobs), the same folders
    second = FakeLocal(tmp_path / "renders2")
    svc2, ap2 = wire(tmp_path, app_loop, second, Director(), Watcher())
    on(app_loop, svc2.resume())
    assert wait_for(lambda: autopilot.alive(run_id) is False and json.loads((folder / "run.json").read_text())["autopilot"]["state"] == "done", 180)
    after = json.loads((folder / "run.json").read_text())
    done = after["autopilot"]
    assert [s["status"] for s in done["segments"]] == ["done"] * 3
    kinds = [c[0] for c in second.calls]
    assert kinds == ["video", "video"]  # piece 2's clip again from piece 1's last frame, then piece 3 (no keyframes)
    assert second.calls[0][3] is None  # it continues the clip before: no picture of its own
    assert piece1.stat().st_mtime_ns == made  # piece 1 was never rendered again
    assert any("Picked up after a restart" in line["text"] for line in done["log"])
    assert len(after["clips"]) == 3 and done["film"]["segments"] == 3
    assert sorted(p.name for p in (folder / "checkpoints").iterdir()) == ["S0001.json", "S0002.json", "S0003.json"]
    snap = svc2.snapshot()
    assert snap["state"] == "done" and snap["film"] and snap["autopilot"]["done"] == 3


def test_the_panel_follows_the_piece_at_work_with_its_live_preview(tmp_path, app_loop):
    renderer = FakeLocal(tmp_path / "renders", hang=lambda n: n == 1)
    svc, ap = wire(tmp_path, app_loop, renderer, Director(), Watcher())
    run_id = on(app_loop, svc.start("a lighthouse keeper's evening", "autopilot", 1, 1))["run"]
    assert wait_for(lambda: svc.live.get("preview") == "engine" and svc.snapshot()["state"] == "rendering")
    snap = svc.snapshot()
    assert snap["current"]["at"] == 3 and snap["current"]["of"] == 8 and snap["autopilot"]["stage"] == "render"
    assert snap["autopilot"]["beat"] and snap["autopilot"]["segment"] == 1
    on(app_loop, svc.stop())
    assert wait_for(lambda: not autopilot.alive(run_id))
    assert json.loads((svc.root / run_id / "run.json").read_text())["autopilot"]["state"] == "paused"


# ---------- a confirmed major problem is retaken; a minor one is a note ----------

def test_a_confirmed_major_retakes_at_the_next_seed_and_a_minor_one_is_a_director_note(loop, monkeypatch):
    def yesno(prompt, frames):
        take1 = len(loop["queue"].bodies) == 1
        rules = {r"change into a different person": "yes" if take1 else "no",  # a morph: major, for anyone
                 r"malformed": "yes" if take1 else "no"}
        return answers_for(rules)(prompt, frames)
    watcher = Watcher(yesno=yesno, second=[True, True, True])
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, continuity={"film_pass": False}))
    first = run["segments"][0]
    assert [t["seed"] for t in first["takes"]] == [1, 2] and first["takes"][0]["passed"] is False
    assert any("identity" in r for r in first["takes"][0]["reasons"]) and first["passed"] is True
    found = first["takes"][0]["continuity"]["findings"]
    assert any(f["category"] == "identity" and f["status"] == "major" and f["action"] == "retake" for f in found)
    assert any(f["status"] == "minor" and f["action"] == "correct" for f in found)  # the malformed hand: a note
    assert any("FAILED ITS CONTINUITY CHECK" in p for p in loop["director"].asked)  # the shot rewritten from the findings


def test_the_retake_budget_ends_with_the_best_take_kept_never_an_approval(loop, monkeypatch):
    watcher = Watcher(yesno=answers_for({r"change into a different person": "yes"}), second=[True] * 20)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(1, retakes=2, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert run["state"] == "done" and len(seg["takes"]) == 3 and [t["seed"] for t in seg["takes"]] == [1, 2, 3]
    assert seg["status"] == "done" and seg["passed"] is False and "no take passed" in seg["reasons"][-1]


# ---------- the eyes: Gemma 4 12B or Cosmos3-Edge, by the profile ----------

def frames_at(tmp_path, count):
    out = []
    for i in range(count):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(b"\x89PNG frame")
        out.append((f"Frame {i + 1}, at {i:.2f} s", p))
    sheet = tmp_path / "sheet.png"
    sheet.write_bytes(b"\x89PNG sheet")
    return out, sheet


def test_the_eyes_swap_between_gemma_and_cosmos_by_the_profile(tmp_path, bolt_like):
    profile = {"autopilot": {"eyes": "gemma", "eyes_choices": {
        "gemma": {"engine": "gemma", "url": bolt_like, "model": "gemma-4-12b"},
        "cosmos": {"engine": "cosmos", "url": bolt_like + "/v1", "model": "cosmos3"}}}}
    frames, sheet = frames_at(tmp_path, 8)
    gemma = eyes.make(profile)
    assert (gemma.engine, gemma.model, gemma.max_images, gemma.label) == ("gemma", "gemma-4-12b", 9, "Gemma 4 12B")
    Server.replies = [(200, {"content": '<image|>{"q1": "yes"}'})]
    answer, who = gemma.look("sys", frames, sheet, "YES/NO CHECK")
    images = [p for p in Server.bodies[-1]["messages"][1]["content"] if p["type"] == "image_url"]
    assert answer == '{"q1": "yes"}' and who == "Gemma 4 12B" and len(images) == 8  # every frame, the leaked token gone
    profile["autopilot"]["eyes"] = "cosmos"
    cosmos = eyes.make(profile)
    Server.replies = [(200, {"content": '{"q1": "no"}'})]
    answer, who = cosmos.look("sys", frames, sheet, "YES/NO CHECK")
    content = Server.bodies[-1]["messages"][1]["content"]
    assert who == "Cosmos3-Edge (frame sheet)" and sum(p["type"] == "image_url" for p in content) == 1  # it takes 4 at most
    assert "contact sheet" in content[-1]["text"] and Server.bodies[-1]["model"] == "cosmos3"
    # the loop's eyes are whichever is attached: the same yes/no questions, decided in code either way
    autopilot.eyes = cosmos
    Server.replies = [(200, {"content": '{"q1": "yes"}'})]
    assert autopilot.ask_vision("sys", frames[:3], sheet, "YES/NO CHECK")[0] == '{"q1": "yes"}'
    assert eyes.make({"eyes": {"url": "http://127.0.0.1:8099/v1", "model": "cosmos3"}}).engine == "cosmos"  # no autopilot section
    autopilot.eyes = None


# ---------- the Spark profile: Gemma is never resident beside a render ----------

class LoadingEngine(FakeEngine):
    """A managed engine that loads (its memory taken from the faked machine) without a process."""

    def start(self, stopped=lambda: False):
        if self.loaded():
            return
        if self.admit is not None:
            self.admit(self)
        self.bring_up()


def test_on_the_spark_profile_the_judge_loads_between_renders_and_gemma_is_the_self_checks_eyes(tmp_path):
    profile = json.loads((ROOT / "profiles" / "spark.example.json").read_text())
    schedules = profile["governor"]["schedules"]
    # d1-2B is the judge; Gemma 4 12B is the self-check's eyes, kept beside the clip and the keyframe where it fits
    # (evicted last: after the brain); the judge stays through a keyframe only
    assert all("gemma" in schedules[n]["keep"] for n in ("live", "keyframe")) and profile["autopilot"]["eyes"] == "gemma"
    assert all("d1" not in s.get("keep", []) for name, s in schedules.items() if name != "keyframe")
    assert "d1" in schedules["keyframe"]["keep"]
    assert schedules["keyframe"]["fallbacks"] == [{"step": "evict", "engine": "qwen"}, {"step": "evict", "engine": "gemma"}]
    assert profile["autopilot"]["judge"] == "d1" and profile["autopilot"]["judge_choices"]["d1"]["url"].endswith(":8412")
    assert profile["engines"]["d1"]["port"] == 8412 and profile["engines"]["gemma"]["role"] == "vision"  # Gemma stays selectable
    # Ghost 3.8 (brain + looker in one) stays selectable, not resident: the user preferred QC v2's film (2026-11-08)
    assert "--vision" in profile["engines"]["ghost"]["argv"] and profile["engines"]["ghost"]["port"] != profile["engines"]["qwen"]["port"]
    assert {"ghost"} <= set(profile["autopilot"]["director_choices"]) & set(profile["autopilot"]["eyes_choices"])
    assert all("ghost" not in schedules[n]["keep"] for n in schedules)
    assert judge.make(profile) is not None and judge.make({**profile, "autopilot": {**profile["autopilot"], "judge": "eyes"}}) is None
    machine = Machine(110_000)
    reader = machine.reader()
    registry = Registry({"engines": {"asr": {"role": "asr", "mode": "inprocess", "resident_gb": 5.4}}}, reader)
    for name in ("qwen", "kokoro", "cosmos", "d1"):
        spec = {**profile["engines"][name], "mode": "managed"}
        registry.engines[name] = LoadingEngine(name, spec, reader, machine)
    governor = Governor({**profile["governor"], "floor_gb": 20}, registry, reader, renderer=FakeRenderer())
    registry.set_admit(governor.admit)
    for name in ("qwen", "kokoro", "cosmos"):
        registry[name].bring_up()
    d1 = judge.make(profile, registry)
    with d1.request():  # a check: the judge loads for it
        assert registry["d1"].loaded()
    assert d1.loads and registry["d1"].loaded()

    async def render():
        grant = await governor.enter("live", frames=243)
        assert "d1" in grant["evicted"] and not registry["d1"].loaded()
        with pytest.raises(Held):  # nothing may load the judge while the render has the memory
            registry["d1"].acquire(wait=False)
        assert registry["qwen"].loaded() and registry["kokoro"].loaded()  # the live schedule keeps them
        await governor.leave("the clip is made")
        grant = await governor.enter("keyframe")
        assert "d1" not in grant["evicted"]  # a keyframe keeps it
        await governor.leave("the keyframe is drawn")
    asyncio.run(render())
    with d1.request():  # between renders again: loaded for the checks
        assert registry["d1"].loaded()
    assert len(d1.loads) == 2


def test_the_status_beside_the_preview_counts_checks_and_retakes():
    run = {"state": "running", "settings": {"segments": 3}, "current": {"stage": "checks", "segment": 2, "text": "Segment 2: the dense checks"},
           "check_stats": {"requests": 12, "questions": 40, "failed_answers": 2, "votes": 1, "seconds": 9.5},
           "story_so_far": {"text": "The keeper lights the lamp."}, "film": {"seconds": 10.1},
           "segments": [{"index": 1, "status": "done", "beat": "lamp lit", "takes": [{}, {}]},
                        {"index": 2, "status": "rendered", "beat": "the storm comes", "takes": [{}], "keyframes": [{"round": 0}, {"round": 0}]}]}
    said = status(run)
    assert said["beat"] == "the storm comes" and said["retakes"] == 1 and said["keyframes_redrawn"] == 1
    assert said["checks"]["questions"] == 40 and said["story_so_far"] == "The keeper lights the lamp." and said["done"] == 1


def test_the_panel_shows_autopilot_beside_the_live_view():
    """The page's apUpdate (voice_agent/static/index.html), run in node with the page's $ stood in."""

    node = shutil.which("node")
    if not node:
        pytest.skip("needs node")
    page = (ROOT / "voice_agent" / "static" / "index.html").read_text()
    assert 'id="vAP"' in page and '<option value="autopilot">' in page
    code = page[page.index("function apUpdate(a) {"):page.index("function loadStill() {")]
    harness = """
const els = {}; const $ = (id) => els[id] || (els[id] = {hidden: false, textContent: ""});
""" + code + """
apUpdate({stage: "render", segment: 2, count: 3, beat: "the storm comes", checks: {requests: 12, questions: 40, failed_answers: 2},
          retakes: 1, keyframes_redrawn: 0, film: 10.12, story_so_far: "The keeper lights the lamp.", line: "Segment 2: rendering."});
const shown = {hidden: els.vAP.hidden, stage: els.apStage.textContent, beat: els.apPiece.textContent, checks: els.apChecks.textContent,
               retakes: els.apRetakes.textContent, film: els.apFilm.textContent, story: els.apStory.textContent};
apUpdate(null);
console.log(JSON.stringify({...shown, after: els.vAP.hidden}));
"""
    done = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    shown = json.loads(done.stdout.strip().splitlines()[-1])
    assert shown["hidden"] is False and shown["after"] is True
    assert shown["stage"] == "render · piece 2 of 3" and "the storm comes" in shown["beat"]
    assert shown["checks"] == "checks 12 asked · 40 questions · 2 failed" and shown["retakes"] == "retakes 1"
    assert shown["film"] == "film 10.1 s" and shown["story"] == "The keeper lights the lamp."
