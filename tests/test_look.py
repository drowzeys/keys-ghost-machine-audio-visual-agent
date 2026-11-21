"""The look (gma/look.py, gma/neon.py): the Neon table, the copies beside a clip, what a route serves, and the voice."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from gma import neon
from gma.look import LOOKS, LookAgent, LookError, LookService, is_copy, parse


# ---------- the Neon table ----------

def test_neon_sends_greys_to_cobalt_and_hot_red_and_keeps_a_strong_hue():
    r, g, b = neon.grade((0.3, 0.3, 0.3))  # a dark grey: cobalt
    assert b > 2 * r and b > 2 * g
    r, g, b = neon.grade((0.8, 0.8, 0.8))  # a bright grey: hot red
    assert r > 2 * g and r > 2 * b
    r, g, b = neon.grade((0.1, 0.9, 0.2))  # a green light stays green
    assert g > r and g > b
    r, g, b = neon.grade((1.0, 0.8, 0.0))  # yellow turns toward red: less green than it had
    assert r == pytest.approx(1.0, abs=0.02) and g < 0.6
    assert neon.grade((0, 0, 0))[0] < 0.02 and min(neon.grade((1, 1, 1))) > 0.8
    assert all(0 <= x <= 1 for v in ((0.5, 0.2, 0.9), (1, 0, 0), (0, 0, 1)) for x in neon.grade(v))


def test_neon_cube_is_written_once(tmp_path):
    path = neon.cube(tmp_path / "t" / "neon.cube")
    rows = path.read_text().splitlines()
    assert rows[1] == f"LUT_3D_SIZE {neon.SIZE}" and len(rows) == 2 + neon.SIZE ** 3
    at = path.stat().st_mtime_ns
    assert neon.cube(path).stat().st_mtime_ns == at
    path.write_text("# another version\n")
    assert len(neon.cube(path).read_text().splitlines()) == 2 + neon.SIZE ** 3


# ---------- the copies ----------

class FakeStudio:
    """The Studio's client as the look uses it: bring, inside, submit, wait, fetch, home."""

    def __init__(self, home: Path, fail: bool = False) -> None:
        self._home, self.fail, self.jobs = home, fail, []
        (home / "ghost-inputs").mkdir(parents=True)
        (home / "renders").mkdir()

    def home(self) -> Path:
        return self._home

    def inside(self, path):
        return None

    def bring(self, path) -> str:
        shutil.copyfile(path, self._home / "ghost-inputs" / "in.mp4")
        return "ghost-inputs/in.mp4"

    def submit(self, kind, params, title=""):
        self.jobs.append((kind, dict(params), title))
        return {"id": "j1"}

    def wait(self, job_id, limit_s=0):
        if self.fail:
            return {"status": "failed", "error": "no libx265"}
        outputs = ["renders/j1-hlg.mp4"] + (["renders/j1-sdr.mp4"] if self.jobs[-1][1]["sdr"] else [])
        for name in outputs:
            (self._home / name).write_bytes(b"hdr" if name.endswith("-hlg.mp4") else b"sdr")
        return {"status": "done", "outputs": outputs}

    def fetch(self, path, dest) -> None:
        shutil.copyfile(self._home / path, dest)


def service(tmp_path, **more) -> tuple[LookService, FakeStudio, Path]:
    studio = FakeStudio(tmp_path / "home", **more)
    clip = tmp_path / "run" / "0.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"plain")
    return LookService(tmp_path / "look", lambda: studio), studio, clip


def test_natural_serves_the_clip_itself_and_asks_for_nothing(tmp_path):
    look, studio, clip = service(tmp_path)
    assert look.shown(clip) == clip and look.shown(clip, hdr=True) == clip
    assert not look.queue and not studio.jobs


def test_hdr_copies_are_made_beside_the_clip_and_served_by_screen(tmp_path):
    look, studio, clip = service(tmp_path)
    assert look.set("hdr")["look"] == "hdr"
    assert look.shown(clip) == clip  # not made yet: the plain clip, and the copy is asked for (no loop here: by hand)
    assert look.queue == [(clip, "hdr")]
    look.finish(clip, "hdr")
    assert studio.jobs == [("hlg", {"video": "ghost-inputs/in.mp4", "strength": 1.0, "sdr": True}, "Ghost Studio · HDR look")]
    assert look.shown(clip, hdr=True).read_bytes() == b"hdr" and look.shown(clip).read_bytes() == b"sdr"
    assert look.shown(clip).name == "0.look-hdr.mp4" and is_copy(look.shown(clip))
    assert look.shown(look.shown(clip)) == look.shown(clip)  # a copy is never finished again
    # nothing of ours is left in the Studio's home
    assert not list((tmp_path / "home" / "renders").iterdir()) and not list((tmp_path / "home" / "ghost-inputs").iterdir())
    clip.write_bytes(b"made again")  # a newer clip: its old copies are not shown
    import os
    os.utime(clip, (clip.stat().st_atime, clip.stat().st_mtime + 5))
    assert look.shown(clip) == clip


def test_a_failed_finish_is_said_and_the_plain_clip_stays(tmp_path):
    look, studio, clip = service(tmp_path, fail=True)
    look.set("hdr")

    async def go():
        look.want(clip)
        await look.task

    asyncio.run(go())
    assert "no libx265" in look.state()["error"] and look.shown(clip) == clip
    assert not look.queue  # not asked for again until the look changes
    assert not list((tmp_path / "home" / "ghost-inputs").iterdir())


def test_the_look_is_kept_and_a_change_finishes_what_is_on_show(tmp_path):
    look, studio, clip = service(tmp_path)
    look.recent = lambda: [clip]
    heard = []
    look.on_change.append(lambda: heard.append(look.stamp))
    look.set("neon")
    assert look.queue == [(clip, "neon")] and heard == [1]
    assert LookService(tmp_path / "look", lambda: studio).look == "neon"
    with pytest.raises(LookError):
        look.set("sepia")
    look.set("natural")
    assert not look.queue and look.state()["working"] == 0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_neon_lays_the_table_then_asks_for_a_gentler_hdr_finish(tmp_path):
    look, studio, clip = service(tmp_path)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x36:r=8:d=0.5", "-pix_fmt", "yuv420p", str(clip)], check=True)
    look.set("neon")
    look.finish(clip, "neon")
    kind, params, _ = studio.jobs[0]
    assert kind == "hlg" and params["sdr"] is False and params["strength"] == LOOKS["neon"]["strength"] < 1
    sdr = look.shown(clip)
    assert sdr.name == "0.look-neon.mp4" and look.shown(clip, hdr=True).read_bytes() == b"hdr"
    pixel = subprocess.run(["ffmpeg", "-v", "error", "-i", str(sdr), "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, check=True).stdout
    assert pixel[2] > pixel[1] + 40  # the grey clip is blue now


# ---------- the Sonar look ----------

def test_sonar_plans_its_pulses_so_the_last_wave_crosses_before_the_end():
    from gma import sonar

    for seconds in (3, 5, 10, 20, 60):
        drawn = sonar.plan(seconds)
        pulses = drawn["pulses"]
        assert 1 <= len(pulses) <= 4 and pulses == sorted(pulses) and 0 < pulses[0] <= 1.0
        assert pulses[-1] + drawn["far"] * 1.25 / drawn["speed"] <= seconds - 0.3  # across, and some colour after
        assert all(b - a >= 1.5 for a, b in zip(pulses, pulses[1:]))
    assert len(sonar.plan(3)["pulses"]) == 1 and sonar.plan(3)["speed"] > sonar.SPEED  # a short clip: one fast wave
    assert len(sonar.plan(20)["pulses"]) == 4 and sonar.plan(20)["speed"] == sonar.SPEED
    assert sonar.plan(10, {"far": 12, "at": [0.5, 0.3]})["at"] == [0.5, 0.3]


def test_sonar_wave_follows_depth_not_a_circle_on_the_picture():
    np = pytest.importorskip("numpy")
    from gma.sonar_worker import Wave, subject

    # a wall 3.5 m away fills the left half; the right half is open, 18 m away; the subject stands in the middle, in front
    depth = np.full((96, 172), 18.0, np.float32)
    depth[:, :86] = 3.5
    depth[30:80, 76:96] = 3.0
    x, y, near = subject(depth)
    assert 0.4 < x < 0.6 and 0.4 < y < 0.7 and near == pytest.approx(3.0, abs=0.1)
    wave = Wave(172, 96, 24.0, [0.0, 5.0], speed=5.0, far=20.0, dist=3.0)
    reveal, light = wave.at(24, depth)  # one second after the first pulse: 5 m of reach
    assert reveal[48, 70] > 0.3 and reveal[48, 20] > 0.3  # the wall beside the subject is lit, to its far end
    assert reveal[48, 110] == 0 and reveal[48, 160] == 0  # the open side, as near on the picture, is not reached yet
    assert light.max() > 0.2
    late, _ = wave.at(24 * 4, depth)  # the early pulse's colour has died back on the wall
    assert late[48, 20] < 0.2
    end, _ = wave.at(24 * 12, depth)  # the last pulse stays, everywhere
    assert end.min() > 0.99


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_sonar_makes_both_copies_with_sound_from_the_hdr_looks_colour(tmp_path):
    np = pytest.importorskip("numpy")
    look, studio, clip = service(tmp_path)
    make = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=8:d=4", "-pix_fmt", "yuv420p"]
    subprocess.run(make + [str(clip)], check=True)

    def wait(job_id, limit_s=0):  # the Studio's finish, stood in for by real small clips (the worker reads them)
        subprocess.run(make + [str(studio.home() / "renders/j1-sdr.mp4")], check=True)
        subprocess.run(make + ["-c:v", "libx264", str(studio.home() / "renders/j1-hlg.mp4")], check=True)
        return {"status": "done", "outputs": ["renders/j1-hlg.mp4", "renders/j1-sdr.mp4"]}

    studio.wait = wait
    depth = tmp_path / "depth.npy"
    np.save(depth, np.full((32, 36, 64), 3.0, np.float16))
    look.spec = {"sonar": {"depth": str(depth)}}
    look.set("sonar")
    look.finish(clip, "sonar")
    assert look.made(clip, "hdr") and look.made(clip, "sonar")
    sdr = look.shown(clip)
    assert sdr.name == "0.look-sonar.mp4" and look.shown(clip, hdr=True).name == "0.look-sonar.hdr.mp4"
    kinds = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(sdr)],
                           capture_output=True, text=True, check=True).stdout.split()
    assert sorted(kinds) == ["audio", "video"]

    def pixel(at):
        return subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(sdr), "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout

    first, last = pixel(0), pixel(3.7)
    assert abs(first[0] - first[1]) < 40 and last[0] > last[1] + 100  # grey before the pulse, the clip's red after it


def test_sonar_failure_is_a_look_error(tmp_path):
    look, studio, clip = service(tmp_path)  # the clip is not a video: nothing to measure
    look.spec = {"sonar": {"python": "/nonexistent/python"}}
    with pytest.raises(LookError):
        look.finish(clip, "sonar")


# ---------- the voice ----------

@pytest.mark.parametrize("words, act", [
    ("neon look", ("set", "neon")), ("Make it HDR.", ("set", "hdr")), ("switch to the neon grade", ("set", "neon")),
    ("natural colours please", ("set", "natural")), ("turn off the HDR", ("set", "natural")), ("turn the look off", ("set", "natural")),
    ("sonar reveal", ("set", "sonar")), ("do the sonar reveal on this one", ("set", "sonar")), ("make it sonar", ("set", "sonar")),
    ("turn off the sonar", ("set", "natural")), ("make a video of a bat with a sonar reveal", None),
    ("Neon.", ("set", "neon")), ("vivid colors", ("set", "neon")), ("change the look", ("ask", "")), ("what looks are there", ("ask", "")),
    ("make a video of a neon city at night", None), ("show me a clip of a normal day with neon colours", None),
    ("that looks normal to me, what do you think about it", None), ("how is the weather", None), ("look at this", None),
])
def test_words_for_the_look(words, act):
    assert parse(words) == act


def test_the_agent_offers_the_looks_and_takes_the_answer(tmp_path):
    look, studio, clip = service(tmp_path)
    agent, sent = LookAgent(look), []

    async def go():
        assert agent.wants("change the look") and not agent.wants("the middle one I think")
        said, _ = await agent.handle("change the look", lambda kind, value: sent.append((kind, value)))
        assert said == "Natural, HDR, Neon, or Sonar. Which look?" and '"what": "ask"' in sent[0][1]
        assert agent.wants("the neon one I think")  # an answer needs no word about looks
        said, detail = await agent.handle("the neon one I think", lambda kind, value: sent.append((kind, value)))
        assert said.startswith("Neon look on.") and detail == "look neon" and look.look == "neon"
        assert (await agent.handle("neon look", lambda *a: None))[0] == "It's on the Neon look already."
        assert (await agent.handle("turn the look off", lambda *a: None))[0] == "Back to the natural look."

    asyncio.run(go())


# ---------- the routes ----------

def test_the_routes_and_the_agent_are_added_ahead_of_the_others(tmp_path):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from gma.look import register

    studio = FakeStudio(tmp_path / "home")
    told = []
    video = SimpleNamespace(root=tmp_path / "video", ffmpeg="ffmpeg", run=None, changed=lambda: told.append(1), look=None)
    engines = SimpleNamespace(video=video, supervisor=SimpleNamespace(renderer=SimpleNamespace(client=studio)), agents={"video": object()})
    app = FastAPI()
    look = register(app, engines, {})
    assert list(engines.agents) == ["look", "video"] and video.look is look and engines.look is look
    page = TestClient(app)
    assert [one["id"] for one in page.get("/api/look").json()["looks"]] == ["natural", "hdr", "neon", "sonar"]
    assert page.post("/api/look", json={"look": "neon"}).json()["look"] == "neon" and told
    assert page.post("/api/look", json={"look": "sepia"}).status_code == 400
    assert page.post("/api/look", content=b"not json").status_code == 400
