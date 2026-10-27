"""The video panel's job service: a run of clips made by a render host, one at a time, kept on disk clip by clip so a
restart picks up where it stopped, with its state pushed to the page as it changes.

    POST /api/video/start   {"prompt": "...", "mode": "single"|"continue", "clips": n, "seconds": s}
    POST /api/video/stop
    GET  /api/video/status  the run as it is now (the page polls this when it cannot hold an event stream)
    GET  /api/video/events  the same, as server-sent events, each time it changes (at most a few a second)
    GET  /api/video/clips   the finished clips of the run
    GET  /api/video/clip/{run}/{n}.mp4 | .jpg   one clip, its poster
    GET  /api/video/preview.jpg                 the picture shown while the current clip renders (see "previews")
    GET  /api/video/film                        the run's clips joined into one film (made on demand)

A profile turns it on with a "video" section:

    "video": {"renderer": "studio", "url": "http://127.0.0.1:7870", "render_host": "remote",
              "key_file": "~/.config/boo/studio-key", "key_env": "STUDIO_ACCESS_KEY_VALUE",
              "preset": "small", "quality": "fh8", "seconds": 5, "max_clips": 6, "keyframe": false,
              "dir": "~/.boo-video", "local_min_free_gb": 64}

Renderers (one interface, ``Renderer``):
  studio     Imagination Studio's own HTTP API: a clip is a "video" job in its render queue (POST /api/jobs), followed
             with GET /api/jobs/{id} and fetched from /files/. A "continue" clip opens on the last frame of the clip
             before (the Studio's ``first_frame_from``); with "keyframe" a still is made first (a "scout" job).
  hyperflow  a stub for a future infinity engine that streams its own clips and previews; it refuses for now.

Where clips are made: never beside the live voice models unless that machine has room. "render_host" says where the
renderer runs: "remote" (another machine; a URL on 127.0.0.1 that is an ssh tunnel counts) or "local" (this one),
which is refused unless this machine has ``local_min_free_gb`` free. The Studio's own memory check (/api/estimate) is
asked too, and its refusal said as it is.

Previews: what the engine exposes is said in each state ("preview": "engine" | "still" | "noise"). The Studio's job
API gives the sampler's step count and an estimate, no pictures, so while a clip renders the page shows the clip's
opening still (the keyframe, or the last frame of the clip before) or noise, sharpening with the step count, labelled
"progress only". A renderer that can give real step previews names each one in ``Progress.preview`` and the page
shows those instead (docs/video-panel.md says what the Studio would need for that).

Continue mode is the start of the infinity loop: Bolt (the profile's brain) plans each next beat from the story so far
and the last clip's prompt, and the next clip opens on the last one's final frame. A ``Checker`` looks at each clip
before it is kept; the one here passes everything, and the dense checks come later behind the same interface.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

STATES = ("idle", "planning", "keyframe", "rendering", "checking", "done", "stopped", "failed")


class Refused(Exception):
    """Said to the person as it is: why a run cannot start or go on."""


# ---------------------------------------------------------------------------------------------------- the renderers

@dataclass
class Progress:
    state: str  # queued, rendering, done, failed, cancelled
    at: int | None = None  # sampler step
    of: int | None = None
    stage: str = ""
    eta_s: float | None = None
    output: str | None = None  # where the renderer keeps the finished clip (its own path)
    error: str | None = None
    preview: str | None = None  # a real picture of the clip as it renders, when the engine gives one (its own path)


class Renderer:
    """One render host. A clip is ``submit``ted, ``poll``ed until done, then ``fetch``ed to a local file."""

    name = "renderer"
    previews = False  # True when its Progress names real pictures of the clip as it renders

    async def room(self, clip: dict) -> str | None:
        """None when the clip can be made now, else why not (said aloud)."""
        return None

    async def keyframe(self, prompt: str, seed: int) -> str:
        """Starts a still for the clip to open on; returns a handle polled like a clip."""
        raise Refused("This render host makes no keyframes.")

    async def submit(self, clip: dict) -> str:
        raise NotImplementedError

    async def poll(self, handle: str) -> Progress:
        raise NotImplementedError

    async def picture(self, ref: str) -> bytes | None:
        """The bytes of a picture the renderer made (a keyframe, a step preview)."""
        return None

    async def fetch(self, output: str, dest: Path) -> None:
        raise NotImplementedError

    async def cancel(self, handle: str) -> None:
        return None


class StudioRenderer(Renderer):
    """Imagination Studio's render queue, through the same routes its page uses."""

    name = "studio"

    def __init__(self, spec: dict, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = spec.get("url", "http://127.0.0.1:7870").rstrip("/")
        self.key_file, self.key_env = spec.get("key_file"), spec.get("key_env")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=4), transport=transport)
        self.home: str | None = None  # the Studio's folder, from /api/state: outputs are paths inside it

    def key(self) -> str:
        if self.key_env and os.environ.get(self.key_env, "").strip():
            return os.environ[self.key_env].strip()
        if self.key_file:
            try:
                return Path(self.key_file).expanduser().read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    async def call(self, method: str, path: str, body: dict | None = None):
        key = self.key()
        try:
            reply = await self.client.request(method, self.url + path, json=body, headers={"X-Studio-Key": key} if key else {})
        except httpx.HTTPError as error:
            raise Refused(f"I can't reach the Studio right now ({type(error).__name__}).") from None
        if reply.status_code == 401:
            raise Refused("The Studio wants its access key, and the one I have doesn't work.")
        if reply.status_code >= 400:
            try:
                said = reply.json()
                said = said.get("error") or said.get("detail") or reply.reason_phrase
            except ValueError:
                said = reply.reason_phrase
            raise Refused(f"The Studio said: {str(said)[:160]}")
        return reply.json()

    async def room(self, clip: dict) -> str | None:
        """The Studio's own memory check for this clip (a DGX Spark answers it; a Mac says nothing)."""
        said = await self.call("POST", "/api/estimate", self.params(clip, None))
        memory = said.get("memory") or {}
        if memory and memory.get("fits") is False:
            return f"The render host has no room for this clip now: {str(memory.get('reason') or '')[:140]}"
        return None

    @staticmethod
    def params(clip: dict, output: str | None) -> dict:
        params = {"prompt": clip["prompt"], "preset": clip.get("preset", "small"), "quality": clip.get("quality", "fh8"),
                  "seconds": clip.get("seconds", 5), "seed": int(clip.get("seed", 0))}
        if clip.get("opens_on"):  # continuity: the last frame of the clip before (a path in the Studio's folder)
            params["first_frame_from"] = clip["opens_on"]
        elif clip.get("image"):  # a keyframe the Studio made
            params["image"] = clip["image"]
        if output:
            params["output"] = output
        return params

    async def keyframe(self, prompt: str, seed: int) -> str:
        job = await self.call("POST", "/api/jobs", {"kind": "scout", "title": "Boo · keyframe",
                                                    "params": {"prompt": prompt, "count": 1, "seed": seed,
                                                               "width": 864, "height": 480}})
        return job["id"]

    async def submit(self, clip: dict) -> str:
        title = f"Boo · {clip.get('title') or 'clip'}"
        job = await self.call("POST", "/api/jobs", {"kind": "video", "title": title[:80], "params": self.params(clip, None)})
        return job["id"]

    async def poll(self, handle: str) -> Progress:
        job = await self.call("GET", f"/api/jobs/{handle}")
        status = job.get("status")
        step = job.get("progress") if isinstance(job.get("progress"), dict) else {}
        eta = None
        if job.get("estimate") and status in ("running", "queued"):
            eta = max(0.0, float(job["estimate"]) - float(job.get("elapsed") or 0))
        outputs = [o for o in job.get("outputs") or [] if isinstance(o, str)]
        output = next((o for o in outputs if o.endswith((".mp4", ".png", ".jpg"))), None)
        stage = str(step.get("stage") or "")
        if not step and status == "running" and re.search(r"starting ComfyUI|loads the models|loading", str(job.get("log") or "")[-400:]):
            stage = "loading the models"  # a word from the job's log, never the log itself (it holds the host's paths)
        state = {"queued": "queued", "running": "rendering", "done": "done", "failed": "failed",
                 "cancelled": "cancelled"}.get(status, "rendering")
        # a real step preview, should the Studio offer one as a job's "preview" path (today's Studio gives none:
        # docs/video-panel.md says what it would take); the step count and estimate are all it gives now
        preview = job.get("preview") if isinstance(job.get("preview"), str) else None
        return Progress(state, step.get("at"), step.get("of"), stage, eta, output,
                        (job.get("error") or None) if state == "failed" else None, preview)

    async def picture(self, ref: str) -> bytes | None:
        return await self.bytes_of(ref)

    async def relative(self, output: str) -> str:
        if self.home is None:
            self.home = str((await self.call("GET", "/api/state")).get("home") or "")
        if self.home and output.startswith(self.home.rstrip("/") + "/"):
            return output[len(self.home.rstrip("/")) + 1:]
        return output.lstrip("/")

    async def bytes_of(self, output: str) -> bytes | None:
        key = self.key()
        try:
            reply = await self.client.get(f"{self.url}/files/{await self.relative(output)}",
                                          headers={"X-Studio-Key": key} if key else {})
        except (httpx.HTTPError, Refused):
            return None
        return reply.content if reply.status_code == 200 else None

    async def fetch(self, output: str, dest: Path) -> None:
        key = self.key()
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            async with self.client.stream("GET", f"{self.url}/files/{await self.relative(output)}",
                                          headers={"X-Studio-Key": key} if key else {}) as reply:
                if reply.status_code != 200:
                    raise Refused(f"The Studio would not hand over the clip ({reply.status_code}).")
                with part.open("wb") as out:
                    async for chunk in reply.aiter_bytes(1 << 16):
                        out.write(chunk)
        except httpx.HTTPError as error:
            raise Refused(f"The clip's download broke off ({type(error).__name__}).") from None
        part.replace(dest)

    async def cancel(self, handle: str) -> None:
        try:
            await self.call("POST", f"/api/jobs/{handle}/cancel", {})
        except Refused:
            pass


class HyperFlowRenderer(Renderer):
    """The future infinity engine: one long-lived generation that hands over clip after clip, each opening on the
    latent tail of the one before, with true step previews. Not built yet; it refuses so a profile can name it."""

    name = "hyperflow"
    previews = True

    def __init__(self, spec: dict) -> None:
        self.url = spec.get("url")

    async def room(self, clip: dict) -> str | None:
        return "The HyperFlow infinity engine isn't built yet; use the Studio renderer for now."

    async def submit(self, clip: dict) -> str:
        raise Refused("The HyperFlow infinity engine isn't built yet.")


RENDERERS = {"studio": StudioRenderer, "hyperflow": HyperFlowRenderer}


# ---------------------------------------------------------------------------------------------------- planning, checks

FIRST = ("Turn this idea into one prompt for a {seconds}-second video clip: {idea}\n\nWrite what the camera sees and "
         "hears in two to four plain sentences: the subject, the place, the light, the movement of subject and camera. "
         "Concrete and physical. Never say what must not be seen (no \"no X\"). No real, named people. Reply with the "
         "prompt only.")
NEXT = ("We are making an open-ended film, {seconds} seconds a clip, each clip opening on the last frame of the one "
        "before. The idea: {idea}\n\nThe story so far, one beat a clip:\n{beats}\n\nThe last clip's prompt:\n{last}\n\n"
        "Write the next clip: something new happens that follows from the last frame (no restaging of an earlier beat, "
        "no repeat of the last one); same people, place and look unless the story moves on. {note}Reply with JSON "
        "only: {{\"beat\": \"the story beat in under twelve words\", \"prompt\": \"two to four plain sentences: what the "
        "camera sees and hears, the movement of subject and camera\"}}.")
PLANNER_SYSTEM = "You are a film director writing prompts for a video model: vivid, concrete, physical, photographable."


class Planner:
    """Writes each clip's prompt with the profile's brain (Bolt). Waits for a quiet moment in the conversation first, so
    the voice never queues behind it; falls back to the idea itself when the brain does not answer."""

    def __init__(self, spec: dict | None, busy: Callable[[], bool] | None = None) -> None:
        self.spec = spec
        self.busy = busy or (lambda: False)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=4))

    async def ask(self, prompt: str, max_tokens: int) -> str:
        if not self.spec:
            return ""
        for _ in range(40):  # at most ~10 s for the conversation to go quiet
            if not self.busy():
                break
            await asyncio.sleep(0.25)
        extra = {k: v for k, v in (self.spec.get("extra") or {}).items() if k not in ("max_tokens", "temperature")}
        body = {"model": self.spec["model"], "messages": [{"role": "system", "content": PLANNER_SYSTEM},
                                                          {"role": "user", "content": prompt}],
                **extra, "max_tokens": max_tokens, "temperature": 0.8, "stream": False}
        try:
            reply = await self.client.post(self.spec["url"].rstrip("/") + "/chat/completions", json=body)
            return (reply.json()["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return ""

    async def first(self, idea: str, seconds: float) -> dict:
        said = clean(await self.ask(FIRST.format(idea=idea, seconds=seconds), 220))
        return {"beat": idea[:80], "prompt": said if len(said) > 20 else idea}

    async def next(self, idea: str, clips: list[dict], seconds: float, note: str = "") -> dict:
        beats = "\n".join(f"{i + 1}. {c.get('beat') or c['prompt'][:80]}" for i, c in enumerate(clips[-8:]))
        said = await self.ask(NEXT.format(idea=idea, beats=beats, last=clips[-1]["prompt"], seconds=seconds,
                                          note=f"Put this right: {note} " if note else ""), 320)
        found = re.search(r"\{.*\}", said, re.S)
        try:
            data = json.loads(found.group(0)) if found else {}
        except ValueError:
            data = {}
        prompt = clean(str(data.get("prompt") or ""))
        if len(prompt) < 20:  # no plan: carry on from the last frame with the same prompt
            return {"beat": "the scene carries on", "prompt": clips[-1]["prompt"]}
        return {"beat": clean(str(data.get("beat") or ""))[:80] or "next", "prompt": prompt}


def clean(text: str) -> str:
    text = re.sub(r"^```\w*|```$", "", text.strip()).strip()
    return re.sub(r"\s+", " ", text.strip().strip('"')).strip()


@dataclass
class Verdict:
    ok: bool = True
    notes: list[str] = field(default_factory=list)  # minor findings: told to the planner for the next clip
    retake: bool = False  # a major finding: the clip is made again (with ``notes`` in the planner's hands)


class Checker:
    """Looks at a finished clip before it is kept. This one passes everything; the dense yes/no checks with Cosmos
    (windows of frames, the join between clips, identity and wardrobe against a continuity bible, pace) come later
    behind this same call (docs/video-panel.md)."""

    async def check(self, clip_file: Path, clip: dict, run: dict) -> Verdict:
        return Verdict()


# ---------------------------------------------------------------------------------------------------- the host's room

def free_gb() -> float | None:
    """Memory this machine can give without swapping: MemAvailable on Linux, free + inactive + speculative on macOS."""

    try:
        if platform.system() == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1 << 20)
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = sum(int(re.search(rf"Pages {name}:\s+(\d+)", out).group(1)) for name in ("free", "inactive", "speculative"))
            return pages * page / (1 << 30)
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None
    return None


def find_ffmpeg(given: str | None = None) -> str | None:
    for path in (given, shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if path and Path(path).expanduser().is_file():
            return str(Path(path).expanduser())
    return None


# ---------------------------------------------------------------------------------------------------- the service

class VideoService:
    """One run at a time; every change is written to ``run.json`` (a checkpoint a clip) and sent to the listeners."""

    def __init__(self, spec: dict, brain: dict | None = None, renderer: Renderer | None = None,
                 checker: Checker | None = None, busy: Callable[[], bool] | None = None) -> None:
        self.spec = spec
        self.renderer = renderer or RENDERERS.get(spec.get("renderer", "studio"), StudioRenderer)(spec)
        self.planner = Planner(spec.get("brain") or brain, busy)
        self.checker = checker or Checker()
        self.root = Path(spec.get("dir", "~/.boo-video")).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = find_ffmpeg(spec.get("ffmpeg"))
        self.poll_s = float(spec.get("poll_s", 1.0))
        self.run: dict | None = None
        self.task: asyncio.Task | None = None
        self.live: dict = {}  # the clip at work: {at, of, stage, eta_s, preview, seq}; not written to disk
        self.still: bytes | None = None  # the picture the page shows while the clip renders
        self.preview_ref: str | None = None
        self.listeners: set[asyncio.Queue] = set()
        self.on_news: list[Callable[[str, dict], None]] = []  # spoken updates: ("clip"|"done"|"failed", run)

    # ----- what the page sees -----

    def snapshot(self) -> dict:
        run = self.run
        if not run:
            return {"run": None, "state": "idle", "clips": [], "renderer": self.renderer.name,
                    "previews": "engine" if self.renderer.previews else "progress"}
        clips = [{"index": c["index"], "beat": c.get("beat"), "prompt": c["prompt"], "seconds": c.get("seconds"),
                  "url": f"/api/video/clip/{run['id']}/{c['index']}.mp4",
                  "poster": f"/api/video/clip/{run['id']}/{c['index']}.jpg" if c.get("poster") else None}
                 for c in run["clips"] if c.get("state") == "done"]
        current = next((c for c in reversed(run["clips"]) if c.get("state") not in ("done",)), None)
        return {"run": run["id"], "state": run["state"], "mode": run["mode"], "idea": run["idea"],
                "count": run.get("count"), "message": run.get("message"), "renderer": self.renderer.name,
                "previews": "engine" if self.renderer.previews else "progress", "clips": clips,
                "current": {"index": current["index"], "beat": current.get("beat"), "prompt": current.get("prompt"),
                            "state": current.get("state"), **self.live} if current and run["state"] not in ("done", "stopped", "failed") else None,
                "film": f"/api/video/film?run={run['id']}" if clips else None}

    def changed(self) -> None:
        state = self.snapshot()
        for queue in list(self.listeners):
            if queue.full():
                try:
                    queue.get_nowait()  # a slow page gets the newest state, never a backlog
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(state)

    def news(self, kind: str) -> None:
        for hear in list(self.on_news):
            try:
                hear(kind, self.run or {})
            except Exception:
                pass

    # ----- disk -----

    def folder(self, run_id: str) -> Path:
        if not re.fullmatch(r"[\w-]{4,40}", run_id):
            raise Refused("No such run.")
        return self.root / run_id

    def save(self) -> None:
        if not self.run:
            return
        folder = self.folder(self.run["id"])
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "run.json.tmp"
        tmp.write_text(json.dumps(self.run, indent=1))
        tmp.replace(folder / "run.json")
        (self.root / "latest").write_text(self.run["id"])

    def load_latest(self) -> dict | None:
        try:
            run_id = (self.root / "latest").read_text().strip()
            return json.loads((self.folder(run_id) / "run.json").read_text())
        except (OSError, ValueError, Refused):
            return None

    def clip_file(self, run_id: str, index: int, ext: str) -> Path:
        return self.folder(run_id) / f"clip{int(index):03d}.{ext}"

    # ----- the host's room -----

    async def room(self, clip: dict) -> str | None:
        host = self.spec.get("render_host")
        if host is None:  # not said: a renderer on this machine's loopback is taken to be on this machine
            host = "local" if re.match(r"https?://(127\.0\.0\.1|localhost|\[::1\])[:/]", self.spec.get("url", "http://127.0.0.1")) else "remote"
        if host == "local":
            need, free = float(self.spec.get("local_min_free_gb", 64)), free_gb()
            if free is None or free < need:
                return (f"I won't make video on this machine while my voice models are on it: it has "
                        f"{'unknown' if free is None else round(free)} gigabytes free and video needs {need:g}. "
                        f"Set a render host in my profile.")
        return await self.renderer.room(clip)

    # ----- runs -----

    async def resume(self) -> None:
        """At start-up: a run that was at work when the agent stopped carries on from its last checkpoint."""
        run = self.load_latest()
        if run:
            self.run = run
            if run["state"] not in ("done", "stopped", "failed"):
                run["message"] = "resumed after a restart"
                self.task = asyncio.create_task(self.work())

    async def start(self, idea: str, mode: str = "single", count: int | None = None, seconds: float | None = None) -> dict:
        idea = re.sub(r"\s+", " ", (idea or "").strip())[:600]
        if not idea:
            raise Refused("Tell me what the video should show.")
        if self.task and not self.task.done():
            raise Refused("A video is already being made. Say stop the video first.")
        mode = "continue" if mode == "continue" else "single"
        most = int(self.spec.get("max_clips", 6))
        count = 1 if mode == "single" else max(1, min(int(count or most), most)) if most else int(count or 0) or None
        seconds = float(seconds or self.spec.get("seconds", 5))
        refusal = await self.room({"prompt": idea, "seconds": seconds, "preset": self.spec.get("preset", "small"),
                                   "quality": self.spec.get("quality", "fh8")})
        if refusal:
            raise Refused(refusal)
        self.run = {"id": time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "idea": idea, "mode": mode,
                    "count": count, "seconds": seconds, "state": "planning", "clips": [], "created": time.time(),
                    "renderer": self.renderer.name, "seed": int(self.spec.get("seed", 42))}
        self.save()
        self.task = asyncio.create_task(self.work())
        self.changed()
        return self.snapshot()

    async def stop(self) -> dict:
        run = self.run
        if not run or run["state"] in ("done", "stopped", "failed"):
            return self.snapshot()
        run["state"], run["message"] = "stopped", "stopped"
        current = next((c for c in reversed(run["clips"]) if c.get("state") not in ("done",)), None)
        if self.task and not self.task.done():
            self.task.cancel()
        if current:
            for handle in (current.get("job"), current.get("keyframe_job")):
                if handle and current.get("state") in ("rendering", "keyframe"):
                    await self.renderer.cancel(handle)
            current["state"] = "stopped"
        self.live, self.still = {}, None
        self.save()
        self.changed()
        return self.snapshot()

    async def work(self) -> None:
        run = self.run
        try:
            while run["count"] is None or len([c for c in run["clips"] if c["state"] == "done"]) < run["count"]:
                clip = next((c for c in run["clips"] if c["state"] not in ("done", "dropped")), None)
                if clip is None:
                    clip = await self.plan(run)
                await self.make(run, clip)
            run["state"], run["message"] = "done", None
            self.live, self.still = {}, None
            self.save()
            self.changed()
            self.news("done")
        except asyncio.CancelledError:
            raise
        except Refused as error:
            self.fail(run, str(error))
        except Exception as error:  # a bug or a host gone: said, the run kept for a retry
            self.fail(run, f"{type(error).__name__}: {str(error)[:160]}")

    def fail(self, run: dict, message: str) -> None:
        run["state"], run["message"] = "failed", message
        self.live, self.still = {}, None
        self.save()
        self.changed()
        self.news("failed")

    async def plan(self, run: dict) -> dict:
        run["state"] = "planning"
        self.live = {}
        self.changed()
        done = [c for c in run["clips"] if c["state"] == "done"]
        if not done:
            planned = await self.planner.first(run["idea"], run["seconds"])
        else:
            planned = await self.planner.next(run["idea"], done, run["seconds"], "; ".join(run.get("notes") or []))
        index = len(run["clips"])
        clip = {"index": index, **planned, "seconds": run["seconds"], "seed": run["seed"] + index, "state": "planned",
                "title": f"{run['idea'][:40]} · {index + 1}"}
        if done and done[-1].get("output"):
            clip["opens_on"] = done[-1]["output"]  # continuity: the next clip opens on this one's last frame
        run["clips"].append(clip)
        self.save()
        return clip

    async def make(self, run: dict, clip: dict) -> None:
        """One clip from plan to checkpoint; a clip with a job already (a restart) is followed, not sent again."""
        seq = [0]
        self.still = self.opening_still(run, clip)
        self.live = {"at": None, "of": None, "stage": "", "eta_s": None, "preview": "still" if self.still else "noise", "seq": 0}
        if self.spec.get("keyframe") and not clip.get("opens_on") and not clip.get("image"):
            run["state"] = clip["state"] = "keyframe"
            self.changed()
            if not clip.get("keyframe_job"):
                clip["keyframe_job"] = await self.renderer.keyframe(clip["prompt"], clip["seed"])
                self.save()
            done = await self.follow(clip["keyframe_job"], seq, stage="keyframe")
            clip["image"] = done.output
            picture = await self.renderer.picture(done.output)
            if picture:
                self.still = picture
                seq[0] += 1
                self.live.update(preview="still", seq=seq[0])
            self.save()
        run["state"] = clip["state"] = "rendering"
        self.changed()
        if not clip.get("job"):
            clip["job"] = await self.renderer.submit(clip)
            self.save()
        done = await self.follow(clip["job"], seq)
        clip["output"] = done.output
        dest = self.clip_file(run["id"], clip["index"], "mp4")
        if not dest.is_file():
            await self.renderer.fetch(done.output, dest)
        run["state"] = clip["state"] = "checking"
        self.live.update(stage="checking", at=None, of=None, eta_s=None)
        self.changed()
        await self.frames(dest, run["id"], clip["index"])
        clip["poster"] = self.clip_file(run["id"], clip["index"], "jpg").is_file()
        verdict = await self.checker.check(dest, clip, run)
        run["notes"] = verdict.notes
        if verdict.retake and clip.get("retakes", 0) < int(self.spec.get("retakes", 1)):
            clip.update(state="planned", job=None, output=None, retakes=clip.get("retakes", 0) + 1, seed=clip["seed"] + 1000)
            dest.unlink(missing_ok=True)
            self.save()
            return
        clip["state"] = "done"
        clip["finished"] = time.time()
        self.live, self.still = {}, None
        self.save()
        self.changed()
        self.news("clip")

    async def follow(self, handle: str, seq: list, stage: str = "") -> Progress:
        """Polls the job until it ends, sending its steps (and any real preview) to the page."""
        while True:
            got = await self.renderer.poll(handle)
            if got.state == "done":
                if not got.output:
                    raise Refused("The render finished without a file.")
                return got
            if got.state in ("failed", "cancelled"):
                raise Refused(f"The render {got.state}" + (f": {str(got.error)[:160]}" if got.error else "."))
            self.live.update(at=got.at, of=got.of, stage=stage or got.stage or got.state, eta_s=got.eta_s, queued=got.state == "queued")
            if got.preview and got.preview != self.preview_ref:  # a new step preview from the engine
                self.preview_ref = got.preview  # the renderer's own path: never sent to the page
                picture = await self.renderer.picture(got.preview)
                if picture:
                    self.still = picture
                    seq[0] += 1
                    self.live.update(preview="engine", seq=seq[0])
            self.changed()
            await asyncio.sleep(self.poll_s)

    def opening_still(self, run: dict, clip: dict) -> bytes | None:
        """The picture the clip opens on, as far as is known here: the last frame of the clip before."""
        if clip["index"] and run["clips"][clip["index"] - 1].get("state") == "done":
            last = self.clip_file(run["id"], clip["index"] - 1, "last.jpg")
            if last.is_file():
                return last.read_bytes()
        return None

    async def frames(self, clip: Path, run_id: str, index: int) -> None:
        """The clip's poster (its first second) and its last frame (the next clip's opening still), with ffmpeg."""
        if not self.ffmpeg:
            return
        for args, out in ((["-ss", "0.4", "-i", str(clip), "-frames:v", "1", "-vf", "scale=320:-2"], "jpg"),
                          (["-sseof", "-0.25", "-i", str(clip), "-frames:v", "1", "-vf", "scale=640:-2", "-update", "1"], "last.jpg")):
            proc = await asyncio.create_subprocess_exec(self.ffmpeg, "-v", "error", "-y", *args, "-q:v", "4",
                                                        str(self.clip_file(run_id, index, out)),
                                                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.wait()

    async def film(self, run_id: str | None = None) -> Path:
        """The run's clips joined in order (stream copy; re-encoded when the clips differ), made again when a clip was
        added since."""
        run = self.run if not run_id or (self.run and self.run["id"] == run_id) else \
            json.loads((self.folder(run_id) / "run.json").read_text())
        if not run:
            raise Refused("There is no video yet.")
        files = [self.clip_file(run["id"], c["index"], "mp4") for c in run["clips"] if c.get("state") == "done"]
        files = [f for f in files if f.is_file()]
        if not files:
            raise Refused("There are no finished clips to join yet.")
        out = self.folder(run["id"]) / f"film-{len(files)}.mp4"
        if out.is_file():
            return out
        if len(files) == 1:
            return files[0]
        if not self.ffmpeg:
            raise Refused("Joining clips needs ffmpeg on this machine.")
        listing = self.folder(run["id"]) / "film.txt"
        listing.write_text("".join(f"file '{f.name}'\n" for f in files))
        for codec in (["-c", "copy"], ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]):
            proc = await asyncio.create_subprocess_exec(self.ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
                                                        "-i", str(listing), *codec, "-movflags", "+faststart", str(out),
                                                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            if await proc.wait() == 0 and out.is_file():
                return out
        raise Refused("ffmpeg could not join the clips.")


# ---------------------------------------------------------------------------------------------------- the voice side

FILLER = (r"(?:\s+(?:me|us|a|an|the|this|that|my|your|our|another|new|short|quick|little|small|some|one|more|long|cool|"
          r"funny|nice|beautiful|endless|infinite|continuous|never-ending|ai))")
VIDEO = r"\s+(video|film|movie|clip)s?\b"
START = re.compile(rf"\b(make|create|render|generate|shoot|start)\b{FILLER}{{0,3}}{VIDEO}")
STOP = re.compile(rf"\b(stop|cancel|end|halt|kill)\b(?:{FILLER}{{0,2}}{VIDEO}|\s+(the\s+)?render(ing)?\b)")
SHOW = re.compile(rf"\b(show|play|watch|see)\b(?:\s+(?:me|us|the|my))*\s+(last|latest|newest|previous|recent|new)\s+(clip|video|shot)\b")
STATUS = re.compile(r"\b(how'?s|how is|status|progress|is .{0,20} done|how far|how long)\b.{0,30}\b(video|film|clip|render)\b")
ENDLESS = re.compile(r"\b(keep going|endless(ly)?|infinite(ly)?|forever|continuous(ly)?|never[- ]ending|series of|story)\b")
MORE = re.compile(r"\b(keep going|continue|another clip|next clip|more clips|go on)\b")
ABOUT = re.compile(r"\b(?:of|about|showing|where|with|in which)\b\s+(.+)$")


class VideoAgent:
    """The voice's handle on the video panel: "make a video of ...", "stop the video", "show me the last clip", "how's
    the video going?". A back agent like the others (backstage.py), so the hand-over, the short "on it" line and the
    report all work as they do for Hermes; it answers in a second, the clips come later."""

    kind = "video"
    always = True
    ack = ("On it.", "Okay.", "On it, one moment.")

    def __init__(self, name: str, spec: dict, service: VideoService) -> None:
        self.name, self.label, self.service = name, spec.get("label", "Video"), service

    def wants(self, text: str) -> bool:
        low = text.lower()
        started = self.service.run is not None
        return bool(START.search(low) or STOP.search(low) or SHOW.search(low)
                    or (started and (STATUS.search(low) or (MORE.search(low) and "video" in low))))

    async def run(self, prompt: str, progress: Callable[[str, str], None], confirm=None) -> str:
        text = prompt.split("\n\n[Do it now", 1)[0].strip()
        low = text.lower()
        svc = self.service
        try:
            if STOP.search(low):
                if not svc.run or svc.run["state"] in ("done", "stopped", "failed"):
                    return "No video is being made right now."
                await svc.stop()
                return "Stopped the video."
            if SHOW.search(low):
                clips = svc.snapshot()["clips"]
                if not clips:
                    return "There's no finished clip yet."
                progress("video", json.dumps({"action": "show", "index": clips[-1]["index"]}))
                return f"Here's clip {clips[-1]['index'] + 1}."
            if STATUS.search(low) and not START.search(low):
                return said_status(svc.snapshot())
            idea = ABOUT.search(text)
            idea = idea.group(1).strip(" .?!") if idea else re.sub(START, "", low).strip(" .?!")
            idea = re.sub(r"^(boo|hey boo)[, ]+", "", idea, flags=re.I)
            if MORE.search(low) and svc.run and not ABOUT.search(text):
                idea, mode = svc.run["idea"], "continue"
            else:
                mode = "continue" if ENDLESS.search(low) else "single"
            if len(idea) < 4:
                return "What should the video show?"
            progress("tool", "video_start")
            await svc.start(idea, mode)
            return ("Starting the video. It keeps going, clip after clip, until you say stop." if mode == "continue"
                    else "Starting the video. I'll tell you when the clip is ready.")
        except Refused as error:
            return str(error)


def said_status(state: dict) -> str:
    if not state.get("run"):
        return "No video has been started."
    done = len(state["clips"])
    now = state.get("current") or {}
    if state["state"] == "done":
        return f"The video is finished: {done} clip{'s' if done != 1 else ''}."
    if state["state"] in ("stopped", "failed"):
        return f"The video {state['state']}" + (f": {state['message']}" if state.get("message") and state["state"] == "failed" else "") + "."
    head = f"Clip {now.get('index', done) + 1} is {state['state']}"
    if now.get("at") and now.get("of"):
        head += f", step {now['at']} of {now['of']}"
    if now.get("eta_s"):
        head += f", about {max(5, round(now['eta_s'] / 5) * 5)} seconds left"
    return head + "."
