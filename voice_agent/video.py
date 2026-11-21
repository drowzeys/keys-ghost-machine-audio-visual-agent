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

A profile turns it on with a "video" section ({"renderer": "studio-headless", "seconds": 5, "max_clips": 6,
"keyframe": true, "dir": ...}); its renderer is this app's own (gma/studio.py ``HeadlessRenderer``): a keyframe is a
job of the headless Imagination Studio, a clip is made by the FP4 clip worker (gma/clips.py, gma/fp4_worker.py:
FastH3 V2 on FastVideo's FP4 sparse-attention recipe), and the memory governor says whether it fits now. A "continue"
clip opens on the last frame of the clip before; with "keyframe" a still is made first.

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
import base64
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

STATES = ("idle", "planning", "keyframe", "loading", "rendering", "checking", "stopping", "done", "stopped", "failed")
ENDED = ("done", "stopped", "failed")


class Refused(Exception):
    """Why a run cannot start or go on. ``str(error)`` is the whole reason, for the panel and the status; ``said`` is the
    short plain line the voice speaks ("The video machine is busy right now, so I can't render yet.")."""

    def __init__(self, detail: str, said: str | None = None) -> None:
        super().__init__(detail)
        self.said = said or (detail if len(detail) <= 90 else "The video machine said no. The panel has the details.")


BUSY = "The video machine is busy right now, so I can't render yet."
DOWN = "The video machine isn't answering right now, so I can't render yet."


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
    timings: dict | None = None  # what the render took (a local renderer says: keyframe, load, render, decode, memory)


class Renderer:
    """One render host. A clip is ``submit``ted, ``poll``ed until done, then ``fetch``ed to a local file."""

    name = "renderer"
    previews = False  # True when its Progress names real pictures of the clip as it renders

    async def room(self, clip: dict) -> Refused | None:
        """None when the clip can be made now, else why not."""
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

    def __init__(self, spec: dict | None, busy: Callable[[], bool] | None = None, lanes=None) -> None:
        self.spec = spec
        self.busy = busy or (lambda: False)
        self.lanes = lanes  # gma.lanes.Lanes: the brain's director lane, which yields to every voice turn
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=4))

    async def ask(self, prompt: str, max_tokens: int) -> str:
        if not self.spec:
            return ""
        if self.lanes is not None:
            try:
                return await self.lanes.director(lambda: self.post(prompt, max_tokens), retries=5)
            except Exception:  # noqa: BLE001 - yielded too often, or no answer: the idea itself is the prompt
                return ""
        for _ in range(40):  # at most ~10 s for the conversation to go quiet
            if not self.busy():
                break
            await asyncio.sleep(0.25)
        return await self.post(prompt, max_tokens)

    async def post(self, prompt: str, max_tokens: int) -> str:
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


def retakes_of(shell: dict) -> int:
    """Pieces sent back to be made again in an AutoPilot run: each failed take, but the last take of a piece that joined
    the film anyway (its retakes ran out)."""

    count = 0
    for s in ((shell or {}).get("autopilot") or {}).get("segments") or []:
        failed = [t for t in s.get("takes") or [] if not t.get("passed")]
        count += len(failed) - (1 if s.get("status") == "done" and failed and failed[-1] is (s.get("takes") or [None])[-1] else 0)
    return count


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


def find_ffmpeg(given: str | None = None) -> str | None:
    for path in (given, shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if path and Path(path).expanduser().is_file():
            return str(Path(path).expanduser())
    return None


# ---------------------------------------------------------------------------------------------------- the service

class VideoService:
    """One run at a time; every change is written to ``run.json`` (a checkpoint a clip) and sent to the listeners."""

    def __init__(self, spec: dict, brain: dict | None = None, renderer: Renderer | None = None,
                 checker: Checker | None = None, busy: Callable[[], bool] | None = None, lanes=None) -> None:
        self.spec = spec
        if renderer is None:
            raise ValueError("the video panel renders with this app's own renderer (gma/studio.py HeadlessRenderer: "
                             "keyframes from the headless Studio, clips from the FP4 clip worker)")
        self.renderer = renderer
        self.planner = Planner(spec.get("brain") or brain, busy, lanes)
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
        self.notice: str | None = None  # why the last start was refused, in full: shown in the panel, never spoken
        self.on_news: list[Callable[[str, dict], None]] = []  # spoken updates: ("clip"|"done"|"failed", run)
        self.autopilot = None  # gma.ap.service.AutoPilotService: "autopilot" runs (AutoPilot directs every piece)
        self.look = None  # gma.look.LookService: the finish laid over what is watched (Natural, HDR, Neon, Sonar)

    # ----- what the page sees -----

    def snapshot(self) -> dict:
        out = self.plain_snapshot()
        if self.look is not None:
            out["look"] = self.look.state()
        return out

    def plain_snapshot(self) -> dict:
        run = self.run
        if run and run.get("mode") == "autopilot":
            return self.autopilot_snapshot(run)
        if not run:
            return {"run": None, "state": "idle", "clips": [], "renderer": self.renderer.name, "message": self.notice,
                    "previews": self.preview_kind()}
        clips = [{"index": c["index"], "beat": c.get("beat"), "prompt": c["prompt"], "seconds": c.get("seconds"),
                  "url": f"/api/video/clip/{run['id']}/{c['index']}.mp4",
                  "poster": f"/api/video/clip/{run['id']}/{c['index']}.jpg" if c.get("poster") else None}
                 for c in run["clips"] if c.get("state") == "done"]
        current = next((c for c in reversed(run["clips"]) if c.get("state") not in ("done",)), None)
        return {"run": run["id"], "state": run["state"], "mode": run["mode"], "idea": run["idea"],
                "count": run.get("count"), "message": self.notice or run.get("message"), "renderer": self.renderer.name,
                "previews": self.preview_kind(), "clips": clips,
                "current": {"index": current["index"], "beat": current.get("beat"), "prompt": current.get("prompt"),
                            "state": current.get("state"), **self.live} if current and run["state"] not in ENDED else None,
                "film": f"/api/video/film?run={run['id']}" if clips else None}

    def autopilot_snapshot(self, run: dict) -> dict:
        """An AutoPilot run (a copy of its run.json shell): its pieces as clips, the piece at work with the live view,
        and AutoPilot's own status (beat, story so far, check counts, retakes) beside it."""

        from gma.ap.service import status

        clips = [{"index": c["index"], "beat": c.get("beat"), "prompt": c.get("prompt"), "seconds": c.get("seconds"),
                  "url": f"/api/video/clip/{run['id']}/{c['index']}.mp4",
                  "poster": f"/api/video/clip/{run['id']}/{c['index']}.jpg" if c.get("poster") else None} for c in run.get("clips") or []]
        info = status(run.get("autopilot"))
        current = None
        if run["state"] not in ENDED and info:
            current = {"index": (info.get("segment") or len(clips) + 1) - 1, "beat": info.get("beat"), "prompt": info.get("beat"),
                       "state": run["state"], **self.live}
        film = (run.get("autopilot") or {}).get("film")
        return {"run": run["id"], "state": run["state"], "mode": "autopilot", "idea": run.get("idea"), "count": run.get("count"),
                "message": self.notice or run.get("message"), "renderer": self.renderer.name, "previews": self.preview_kind(),
                "clips": clips, "current": current, "autopilot": info,
                "film": f"/api/video/film?run={run['id']}" if film else None}

    def autopilot_changed(self, view: dict) -> None:
        """AutoPilot saved (from its thread, through the loop): the panel's copy of the run, and spoken news."""

        before = self.run if self.run and self.run.get("id") == view.get("id") else None
        self.run = view
        if view.get("state") not in ("rendering", "keyframe"):
            self.live.update(at=None, of=None, eta_s=None)
        self.changed()
        had = len((before or {}).get("clips") or [])
        if len(view.get("clips") or []) > had and before is not None:
            self.news("clip")
        if before is not None and retakes_of(view) > retakes_of(before):
            self.news("retake")
        if before is not None and before.get("state") != view.get("state") and view.get("state") in ("done", "failed"):
            if view["state"] == "failed":
                view["said"] = "AutoPilot stopped on a problem. The panel has the details."
            self.news(view["state"])

    def preview_kind(self) -> str:
        """What the panel's live view is made of: "engine" (real step pictures), "stages" (the opening still with the
        stage at work, from a renderer that hands over no pictures between passes) or "progress" (the step count)."""
        if self.renderer.previews:
            return "engine"
        return getattr(self.renderer, "preview_kind", "progress")

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
        if not self.run or self.run.get("mode") == "autopilot":  # AutoPilot writes its own run.json shell
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

    async def room(self, clip: dict) -> Refused | None:
        return await self.renderer.room(clip)  # this app's own renderer: the memory governor decides

    # ----- runs -----

    async def resume(self) -> None:
        """At start-up: a run that was at work when the agent stopped carries on from its last checkpoint."""
        run = self.load_latest()
        if run and run.get("mode") == "autopilot":
            self.run = run
            if self.autopilot is not None and (run.get("autopilot") or {}).get("state") == "running":
                self.autopilot.bind(asyncio.get_running_loop())
                await asyncio.to_thread(self.autopilot.resume, run["id"])  # from its checkpoint: done pieces stay done
            return
        if run:
            self.run = run
            if run["state"] not in ENDED:
                if "stream" in run:  # a stream of the HyperFlow service this app no longer has: it cannot go on
                    run["state"], run["message"] = "stopped", "stopped: its renderer is gone"
                    self.save()
                    return
                run["message"] = "resumed after a restart"
                self.task = asyncio.create_task(self.work())

    async def start(self, idea: str, mode: str = "single", count: int | None = None, seconds: float | None = None,
                    seed: dict | None = None, words: str | None = None) -> dict:
        try:
            if mode == "autopilot":
                return await self.start_autopilot(idea, count, seconds, seed, words)
            return await self._start(idea, mode, count, seconds)
        except Refused as error:  # the whole reason goes to the panel; the voice says ``error.said``
            self.notice = str(error)
            self.changed()
            raise

    def check_autopilot(self, idea: str, seed: dict | None = None) -> None:
        """Refused when an AutoPilot film cannot start now (said before anything is made, so the voice can say why)."""

        if not (idea or "").strip() and not seed:
            raise Refused("Tell me what the film should be about.", "What should the film be about?")
        if self.autopilot is None:
            raise Refused("AutoPilot is not set up in this profile (its \"autopilot\" section).", "AutoPilot isn't set up here.")
        if (self.task and not self.task.done()) or (self.run and self.run.get("mode") == "autopilot"
                                                    and self.run.get("state") not in ENDED + ("stopped",)):
            raise Refused("A video is already being made. Say stop the video first.")

    async def start_autopilot(self, idea: str, count: int | None, seconds: float | None, seed: dict | None,
                              words: str | None = None) -> dict:
        """An AutoPilot film from the idea (the brief): it directs, renders, checks and retakes every piece itself.
        ``words`` (a spoken request): the brain turns them into AutoPilot's seed first (gma/ap/service.py)."""

        idea = re.sub(r"\s+", " ", (idea or "").strip())[:600]
        self.check_autopilot(idea, seed)
        self.notice = None
        self.autopilot.bind(asyncio.get_running_loop())
        run = await asyncio.to_thread(self.autopilot.create, idea, count, seconds, seed, None, words)
        self.autopilot_changed(self.autopilot.shell_view(run))
        return self.snapshot()

    def autopilot_run_id(self) -> str | None:
        run = self.run
        return run["id"] if run and run.get("mode") == "autopilot" else None

    async def autopilot_voice(self, action: str, text: str = "", pieces: int = 1) -> dict:
        """The person's word during an AutoPilot film: "steer" (a Director note and a bible fact from the next piece),
        "scene" (a new scene), "pace" (faster), "extend" (keep going / N seconds more), "redo" (the piece shown, again),
        "sharp" (1376x768, the brain evicted for it), "pause" (stop after this piece), "cancel" (stop now)."""

        run_id = self.autopilot_run_id()
        if run_id is None or self.autopilot is None:
            raise Refused("No AutoPilot film to steer. Start one first.", "There's no film to change yet.")
        self.autopilot.bind(asyncio.get_running_loop())
        if action == "cancel":
            await self.autopilot.stop(run_id)
        else:
            await asyncio.to_thread(self.autopilot.voice, run_id, action, text, pieces)
        return self.snapshot()

    async def _start(self, idea: str, mode: str, count: int | None, seconds: float | None) -> dict:
        idea = re.sub(r"\s+", " ", (idea or "").strip())[:600]
        if not idea:
            raise Refused("Tell me what the video should show.", "What should the video show?")
        if self.task and not self.task.done():
            raise Refused("A video is already being made. Say stop the video first.")
        mode = "continue" if mode == "continue" else "single"
        most = int(self.spec.get("max_clips", 6))
        count = 1 if mode == "single" else max(1, min(int(count or most), most)) if most else int(count or 0) or None
        seconds = float(seconds or self.spec.get("seconds", 5))
        refusal = await self.room({"prompt": idea, "seconds": seconds, "preset": self.spec.get("preset", "small"),
                                   "quality": self.spec.get("quality", "fh8")})
        if refusal:
            raise refusal if isinstance(refusal, Refused) else Refused(str(refusal), BUSY)
        self.notice = None
        self.run = {"id": time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "idea": idea, "mode": mode,
                    "count": count, "seconds": seconds, "state": "planning", "clips": [], "created": time.time(),
                    "renderer": self.renderer.name, "seed": int(self.spec.get("seed", 42))}
        self.save()
        self.task = asyncio.create_task(self.work())
        self.changed()
        return self.snapshot()

    async def stop(self) -> dict:
        run = self.run
        if run and run.get("mode") == "autopilot":
            if self.autopilot is not None:
                await self.autopilot.stop(run["id"])
            return self.snapshot()
        if not run or run["state"] in ENDED:
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
            self.fail(run, str(error), error.said)
        except Exception as error:  # a bug or a host gone: said, the run kept for a retry
            self.fail(run, f"{type(error).__name__}: {str(error)[:160]}")

    def fail(self, run: dict, message: str, said: str | None = None) -> None:
        run["state"], run["message"] = "failed", message
        run["said"] = said or "Something went wrong. The panel has the details."
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
            if done.timings:
                clip["keyframe_timings"] = done.timings
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
        if done.timings:
            clip["timings"] = done.timings
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
                raise Refused(f"The render {got.state}" + (f": {str(got.error)[:160]}" if got.error else "."),
                              f"The render {got.state}. The panel has the details.")
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
        if run.get("mode") == "autopilot":  # AutoPilot's own film: joined after every piece, with its fades
            film = (run.get("autopilot") or {}).get("film") or {}
            path = self.root / film["path"] if film.get("path") else None
            if path is None or not path.is_file():
                raise Refused("There are no finished pieces to join yet.")
            return path
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

# A request for a video, found in code before any model sees the turn: "generate a video of a dragon over a castle",
# "can you make me a short film about ...", "make a video" (no subject: the agent asks what it should show), "show me a
# clip of ...", "I want an endless video of ...". Not "I watched a video yesterday", "I made a video", "a video call".
FILLER = (r"(?:\s+(?:me|us|a|an|the|this|that|my|your|our|another|new|short|quick|little|small|some|one|more|long|cool|"
          r"funny|fun|nice|beautiful|epic|cinematic|endless|infinite|continuous|never-ending|ai|"
          r"(?:\d+|one|two|three|five|ten|fifteen|twenty|thirty)[- ]second))")
VIDEO = r"\s+(video|film|movie|clip|animation)s?\b(?!\s+(?:call|chat|game|camera|card|editor|player|stream)s?\b)"
VERB = r"(?:make|create|render|generate|shoot|start|produce|animate|do|build|whip up|cook up|put together)"
ASK = r"(?:show me|show us|i want|i'?d like|i would like|give me|get me|can i (?:get|have)|could i (?:get|have))"
START = re.compile(rf"\b(?:{VERB}|{ASK})\b{FILLER}{{0,4}}{VIDEO}")
# the person talking about their own video, not asking for one: "I'm going to make a video", "we were making a film"
THEIRS = re.compile(r"\b(?:i|we)(?:'m| am|'re| are| was| were|'ve| have| had| used to| tried| will|'ll)(?:\s+[\w']+){0,2}\s*$")
HOW_TO = re.compile(r"\bhow (?:do|does|did|would|should) (?:i|we|you|people|one|they)\b|\bhow to\b")
STOP = re.compile(rf"\b(stop|cancel|end|halt|kill)\b(?:{FILLER}{{0,2}}{VIDEO}|\s+(the\s+)?render(ing)?\b)")
SHOW = re.compile(r"\b(show|play|watch|see)\b(?:\s+(?:me|us|the|my))*\s+(last|latest|newest|previous|recent|new)\s+(clip|video|shot)\b"
                  r"|\b(?:show|play)\b(?:\s+(?:me|us))?\s+(?:the|that|this|my)\s+(?:clip|video|film)\b(?:\s+(?:again|now|please))?\s*[.!?]*$")
STATUS = re.compile(r"\b(how'?s|how is|status|progress|is .{0,20} done|how far|how long)\b.{0,30}\b(video|film|clip|render)\b")
ENDLESS = re.compile(r"\b(keep going|endless(ly)?|infinite(ly)?|forever|continuous(ly)?|never[- ]ending|series of|story)\b")
MORE = re.compile(r"\b(keep going|continue|another clip|next clip|more clips|go on)\b")
# steering words: "make it longer", "carry on", "now make it rain", "let it snow", "add a dog"
LONGER = re.compile(r"\b(make it longer|longer|extend it|carry on|keep it going|more of (?:it|this|that)|"
                    r"(?:\d+|ten|fifteen|twenty|thirty|sixty) more seconds)\b")
STEER_LEAD = re.compile(r"^(?:[\s,.]|\b(?:and|okay|ok|so|boo|then|um+|uh+)\b)*", re.I)
STEER = re.compile(r"^(?:now\s+)?(?:make (?:it|him|her|them)|let (?:it|him|her|them)|have (?:it|him|her|them)|turn (?:it|this)|"
                   r"add|bring in|cut to|switch to|change (?:it|the)|give (?:it|him|her|them))\b")
WORDS = {"ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60}
RECENT_S = 600  # a finished film still takes "keep going" for this long
# what is left around the idea once the request is taken off: "make a video [of] a dragon [please]"
LEAD = re.compile(r"^(?:[\s,.:;-]|\b(?:for (?:me|us)|please|now|right now|real quick|quickly)\b)*"
                  r"(?:\b(?:of|about|showing|that shows|which shows|where|in which|with|featuring|depicting)\b\s*)?", re.I)
TAIL = re.compile(r"(?:[\s,]*\b(?:please|for (?:me|us)|now|right now|boo|thanks|thank you|okay|ok)\b)*[\s.?!,]*$", re.I)
# the answer to "What should the video show?": "um, a red kite over green hills", "how about a fox in the snow"
REPLY_LEAD = re.compile(r"^(?:[\s,.]|\b(?:um+|uh+|er+|hmm+|okay|ok|so|well|maybe|let'?s say|how about|what about|"
                        r"it should show|show|make it(?: about)?|it'?s|let'?s do|i want|i'?d like|a video of|one of|of|"
                        r"about|boo)\b)*", re.I)
NOTHING = {"", "it", "one", "something", "anything", "a video", "video", "videos", "a clip", "a film", "a movie",
           "for me", "please", "now", "yes", "yeah", "sure"}
NEVERMIND = re.compile(r"^\W*(?:no|nope|nah)\W*$|\b(never ?mind|forget (?:it|about it)|cancel|no thanks|nothing|don'?t bother|"
                       r"not now|no video|skip it)\b")
QUESTION = re.compile(r"^\W*(?:what|who|why|when|where|which|how|is|are|do|does|did|can you|could you|will you|tell me)\b"
                      r"(?! about\b)")
WAITS_S = 90  # how long "What should the video show?" waits for its answer

# ----- an AutoPilot film, steered by voice (mapped in code, no brain asked) -----
# The words that steer a film ("add a dog", "make it rain") are also everyday words ("add milk to my list", "make it a
# double"). While a film runs they are the film's only when they are about the film: a film word or a word the film
# itself uses, or a picture word (weather, light, camera), or the person spoke about the film moments ago; and never with
# words of an errand, a time or the conversation itself.
FILM_WORDS = re.compile(r"\b(video|film|movie|clip|scene|shot|camera|frame|piece|story|panel|keyframe|footage|picture)s?\b")
VISUAL = re.compile(r"\b(rain\w*|snow\w*|storm\w*|fog\w*|foggy|mist\w*|thunder\w*|lightning|sunset|sunrise|dawn|dusk|night|"
                    r"daylight|darker|brighter|moon\w*|windy|wind|waves?|sea|ocean|sky|cloud\w*|fire|smoke|flames?|lamp|lantern|"
                    r"close[- ]?ups?|zoom\w*|wide shot|slow motion|cinematic|dramatic|colou?rs?|black and white|sepia)\b")
NOT_FILM = re.compile(r"\b(list|shopping|grocer\w*|reminders?|remind|calendar|meeting|appointment|timer|alarm|notes?|e-?mails?|"
                      r"messages?|text (?:him|her|them|me)|cart|basket|order|coffee|espresso|latte|tea|drinks?|pizza|dinner|lunch|"
                      r"breakfast|recipe|tomorrow|tonight|o'?clock|\d+\s*(?:am|pm)|[ap]\.m\.|playlist|song|music volume|volume|"
                      r"louder|quieter|thermostat|heating|laundry|call|talk|speak|speaking|answer|reply|respond|read)\b")
FASTER = re.compile(r"\b(faster|more action|speed it up|pick up the pace|more energy|quicker)\b")
REDO = re.compile(r"\b(redo (?:that|it|this|the last (?:one|piece|part|clip))|do (?:that|it) again|try (?:that|it) again|"
                  r"take (?:that|it) again|retake(?: it| that)?|make (?:that|it) again)\b")
SHARPER = re.compile(r"\b(go sharper|sharper|higher (?:quality|resolution)|in hd|full hd|more detail)\b")
NEW_SCENE = re.compile(r"^(?:now\s+)?(?:new scene|next scene|cut to|switch to|change the scene to)\b[\s:,-]*(.*)$")
STOP_NOW = re.compile(r"\b(stop (?:it |the (?:video|film|movie|render(?:ing)?) )?(?:right )?now|cancel (?:the )?(?:render(?:ing)?|video|film)|"
                      r"kill (?:the )?(?:render|video|film)|abort)\b")
STOP_SOFT = re.compile(r"^\W*(?:okay |ok |boo,? )?(?:stop|pause|that'?s enough|wrap (?:it )?up)\W*$|\b(?:stop|pause|end) (?:the )?(?:video|film|movie)\b"
                       r"|\bstop after (?:this|that)\b")
MORE_SECONDS = re.compile(r"\b(\d+|ten|fifteen|twenty|thirty|sixty) (?:more )?seconds?(?: more| longer)?\b|\b(?:a|one|two|\d+) more (?:piece|part|clip)s?\b")
FILM_STEER = re.compile(r"^(?:now\s+)?(?:make|let|have|turn|give|add|bring in|show|put|change)\b")
VIDEO_RECENT_S = 20  # words said this soon after the person last spoke about the film count as the film's
BARGED_S = 3  # a bare "stop" this soon after the person cut Boo off was for Boo's words, not the film
# a spoken request is held this long before its film starts: words that follow it at once are the rest of the request
# (on the Spark smoke, 2026-10-27, "make a video of a lighthouse keeper ... lighting the lamp as a storm rolls in" came
# in as two turns at a breath, and the film was made of "a lighthouse keeper" alone)
MERGE_S = 2.5
MERGE_MAX_S = 12.0


def asked(low: str) -> re.Match | None:
    """The request for a new video in these (lower-case) words, if they hold one."""
    if HOW_TO.search(low):
        return None
    for found in START.finditer(low):
        if not THEIRS.search(low[: found.start()]):
            return found
    return None


def idea_in(words: str) -> str:
    """The subject of the video from what follows the request ("of a dragon over a castle, please")."""
    idea = TAIL.sub("", LEAD.sub("", words.strip(), count=1)).strip(" ,.?!\"'")
    return "" if idea.lower() in NOTHING or len(idea) < 3 else idea


def idea_said(words: str) -> str:
    """The subject from the answer to "What should the video show?"."""
    return idea_in(REPLY_LEAD.sub("", words.strip(), count=1))


class VideoAgent:
    """The voice's handle on the video panel: "make a video of ...", "stop the video", "show me the last clip", "how's
    the video going?". Its requests are found in code (``wants``) before the front brain sees the turn, and it answers
    that turn itself, in a second, with one short line (the server speaks it as the turn's only reply); the clips come
    later. With no subject it asks "What should the video show?", and the next words are taken as the idea."""

    kind = "video"
    always = True
    ack = ("On it.", "Okay.", "On it, one moment.")
    ASKING = "What should the video show?"

    def __init__(self, name: str, spec: dict, service: VideoService) -> None:
        self.name, self.label, self.service = name, spec.get("label", "Video"), service
        self.spec = spec
        self.asked_at: float | None = None  # when it asked what the video should show
        self.video_at = 0.0  # when the person last spoke about the film (a turn this agent took)
        self.pending: dict | None = None  # a spoken request held for MERGE_S: {"idea", "words", "until"}
        self.busy = lambda: False  # someone is talking or being answered (the server says)
        self.barged_at = 0.0  # when the person last cut the voice off (the server says)

    # ----- an AutoPilot film -----

    def film_live(self) -> bool:
        """An AutoPilot film that is being made, or ended (stopped, paused, done) not long ago."""

        run = self.service.run
        if not run or run.get("mode") != "autopilot" or self.service.autopilot is None:
            return False
        if run.get("state") not in ENDED:
            return True
        if time.monotonic() - self.video_at < RECENT_S:
            return True
        try:  # ended (done, stopped, failed) not long ago, even before a restart of the app
            ended = time.mktime(time.strptime(str((run.get("autopilot") or {}).get("updated")), "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return False
        return time.time() - ended < RECENT_S

    def vocabulary(self) -> set[str]:
        """The film's own words (its idea, seed and beats): "the keeper", "the lamp" are the film's."""

        run = self.service.run or {}
        ap = run.get("autopilot") or {}
        seed = ap.get("seed") or {}
        text = " ".join([str(run.get("idea") or ""), str(seed.get("world") or ""), str(seed.get("situation") or ""),
                         *[str(c.get("name") or "") for c in seed.get("characters") or []],
                         *[str(x.get("beat") or "") for x in ap.get("segments") or []]]).lower()
        return {w for w in re.findall(r"[a-z']{5,}", text)} - {"there", "their", "about", "which", "while", "where", "these", "those"}

    def about_film(self, low: str) -> bool:
        """Steering words that are about the film (see FILM_WORDS): never an errand's or the conversation's."""

        if NOT_FILM.search(low) and not FILM_WORDS.search(low):
            return False
        if FILM_WORDS.search(low) or VISUAL.search(low):
            return True
        if set(re.findall(r"[a-z']{5,}", low)) & self.vocabulary():
            return True
        return time.monotonic() - self.video_at < VIDEO_RECENT_S

    def film_action(self, low: str) -> tuple[str, str] | None:
        """(action, text) for words that steer a live AutoPilot film, else None."""

        if not self.film_live():
            return None
        bare = STEER_LEAD.sub("", low, count=1).strip()
        if STOP_NOW.search(low):
            return "cancel", ""
        if STOP_SOFT.search(bare):
            if re.fullmatch(r"\W*(?:okay |ok |boo,? )?(?:stop|pause)\W*", bare) and time.monotonic() - self.barged_at < BARGED_S:
                return None  # a bare "stop" right after cutting Boo off was for Boo
            return "pause", ""
        if REDO.search(low):
            return "redo", ""
        if SHARPER.search(low) and (FILM_WORDS.search(low) or re.fullmatch(r"\W*(?:go |make it |can you go )?sharper\W*", bare) or "go sharper" in low):
            return "sharp", ""
        found = NEW_SCENE.match(bare)
        if found and len(found.group(1).strip(" .,!?")) >= 3:
            return "scene", found.group(1).strip(" .,!?")
        if MORE.search(low) or LONGER.search(low) or MORE_SECONDS.search(low):
            return "extend", ""
        if FASTER.search(low) and (self.about_film(low) or "action" in low) and not NOT_FILM.search(low):
            return "pace", bare
        if (STEER.search(bare) or FILM_STEER.search(bare)) and self.about_film(low):
            return "steer", steer_in(low) or bare
        return None

    def waiting(self) -> bool:
        return self.asked_at is not None and time.monotonic() - self.asked_at < WAITS_S

    def merging(self) -> bool:
        return self.pending is not None

    async def start_after_merge(self) -> None:
        """The held request's film, once the conversation has been quiet for MERGE_S after its last words (the
        person speaking, or being answered, keeps it waiting; MERGE_MAX_S at most)."""

        began = time.monotonic()
        while self.pending is not None and time.monotonic() - began < MERGE_MAX_S:
            if self.busy():
                self.pending["until"] = time.monotonic() + MERGE_S
            elif time.monotonic() >= self.pending["until"]:
                break
            await asyncio.sleep(0.1)
        held, self.pending = self.pending, None
        if held is not None:
            try:
                await self.service.start(held["idea"], "autopilot", None, None, None, words=held["words"])
            except Refused:
                pass  # its reason is in the panel (the service's notice)

    def wants(self, text: str) -> bool:
        low = text.lower()
        started = self.service.run is not None
        if self.merging() and not QUESTION.search(low):
            return True
        if asked(low) or STOP.search(low) or SHOW.search(low):
            return True
        if self.film_action(low):
            return True
        if started and (STATUS.search(low) or (MORE.search(low) and "video" in low)):
            return True
        # the answer to its question; a question of the person's own goes to the front brain as usual
        return self.waiting() and not (QUESTION.search(low) and not NEVERMIND.search(low))

    async def run(self, prompt: str, progress: Callable[[str, str], None], confirm=None) -> str:
        return (await self.handle(prompt, progress))[0]

    async def handle(self, prompt: str, progress: Callable[[str, str], None]) -> tuple[str, str]:
        """(what the voice says, the whole story for the page): the two differ only when a run is refused."""
        text = prompt.split("\n\n[Do it now", 1)[0].strip()
        low = text.lower()
        svc = self.service
        waiting, self.asked_at = self.waiting(), None
        if self.merging() and not QUESTION.search(low):
            # the rest of the request just said: the film starts with it (nothing is said, the person is mid-thought)
            self.pending["idea"] = (self.pending["idea"] + " " + text.strip(" .")).strip()
            self.pending["words"] = (self.pending["words"] + " " + text).strip()
            self.pending["until"] = time.monotonic() + MERGE_S
            return "", f"(the request goes on: {text})"
        try:
            found = asked(low)
            act = self.film_action(low) if not (found and idea_in(text[found.end():] if len(text) == len(low) else low[found.end():])) else None
            if act:
                self.video_at = time.monotonic()
                said = await self.steer_film(act, low)
                return said, f"{said} ({act[0]}{': ' + act[1] if act[1] else ''})"
            if STOP.search(low):
                if not svc.run or svc.run["state"] in ENDED:
                    return ("No video is being made right now.",) * 2
                await svc.stop()
                return ("Stopped the video.",) * 2
            found = asked(low)
            if SHOW.search(low) and not (found and idea_in(text[found.end():] if len(text) == len(low) else low[found.end():])):
                clips = svc.snapshot()["clips"]
                if not clips:
                    return ("There's no finished clip yet.",) * 2
                progress("video", json.dumps({"action": "show", "index": clips[-1]["index"]}))
                return (f"Here's clip {clips[-1]['index'] + 1}.",) * 2
            if STATUS.search(low) and not found:
                return (said_status(svc.snapshot()),) * 2
            if found:
                idea = idea_in(text[found.end():] if len(text) == len(low) else low[found.end():])
            elif waiting:
                if NEVERMIND.search(low):
                    return ("Okay, no video then.",) * 2
                idea = idea_said(text)
            else:
                idea = ""
            if not idea and MORE.search(low) and svc.run:
                idea, mode = svc.run["idea"], "continue"
            else:
                mode = "continue" if ENDLESS.search(low) else "single"
            if not idea:
                self.asked_at = time.monotonic()
                return (self.ASKING,) * 2
            progress("tool", "video_start")
            self.video_at = time.monotonic()
            if svc.autopilot is not None and self.spec.get("voice_mode", "autopilot") == "autopilot":
                # an AutoPilot film: the brain turns the words into its seed in the background, and the first keyframe
                # is drawn while the Director plans; the voice says one short line now
                svc.check_autopilot(idea)
                length = length_in(low)
                self.pending = {"idea": idea, "words": text, "until": time.monotonic() + MERGE_S}
                job = asyncio.create_task(self.start_after_merge())
                job.add_done_callback(lambda t: t.cancelled() or t.exception())
                said = ("Okay, I'm making that film. The first frame will show up in the panel shortly."
                        if not length.get("infinity") else "Okay, I'm making that film, and it keeps going until you say stop.")
                return said, f"{said} ({idea})"
            await svc.start(idea, mode)
            said = ("Starting the video. It keeps going, clip after clip, until you say stop." if mode == "continue"
                    else "Starting the video. I'll tell you when the clip is ready.")
            return said, f"{said} ({idea})"
        except Refused as error:
            return error.said, str(error)


    async def steer_film(self, act: tuple[str, str], low: str) -> str:
        """One short line for a steer of the AutoPilot film; the steer itself goes to AutoPilot."""

        action, text = act
        svc = self.service
        if action == "extend":
            more = MORE_SECONDS.search(low)
            seconds = 0
            if more and more.group(1):
                seconds = int(more.group(1)) if more.group(1).isdigit() else WORDS.get(more.group(1), 10)
            each = float(((svc.run or {}).get("autopilot") or {}).get("settings", {}).get("seconds") or 10)
            pieces = max(1, round(seconds / each)) if seconds else 1
            await svc.autopilot_voice("extend", "", pieces)
            return "Okay, I'll keep the film going." if not seconds else f"Okay, {seconds} more seconds."
        await svc.autopilot_voice(action, text)
        return {"steer": "Okay, the next part will show that.", "pace": "Okay, more action from the next part.",
                "scene": "Okay, a new scene next.", "redo": "Okay, I'll make that part again.",
                "sharp": "Okay, going sharper from the next part. I'll be slower to answer while it renders.",
                "pause": "Okay, I'll stop after this part.", "cancel": "Stopped the film."}[action]


def length_in(low: str) -> dict:
    """A spoken film's length: {"infinity": True} for an endless one, {"segments": n} for "a 30 second film" (pieces of
    10 s), else {} (the profile's)."""

    if re.search(r"\b(endless(?:ly)?|infinite(?:ly)?|forever|never[- ]ending|until i say stop|keep going)\b", low):
        return {"infinity": True}
    found = re.search(r"\b(\d+|ten|fifteen|twenty|thirty|forty|sixty)[- ](second|minute)s?\b", low)
    if not found:
        return {}
    number = int(found.group(1)) if found.group(1).isdigit() else {**WORDS, "forty": 40}[found.group(1)]
    seconds = number * (60 if found.group(2) == "minute" else 1)
    return {"segments": max(1, min(30, round(seconds / 10)))}


def steer_in(low: str) -> str:
    """What the next part should show, from steering words ("now make it rain" -> "make it rain"); "" when they only
    ask for more ("keep going", "make it twenty more seconds longer")."""
    rest = MORE.sub(" ", LONGER.sub(" ", low))
    rest = re.sub(r"\s+", " ", TAIL.sub("", rest)).strip(" ,.?!\"'")
    rest = re.sub(r"^(?:now|and|then)\s+", "", STEER_LEAD.sub("", rest, count=1)).strip(" ,.?!")
    found = STEER.search(rest)
    return rest if found and len(rest[found.end():].strip(" ,.?!")) >= 2 else ""


def said_status(state: dict) -> str:
    if not state.get("run"):
        return "No video has been started."
    done = len(state["clips"])
    now = state.get("current") or {}
    if state["state"] == "done":
        return f"The video is finished: {done} clip{'s' if done != 1 else ''}."
    if state["state"] in ("stopped", "failed"):
        return f"The video {state['state']}."  # the reason is in the panel: too long and technical to speak
    head = f"Clip {now.get('index', done) + 1} is {state['state']}"
    if now.get("at") and now.get("of"):
        head += f", step {now['at']} of {now['of']}"
    if now.get("eta_s"):
        head += f", about {max(5, round(now['eta_s'] / 5) * 5)} seconds left"
    return head + "."
