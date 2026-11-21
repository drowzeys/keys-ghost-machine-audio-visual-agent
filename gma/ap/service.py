"""AutoPilot inside the app: the bridge between AutoPilot's loop (gma/ap/autopilot.py, a thread of its own, as the
Studio ran it) and the app's pieces, which live on the asyncio loop.

- **render queue** (``submit`` / ``job_of`` / ``cancel``): a keyframe or a clip is a job of the headless Imagination
  Studio (gma/studio.py, ``HeadlessRenderer``: its ``scout``, ``edit`` and ``video`` jobs) under the memory governor; each look at a job relays its step and its live preview to the video
  panel (as VideoService.follow does) and, when it is done, puts the file where AutoPilot wants it (the run's
  segments folder). After a keyframe the renderer's session ends at once, so the eyes may load to check it.
- **the Director** (``director``): the app's brain (Bolt) through the brain lanes' Director lane, which yields to every
  voice turn; reasoning first with a bounded ``thinking_budget``.
- **the eyes and the ears**: gma/ap/eyes.py and gma/ap/ears.py.
- **the panel**: every AutoPilot save writes the panel's run.json shell (``mode: "autopilot"``, the panel's own fields,
  the AutoPilot run under ``"autopilot"``) and sends the page a copy (current beat, story so far, check counts,
  retakes). The finished pieces are the panel's clips; the film is AutoPilot's, joined again after every piece.

- **the film's sound** (``finish``): the narration (Drama 3) and the score (MiniMax Music 3), by the Studio
  (gma/ap/sound.py), inside the governor's ``audio`` schedule.

Nothing here asks anyone to approve anything: the loop runs to its end, retakes included.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from gm_core.engine import Held

from . import autopilot as ap

HELD_WAITS = 900  # seconds the Director waits for its brain while a render holds it

STAGES = {"director": "planning", "planned": "planning", "keyframe": "keyframe", "render": "rendering",
          "watch": "checking", "continuity": "checking", "checks": "checking", "committed": "checking", "preview": "checking",
          "film-check": "checking"}


SEED_SYSTEM = ("You turn a spoken request for a short film into the starting sheet of an autonomous film director. Answer with "
               "one JSON object only.")
SEED_PROMPT = """The person said: "{words}"
The film is about: {idea}

Write the film's starting sheet. Keep the person's subject exactly; add only what a film needs. Suitable for a general \
audience; no fights; no real public figures.
{{"title": "<3 to 6 words>",
  "world": "<the place and time, 25 words max, concrete and visual>",
  "situation": "<where the film starts and what is about to happen, 30 words max>",
  "characters": [{{"name": "<a first name>", "look": "<age, face, hair, build, 25 words max>", "wardrobe": "<exactly what they wear, \
one line, 15 words max>"}}],
  "style": "<the visual style, 15 words max: film stock, light, colour>",
  "pace": "calm | action",
  "opening_keyframe": "<the first still of the film, 50 words max: who is where, the framing, the place, the light>"}}"""


def run_coro(loop: asyncio.AbstractEventLoop, coro, timeout: float | None = None):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


class AutoPilotService:
    def __init__(self, video, spec: dict | None = None, brain=None, lanes=None, eyes=None, ears=None,
                 loop: asyncio.AbstractEventLoop | None = None, judge=None) -> None:
        self.video = video  # voice_agent.video.VideoService: the panel, its renderer and its run folder
        self.spec = dict(spec or {})
        # d1-2B judges every take when the profile has it (the default); the eyes then stay unloaded (no Gemma)
        self.brain, self.lanes, self.ears, self.judge = brain, lanes, ears, judge
        self.eyes = eyes if judge is None else None
        self.loop = loop
        self.jobs: dict[str, dict] = {}  # job handle -> {kind, dest}
        self.lock = threading.RLock()
        self.timings: dict = {}  # per piece: director, keyframe, render, checks, retakes (the smoke's report)
        ap.attach(video.root, submit=self.submit, job_of=self.job_of, cancel=self.cancel, director=self.director, director_warm=self.warm_director,
                  eyes=self.eyes, qc_eyes=eyes, judge=judge, ears=ears, on_change=self.changed, shell=self.shell,
                  finish=self.finish if getattr(video.renderer, "client", None) is not None else None)

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # ----- runs -----

    def settings(self, count: int | None, seconds: float | None) -> dict:
        s = dict(self.spec.get("settings") or {})
        if count:
            s["segments"] = int(count)
        if seconds:
            s["seconds"] = float(seconds)
        return s

    def create(self, idea: str, count: int | None = None, seconds: float | None = None, seed: dict | None = None,
               name: str | None = None, words: str | None = None) -> dict:
        """A new AutoPilot run from a typed or spoken idea (the brief), started at once. From spoken ``words`` the
        brain writes the seed first (``seed_from_words``): the world, the cast with a wardrobe line each, the style,
        the pace and the opening still; the length is read from the words in code."""

        settings = self.settings(count, seconds)
        made = None
        if not seed and words and self.brain is not None and self.spec.get("seed_from_words", True):
            made = self.seed_from_words(idea, words)
        if made:
            seed = made["seed"]
            settings.update(made["settings"])
            name = name or made.get("name")
        seed = dict(seed or {}) or {"inspiration": idea, "situation": idea}
        if words:
            from voice_agent.video import length_in
            length = length_in(words.lower())
            if length.get("infinity"):
                settings["infinity"] = True
            elif length.get("segments"):
                settings["segments"] = length["segments"]
        run = ap.create_run({"name": name or idea[:80], "seed": seed, "settings": settings})
        run["idea"] = idea
        if made:
            run["seed_made"] = {"by": "brain", "seconds": made.get("seconds"), "words": words[:300]}
        return ap.begin(run)

    def seed_from_words(self, idea: str, words: str) -> dict | None:
        """AutoPilot's seed from a spoken request, by the brain (no reasoning, a short JSON answer): None when it cannot
        be made (the idea alone is the seed then)."""

        began = time.time()
        prompt = SEED_PROMPT.format(words=words.strip()[:600], idea=idea.strip()[:400])
        try:
            data = ap.extract_json(self.director(SEED_SYSTEM, prompt, 0))
        except Exception:  # noqa: BLE001 - the idea alone still makes a film
            return None
        if not isinstance(data, dict) or not str(data.get("world") or data.get("situation") or "").strip():
            return None
        characters = []
        for c in (data.get("characters") or [])[:4]:
            if isinstance(c, dict) and str(c.get("name") or "").strip():
                characters.append({"name": str(c["name"]).strip()[:60], "look": str(c.get("look") or "").strip()[:400],
                                   **({"wardrobe": str(c["wardrobe"]).strip()[:200]} if str(c.get("wardrobe") or "").strip() else {})})
        seed = {"inspiration": idea, "world": str(data.get("world") or "").strip()[:800],
                "situation": str(data.get("situation") or idea).strip()[:800], "characters": characters,
                **({"opening_keyframe": str(data["opening_keyframe"]).strip()[:500]} if str(data.get("opening_keyframe") or "").strip() else {})}
        settings = {}
        if str(data.get("style") or "").strip():
            settings["style"] = str(data["style"]).strip()[:400]
        pace = str(data.get("pace") or "").strip().lower()
        if pace in ("action", "fast"):
            settings["guidance"] = "Brisk pace: physical action, real-time speed."
        elif pace in ("calm", "slow"):
            settings["guidance"] = "A calm, unhurried pace, every piece still moving the story one visible step."
        return {"seed": seed, "settings": settings, "name": str(data.get("title") or "").strip()[:80] or None,
                "seconds": round(time.time() - began, 1)}

    def voice(self, run_id: str, action: str, text: str = "", pieces: int = 1) -> dict:
        """A spoken steer of a live film (voice_agent.video.VideoAgent maps the words in code)."""

        if action in ("steer", "scene", "pace"):
            ap.steer(run_id, text or action, {"steer": "note"}.get(action, action))
            run = ap.runs.get(run_id, {}).get("run") or ap.load(run_id)
            ap.save(run)
            return run
        if action == "extend":
            return ap.extend(run_id, pieces)
        if action == "redo":
            return ap.redo(run_id)
        if action == "sharp":
            run = ap.sharpen(run_id)
            return run
        if action == "pause":
            ap.pause(run_id)
            return ap.runs.get(run_id, {}).get("run") or ap.load(run_id)
        raise ValueError(f"no such film action: {action}")

    def resume(self, run_id: str) -> dict | None:
        try:
            run = ap.load(run_id)
        except ap.Bad:
            return None
        if run.get("state") in ap.ACTIVE and not ap.alive(run_id):
            ap.say(run, "Picked up after a restart.")
            return ap.begin(run)
        return run

    async def stop(self, run_id: str) -> None:
        live = ap.runs.get(run_id)
        if live:
            seg = next((s for s in live["run"].get("segments") or [] if s.get("status") in ("queued", "keyframe_queued")), None)
            ap.stop_run(run_id)
            if seg and seg.get("job"):
                await self.video.renderer.cancel(seg["job"])

    # ----- the render queue -----

    def submit(self, body: dict) -> dict:
        kind, params = body.get("kind"), body.get("params") or {}
        where = params.get("autopilot") or {}
        folder = ap.run_dir(where.get("run", "x")) / "segments" / f"S{int(where.get('segment', 0)):04d}"
        size = (int(params["width"]), int(params["height"])) if params.get("width") and params.get("height") else None
        if kind in ("scout", "edit"):
            # an "edit" keyframe is drawn beside its reference pictures: the place's first, then the cast's portraits
            refs = [str(ap.HOME / r) for r in ([params["image"]] if params.get("image") else []) + list(params.get("images") or [])] \
                if kind == "edit" else []
            dest = folder / f"keyframe{int(where.get('keyframe', 1))}.png"
            kwargs = {"references": refs} if refs else {}
            if size:
                kwargs["size"] = size
            handle = run_coro(self.loop, self.video.renderer.keyframe(params["prompt"], int(params.get("seed", 0)), **kwargs), 60)
            kind = "scout"
        elif kind == "video":
            clip = {"prompt": params["prompt"], "composed": True, "frames": int(params["frames"]), "seed": int(params.get("seed", 0)),
                    "title": body.get("title", "")}
            if params.get("image"):
                clip["image_file"] = str(ap.HOME / params["image"])
            if params.get("first_frame_from"):
                clip["opens_on"] = str(ap.HOME / params["first_frame_from"])
            if size:
                clip.update(width=size[0], height=size[1])
            if params.get("schedule"):
                clip["schedule"] = params["schedule"]  # "go sharper": the governor's sharp schedule (the brain evicted)
            refused = run_coro(self.loop, self.video.renderer.room(clip), 240)  # it may wait for memory to come back
            if refused is not None:
                raise ap.Failed(f"the renderer refused the piece: {refused}")
            dest = ap.HOME / params["output"]
            handle = run_coro(self.loop, self.video.renderer.submit(clip), 60)
        else:
            raise ap.Failed(f"this app's renderer makes no {kind} jobs (an edited keyframe needs a picture model that edits)")
        with self.lock:
            self.jobs[handle] = {"kind": kind, "dest": str(dest), "began": time.time(), "where": where}
        if kind == "scout" and self.loop is not None:
            # the keyframe goes to the panel the moment it is made, even while the loop is busy (the Director
            # planning beside the opening keyframe of a spoken request)
            asyncio.run_coroutine_threadsafe(self.show_when_made(handle), self.loop)
        return {"id": handle, "status": "queued"}

    async def show_when_made(self, handle: str) -> None:
        while True:
            with self.lock:
                meta = self.jobs.get(handle)
            job = getattr(self.video.renderer, "jobs", {}).get(handle)
            if meta is None or job is None or job.get("state") in ("failed", "cancelled"):
                return
            if job.get("state") == "done":
                await self.look(handle, meta)
                return
            await asyncio.sleep(0.5)

    def job_of(self, job_id: str) -> dict | None:
        with self.lock:
            meta = self.jobs.get(job_id)
        if meta is None or job_id not in getattr(self.video.renderer, "jobs", {}):
            return None  # a job of the process before a restart: AutoPilot looks for its file, or renders it again
        return run_coro(self.loop, self.look(job_id, meta), 600)

    def cancel(self, job_id: str) -> None:
        if self.loop is not None:
            run_coro(self.loop, self.video.renderer.cancel(job_id), 60)

    async def look(self, handle: str, meta: dict) -> dict:
        """One look at a job: its step and preview to the panel; when done, its file where AutoPilot wants it."""

        renderer, svc = self.video.renderer, self.video
        got = await renderer.poll(handle)
        if got.state == "done":
            dest = Path(meta["dest"])
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if meta["kind"] == "scout":
                    picture = await renderer.picture(got.output)
                    if picture:
                        dest.write_bytes(picture)
                        svc.still = picture
                        svc.live.update(preview="still", seq=svc.live.get("seq", 0) + 1)
                    # the keyframe's session ends now: the eyes check it before the clip's render begins
                    await renderer.end("the keyframe is checked before the clip")
                else:
                    await renderer.fetch(got.output, dest)
            meta["timings"] = got.timings or {}
            self.note_timings(meta)
            return {"id": handle, "status": "done", "outputs": [str(dest.relative_to(ap.HOME))], "timings": got.timings}
        if got.state in ("failed", "cancelled"):
            return {"id": handle, "status": got.state, "error": got.error}
        svc.live.update(at=got.at, of=got.of, stage=("keyframe" if meta["kind"] == "scout" else got.stage or got.state),
                        eta_s=got.eta_s, queued=got.state == "queued")
        if got.preview and got.preview != svc.preview_ref:
            svc.preview_ref = got.preview
            picture = await renderer.picture(got.preview)
            if picture:
                svc.still = picture
                svc.live.update(preview="engine", seq=svc.live.get("seq", 0) + 1)
        svc.changed()
        return {"id": handle, "status": "rendering" if got.state == "rendering" else "queued"}

    def note_timings(self, meta: dict) -> None:
        where = meta.get("where") or {}
        piece = self.timings.setdefault(str(where.get("segment")), {"keyframes": [], "renders": []})
        entry = {"wall_s": round(time.time() - meta["began"], 1), **{k: v for k, v in (meta.get("timings") or {}).items()
                                                                     if k in ("keyframe_s", "load_s", "render_s", "decode_s", "total_s",
                                                                              "min_available_gb", "server_start_s", "picture")}}
        if entry not in piece["keyframes" if meta["kind"] == "scout" else "renders"]:
            piece["keyframes" if meta["kind"] == "scout" else "renders"].append(entry)

    # ----- the film's sound (the headless Studio: Drama 3 narration, MiniMax Music 3 score) -----

    def finish(self, run: dict) -> None:
        """After the last piece: the narration and the score, inside the governor's ``audio`` schedule (the brain and
        the judge out for Drama 3 and Music 3), then the session ends and the voice loop's engines may load again."""

        from . import sound

        s = run["settings"]
        if not (s.get("narration") or s.get("score")):
            return
        renderer = self.video.renderer
        text = None
        if s.get("narration"):
            # the Director writes the words while its brain is still resident, before the audio schedule evicts it
            try:
                text = sound.narration_text(run, self.director, float((run.get("film") or {}).get("seconds") or 0))
            except Exception as error:  # noqa: BLE001
                ap.say(run, "The narration could not be written: " + str(error)[:200])
        schedule = "audio" if "audio" in (getattr(renderer.governor, "schedules", None) or {}) else None
        began = time.time()
        entered = False
        try:
            if schedule and self.loop is not None:
                run_coro(self.loop, renderer.begin("sound", None, schedule), 600)
                entered = True
            timings = self.timings.setdefault("sound", {})
            sound.finish_film(run, renderer.client, (lambda *_: text or "") if s.get("narration") else self.director,
                              ap.say, ap.HOME, timings)
            timings["total_s"] = round(time.time() - began, 1)
        finally:
            if entered:
                run_coro(self.loop, renderer.end("the film's sound is made"), 600)

    # ----- the Director -----

    def warm_director(self) -> None:
        """The brain loaded now (its engine through the registry), so the next Director question does not wait for it;
        nothing when it is held for a render or there is no room (the question then loads it, or waits)."""

        use = getattr(self.brain, "use", None)
        if use is None or self.loop is None:
            return

        async def warm():
            for _ in range(120):  # the clip's render may still hold it for a moment (its leave frees the renderer first)
                try:
                    async with use():
                        return
                except Held:
                    await asyncio.sleep(1.0)
        try:
            run_coro(self.loop, warm(), 900)
        except Exception:  # noqa: BLE001 - only a head start
            pass

    def director(self, system: str, prompt: str, think: int) -> str:
        if self.brain is None:
            raise RuntimeError("no brain for the Director")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        began = time.time()

        def asked(settings: dict):
            async def call():
                # a brain held for a render (a keyframe evicts it where the eyes and the picture leave no room) is
                # waited for, never a failed piece
                for _ in range(HELD_WAITS):
                    try:
                        return await self.brain.complete(messages, **settings)
                    except Held:
                        await asyncio.sleep(1.0)
                return await self.brain.complete(messages, **settings)
            if self.lanes is not None:
                return run_coro(self.loop, self.lanes.director(call, retries=50), None)
            return run_coro(self.loop, call(), None)

        try:
            if think > 0:
                said = asked({"temperature": 0.7, "chat_template_kwargs": {"enable_thinking": True}, "thinking_budget": int(think),
                              "max_tokens": int(think) + 8000, "timeout": 900})
                said = re.sub(r"<think>.*?</think>|^.*?</think>", "", said or "", flags=re.S).strip()
                if said:
                    return said
            # the reasoning used the whole budget, or no thinking asked: once more without it, its length bounded
            return asked({"temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 9000, "timeout": 900})
        finally:
            self.director_s = round(time.time() - began, 1)

    # ----- the panel -----

    def shell(self, run: dict) -> dict:
        """The panel's run.json around the AutoPilot run (written by AutoPilot's save). The finished pieces are copied
        to the panel's clip files (clip000.mp4 ...) with their posters, once each."""

        svc = self.video
        done = ap.committed(run)
        clips = []
        for seg in done:
            index = seg["index"] - 1
            out = svc.clip_file(run["id"], index, "mp4")
            src = ap.HOME / seg["output"] if seg.get("output") else None
            if src and src.is_file() and (not out.is_file() or out.stat().st_mtime < src.stat().st_mtime):
                shutil.copyfile(src, out)
                self.posters(out, run["id"], index)
            clips.append({"index": index, "beat": seg.get("beat"), "prompt": (seg.get("fields") or {}).get("action") or seg.get("beat"),
                          "seconds": seg.get("seconds"), "state": "done", "output": seg.get("output"),
                          "poster": svc.clip_file(run["id"], index, "jpg").is_file(), "passed": seg.get("passed")})
        state = {"running": "running", "paused": "stopped", "idle": "planning"}.get(run.get("state"), run.get("state"))
        if state == "running":
            state = STAGES.get((run.get("current") or {}).get("stage"), "planning")
        shell = {"id": run["id"], "idea": run.get("idea") or run.get("name"), "mode": "autopilot",
                 "count": run["settings"].get("segments") or None, "seconds": run["settings"].get("seconds"), "state": state,
                 "clips": clips, "created": run.get("created"), "renderer": svc.renderer.name, "seed": run["settings"].get("seed"),
                 "message": run.get("error") or None, "autopilot": run}
        try:
            (svc.root / "latest").write_text(run["id"])
        except OSError:
            pass
        return shell

    def posters(self, clip: Path, run_id: str, index: int) -> None:
        ffmpeg = self.video.ffmpeg
        if not ffmpeg:
            return
        for args, out in ((["-ss", "0.4", "-i", str(clip), "-frames:v", "1", "-vf", "scale=320:-2"], "jpg"),
                          (["-sseof", "-0.25", "-i", str(clip), "-frames:v", "1", "-vf", "scale=640:-2", "-update", "1"], "last.jpg")):
            subprocess.run([ffmpeg, "-v", "error", "-y", *args, "-q:v", "4", str(self.video.clip_file(run_id, index, out))],
                           capture_output=True, timeout=60)

    def changed(self, run: dict) -> None:
        """Each AutoPilot save: a copy of the shell to the panel (on the loop), never the live dict."""

        with ap.guard:
            view = self.shell_view(run)  # read back from the file just written: never the dict the loop mutates
        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.video.autopilot_changed, view)

    def shell_view(self, run: dict) -> dict:
        try:
            return json.loads((ap.run_dir(run["id"]) / "run.json").read_text())
        except (OSError, ValueError):
            return self.shell(run)


def status(run: dict | None, timings: dict | None = None) -> dict | None:
    """What the panel shows beside the live preview: the current beat, the story so far, the check counts, the
    retakes, the last lines of the log and the eyes at work."""

    if not run:
        return None
    segs = run.get("segments") or []
    current = run.get("current") or {}
    at = next((s for s in segs if s.get("index") == current.get("segment")), None) or (segs[-1] if segs else None)
    retakes = sum(max(0, len(s.get("takes") or []) - 1) for s in segs)
    redrawn = sum(max(0, len([k for k in s.get("keyframes") or [] if k.get("round", 0) == 0]) - 1) for s in segs)
    return {"state": run.get("state"), "stage": current.get("stage"), "line": current.get("text"),
            "segment": (at or {}).get("index"), "beat": (at or {}).get("beat"), "scene": (at or {}).get("scene"),
            "done": len(ap.committed(run)), "count": run["settings"].get("segments"),
            "story_so_far": (run.get("story_so_far") or {}).get("text") or " ".join(
                f"{s['index']}. {s.get('beat', '')}" for s in ap.committed(run))[-600:],
            "checks": run.get("check_stats") or {}, "retakes": retakes, "keyframes_redrawn": redrawn,
            "film": (run.get("film") or {}).get("seconds"), "checkpoint": run.get("checkpoint"),
            "log": [line["text"] for line in (run.get("log") or [])[-4:]], "error": run.get("error") or None}
