"""The look: a finish laid over what the person watches - a clip of the video panel, a film - and never over the clips
the work goes on from (a clip that continues another opens on its plain last frame, and AutoPilot checks plain pictures).

Four looks, one at a time, kept across restarts (``look.json``):

- ``natural``: the clip as the model made it;
- ``hdr``: the Imagination Studio's HDR finish (its ``hlg`` job: real HDR, BT.2020 / HLG in 10-bit HEVC, graded bright
  and contrasty, with an SDR copy beside it), at full strength;
- ``neon``: the Neon colour table first (gma/neon.py: cobalt shadows, hot red and magenta lights; ffmpeg ``lut3d``), then
  the same HDR finish, gentler (the table has done the grading). Its SDR copy is the table's own clip;
- ``sonar``: the Sonar reveal (gma/sonar.py): the clip dark and black-and-white until pulses of echolocation run over the
  surfaces of the scene and leave it in the HDR look's colour, a sonar ping with each.

A finish is a copy beside its source (``<name>.look-neon.hdr.mp4``, ``<name>.look-neon.mp4``), made in the background,
one at a time (CPU and ffmpeg; Sonar's depth estimate uses the GPU where there is one, in its own process); until it is there the plain clip is shown. ``shown(path, hdr)`` is what a route
serves: the HDR copy to a screen that said it shows HDR, else the SDR copy.

``LookAgent`` is the voice's handle (as the video agent is): "neon look", "make it HDR", "sonar reveal", "natural colours", and "change
the look", which offers the looks to pick from by voice, by a click on the page's chips, or by typing.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse

from . import neon

LOOKS = {
    "natural": {"label": "Natural", "about": "the clip as it was made"},
    "hdr": {"label": "HDR", "about": "real HDR, bright and contrasty", "strength": 1.0},
    "neon": {"label": "Neon", "about": "HDR with strong neon colour: cobalt shadows, hot red lights", "strength": 0.6, "table": True},
    "sonar": {"label": "Sonar", "about": "black-and-white until sonar pulses run over the scene and leave it in HDR colour", "sonar": True},
}
MARK = ".look-"
ASKED_S = 60  # how long "Which look?" waits for its answer


class LookError(RuntimeError):
    pass


def is_copy(path: str | Path) -> bool:
    return MARK in Path(path).name


class LookService:
    def __init__(self, root: str | Path, client: Callable[[], object], spec: dict | None = None, ffmpeg: str | None = "ffmpeg",
                 ready: Callable | None = None) -> None:
        self.spec = spec or {}
        self.root = Path(root).expanduser()
        self.client, self.ready, self.ffmpeg = client, ready, ffmpeg or "ffmpeg"
        self.look = "natural"
        self.stamp = 0  # goes up when the look changes or a finish is made: a page plays its clip again from the new copy
        self.queue: list[tuple[Path, str]] = []
        self.working: tuple[Path, str] | None = None
        self.failed: dict[tuple[str, str], str] = {}
        self.task: asyncio.Task | None = None
        self.on_change: list[Callable[[], None]] = []
        self.recent: Callable[[], list[Path]] = lambda: []  # the clips and the film on show now: finished when the look changes
        try:
            saved = json.loads((self.root / "look.json").read_text()).get("look")
            if saved in LOOKS:
                self.look = saved
        except (OSError, ValueError, AttributeError):
            pass

    # ----- what the page sees -----

    def state(self) -> dict:
        return {"look": self.look, "stamp": self.stamp,
                "looks": [{"id": k, "label": v["label"], "about": v["about"]} for k, v in LOOKS.items()],
                "working": len(self.queue) + (1 if self.working else 0),
                "error": next(reversed(self.failed.values()), None) if self.failed else None}

    def changed(self) -> None:
        self.stamp += 1
        for hear in list(self.on_change):
            try:
                hear()
            except Exception:  # noqa: BLE001 - a listener's fault is not the look's
                pass

    def set(self, look: str) -> dict:
        if look not in LOOKS:
            raise LookError("the looks are " + ", ".join(LOOKS))
        if look != self.look:
            self.look = look
            self.failed.clear()
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                (self.root / "look.json").write_text(json.dumps({"look": look}) + "\n")
            except OSError:
                pass
            self.queue.clear()
            for path in self.recent():
                self.want(path)
            self.changed()
        return self.state()

    # ----- the copies -----

    def copy(self, source: Path, look: str, hdr: bool) -> Path:
        return source.with_name(f"{source.stem}{MARK}{look}{'.hdr' if hdr else ''}.mp4")

    def made(self, source: Path, look: str) -> bool:
        try:
            at = source.stat().st_mtime
            return all(self.copy(source, look, hdr).stat().st_mtime >= at for hdr in (True, False))
        except OSError:
            return False

    def shown(self, source: str | Path, hdr: bool = False) -> Path:
        """What to serve for ``source``: its finished copy in the look set now, else itself (and the copy is asked for)."""

        source = Path(source)
        if self.look == "natural" or source.suffix.lower() != ".mp4" or is_copy(source) or not source.is_file():
            return source
        if self.made(source, self.look):
            return self.copy(source, self.look, hdr)
        self.want(source)
        return source

    def want(self, source: str | Path) -> None:
        source, look = Path(source), self.look
        if look == "natural" or is_copy(source) or not source.is_file() or self.made(source, look):
            return
        one = (source, look)
        if one in self.queue or one == self.working or (str(source), look) in self.failed:
            return
        self.queue.append(one)
        try:
            if self.task is None or self.task.done():
                self.task = asyncio.get_running_loop().create_task(self.work())
        except RuntimeError:  # no loop (a test, a script): finish() is called by hand
            pass

    async def work(self) -> None:
        while self.queue:
            self.working = self.queue.pop(0)
            source, look = self.working
            self.changed()
            try:
                if self.ready is not None:
                    await self.ready()
                await asyncio.to_thread(self.finish, source, look)
            except Exception as error:  # noqa: BLE001 - said on the page; the plain clip stays on show
                self.failed[(str(source), look)] = f"the {LOOKS[look]['label']} look could not be made: {str(error)[:200]}"
            finally:
                self.working = None
                self.changed()

    def finish(self, source: Path, look: str) -> None:
        """Both copies of ``source`` in ``look`` (blocking: minutes for a film)."""

        spec = {**LOOKS[look], **(self.spec.get(look) or {})}
        if spec.get("sonar"):
            from . import sonar

            try:
                return sonar.finish(self, source, look, spec)
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                raise LookError(str(error)) from error
        hdr, sdr = self.copy(source, look, True), self.copy(source, look, False)
        graded = source
        if spec.get("table"):
            table = neon.cube(self.root / f"neon-v{neon.VERSION}.cube")
            part = sdr.with_name(sdr.name + ".part.mp4")
            # ffmpeg reads the table's path inside a filter: its ':' and '\' are escaped
            where = str(table).replace("\\", "/").replace(":", "\\:")
            done = subprocess.run([self.ffmpeg, "-y", "-v", "error", "-i", str(source), "-vf", f"lut3d=file='{where}':interp=tetrahedral",
                                   "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy",
                                   "-movflags", "+faststart", str(part)], capture_output=True, text=True, timeout=3600)
            if done.returncode != 0 or not part.is_file():
                raise LookError("ffmpeg could not lay the Neon table: " + done.stderr.strip()[-200:])
            part.replace(sdr)
            graded = sdr
        client = self.client()
        inside = client.bring(graded)
        brought = client.inside(graded) is None
        try:
            job = client.submit("hlg", {"video": inside, "strength": float(spec.get("strength", 1.0)), "sdr": not spec.get("table")},
                                f"Ghost Studio · {spec['label']} look")
            job = client.wait(job["id"], limit_s=float(self.spec.get("limit_s", 3600)))
            if job.get("status") != "done":
                raise LookError("the Studio's HDR finish failed: " + str(job.get("error") or job.get("status"))[:200])
            outputs = [o for o in job.get("outputs") or [] if isinstance(o, str) and o.endswith(".mp4")]
            plain = next((o for o in outputs if o.endswith("-sdr.mp4")), None)
            master = next((o for o in outputs if o != plain), None)
            if not master or (not spec.get("table") and not plain):
                raise LookError("the Studio's HDR finish ended without its files")
            client.fetch(master, hdr)
            if not spec.get("table"):
                client.fetch(plain, sdr)
            home = client.home()
            for name in outputs:  # the copies are ours now: the Studio's Library keeps none
                (home / name).unlink(missing_ok=True)
        finally:
            if brought:
                (client.home() / inside).unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------- the voice side

NAMES = {"sonar reveal": "sonar", "sonar": "sonar", "echolocation": "sonar", "neon": "neon", "vivid": "neon", "hdr": "hdr",
         "h d r": "hdr", "high dynamic range": "hdr", "natural": "natural",
         "normal": "natural", "plain": "natural", "original": "natural", "no look": "natural"}
_NAME = "(" + "|".join(sorted(NAMES, key=len, reverse=True)) + ")"
NAME = re.compile(rf"\b{_NAME}\b")
# a look asked for: its name with a look word after it ("neon look", "natural colours"), or told to ("make it HDR")
ASKS = re.compile(rf"\b{_NAME}\s+(?:\w+\s+)?(?:looks?|grade|grading|finish|colou?rs?|mode|version|effect)\b|"
                  rf"\b(?:make (?:it|this|that|them)|(?:switch|change)(?: it| this| that)? to|turn on|set it to|use)\s+(?:the |a )?{_NAME}\b")
OFF = re.compile(r"\b(?:turn|switch|take)\b.{0,12}\boff\b.{0,24}\b(look|grade|finish|hdr|h d r|neon|sonar)\b|"
                 r"\b(?:turn|switch|take)\b.{0,24}\b(look|grade|finish|hdr|h d r|neon|sonar)\b.{0,12}\boff\b|\bno (?:look|grade|finish)\b")
WHICH = re.compile(r"\b(?:change|switch|pick|choose|set|what|which|show me|another|different)\b.{0,24}\blooks?\b|"
                   r"\bwhat (?:looks|grades|finishes)\b")
# words that make a clip, not a look: "make a video of a neon city" is the video agent's
MAKES = re.compile(r"\b(video|film|movie|clip|animation)s?\b.{0,12}\b(of|about|showing|where|with)\b")


def parse(text: str, asked: bool = False) -> tuple[str, str] | None:
    """("set", look) | ("ask", "") | None for these words. ``asked``: they answer "Which look?"."""

    low = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
    if MAKES.search(low):
        return None
    if OFF.search(low):
        return "set", "natural"
    told = ASKS.search(low)
    if told:
        return "set", NAMES[told.group(1) or told.group(2)]
    if "sonar reveal" in low:  # said in any sentence that is not a clip's description: "do the sonar reveal on this one"
        return "set", "sonar"
    named = NAME.search(low)
    if named and (asked or len(low.split()) <= 2):
        return "set", NAMES[named.group(1)]
    if WHICH.search(low):
        return "ask", ""
    return None


class LookAgent:
    kind = "video"  # the server gives a "video" agent's turn to handle() and speaks its line alone
    always = True
    name = "look"
    label = "Look"
    system = None
    ack = ("On it.",)

    def __init__(self, service: LookService) -> None:
        self.service = service
        self.asked_at: float | None = None

    def asked(self) -> bool:
        return self.asked_at is not None and time.monotonic() - self.asked_at < ASKED_S

    def wants(self, text: str) -> bool:
        return parse(text, self.asked()) is not None

    async def run(self, prompt: str, progress, confirm=None) -> str:
        return (await self.handle(prompt, progress))[0]

    async def handle(self, text: str, progress) -> tuple[str, str]:
        act = parse(text, self.asked())
        if act is None:
            return "", ""
        if act[0] == "ask":
            self.asked_at = time.monotonic()
            progress("video", json.dumps({"action": "look", "what": "ask", **self.service.state()}))
            labels = [one["label"] for one in LOOKS.values()]
            return (", ".join(labels[:-1]) + f", or {labels[-1]}. Which look?",) * 2
        self.asked_at = None
        before = self.service.look
        state = self.service.set(act[1])
        progress("video", json.dumps({"action": "look", "what": "set", **state}))
        label = LOOKS[act[1]]["label"]
        if act[1] == before:
            return f"It's on the {label} look already.", f"look {act[1]} (no change)"
        if act[1] == "natural":
            return "Back to the natural look.", "look natural"
        long = "a few minutes" if LOOKS[act[1]].get("sonar") else "a minute or so"
        wait = f" I'm finishing what's on show now; it takes {long}." if state["working"] else ""
        return f"{label} look on.{wait}", f"look {act[1]}"


def register(app, engines, profile: dict) -> LookService | None:
    """Adds the look to the app: the service on ``engines.look``, its routes, and its voice agent ahead of the others."""

    video = getattr(engines, "video", None)
    renderer = getattr(getattr(engines, "supervisor", None), "renderer", None) or getattr(video, "renderer", None)
    if video is None or getattr(renderer, "client", None) is None:
        return None
    service = LookService(Path(video.root) / "look", lambda: renderer.client, profile.get("look"),
                          ffmpeg=getattr(video, "ffmpeg", None), ready=getattr(renderer, "ready", None))

    def recent() -> list[Path]:
        run = video.run or {}
        out = []
        try:
            out = [video.clip_file(run["id"], c["index"], "mp4") for c in (run.get("clips") or [])[-3:] if c.get("state", "done") == "done"]
            film = (run.get("autopilot") or {}).get("film") or {}
            if film.get("path"):
                out.append(Path(video.root) / film["path"])
        except Exception:  # noqa: BLE001 - no run to finish is not a fault
            pass
        return out

    service.recent = recent
    service.on_change.append(video.changed)
    video.look = service
    engines.look = service
    agents = getattr(engines, "agents", None)
    if isinstance(agents, dict):
        rest = dict(agents)
        agents.clear()
        agents["look"] = LookAgent(service)
        agents.update(rest)

    @app.get("/api/look")
    def look_state():
        return JSONResponse(service.state())

    @app.post("/api/look")
    async def look_set(request: Request):
        try:
            body = await request.json()
            return JSONResponse(service.set(str((body or {}).get("look") or "")))
        except (ValueError, AttributeError, LookError) as error:
            return JSONResponse({"error": str(error) or "say which look"}, status_code=400)

    return service
