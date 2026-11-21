"""The set of a rehearsed scene, made with Ghost Studio's own picture engines (the Studio's keyframe and the FastH3
clip worker, through the app's renderer), for a stage whose profile says ``"render": "h3"``:

* the reference shot: one still of the opening picture, made when the rehearsal begins and again (from the one before,
  so the people stay the same people) when the director changes how the scene opens;
* the character's presence: two short loops from that still, one where the character holds still and listens and one
  where the character talks. The page plays the first, and the second while a line is spoken, so the director talks
  to the person in the picture and not to a text box;
* the take: the plan filmed five seconds at a time, each part opening on the last frame of the part before, with the
  cast's lines spoken in the picture, joined into one film. "Cut" ends it after the parts already made.

    set = Set(renderer, run_folder, tell)
    set.dress(scene, cast, seed)                    # the still, then the loops, in the background
    part = await set.shoot(fields, seed, opens_on)  # one part of the take: the renderer's file
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from gm_core import h3_prompt

from .clock import COLD, Clock

PART_FRAMES = 124  # MiniMax H3's shortest usual clip: 5.17 s at 24 fps
PART_S = PART_FRAMES / 24
STILL = ("{who} stays in place and holds still, looking toward the camera and listening, breathing gently and blinking, "
         "with small natural movements of the eyes and face. The mouth stays closed.")
TALKS = ("{who} stays in place, looks into the camera and talks quietly and calmly the whole time, the mouth moving "
         "naturally with the words, with small changes of expression and the head almost still.")
# The clip model moves the camera toward whatever shot the words name, and reads "no zoom, no push-in" as moves
# (measured 2026-11-19 on one still, zoom over 5 s: 1.37 with those words after the scene; 1.14 with a shot size kept;
# 1.01 with this sentence first and no shot size). The picture it opens on already says how the shot is framed.
HOLD = "Static locked-off shot on a tripod. The framing and the distance to the subject stay the same from the first frame to the last."
FRAMED = re.compile(r"\b(shot|close-?up|camera|angle|framing|framed|handheld|tripod|lens)\b", re.IGNORECASE)


def unframed(scene: str) -> str:
    """``scene`` without the sentence it opens on when that one only names the shot ("Medium close-up, static camera.")."""

    first, _, rest = scene.strip().partition(". ")
    return rest.strip() if rest and len(first.split()) <= 8 and FRAMED.search(first) else scene.strip()


def held(scene: str) -> str:
    """``scene`` for a clip whose camera holds: HOLD first, and no shot size after it."""

    return f"{HOLD} {unframed(scene)}"


# the picture a part of the take ends on, made before the take is filmed: the part is then filmed between two fixed
# pictures (the clip model holds a first and a last frame), and the next part opens on this one and not on a frame of
# video, so faces do not wear from part to part; the cast's close-ups say who is who
ENDS = ("The first reference picture a few seconds later, seen from exactly the same camera position: {scene} The same place, "
        "the same light, the same framing and the same things in the background as in the first reference picture.{people} "
        "Photographic, realistic.")


MOVES = ("push", "pull", "zoom", "dolly", "pan", "tilt", "orbit", "track", "crane", "handheld", "close-up", "closer", "wide", "follow", "move")
FILLER = "Well, I have been thinking about it for a while now, and honestly I am still not sure what I should tell you."


# a character's close-up for the cast's roster, made from the reference shot so it is the same person there
PORTRAIT = ("A close-up portrait of {who}: {look}. The same person as in the reference picture, in the same clothes, the same place "
            "and the same light. Head and shoulders, facing the camera, a calm face with the mouth closed and the eyes open, "
            "the background softly out of focus. Photographic, realistic.")
FACE = 640  # the close-up's side


def face_film(still: Path, film: Path, ffmpeg: str = "ffmpeg") -> None:
    """A still as one second of film: what the lip-sync service takes as a face."""

    done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", str(still), "-t", "1", "-r", "24", "-vf", "scale=512:512,format=yuv420p",
                           "-an", "-crf", "16", str(film)], capture_output=True, text=True, timeout=120)
    if done.returncode != 0 or not film.is_file():
        raise SetError(f"The close-up could not be prepared: {done.stderr.strip()[-200:]}")


LOOP_FRAMES = 66  # of the clip's 124: its first 2.75 s, where the drift toward the face is still small


def there_and_back(clip: Path, loop: Path, ffmpeg: str = "ffmpeg", frames: int = LOOP_FRAMES) -> None:
    """The first ``frames`` of ``clip`` played forward and then backward, without sound: a loop with no jump where it
    starts again (the clip model drifts toward the face over five seconds whatever it is told, measured 2026-11-19,
    and cannot be given a last frame; most of the drift is in the second half). A clip that cannot be read is kept
    as it is."""

    done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(clip), "-filter_complex",
                           f"[0:v]trim=end_frame={int(frames)},setpts=PTS-STARTPTS,split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0", "-an", "-crf", "20", "-movflags", "+faststart",
                           str(loop)], capture_output=True, timeout=180)
    if done.returncode != 0 or not loop.is_file():
        loop.write_bytes(clip.read_bytes())
    clip.unlink(missing_ok=True)


class SetError(RuntimeError):
    """The set could not make something; the message is for the person."""


class Set:
    def __init__(self, renderer, run: Path, tell, size: tuple[int, int] | None = None, lips=None, roster: bool = False) -> None:
        # the take's two qualities: the draft at the renderer's own size, and (where the governor has a "sharp" schedule,
        # which gives the clip model the brain's room) a sharp one; the stills are made at the sharp size for both
        schedules = getattr(getattr(renderer, "governor", None), "schedules", None) or {}
        sharp = (schedules.get("sharp") or {}).get("render") or {}
        self.sharp = (int(sharp["width"]), int(sharp["height"])) if sharp.get("width") and sharp.get("height") else None
        self.quality = "draft"
        self.roster = roster  # the cast as close-ups that answer (their lips), the big picture left alone; else the two loops
        self.faces: dict[str, str] = {}  # who -> the close-up's file in the set's folder
        self.renderer, self.tell = renderer, tell
        self.lips = lips  # the lip-sync service (gma/live/lips.py), given the listening loop when there is one
        self.folder, self.takes = Path(run) / "set", Path(run) / "takes"
        self.size = size or (renderer.width, renderer.height)
        self.still: str | None = None  # the reference shot's file in the set's folder
        self.loops: dict[str, str] = {}  # "idle", "talk": their files
        self.doing, self.error = "", ""
        self.source: str | None = None  # the reference shot as the renderer keeps it
        self.task: asyncio.Task | None = None
        self.wanted: tuple | None = None  # the scene, cast and seed the set was last dressed for
        self.rolling = False  # a take is being filmed (or was): the loops wait for the next rehearsal
        self.handle: str | None = None
        self.count = 0
        self.clock = Clock(Path(run) / "usual.json")  # the steps and their times, for the page's progress track

    def cold(self) -> float:
        """What a clip takes on top when the clip model is not loaded now."""

        try:
            one = self.renderer.clip_engine()
            return 0.0 if one is None or one.loaded() else COLD
        except Exception:  # noqa: BLE001 - a renderer that cannot say: the usual time
            return 0.0

    def state(self) -> dict:
        return {"still": self.still, "loops": dict(self.loops), "doing": self.doing, "error": self.error, "roster": self.roster,
                "quality": self.quality, "sizes": {"draft": list(self.size), **({"sharp": list(self.sharp)} if self.sharp else {})},
                "faces": [{"name": c["name"], "file": self.faces.get(c["name"])} for c in (self.wanted[1] if self.wanted else [])] if self.roster else [],
                "lips": self.lips.state() if self.lips is not None else None}

    def file(self, name: str) -> Path | None:
        path = self.folder / name
        return path if path.is_file() and path.parent == self.folder else None

    def say(self, doing: str) -> None:
        if doing != self.doing:
            self.doing = doing
            self.tell()

    # ----- the renderer, one job at a time -----

    async def follow(self, handle: str, what: str) -> str:
        """The job to its end: where the renderer keeps what it made. The page is told each step."""

        self.handle = handle
        try:
            while True:
                got = await self.renderer.poll(handle)
                if got.state == "done" and got.output:
                    return got.output
                if got.state in ("done", "failed", "cancelled"):
                    raise SetError(f"{what} {got.state if got.state != 'done' else 'ended without a file'}"
                                   + (f": {str(got.error)[:200]}" if got.error else "."))
                step = f" {got.at}/{got.of}" if got.at is not None and got.of else ""
                self.say(f"{what}: {got.stage or got.state}{step}")
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            await self.renderer.cancel(handle)
            raise
        finally:
            self.handle = None

    async def clip(self, clip: dict, what: str) -> str:
        """One clip from the FastH3 worker; the renderer says no when the machine has no room for it."""

        for _ in range(240):  # the clip a cut left in the worker is finished first
            if not any(j["state"] in ("queued", "rendering") for j in getattr(self.renderer, "jobs", {}).values()):
                break
            self.say(f"{what}: waiting for the clip before it")
            await asyncio.sleep(0.5)
        refused = await self.renderer.room(clip)
        if refused is not None:
            raise SetError(str(refused))
        return await self.follow(await self.renderer.submit(clip), what)

    # ----- the reference shot and the character's presence -----

    def dress(self, scene: str, cast: list[dict], seed: int, keep: bool = False) -> None:
        """The still and the loops for ``scene``, in the background; ``keep`` makes the still from the one before."""

        self.strike(all=True)
        who = cast[0]["name"] if cast else "The character"
        if self.roster:
            self.clock.plan([("still", "still", "Reference shot"), ("blocking", "blocking", "Blocking"),
                             *[(f"face-{i}", "face", f"{c['name']}'s close-up") for i, c in enumerate(cast)], ("lips", "lips", "Lip-sync")])
            self.faces = {}
            if self.lips is not None:
                self.lips.clear()
        else:
            self.clock.plan([("still", "still", "Reference shot"), ("blocking", "blocking", "Blocking"), ("idle", "clip", f"{who} listening"),
                             ("talk", "clip", f"{who} talking"), ("lips", "lips", "Lip-sync")])
        self.wanted, self.rolling = (scene, cast, int(seed)), False
        self.task = asyncio.create_task(self._dress(scene, cast, int(seed), keep))

    def recast(self, cast: list[dict]) -> None:
        """The cast was found after the set was dressed (a set-up that named nobody): their close-ups are made too."""

        if self.wanted is None:
            return
        self.wanted = (self.wanted[0], list(cast), self.wanted[2])
        if self.roster:
            old = [(key, step["kind"], step["label"]) for key, step in self.clock.steps.items() if not key.startswith("face-") and key != "lips"]
            self.clock.plan(old + [(f"face-{i}", "face", f"{c['name']}'s close-up") for i, c in enumerate(cast)] + [("lips", "lips", "Lip-sync")],
                            keep=tuple(key for key, _, _ in old))
        if not self.rolling:
            self.resume()  # the reference shot is there already: the close-ups now; else they follow it
        self.tell()

    def strike(self, all: bool = False) -> None:
        """The take has the machine now: the character's loops still being made are left. A reference shot still being
        made is finished first, since the take opens on it (``all``: that too is dropped, for a new scene)."""

        self.rolling = True
        if self.task is not None and not self.task.done():
            if (self.source is None or self.roster) and not all:
                return  # the still (or a close-up) is on its way: the task ends after it (rolling), and the take waits for it
            self.task.cancel()
        self.task = None

    async def _dress(self, scene: str, cast: list[dict], seed: int, keep: bool) -> None:
        self.error = ""
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            references = [self.source] if keep and self.source else None
            self.clock.begin("still")
            made = await self.follow(await self.renderer.keyframe(scene, seed, references=references, size=self.sharp or self.size),
                                     "the reference shot")
            self.count += 1
            name = f"ref-{self.count}{Path(made).suffix.lower() or '.png'}"
            await self.renderer.fetch(made, self.folder / name)
            self.source, self.still, self.loops = str(self.folder / name), name, {}
            self.clock.end("still")
            self.tell()
            if self.rolling:  # "action" came while the still was made: the take has the machine, no loops now
                self.say("")
                return
            await (self._faces(*self.wanted) if self.roster else self._loops(*self.wanted))  # the cast as it is now (recast)
        except asyncio.CancelledError:
            self.doing = ""
            self.clock.drop("still")
            raise
        except Exception as wrong:  # noqa: BLE001 - the rehearsal goes on without the picture; the page says why
            self.error, self.doing = str(wrong)[:300], ""
            self.clock.end("still", failed=True)
            self.tell()

    def resume(self) -> None:
        """Back at the table after a take: the character's loops the take interrupted are made now."""

        self.rolling = False
        if self.source is None or self.wanted is None or (self.task is not None and not self.task.done()):
            return
        if self.roster and len(self.faces) < len(self.wanted[1]):
            self.task = asyncio.create_task(self._faces(*self.wanted))
        elif not self.roster and len(self.loops) < 2:
            self.task = asyncio.create_task(self._loops(*self.wanted))

    async def _faces(self, scene: str, cast: list[dict], seed: int) -> None:
        """The cast's close-ups, one after the other; each is given to the lip-sync service as it is made."""

        for i, one in enumerate(cast):
            who, key = one["name"], f"face-{i}"
            if who in self.faces:
                continue
            try:
                self.clock.begin(key)
                made = await self.follow(await self.renderer.keyframe(PORTRAIT.format(who=who, look=one.get("look") or "a person in the scene"), seed + 1 + i,
                                                                      references=[self.source], size=(FACE, FACE)), f"{who}'s close-up")
                name = f"face-{self.count}-{i}{Path(made).suffix.lower() or '.png'}"
                await self.renderer.fetch(made, self.folder / name)
                if self.lips is not None:
                    film = self.folder / f"face-{self.count}-{i}.mp4"
                    await asyncio.to_thread(face_film, self.folder / name, film)
                    self.lips.face(who, film)
                self.faces[who] = name
                self.clock.end(key)
                self.tell()
            except asyncio.CancelledError:
                self.doing = ""
                self.clock.drop(key)
                raise
            except Exception as wrong:  # noqa: BLE001 - that character answers in sound only; the page says why
                self.error = str(wrong)[:300]
                self.clock.end(key, failed=True)
                self.tell()
            if self.rolling:
                break  # a take began meanwhile: the rest after it
        self.say("")

    async def _loops(self, scene: str, cast: list[dict], seed: int) -> None:
        try:
            who = cast[0]["name"] if cast else "The person in the picture"
            for kind, action, lines in (("idle", STILL, []), ("talk", TALKS, [{"speaker": who, "voice": "a quiet natural voice", "text": FILLER}])):
                if kind in self.loops:
                    continue
                prompt = h3_prompt.compose({"scene": held(scene), "action": action.format(who=who), "dialogue": lines,
                                            "soundscape": "Quiet ambient sound of the place."}, True, frames=PART_FRAMES)
                self.clock.begin(kind, self.cold())
                self.tell()
                made = await self.clip({"prompt": prompt, "composed": True, "frames": PART_FRAMES, "seed": seed, "width": self.size[0],
                                        "height": self.size[1], "image_file": self.source},
                                       f"{who} is getting ready ({'listening' if kind == 'idle' else 'talking'})")
                if self.rolling:
                    self.clock.drop(kind)
                    break  # a take began meanwhile
                name = f"{kind}-{self.count}.mp4"
                await self.renderer.fetch(made, self.folder / f"raw-{name}")
                await asyncio.to_thread(there_and_back, self.folder / f"raw-{name}", self.folder / name)
                self.loops[kind] = name
                self.clock.end(kind)
                if kind == "idle" and self.lips is not None:
                    self.lips.follow(self.folder / name)  # her lips follow her lines from now on
                self.tell()
            self.say("")
        except asyncio.CancelledError:
            self.doing = ""
            for kind in ("idle", "talk"):
                self.clock.drop(kind)
            raise
        except Exception as wrong:  # noqa: BLE001 - the rehearsal goes on without the picture; the page says why
            self.error, self.doing = str(wrong)[:300], ""
            for kind in ("idle", "talk"):
                if self.clock.steps.get(kind, {}).get("state") == "running":
                    self.clock.end(kind, failed=True)
            self.tell()

    async def ready(self, timeout: float = 600) -> str:
        """The reference shot's file, waited for when it is still being made."""

        deadline = asyncio.get_running_loop().time() + timeout
        if self.source is None and (self.task is None or self.task.done()) and self.wanted is not None:
            self.error, self.rolling = "", True  # none was made (an earlier try failed or was dropped): made now
            self.task = asyncio.create_task(self._dress(*self.wanted, False))
        while self.source is None:
            if self.error or self.task is None or asyncio.get_running_loop().time() > deadline:
                raise SetError(self.error or "There is no reference shot to film from.")
            await asyncio.sleep(0.3)
        return self.source

    # ----- the take -----

    async def ends(self, scene: str, cast: list[dict], seed: int, opens: str, file: Path, what: str) -> str:
        """The picture a part ends on (ENDS): ``scene`` as it stands after the part, made from the picture the part
        opens on and the close-ups of those of the cast it names. Its file."""

        named = [c for c in cast if c["name"] in self.faces and c["name"].lower() in scene.lower()][:3]
        people = "".join(f" {c['name']} is the person in reference picture {i + 2} and has exactly that face." for i, c in enumerate(named))
        made = await self.follow(await self.renderer.keyframe(ENDS.format(scene=unframed(scene), people=people), int(seed),
                                                              references=[opens] + [str(self.folder / self.faces[c["name"]]) for c in named],
                                                              size=self.sharp or self.size), what)
        await self.renderer.fetch(made, file)
        return str(file)

    async def shoot(self, fields: dict, seed: int, what: str, opens_on: str | None = None, moves: str = "",
                    first: str | None = None, last: str | None = None) -> str:
        """One part of the take: ``fields`` (scene, dialogue) filmed for five seconds from the reference shot, or from
        the last frame of the part before (``opens_on``). ``moves`` is what the director asked for in this part: with
        no word about the camera in it, the camera is told to hold."""

        if not fields.get("camera") and not any(word in moves.lower() for word in MOVES):
            fields = {**fields, "scene": held(str(fields.get("scene") or ""))}  # no camera move was asked for in this part
        prompt = h3_prompt.compose(fields, True, bool(last), frames=PART_FRAMES)
        width, height = self.sharp if self.quality == "sharp" and self.sharp else self.size
        clip = {"prompt": prompt, "composed": True, "frames": PART_FRAMES, "seed": int(seed), "width": width, "height": height}
        if (width, height) != tuple(self.size):
            clip["schedule"] = "sharp"  # the governor's own: the clip model has the brain's room for it
        clip.update({"image_file": first} if first else {"opens_on": opens_on} if opens_on else {"image_file": await self.ready()})
        if last:  # the part is filmed between two pictures: the one it opens on and the one made for its end
            clip["last_file"] = last
        return await self.clip(clip, what)

    async def join(self, parts: list[Path], film: Path, ffmpeg: str = "ffmpeg") -> None:
        """The parts as one film, and its poster beside it."""

        listing = film.with_suffix(".txt")
        listing.write_text("".join(f"file '{p}'\n" for p in parts))

        def run() -> None:
            done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                                   "-movflags", "+faststart", str(film)], capture_output=True, text=True, timeout=300)
            if done.returncode != 0 or not film.is_file():
                raise SetError(f"The parts could not be joined: {done.stderr.strip()[-200:]}")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(film), "-frames:v", "1", "-vf", "scale=320:-2",
                            str(film.with_suffix(".jpg"))], capture_output=True, timeout=60)
        try:
            await asyncio.to_thread(run)
        finally:
            listing.unlink(missing_ok=True)
