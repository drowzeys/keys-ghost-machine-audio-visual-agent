"""The rehearsal's blocking preview: where the cast stand and walk, drawn as stick figures on a floor plan in about
two seconds, with no picture model. The director's plan (the opening picture and its beats) is turned into marks, start
positions and timed walks and turns by the brain, and NVIDIA's MotionBricks (SOMA, on the CPU through ONNX Runtime)
makes the body motion between them. It runs in its own container with no network (gma/live/engine/mb.py).

The preview shows the logistics before anything is rendered; it does not steer the take, which is given the same beats
as text (the clip model takes a first frame and words, nothing else).

The profile's ``live.blocking``: {"home": "~/motionbricks-try" (src, lfs, deps), "image": "vllm-audio:local"}.

    blocking = Blocking(spec, run_folder, tell, brain)
    blocking.draw(scene, cast, beats)       # in the background; state() says how it goes and names the file
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

FLOOR_M = 10.0
SYSTEM = """You block one film scene on a floor plan, for a stick-figure preview. The floor is 10 metres square: x runs \
from -5 (screen left) to 5 (screen right), z from -5 (far from the camera) to 5 (near it). The camera stands at z 7 and \
looks at the middle. "facing" is in degrees: 0 faces the camera, 90 faces screen right, -90 screen left, 180 away.

You get the opening picture, the cast and the beats (seconds after "action"). Reply with one JSON object and nothing else:
{"marks": [{"name": "lamp", "at": [-2, -1]}],
 "cast": [{"name": "Mara", "start": [-1, -1], "facing": 0,
           "cues": [{"t": 4, "do": "walk", "to": [-1, 2.5], "facing": 0}, {"t": 9, "do": "turn", "facing": 90}]}]}

Rules:
- One entry for every cast member, with the name exactly as given. "start" is where the opening picture has them.
- Every beat in which a person does something with their body gives that person at least one cue; "t" is that beat's \
seconds, or a little after it when one beat holds several things. "do" is one of:
  "walk" or "run" with "to" (and "facing" on arrival); "turn" with "facing";
  "fight" (a fighting stance with quick steps), "crouch", "dance", "sit", "look" (looks around): each goes on until that \
person's next cue; "jump" (one leap);
  and quick gestures, done on top of whatever the body is doing: "punch", "kick", "blast" (both hands thrown forward: \
an energy blast, a spell, a shot), "wave", "fly" (rises off the floor and comes down; "secs" says how long, up to 6).
- People walk about 1.4 metres a second and run about 3; "toward the camera" is a larger z, "away" a smaller one. Someone \
who enters starts at the edge of the floor and walks in at that beat.
- In a fight, first bring the fighters to about 1.2 metres from each other, each "facing" the other, and give both \
"fight"; then give them punches and kicks in turn, about one a second, for as long as the beat lasts.
- Camera moves, spoken lines, looks and expressions get no cue.
- "marks": up to four fixed things the picture names (a lamp, a door, a table), where they stand."""

MOVES, ACTIONS, GESTURES = ("walk", "run"), ("fight", "jump", "crouch", "dance", "sit", "look"), ("punch", "kick", "blast", "wave", "fly")


def plain(cast: list[dict]) -> dict:
    """Everyone standing in a row, facing the camera: the plan when the brain is away."""

    names = [c["name"] for c in cast] or ["Lead"]
    return {"marks": [], "cast": [{"name": n, "start": [(i - (len(names) - 1) / 2) * 1.5, 0.0], "facing": 0, "cues": []}
                                  for i, n in enumerate(names[:4])]}


def checked(said, cast: list[dict], beats: list[dict]) -> dict:
    """The brain's floor plan made safe to draw: known people only, everything on the floor, cues in order."""

    def spot(value, fallback=(0.0, 0.0)) -> list[float]:
        try:
            x, z = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            x, z = fallback
        edge = FLOOR_M / 2
        return [round(max(-edge, min(edge, x)), 2), round(max(-edge, min(edge, z)), 2)]

    def angle(value) -> float:
        try:
            return float(value) % 360
        except (TypeError, ValueError):
            return 0.0

    base = plain(cast)
    if not isinstance(said, dict):
        return base
    given = {str(p.get("name", "")).strip().lower(): p for p in said.get("cast") or [] if isinstance(p, dict)}
    people = []
    for person in base["cast"]:
        found = given.get(person["name"].lower())
        if found is None:
            people.append(person)
            continue
        cues = []
        for cue in found.get("cues") or []:
            if not isinstance(cue, dict) or cue.get("do") not in (*MOVES, "turn", *ACTIONS, *GESTURES):
                continue
            try:
                one = {"t": round(max(0.0, float(cue.get("t", 0))), 1), "do": cue["do"]}
            except (TypeError, ValueError):
                continue
            if cue["do"] in MOVES:
                if cue.get("to") is None:
                    continue
                one["to"] = spot(cue.get("to"))
                if cue.get("facing") is not None:
                    one["facing"] = angle(cue["facing"])
            elif cue["do"] == "turn":
                one["facing"] = angle(cue.get("facing"))
            elif cue["do"] in GESTURES and cue.get("secs") is not None:
                try:
                    one["secs"] = round(max(0.3, min(6.0, float(cue["secs"]))), 1)
                except (TypeError, ValueError):
                    pass
            cues.append(one)
        people.append({"name": person["name"], "start": spot(found.get("start"), person["start"]), "facing": angle(found.get("facing")),
                       "cues": sorted(cues, key=lambda c: c["t"])[:40]})
    marks = [{"name": str(m.get("name", ""))[:24], "at": spot(m.get("at"))} for m in (said.get("marks") or [])[:4]
             if isinstance(m, dict) and m.get("name")]
    return {"marks": marks, "cast": people}


def whole(plan: dict, beats: list[dict]) -> dict:
    """The plan as the drawing program reads it: its length (the last beat and five seconds), the camera, the beats."""

    last = max([float(b["s"]) for b in beats] + [c["t"] for p in plan["cast"] for c in p["cues"]] + [0.0])
    return {**plan, "seconds": int(min(60, max(10, last + 5))), "floor_m": FLOOR_M,
            "camera": {"at": [0, 1.5, 7.0], "look": [0, 1.0, 0]}, "beats": [{"s": float(b["s"]), "note": str(b["note"])} for b in beats]}


def faces(degrees: float) -> str:
    turn = (float(degrees) + 180) % 360 - 180
    return "faces the camera" if abs(turn) <= 45 else "has their back to the camera" if abs(turn) >= 135 else \
        "faces screen right" if turn > 0 else "faces screen left"


def staging(plan: dict | None, start: float, end: float) -> str:
    """What the floor plan has the cast do between ``start`` and ``end`` seconds, as the take's clip model is told it:
    the walks and turns the blocking preview shows, in the frame's own words (screen left, toward the camera). Where
    people stand at the start is left to the picture the part opens on."""

    said = []
    for person in (plan or {}).get("cast", []):
        at, name = list(person["start"]), person["name"]
        for cue in person["cues"]:
            if cue["t"] >= end:
                break
            if cue.get("do") in MOVES and cue.get("to"):
                to = cue["to"]
                if cue["t"] >= start:
                    dx, dz = to[0] - at[0], to[1] - at[1]
                    way = [w for w in ("toward the camera" if dz > 0.8 else "away from the camera" if dz < -0.8 else "",
                                       "to screen right" if dx > 0.8 else "to screen left" if dx < -0.8 else "") if w]
                    far = "a few steps" if (dx * dx + dz * dz) ** 0.5 < 2.5 else "across the frame" if abs(dx) > abs(dz) else "several steps"
                    enters = abs(at[0]) >= FLOOR_M / 2 - 0.6
                    ends = "screen left" if to[0] < -1.5 else "screen right" if to[0] > 1.5 else "the middle of the frame"
                    said.append((cue["t"], f"{name} {'comes in from screen ' + ('right' if at[0] > 0 else 'left') + ' and walks' if enters else 'walks ' + far} "
                                           f"{' and '.join(way) or 'a little'}, stops at {ends} and {faces(cue.get('facing', 0))}."))
                at = list(to)
            elif cue.get("do") == "turn" and cue["t"] >= start:
                said.append((cue["t"], f"{name} turns and {faces(cue.get('facing', 0))}."))
    return " ".join(text for _, text in sorted(said))


async def floor_plan(brain, scene: str, cast: list[dict], beats: list[dict]) -> tuple[dict, str]:
    """The floor plan for the scene and its beats, and who made it ("brain" or "plain")."""

    if brain is None:
        return checked(None, cast, beats), "plain"
    people = "; ".join(f"{c['name']} ({c['look']})" for c in cast) or "Lead (the person in the picture)"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Opening picture:\n{scene}\n\nCast: {people}\n\nBeats: {json.dumps(beats)}"}]
    try:
        reply = await brain.complete(messages, max_tokens=1500, temperature=0.2, timeout=75)
        found = re.search(r"\{.*\}", reply or "", re.S)
        said = json.loads(found.group(0)) if found else None
    except Exception:  # noqa: BLE001 - away, held, or not the JSON asked for
        said = None
    return checked(said, cast, beats), ("brain" if isinstance(said, dict) else "plain")


def container(spec: dict, work: Path, name: str = "ghost-live-blocking") -> list[str]:
    """The drawing program's container: no network, read-only, the CPU only, a small memory cap."""

    home = Path(str(spec.get("home", "~/motionbricks-try"))).expanduser()
    code = Path(__file__).resolve().parent / "engine"
    return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:exec,size=256m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--pids-limit", "256", "--memory", str(spec.get("memory", "8g")),
            "--cpus", str(spec.get("cpus", 8)), "--ipc", "private",
            "-v", f"{home / 'src'}:/mb/src:ro", "-v", f"{home / 'lfs'}:/mb/lfs:ro", "-v", f"{home / 'deps'}:/deps:ro",
            "-v", f"{code}:/mb/work:ro", "-v", f"{work}:/out",
            "-e", "PYTHONPATH=/deps", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", "/tmp", "--entrypoint", "python3",
            str(spec.get("image", "vllm-audio:local")), "-I", "/mb/work/mb.py", "/out/plan.json", "/out"]


def drawn(spec: dict, plan: dict, work: Path, film: Path, ffmpeg: str = "ffmpeg") -> dict:
    """``plan`` drawn into ``film`` (the default drawer; tests give their own): the program's report."""

    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps(plan))
    raw = work / "preview.rgb"
    name = f"{spec.get('name', 'ghost-live-blocking')}-{work.name}"  # a drawing dropped for a newer plan ends on its own, beside it
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
        done = subprocess.run(container(spec, work, name), capture_output=True, text=True, timeout=float(spec.get("timeout_s", 90)))
        if done.returncode != 0 or not raw.is_file():
            raise RuntimeError((done.stderr or done.stdout or "no output").strip()[-300:])
        made = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30",
                               "-i", str(raw), "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(film)],
                              capture_output=True, text=True, timeout=120)
        if made.returncode != 0 or not film.is_file():
            raise RuntimeError(made.stderr.strip()[-300:] or "the preview could not be written")
        return json.loads((work / "report.json").read_text())
    finally:
        shutil.rmtree(work, ignore_errors=True)


class Blocking:
    def __init__(self, spec: dict, run: Path, tell, brain=None, drawer=None, free_gb=None) -> None:
        self.spec, self.tell, self.brain = spec, tell, brain
        self.folder, self.work = Path(run) / "set", Path(run) / "blocking"
        self.drawer = drawer or drawn
        self.free_gb = free_gb
        self.file_name: str | None = None
        self.plan: dict | None = None
        self.doing, self.error, self.by, self.took_s = "", "", "", None
        self.task: asyncio.Task | None = None
        self.count = 0

    def state(self) -> dict:
        return {"file": self.file_name, "doing": self.doing, "error": self.error, "by": self.by, "took_s": self.took_s,
                "seconds": self.plan and self.plan["seconds"]}

    def file(self, name: str) -> Path | None:
        path = self.folder / name
        return path if path.is_file() and path.parent == self.folder else None

    def draw(self, scene: str, cast: list[dict], beats: list[dict]) -> None:
        """The preview for the plan as it stands now, in the background; one being drawn for an older plan is dropped."""

        self.clear()
        self.task = asyncio.create_task(self._draw(scene, list(cast), [dict(b) for b in beats]))

    def clear(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None

    async def _draw(self, scene: str, cast: list[dict], beats: list[dict]) -> None:
        began = time.monotonic()
        self.doing, self.error = "Blocking the scene", ""
        self.tell()
        try:
            await asyncio.sleep(float(self.spec.get("wait_s", 0.4)))  # a second note right behind this one replaces it
            floor = float(self.spec.get("floor_gb", 16))
            if self.free_gb is not None and self.free_gb() < floor:
                raise RuntimeError(f"the machine has under {floor:.0f} GB of memory free")
            found, self.by = await floor_plan(self.brain, scene, cast, beats)
            plan = whole(found, beats)
            self.folder.mkdir(parents=True, exist_ok=True)
            self.count += 1
            name = f"block-{self.count}.mp4"
            await asyncio.to_thread(self.drawer, self.spec, plan, self.work / str(self.count), self.folder / name)
            self.file_name, self.plan = name, plan
            for old in self.folder.glob("block-*.mp4"):  # the ones before, and any a dropped drawing left
                if old.name != name:
                    old.unlink(missing_ok=True)
            self.took_s = round(time.monotonic() - began, 1)
        except asyncio.CancelledError:
            raise  # a newer plan is being drawn: what the page is told is that drawing's
        except Exception as wrong:  # noqa: BLE001 - the rehearsal goes on without the preview; the page says why
            self.error = str(wrong)[:200]
        self.doing = ""
        self.tell()
