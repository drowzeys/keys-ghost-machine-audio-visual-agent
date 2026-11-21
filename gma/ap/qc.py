"""AutoPilot's self-check: whether a take shows what its shot asked for, from what is actually in the frames.

The Director names, with every shot, what can be checked (``qc``):

    {"start": ["the oil lamp is unlit, its wick dark"], "end": ["the oil lamp is lit, its flame inside the glass chimney"],
     "motion": "Arthur climbs the stairs upward, away from the camera",
     "wrong_motion": "Arthur walks down the stairs toward the camera",
     "contact": "the match flame touches the wick inside the glass chimney",
     "start_place": "...", "end_place": "...", "wrong_place": "...", "people": [{"who": "...", "count": 1}]}

A **looker** answers about pictures (``Looker``): ``yes`` (p of yes to a statement), ``pick`` (p that the right one of
two descriptions holds, asked in both orders), ``describe`` (free text of what is visible, never the plan). On a Spark
it is Gemma 4 12B with vision (vLLM, log-probabilities; gma/ap/eyes.py ``odds``), loaded for the check and released
after; d1-2B's yes/no (``D1Looker``) where no such model runs.

Continuity outranks the per-shot checks, and a retake is only for a **confident, visible** failure:

| check | asked of | retake when (confident) | else |
|---|---|---|---|
| seam | the piece before's kept last frame against this take's first frame | layout correlation < 0.5, or the looker's "same place, person, light, framing" <= 0.2 | note |
| people | first, middle, last frame | "two or more of this person" >= 0.8 | note |
| end place | last frame, ``end_place`` against ``wrong_place`` | p(right) <= 0.25 | note |
| direction | first, middle, last, ``motion`` against ``wrong_motion`` | p(right) <= 0.25 | note |
| start state / carried | first frame (a lamp lit before the match; lit/unlit, open/shut only) | p(yes) <= 0.15, and only where the piece opens (a continuing piece inherits its first frame) | note |
| hard cut | every frame at 12 fps (numeric) | layout correlation over 1/3 s < 0.3 | - |
| keyframe start | the opening keyframe (a lamp already lit) | p(yes) <= 0.15: drawn again once | note |
| end state, start place, contact | | never (contact: no model here sees a flame on a wick) | note |

Notes are not retakes: they go to the Director with the next piece, beside what the describer saw. Every answer is
written to the segment's ``qc-take<N>.json``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

CUT_FPS = 12
CUT_SIZE = (64, 36)
CUT_LAG = 4  # frames (1/3 s): a camera move or a flash keeps the picture's layout across it, a cut does not
CUT_NCC = 0.3  # the layouts' correlation under this is a cut (measured on the DGX Spark: the stair cut -0.11; continuous takes 0.52-0.92)

YES = 0.5  # p(yes) at or above this is a yes
CONFIDENT = {"people": 0.8, "end place": 0.25, "direction": 0.25, "start state": 0.15, "carried": 0.15, "seam": 0.2,
             "keyframe start": 0.15}
# a start state retakes only when it names something a picture shows as one of two states (a lamp lit or unlit, a door
# open or shut); "the railing untouched" and the like are notes (Gemma said 0.00 to it on a good take)
BINARY = re.compile(r"\b(un)?lit\b|\bburning\b|\bflame\b|\bdark\b|\bopen\b|\bclosed\b|\bshut\b", re.I)
RETAKE = ("seam", "people", "end place", "direction", "start state", "carried", "cut", "keyframe start")
SEAM_CORR = 0.5  # the seam's 16x9 layout correlation under this is a jump (continuation keeps the layout: ~0.9+)
YES_SYSTEM = "You answer one question about pictures, only from what is visible in them. Answer with one word: yes or no."
PICK_SYSTEM = "You compare two descriptions with pictures, only from what is visible. Answer with one letter."
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
RATE_SYSTEM = "You judge statements about pictures, only from what is visible. Answer with one number."
COUNT_SYSTEM = "You count people in a picture, only from what is visible. Answer with one number."
COUNT = ("How many different people are visible in this picture, whole or in part (a person seen through a window, in "
         "a doorway or in a reflection counts once more)? Answer with one number.")
SEE_SYSTEM = "You describe video frames exactly as they look. Never guess what was meant to happen."
SEE = ("Describe what is visible in each frame, in order (where each person is, inside or outside and in which room, what "
       "they hold and do, the state of lamps, doors and lights), then how they move across the frames. 80 words.")


class Looker:
    """The vision model's answers: ``eyes`` is gma/ap/eyes.py's Eyes (``odds`` and ``look``)."""

    def __init__(self, eyes) -> None:
        self.eyes = eyes

    def yes(self, pictures: list, statement: str) -> float:
        return self.eyes.odds(YES_SYSTEM, pictures, f"Is this true: {statement}?\nAnswer yes or no.", ("yes", "no"))

    def pick(self, pictures: list, question: str, right: str, wrong: str) -> float:
        """p that ``right`` holds rather than ``wrong``, asked with each first (a looker leans to one letter)."""

        ask = lambda a, b: self.eyes.odds(PICK_SYSTEM, pictures, f"{question}\nA: {a}\nB: {b}\nAnswer with one letter: A or B.", ("A", "B"))
        return (ask(right, wrong) + 1 - ask(wrong, right)) / 2

    def describe(self, pictures: list) -> str:
        return self.eyes.look(SEE_SYSTEM, pictures, None, SEE)[0]

    def count(self, picture: tuple[str, Path]) -> int | None:
        return number(self.eyes.look(COUNT_SYSTEM, [picture], None, COUNT)[0])


class GhostLooker:
    """Ghost 3.8 27B (TensorFold, --vision) as the looker: ONE picture a request (several frames are tiled into one
    labelled sheet), a 0-10 rating for a statement, and an A/B choice it explains frame by frame before it answers,
    asked in both orders (p 1.0 when both answers name the right one, 0.0 when both name the wrong one, 0.5 split).
    Measured on the original film's frames (DGX Spark, 2026-11-08): lamp unlit 0/10, wrong way on the stair, out through the
    door and the flame outside the glass all named right in both orders; the end place split (a note)."""

    def __init__(self, eyes, folder: Path | None = None) -> None:
        self.eyes, self.folder = eyes, Path(folder or "/tmp")
        self.folder.mkdir(parents=True, exist_ok=True)

    def one(self, pictures: list) -> tuple[str, Path]:
        if len(pictures) == 1:
            return pictures[0]
        target = self.folder / f"sheet-{abs(hash(tuple(str(p) for _, p in pictures)))}.png"
        if not target.is_file():
            tile(pictures, target)
        return ("Frames side by side in time order, left to right: " + ", ".join(c for c, _ in pictures), target)

    def yes(self, pictures: list, statement: str) -> float:
        said = self.eyes.look(RATE_SYSTEM, [self.one(pictures)], None,
                              f"Statement: {statement}\nHow true is the statement, from what is visible? Answer with one "
                              "number from 0 (clearly false) to 10 (clearly true).")[0]
        found = re.search(r"\d+", said or "")
        return min(10, int(found.group(0))) / 10 if found else 0.5

    def pick(self, pictures: list, question: str, right: str, wrong: str) -> float:
        picture = self.one(pictures)

        def ask(a: str, b: str) -> str:
            said = self.eyes.look(PICK_SYSTEM, [picture], None,
                                  f"{question}\nA: {a}\nB: {b}\nFirst say in one short sentence per frame where the person is "
                                  "and which way they face and move. Then write the last line as ANSWER: A or ANSWER: B.")[0]
            found = re.findall(r"ANSWER:\s*\**([AB])", said or "")
            return found[-1] if found else "?"
        return ((ask(right, wrong) == "A") + (ask(wrong, right) == "B")) / 2

    def describe(self, pictures: list) -> str:
        return self.eyes.look(SEE_SYSTEM, [self.one(pictures)], None, SEE)[0]

    def count(self, picture: tuple[str, Path]) -> int | None:
        return number(self.eyes.look(COUNT_SYSTEM, [picture], None, COUNT)[0])


STRICT_SYSTEM = "You check statements about video frames, only from what is visible. Never guess what was meant to happen."
STRICT_PASS = 9  # the rating (0-10) both orders must reach for a pass


class StrictGhostLooker(GhostLooker):
    """Ghost 3.8 held to Gemma's standard (2026-11-08, after its first film passed nearly everything: piece 2's own
    description said "no visible lamps" while its rating passed "a flame flickers in the wick"). The frames are described
    first (once a picture), then every statement is asked as a yes/no choice in both orders with that description in
    front of it, each answer with a 0-10 rating. It passes only when both orders say it holds and both ratings are 9 or
    10; anything less fails and goes to the Director as a fix note. p: both orders false 0.0-0.1 (a confident failure),
    the orders split 0.3, both true but rated under 9 0.45, a pass the lower rating / 10."""

    def __init__(self, eyes, folder: Path | None = None) -> None:
        super().__init__(eyes, folder)
        self.seen: dict[str, str] = {}

    def described(self, picture: tuple[str, Path]) -> str:
        key = str(picture[1])
        if key not in self.seen:
            self.seen[key] = (self.eyes.look(SEE_SYSTEM, [picture], None, SEE)[0] or "").strip()
        return self.seen[key]

    def verdict(self, picture: tuple[str, Path], question: str, a: str, b: str) -> tuple[str, int]:
        said = self.eyes.look(STRICT_SYSTEM, [picture], None,
                              f"What the frames show (described before this question): {self.described(picture)}\n\n"
                              f"{question}\nA: {a}\nB: {b}\nCheck the picture and the description against A and B, one short "
                              "sentence. Then write RATING: <0-10, how clearly the answer you choose is visible> and, on the "
                              "last line, ANSWER: A or ANSWER: B.")[0] or ""
        found = re.findall(r"ANSWER:\s*\**([AB])", said)
        rating = re.findall(r"RATING:\s*\**(\d+)", said)
        return (found[-1] if found else "?"), (min(10, int(rating[-1])) if rating else 0)

    def strict(self, picture: tuple[str, Path], question: str, right: str, wrong: str) -> float:
        first, r1 = self.verdict(picture, question, right, wrong)
        second, r2 = self.verdict(picture, question, wrong, right)
        votes = (first == "A") + (second == "B")
        if votes == 2:
            low = min(r1, r2)
            return low / 10 if low >= STRICT_PASS else 0.45
        if votes == 1:
            return 0.3
        return max(0, 10 - min(r1, r2)) / 100  # false in both orders: 0.0-0.1, the surer the lower

    def yes(self, pictures: list, statement: str) -> float:
        return self.strict(self.one(pictures), "Is this statement true in the frames?", f"Yes: {statement}",
                           f"No: it is not the case that {statement[0].lower() + statement[1:] if statement else statement}")

    def pick(self, pictures: list, question: str, right: str, wrong: str) -> float:
        return self.strict(self.one(pictures), question, right, wrong)

    def describe(self, pictures: list) -> str:
        return self.described(self.one(pictures))

    def count(self, picture: tuple[str, Path]) -> int | None:
        """Counted with its own description in front (the description is the honest part of Ghost's answers)."""

        return number(self.eyes.look(COUNT_SYSTEM, [picture], None, f"What the picture shows (described before this "
                                     f"question): {self.described(picture)}\n\n{COUNT}")[0])


def tile(pictures: list, target: Path, ffmpeg: str = "ffmpeg") -> Path:
    """Frames side by side in one picture, each labelled with its number (a looker that takes one picture a request)."""

    args = [ffmpeg, "-v", "error", "-y"]
    for _, p in pictures:
        args += ["-i", str(p)]
    font = next((f for f in FONTS if Path(f).is_file()), None)
    label = (lambda i: f",drawtext=fontfile={font}:text='{i + 1}':x=8:y=6:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6") if font else (lambda i: "")
    parts = [f"[{i}:v]scale=448:256{label(i)}[v{i}]" for i in range(len(pictures))]
    graph = ";".join(parts) + ";" + "".join(f"[v{i}]" for i in range(len(pictures))) + f"hstack=inputs={len(pictures)}"
    subprocess.run(args + ["-filter_complex", graph, "-frames:v", "1", str(target)], check=True, capture_output=True, timeout=60)
    return target


class D1Looker:
    """d1-2B's yes/no where no vision model that describes runs: ``ask(state, {key: question}, [paths]) -> {key: p}``."""

    def __init__(self, ask) -> None:
        self.ask = ask

    def yes(self, pictures: list, statement: str) -> float:
        state = "This is one picture." if len(pictures) == 1 else "These are frames in time order from one video clip."
        return self.ask(state, {"q": f"Is this true: {statement}?"}, [p for _, p in pictures])["q"]

    def pick(self, pictures: list, question: str, right: str, wrong: str) -> float:
        r, w = self.yes(pictures, right), self.yes(pictures, wrong)
        return r / (r + w) if r + w > 0 else 0.5

    def describe(self, pictures: list) -> str:
        return ""


def clean_qc(data) -> dict:
    """The Director's ``qc`` as lists and strings (anything else dropped)."""

    data = data if isinstance(data, dict) else {}

    def states(value) -> list[str]:
        return [re.sub(r"\s+", " ", str(v)).strip()[:120] for v in (value if isinstance(value, list) else []) if str(v).strip()][:2]
    def line(key: str) -> str:
        return re.sub(r"\s+", " ", str(data.get(key) or "")).strip()[:160]
    people = []
    for one in data.get("people") if isinstance(data.get("people"), list) else []:
        who = re.sub(r"\s+", " ", str((one or {}).get("who") if isinstance(one, dict) else one or "")).strip()[:100]
        if who:
            people.append({"who": who, "count": max(1, int((one.get("count") if isinstance(one, dict) else 1) or 1))})
    return {"start": states(data.get("start")), "end": states(data.get("end")), "motion": line("motion"),
            "wrong_motion": line("wrong_motion"), "contact": line("contact"), "start_place": line("start_place"),
            "end_place": line("end_place"), "wrong_place": line("wrong_place"), "people": people[:3]}


def start_line(qc: dict) -> str:
    """The keyframe's explicit opening state, for its prompt: "At this moment: oil lamp: unlit, dark wick."."""

    return ("At this moment, before anything happens: " + "; ".join(qc["start"]) + ".") if qc.get("start") else ""


# ----- hard cuts, numerically -----

def grey_frames(clip: Path, ffmpeg: str = "ffmpeg") -> np.ndarray:
    """The clip at CUT_FPS as (n, h, w) grey levels."""

    w, h = CUT_SIZE
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(clip), "-vf", f"fps={CUT_FPS},scale={w}:{h}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True, timeout=120).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)


def layout_corr(a: np.ndarray, b: np.ndarray) -> float:
    """The correlation of two frames' 16x9 layouts (brightness and contrast taken out: a lightning flash is no cut)."""

    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-6))


def cuts_of(frames: np.ndarray, fps: float = CUT_FPS) -> dict:
    """Hard cuts in ``frames`` ((n, h, w) grey levels, h and w multiples of 9 and 16): the layout of the picture a
    third of a second later no longer correlates with this one (CUT_NCC); one cut a second at most."""

    n, h, w = frames.shape
    if n <= CUT_LAG:
        return {"cuts": [], "min": None}
    small = frames.reshape(n, 9, h // 9, 16, w // 16).mean(axis=(2, 4))
    corr = [layout_corr(small[i], small[i + CUT_LAG]) for i in range(n - CUT_LAG)]
    cuts = []
    for i, c in enumerate(corr):
        if c < CUT_NCC:
            t = round((i + CUT_LAG / 2) / fps, 2)
            if cuts and t - cuts[-1]["t"] < 1.0:
                if c < cuts[-1]["corr"]:
                    cuts[-1].update(t=t, corr=round(c, 2))
                continue
            cuts.append({"t": t, "corr": round(c, 2)})
    return {"cuts": cuts, "min": round(min(corr), 2), "median": round(float(np.median(corr)), 2)}


def hard_cuts(clip: Path, ffmpeg: str = "ffmpeg") -> dict:
    try:
        return cuts_of(grey_frames(clip, ffmpeg))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {"cuts": [], "error": str(error)[:200]}


# ----- the checks -----

def number(said: str | None) -> int | None:
    found = re.search(r"\d+", said or "")
    if found:
        return int(found.group(0))
    words = {"zero": 0, "none": 0, "no": 0, "one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    first = re.findall(r"[a-z]+", (said or "").lower())
    return words.get(first[0]) if first else None


def surplus(look, pictures: list, expected: int) -> tuple[float, list]:
    """The people check against the cast count: the share of ``pictures`` (caption, path) that show MORE people than
    the shot plans (a second copy of someone), by the looker's own count; one person plainly shown is never a failure.
    A looker without a count (d1-2B) asks "two or more" of the first planned person instead."""

    counts = []
    for caption, path in pictures:
        if hasattr(look, "count"):
            n = look.count((caption, path))
            counts.append(n)
        else:
            counts.append(None)
    if all(n is None for n in counts):
        return -1.0, counts
    over = [n is not None and n > expected for n in counts]
    return sum(over) / len(over), counts


def many(who: str) -> str:
    return (f"two or more of this person are visible anywhere in the picture, counting any seen through a window, in a "
            f"doorway or in a reflection: {who}")


def result(check: str, state: str, where: str, p: float, failed: bool, **more) -> dict:
    """One check: ``passed`` (no failure), ``confident`` (a failure sure enough to retake), its p."""

    limit = CONFIDENT.get(check)
    if check == "people":
        confident = failed and p >= limit
    else:
        confident = failed and limit is not None and p <= limit
    if check in ("start state", "carried", "keyframe start") and not BINARY.search(state):
        confident = False
    return {"check": check, "state": state, "frame": where, "p": round(p, 3), "passed": not failed,
            "confident": bool(confident), **more}


def two_or_more(look, picture: Path, who: str) -> float:
    """p that a second copy of the person shows: asked as a choice against "exactly one" (a looker that agrees with any
    statement, as Ghost's ratings do, said 10 of 10 to "two or more" on frames with one man)."""

    return 1 - look.pick([("The picture", picture)], "Which is true in the picture?",
                         f"exactly one of this person is visible: {who}", many(who))


def keyframe_checks(look, plan: dict, picture: Path) -> list[dict]:
    """The opening keyframe: its start states and each planned person once (notes and one redraw at most)."""

    out = []
    for x in plan.get("start") or []:
        p = look.yes([("The picture", picture)], x)
        out.append(result("keyframe start", x, "the keyframe", p, p < YES))
    out += people_checks(look, plan, [("the keyframe", ("The picture", picture))])
    return out


def people_checks(look, plan: dict, where: list) -> list[dict]:
    """Each planned person once: counted against the cast (``surplus``) where the looker counts, else "two or more"
    of each person asked (d1-2B). ``where``: [(label, (caption, path))]."""

    people = [one for one in plan.get("people") or [] if one.get("count")]
    if not people:
        return []
    expected = sum(int(one["count"]) for one in people)
    p, counts = surplus(look, [pic for _, pic in where], expected)
    who = "; ".join(one["who"] for one in people)
    if p >= 0:
        labels = ", ".join(f"{label} {n if n is not None else '?'}" for (label, _), n in zip(where, counts))
        return [result("people", f"{expected} {'person' if expected == 1 else 'people'} ({who}), never a second copy",
                       labels, p, p >= YES, counted=counts, expected=expected)]
    out = []
    for one in people:
        if one["count"] == 1:
            worst = max((two_or_more(look, pic[1], one["who"]), label) for label, pic in where)
            out.append(result("people", f"exactly one {one['who']}", worst[1], worst[0], worst[0] >= YES))
    return out


def take_checks(look, plan: dict, frames: list, carried: list[str] | None = None, opens_on: Path | None = None) -> list[dict]:
    """``frames``: the take's (caption, path) frames in time order, its last frame last. ``opens_on``: the picture the
    take was made from (its keyframe, or the piece before's kept last frame): when the first frame keeps that picture's
    layout, the take continues it, and its start (states, place, the carried end) is that picture's, already checked
    there; it is not asked again of the first frame (the strict looker failed true starts on long compound statements)."""

    out = []
    first, middle, last = frames[0], frames[len(frames) // 2], frames[-1]
    three = [("First frame", first[1]), ("Middle frame", middle[1]), ("Last frame", last[1])]
    opened = None
    if opens_on is not None and Path(opens_on).is_file():
        try:
            opened = seam_numbers(picture_grey(Path(opens_on)), picture_grey(first[1]))
        except Exception:  # noqa: BLE001 - the start checks are asked instead
            opened = None
    continues = opened is not None and opened["corr"] >= SEAM_CORR
    if opened is not None:
        out.append(result("opening", "the first frame continues the picture the take was made from", "the first frame",
                          1.0 if continues else 0.0, not continues, **opened))
    starts = [] if continues else [("carried", list(carried or []), first, "the first frame"),
                                   ("start state", plan.get("start") or [], first, "the first frame"),
                                   ("start place", [plan["start_place"]] if plan.get("start_place") else [], first, "the first frame")]
    for check, states, picture, where in starts + [("end state", plan.get("end") or [], last, "the last frame")]:
        for x in states:
            p = look.yes([("The picture", picture[1])], x)
            out.append(result(check, x, where, p, p < YES))
    if plan.get("end_place"):
        if plan.get("wrong_place"):
            p = look.pick([("The picture", last[1])], "Which is true in the picture?", plan["end_place"], plan["wrong_place"])
        else:
            p = look.yes([("The picture", last[1])], plan["end_place"])
        out.append(result("end place", plan["end_place"], "the last frame", p, p < YES))
    if plan.get("motion"):
        if plan.get("wrong_motion"):
            p = look.pick(three, "These three frames are in time order. Which is true across them?", plan["motion"], plan["wrong_motion"])
        else:
            p = look.yes(three, plan["motion"])
        out.append(result("direction", plan["motion"], "first, middle, last", p, p < YES))
    if plan.get("contact"):
        p = max(look.yes([("The picture", f[1])], plan["contact"]) for f in frames[-3:])
        out.append(result("contact", plan["contact"], "the last three frames", p, p < YES))
    out += people_checks(look, plan, [("the first frame", ("The picture", first[1])), ("the middle frame", ("The picture", middle[1])),
                                      ("the last frame", ("The picture", last[1]))])
    return out


def picture_grey(picture: Path, ffmpeg: str = "ffmpeg") -> np.ndarray:
    w, h = CUT_SIZE
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(picture), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray",
                          "-frames:v", "1", "-"], capture_output=True, timeout=60).stdout
    return np.frombuffer(raw[: w * h], dtype=np.uint8).reshape(h, w).astype(np.float32)


def seam_numbers(a: np.ndarray, b: np.ndarray) -> dict:
    """Two frames' 16x9 layout correlation and their mean brightness gap (0-255)."""

    h, w = a.shape
    small = lambda x: x.reshape(9, h // 9, 16, w // 16).mean(axis=(1, 3))
    return {"corr": round(layout_corr(small(a), small(b)), 3), "luma_gap": round(abs(float(a.mean()) - float(b.mean())), 1)}


def seam(look, before: Path, first: Path, ffmpeg: str = "ffmpeg") -> dict:
    """The cut into this piece: the piece before's kept last frame against this take's first frame."""

    numbers = seam_numbers(picture_grey(before, ffmpeg), picture_grey(first, ffmpeg))
    p = look.yes([("Last frame of the shot before", before), ("First frame of this shot", first)],
                 "these two pictures continue each other: the same place, the same person in the same spot, the same light "
                 "and nearly the same framing") if look is not None else None
    jump = numbers["corr"] < SEAM_CORR
    r = result("seam", "the shot continues the last frame of the one before", "the cut", p if p is not None else 1.0,
               jump or (p is not None and p < YES), **numbers)
    r["confident"] = jump or (p is not None and p <= CONFIDENT["seam"])
    return r


def verified_end(record: dict) -> list[str]:
    """What a kept take showed at its end, verified: its passed end states and end place (the next piece's carried)."""

    return [r["state"] for r in record.get("checks") or [] if r["passed"] and r["check"] in ("end state", "end place")]


def inherit(results: list[dict]) -> None:
    """A continuing take whose seam passed opens on the piece before's own last frame: what its first frame shows was
    left by that piece, and a retake here cannot change it (pieces are never sent back). Its start checks are notes."""

    if any(r["check"] == "seam" and r["passed"] for r in results):
        for r in results:
            if r["check"] in ("carried", "start state", "start place") and not r["passed"]:
                r.update(confident=False, inherited=True)


def notes_of(record: dict) -> list[str]:
    """The take's failed checks that are not retakes: notes for the Director's next piece."""

    return [f"{f['check']}: {f['evidence']}" for f in record.get("findings") or [] if not f.get("retake")]


def seen(results: list[dict]) -> str:
    """The checks' answers as one line: "the first frame: the oil lamp is unlit (failed, 0.04); ..."."""

    return "; ".join(f"{r['frame']}: {r['state']} ({'ok' if r['passed'] else 'failed'}, {r['p']:.2f})" for r in results)


FIX = {
    "keyframe start": "the keyframe must show the START of the shot: {state}. Write the keyframe with that state named "
                      "plainly and nothing of the outcome (no light, flame or result that comes later)",
    "start state": "the first frame must show {state}; it did not. Keep the opening in that state",
    "end state": "the last frame must show {state}; it did not. Make the action reach that result on screen",
    "direction": "the movement must be: {state}; the take showed otherwise. Anchor the camera (where it stands, which way "
                 "it looks) and say the direction relative to it, one continuous movement",
    "contact": "the contact must be visible: {state}; it was not. Spell out where the two things meet and frame them close",
    "cut": "the take has a hard cut at {state}; write one continuous shot with no cut, the camera moving with the action",
    "keyframe place": "the keyframe must show the start place: {state}. Name the room and which side of the door the "
                      "person is on",
    "start place": "the first frame must show the start place: {state}; it did not",
    "end place": "the shot must end here: {state}; it ended elsewhere (the wrong side of a door, or outside). Say which "
                 "way through the door and where the person stops",
    "people": "the frame must show {state}, never a second copy (through a window, a doorway or in a reflection). Keep "
              "the person in one place in the shot and name where they stand",
    "carried": "the piece before ended with {state}; this piece must open exactly so (it continues from that last frame)",
    "opening": "{state}: it did not (the shot left its opening picture at once); open on that picture and move from it",
    "seam": "{state}: the cut into this piece jumps (another place, person, light or framing); continue the shot before",
}


def failures(results: list[dict], cuts: dict, judge_cut: float | None = None) -> list[dict]:
    """The failed checks as findings: ``retake`` only for a confident failure of a check in RETAKE, else a note."""

    found = []
    for r in results:
        if not r["passed"]:
            retake = r["check"] in RETAKE and r.get("confident", False)
            found.append({"category": "qc", "check": r["check"], "status": "major" if retake else "minor", "frame": 0, "source": "qc",
                          "retake": retake, "p": r.get("p"), "evidence": FIX[r["check"]].format(state=r["state"])})
    for c in cuts.get("cuts") or []:
        found.append({"category": "qc", "check": "cut", "status": "major", "frame": 0, "source": "qc", "retake": True,
                      "evidence": FIX["cut"].format(state=f"{c['t']:.1f} s")
                      + (f" (the judge's cut answer {judge_cut:.2f})" if judge_cut is not None else "")})
    return found
