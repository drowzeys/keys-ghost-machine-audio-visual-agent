# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, app/checks.py at 6cc6e67c5e138499ea2c17183eda261b710d1217.
# Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: nothing; the code is unchanged.
"""AutoPilot's dense checks: many small yes/no questions to the vision brain, decided in code (autopilot.py asks,
this file says what to ask and what the answers mean), and the pace of a piece measured from its pixels.

Why yes/no and why many. On the Viking v2 test film (six 10 s pieces, one Cosmos3-Edge verdict a piece) the eyes were
noisy: a scar was read as a wound, a carved training-dummy face as an extra person, the same picture got different
answers twice, and one answer came back as nested JSON that was read as "no answer" to every question and rendered a
good piece again. So:

  windows   a piece is looked at in windows (``every`` seconds, 2 by default), each a few frames with their times
            (``frames``, 3), not one sheet of eight; the keyframe is looked at before anything moves, and a
            continuation's join (the last frame of the piece before beside this one's first) is a window of its own
  yes/no    each window gets short questions built from the continuity bible (is Eirik still the man in the
            reference? does he wear the brown coat? is this the cobbled street? anything from a later era? does a person
            turn into another?), each answered "yes" or "no"; pass or fail is decided here, against the answer the bible
            expects. A missing answer (or an "unclear" given anyway) is never a failure; an answer that is not JSON, or that
            answers less than ``ANSWERED`` of the questions, is asked again
  votes     a failed question of a major kind is asked again on its own, ``votes`` times in all, and fails only by a
            majority of the yes/no answers; the second look of autopilot.second_look still has the last word
  pace      a fight or an action piece is measured (the mean difference between frames, a second at a time) and the
            eyes are asked "is this slow motion?" in each window; a piece the Director planned fast that comes out slow
            throughout fails with a pace finding

Every request is logged with its window, questions, answers, votes and seconds (``log``), so the page and the run's
report say how many checks ran and what each found.
"""

from __future__ import annotations

import re
import subprocess
import time

DEFAULT = {"dense": True, "every": 2.0, "frames": 3, "votes": 3, "join": True, "keyframe": True, "pace": True}
ANSWERED = 0.7  # an answer that answers fewer of the questions than this is asked again, once
FAST_KINDS = ("fight", "action", "chase")
PACE_KINDS = FAST_KINDS + ("calm", "dialogue")
SPEEDS = ("fast", "real", "slow")
SLOW_WORDS = re.compile(r"\bslow[- ]?(motion|mo)\b|\bslowly\b|\bslow\b", re.I)
# The questions offer only yes and no. With "unclear" offered as well, Cosmos3-Edge answered "unclear" to nearly every
# question that expects "no" (a morph, a malformed hand, an out-of-era object, slow motion) on the first live Infinity
# run (2026-10-26, a DGX Spark): 3 of 6 questions a window unanswered, every window asked twice. Asked yes or no only,
# the same windows were answered in full in about 1 s each; with its reasoning on it took 3 to 11 s, refused once and
# flipped two answers. An "unclear" that comes back anyway still counts as no answer.
SYSTEM = ("You inspect frames of a film carefully and answer each question only from what you can see. "
          "Answer with JSON only: one object mapping each key to \"yes\" or \"no\".")


def clip(text, words: int) -> str:
    return " ".join(str(text or "").split()[:words])


def options(settings: dict) -> dict:
    given = (settings or {}).get("checks") or {}
    out = dict(DEFAULT)
    for key in ("dense", "join", "keyframe", "pace"):
        if given.get(key) is not None:
            out[key] = bool(given[key])
    for key, low, high, kind in (("every", 1.0, 10.0, float), ("frames", 1, 4, int), ("votes", 1, 7, int)):
        if given.get(key) not in (None, ""):
            try:
                out[key] = max(low, min(high, kind(given[key])))
            except (TypeError, ValueError):
                raise ValueError(f"checks.{key} is a number from {low:g} to {high:g}")
    return out


# ---------- the Director's pace plan ----------

def pace_of(data) -> dict:
    """The Director's pace plan cleaned: {"kind", "beats": [{"from", "to", "speed", "what"}]}; calm when it gave none."""

    data = data if isinstance(data, dict) else {}
    kind = str(data.get("kind") or "calm").strip().lower()
    kind = kind if kind in PACE_KINDS else ("fight" if "fight" in kind else "action" if "action" in kind else "calm")
    beats = []
    for b in data.get("beats") or []:
        if not isinstance(b, dict):
            continue
        try:
            a, z = float(b.get("from")), float(b.get("to"))
        except (TypeError, ValueError):
            continue
        speed = str(b.get("speed") or "real").strip().lower()
        speed = "slow" if "slow" in speed else "fast" if speed in ("fast", "quick", "rapid") else "real"
        if z > a:
            beats.append({"from": round(a, 2), "to": round(z, 2), "speed": speed, "what": clip(b.get("what"), 8)})
    return {"kind": kind, "beats": sorted(beats, key=lambda b: b["from"])}


def pace_problem(pace: dict, action: str, camera: str, seconds: float) -> str:
    """Why a pace plan cannot be used ("" when it can). A fight or an action piece is mostly fast: slow motion only for
    one or two key moves, a third of the piece at most; outside those spans neither the action nor the camera says
    "slow". The key move's seconds need not be in the action: pace_line writes them into the video prompt (on the
    first live Infinity run Bolt's timings were turned back three times by a stricter rule)."""

    slow = [b for b in pace["beats"] if b["speed"] == "slow"]
    slow_s = sum(b["to"] - b["from"] for b in slow)
    if pace["kind"] in FAST_KINDS:
        if not pace["beats"]:
            return ('a fight or action piece needs "pace.beats": its seconds, fast for the exchange and slow only for a key move, '
                    'like [{"from": 0, "to": 6, "speed": "fast"}, {"from": 6, "to": 8, "speed": "slow", "what": "the decisive blow"}, '
                    '{"from": 8, "to": %g, "speed": "fast"}]' % seconds)
        if len(slow) > 2:
            return f"{len(slow)} slow-motion spans: a fight or action piece keeps slow motion for one or two key moves"
        if slow_s > max(3.0, seconds / 3):
            return (f"{slow_s:g} s of {seconds:g} in slow motion: a fight or action piece is mostly fast, real-time; slow motion "
                    "only for the key move, a third of the piece at most")
        if SLOW_WORDS.search(camera or ""):
            return '"camera" says slow: a fight or action camera is snappy (whip pans, quick reframes, handheld); the key move is slowed in "action" only'
    if SLOW_WORDS.search(action or "") and not slow:
        return '"action" says slow motion but "pace.beats" has no slow span: say real-time, or plan the key move\'s span'
    return ""


def pace_line(pace: dict, seconds: float) -> str:
    """The words the video model reads for the pace (MiniMax H3 has no speed control of its own: the prompt is all)."""

    if pace["kind"] not in FAST_KINDS:
        return ""
    slow = [b for b in pace["beats"] if b["speed"] == "slow"]
    line = ("Pace: fast, real-time speed throughout, quick strikes and rapid exchanges, sharp impacts, snappy whip pans and "
            "quick handheld reframes")
    if slow:
        spans = "; ".join(f"from {b['from']:g} to {b['to']:g} s" + (f" on {b['what']}" if b["what"] else "") for b in slow)
        line += f", with slow motion only {spans}, then straight back to full real-time speed"
    return line + "."


def pace_opening(pace: dict) -> str:
    """The first words of a fight's action: the speed from the very first frame. On the merge gate's films the openings
    of fight pieces measured 3 to 5 against 13 to 22 in their exchanges: H3 eased in from a still frame."""

    if pace["kind"] not in FAST_KINDS:
        return ""
    first = next((b for b in pace["beats"] if b["from"] <= 0.5), None)
    if first and first["speed"] == "slow":
        return ""
    return "From the very first frame, already in motion at full real-time speed:"


def planned(pace: dict, at: float, seconds: float) -> str:
    """The speed the plan has at second ``at``: fast in a fight or action piece unless a slow span covers it."""

    for b in pace["beats"]:
        if b["from"] <= at < b["to"]:
            return b["speed"] if pace["kind"] in FAST_KINDS or b["speed"] == "slow" else "real"
    return "fast" if pace["kind"] in FAST_KINDS else "real"


# ---------- the windows ----------

def windows(length: float, every: float, frames: int) -> list[list[float]]:
    """The times of each window's frames: ``length`` cut into windows of about ``every`` seconds, ``frames`` evenly
    in each (none at the very end: the last frame of a clip is its own)."""

    count = max(1, round(length / max(every, 0.5)))
    span = length / count
    end = max(0.0, length - 0.05)
    return [[round(min(end, i * span + (j + 0.5) * span / frames), 2) for j in range(frames)] for i in range(count)]


def grab(clip_path, at: float, target, size=(448, 256), env=None) -> bool:
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.3f}", "-i", str(clip_path), "-frames:v", "1", "-vf",
                        f"scale={size[0]}:{size[1]}", str(target)], check=True, env=env, timeout=60, capture_output=True)
        return target.is_file()
    except (subprocess.SubprocessError, OSError):
        return False


# ---------- the questions ----------

def questions(bible: dict, index: int, seg: dict, cast: dict, categories: list, continuity) -> list[dict]:
    """The yes/no questions for one window of segment ``index``: {key, category, text, expect, major}. Built from the
    bible's canonical facts for the people the shot names, its setting and its era."""

    fields = seg.get("fields") or {}
    text = " ".join(str(fields.get(k) or "") for k in ("setting", "action")).lower()
    named = [n for n in cast if n and n.lower().split()[0] in text][:3] or list(cast)[:1]
    out = []

    def ask(category: str, question: str, expect: str, major: bool = True) -> None:
        if category in categories:
            out.append({"key": f"q{len(out) + 1}", "category": category, "text": question, "expect": expect, "major": major})

    for name in named:
        sid = continuity.subject_of(bible, name, "character", create=False)
        facts = continuity.current(bible, index, sid) if sid else []
        look = clip(cast.get(name), 16)
        # the user's cast (a seed sheet) is held hard; someone the Director brought in (a thief, a guard) is a note:
        # on the pace and wardrobe run (2026-10-26) "is the first thief the same person" redrew three keyframes and
        # rendered two pieces again
        hero = any(f["key"] == "look" and f["source"] == "seed" for f in facts)

        def ask_about(category: str, question: str, expect: str, major: bool = True) -> None:
            ask(category, question, expect, major and hero)
        ask_about("identity", f"Where {name} ({look}) is visible, is it clearly the same person in every frame"
                              + (" and the same person as in the reference picture" if sid and (bible["subjects"].get(sid) or {}).get("ref") else "") + "?", "yes")
        # a double: generated crowds copy the hero (the gate's 4-piece film, 2026-10-26: a second braided man in the
        # same grey tunic beside Eirik, unseen by every check of that run)
        ask_about("identity", f"Is there a second person who looks like {name}: the same hair, beard and clothes?", "no")
        wear = next((f["value"] for f in facts if f["category"] == "wardrobe"), "")
        if wear:
            ask_about("wardrobe", f"Where {name} is visible, does {name} wear {clip(wear, 30)}?", "yes")
        elif look:
            # no wardrobe fact yet: the cast sheet's look (on the gate's Infinity run the brown wool coat became a
            # waistcoat and nothing asked about it)
            ask_about("wardrobe", f"Where {name} is visible, is {name} dressed as described here: {clip(cast.get(name), 30)}?", "yes")
        held = next((f["value"] for f in facts if f["category"] == "props" and f["key"] == "held"), "")
        if held:
            ask_about("props", f"Where {name} is visible, is {name} holding {clip(held, 10)}?", "yes", major=False)
    ask("identity", "Does any person change into a different person, melt, or morph between these frames?", "no")
    ask("identity", "Is any face, hand or body clearly malformed, duplicated or floating?", "no", major=False)
    if fields.get("setting"):
        ask("setting", f"Is the place in these frames this one: {clip(fields['setting'], 18)}?", "yes")
    era = continuity.era_at(bible, index)
    if era.get("label") or era.get("rules"):
        when = era.get("label") or "; ".join(era["rules"][:2])
        ask("era", f"Is there anything clearly out of place for {clip(when, 14)}: a modern vehicle, machine, electric light, "
                   "modern clothing or a sign with text?", "no")
    light = next((f["value"] for f in continuity.current(bible, index, "_scene") if f["category"] == "lighting"), "")
    if light and seg.get("mode") == "previous":
        ask("lighting", f"Do the light, time of day and weather look like this: {clip(light, 12)}?", "yes", major=False)
    return out


def pace_question(expect_slow: bool) -> dict:
    return {"key": "pace", "category": "pace", "text": "Is the motion in these frames slow motion: people and things moving "
            "clearly slower than real life?", "expect": "yes" if expect_slow else "no", "major": False}


# a fight's keyframe: a still pose makes the video model ease in slowly (the merge gate's openings, 3 to 5 a second)
MID_MOTION = {"key": "m1", "category": "pace", "text": "Is someone caught in the middle of a fast movement, a blow or a swing "
              "already in motion, rather than standing still or squaring up?", "expect": "yes", "major": True}


def join_questions(categories: list) -> list[dict]:
    """A continuation's join: picture 1 is the last frame of the piece before, picture 2 this one's first."""

    out = []
    if "identity" in categories or "wardrobe" in categories:
        out.append({"key": "j1", "category": "identity" if "identity" in categories else "wardrobe",
                    "text": "Is the main person in the second picture the same person, in the same clothes, as in the first?", "expect": "yes", "major": True})
    if "location" in categories or "setting" in categories:
        out.append({"key": "j2", "category": "location" if "location" in categories else "setting",
                    "text": "Is the second picture the same place as the first, in the same light?", "expect": "yes", "major": True})
    return out


def prompt(header: str, asked: list[dict]) -> str:
    listed = "\n".join(f'"{q["key"]}": {q["text"]}' for q in asked)
    return (f"YES/NO CHECK. {header}\nAnswer each question with \"yes\" or \"no\", only from what the pictures show.\n{listed}\n\n"
            "Answer with one flat JSON object mapping each key to \"yes\" or \"no\", and nothing else.")


# ---------- the answers ----------

def flat(data, out=None) -> dict:
    """Nested answers flattened ({"q1": {"answer": "yes"}} and {"same": {"copper_red": "yes"}} alike)."""

    out = {} if out is None else out
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                inner = value.get("answer", value.get("value"))
                if isinstance(inner, (str, bool)):
                    out[str(key).strip().lower()] = inner
                else:
                    flat(value, out)
            elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                for v in value:
                    flat(v, out)
            else:
                out[str(key).strip().lower()] = value
    return out


def word(value) -> str | None:
    """"yes", "no", or None for anything else (unclear, missing, a sentence that says neither first)."""

    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value or "").strip().lower()
    if re.match(r"^(yes|y|true|1)\b", text):
        return "yes"
    if re.match(r"^(no|n|false|0)\b", text):
        return "no"
    return None


def read(text: str, keys: list[str], extract) -> dict:
    """{key: "yes" | "no" | None} from an answer; a reply with no JSON reads as nothing answered."""

    try:
        data = flat(extract(text))
    except ValueError:
        data = {}
    return {k: word(data.get(k)) for k in keys}


def answered(found: dict) -> float:
    return sum(1 for v in found.values() if v) / max(1, len(found))


def ask_window(asker, pictures: list, header: str, asked: list[dict], extract, log: list, where: dict) -> dict:
    """One window's questions in one request (again once when too few come back): {key: yes|no|None}."""

    keys = [q["key"] for q in asked]
    found, tries = {k: None for k in keys}, 0
    started = time.time()
    for _ in range(2):
        tries += 1
        try:
            said = read(asker(SYSTEM, pictures, prompt(header, asked)), keys, extract)
        except Exception as error:  # noqa: BLE001 - an eye that does not answer leaves the questions unanswered
            log.append({**where, "error": str(error)[:200], "seconds": round(time.time() - started, 2)})
            break
        found = {k: found[k] or said[k] for k in keys}
        if answered(found) >= ANSWERED:
            break
    log.append({**where, "requests": tries, "questions": [{"key": q["key"], "category": q["category"], "text": q["text"], "expect": q["expect"],
                                                          "answer": found[q["key"]]} for q in asked],
                "seconds": round(time.time() - started, 2)})
    return found


def vote(asker, pictures: list, header: str, q: dict, first: str | None, votes: int, extract, log: list, where: dict) -> str | None:
    """A failed question asked again on its own until ``votes`` answers in all: the majority of yes/no, None on a tie."""

    tally = [first] if first else []
    started = time.time()
    for _ in range(max(0, votes - 1)):
        try:
            tally.append(read(asker(SYSTEM, pictures, prompt(header, [q])), [q["key"]], extract)[q["key"]])
        except Exception:  # noqa: BLE001
            tally.append(None)
    yes, no = tally.count("yes"), tally.count("no")
    result = "yes" if yes > no else "no" if no > yes else None
    log.append({**where, "vote": q["key"], "category": q["category"], "text": q["text"], "expect": q["expect"], "tally": tally,
                "answer": result, "requests": max(0, votes - 1), "seconds": round(time.time() - started, 2)})
    return result


# ---------- the pace, measured ----------

# Mean absolute difference between consecutive frames, 0-255, at 64x36 grey, averaged over a second (camera and
# subject motion together). It scales with speed: the same clip made 0.5x slow motion (motion-interpolated) measures
# 0.31 to 0.58 of itself, 0.49 at the median (51 clips, 433 s, measured 2026-10-26 on a DGX Spark). On the 17 action
# clips among them (fights, a sprint, dancers, the Viking v2 pieces, the merge gate's films), SLOW at 6 calls 87 % of
# the seconds of their 0.5x copies slow and 40 % of their own seconds (most of which the owner had found too slow:
# the Viking v2 pieces, the openings of the gate's fights at 2.0 to 5.7); 8 caught 95 % but called 65 % slow, 5 caught
# 77 % and 31 %. Calm clips sit at 0.4 to 3: a calm piece is never measured against it.
SLOW = 6.0
OPENING_S = 2  # a fight's first seconds that must not all be slow


def motion(clip_path, env=None, fps: int = 24) -> list[float]:
    """The motion of each second of a clip (see SLOW), from ffmpeg's grey 64x36 frames; [] when it cannot be read."""

    w, h = 64, 36
    try:
        done = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip_path), "-vf", f"fps={fps},scale={w}:{h},format=gray",
                               "-f", "rawvideo", "-"], capture_output=True, env=env, timeout=300)
    except (subprocess.SubprocessError, OSError):
        return []
    raw, size = done.stdout, w * h
    frames = [raw[i:i + size] for i in range(0, len(raw) - size + 1, size)]
    if len(frames) < 2:
        return []
    diffs = []
    for a, b in zip(frames, frames[1:]):
        diffs.append(sum(abs(x - y) for x, y in zip(a, b)) / size)
    out = []
    for s in range(0, len(diffs), fps):
        part = diffs[s:s + fps]
        if len(part) >= fps // 2 or not out:
            out.append(round(sum(part) / len(part), 3))
    return out


def pace_verdict(pace: dict, seconds_motion: list[float], slow_answers: list[tuple[float, str | None]], length: float) -> dict:
    """Whether a piece came out at the pace it was planned: {"status": ok|major|not_planned|unknown, "evidence", ...}.
    A fight or action piece fails (major) when the seconds the plan has fast are slow by measurement in
    most of them, and the eyes, where they answered, said slow motion in most of their fast windows. Without an answer
    from the eyes the measurement alone decides, with a stricter share."""

    if pace["kind"] not in FAST_KINDS:
        return {"status": "not_planned", "kind": pace["kind"]}
    if not seconds_motion:
        return {"status": "unknown", "kind": pace["kind"], "evidence": "the clip's motion could not be measured"}
    fast = [s for s in range(len(seconds_motion)) if planned(pace, s + 0.5, length) != "slow"]
    slow_fast = [s for s in fast if seconds_motion[s] < SLOW]
    share = len(slow_fast) / max(1, len(fast))
    eyes = [a for t, a in slow_answers if a and planned(pace, t, length) != "slow"]
    eyes_slow = eyes.count("yes") / len(eyes) if eyes else None
    failed = share >= 0.85 if eyes_slow is None else (share >= 0.6 and eyes_slow > 0.5)
    # the opening: the first OPENING_S seconds, planned fast, all measuring slow (an ease-in from a still frame)
    head = [s for s in range(min(OPENING_S, len(seconds_motion))) if planned(pace, s + 0.5, length) == "fast"]
    slow_open = bool(head) and len(head) == min(OPENING_S, len(seconds_motion)) and all(seconds_motion[s] < SLOW for s in head)
    out = {"kind": pace["kind"], "motion": seconds_motion, "slow_share": round(share, 2), "eyes_slow": None if eyes_slow is None else round(eyes_slow, 2),
           "threshold": SLOW, "opening": [seconds_motion[s] for s in head]}
    if failed:
        out.update(status="major", evidence=(f"planned fast, it comes out slow: {len(slow_fast)} of {len(fast)} fast seconds measure under "
                                             f"{SLOW:g}" + (f", and the eyes saw slow motion in {round(100 * eyes_slow)} % of its fast windows" if eyes_slow is not None else "")))
    elif slow_open:
        # the film cuts a slow lead that a fast second follows (autopilot.ease_in): then it is a note for the Director,
        # not a retake (on the pace and wardrobe run the retakes for it cost a render each and fixed one in three)
        rescued = any(m >= SLOW for m in seconds_motion[len(head):])
        out.update(status="minor" if rescued else "major", evidence=(f"the fight opens slowly: its first {len(head)} seconds, planned fast, measure "
                                             + ", ".join(f"{seconds_motion[s]:.1f}" for s in head) + f" (under {SLOW:g}); start it already in motion"))
    else:
        out["status"] = "ok"
    return out


def summary(log: list) -> dict:
    """How many checks ran and what they took: requests, questions, failed answers, seconds."""

    requests = sum(int(e.get("requests") or 0) for e in log)
    questions = sum(len(e.get("questions") or []) for e in log) + sum(1 for e in log if e.get("vote"))
    failed = sum(1 for e in log for q in e.get("questions") or [] if q.get("answer") and q["answer"] != q["expect"])
    return {"requests": requests, "questions": questions, "failed_answers": failed, "votes": sum(1 for e in log if e.get("vote")),
            "seconds": round(sum(float(e.get("seconds") or 0) for e in log), 1)}

