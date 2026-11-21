"""The film's arc, its place and the person's spoken steers: what AutoPilot keeps across pieces besides the scene graph
and the bible (gma/ap/autopilot.py reads and writes them; this file says what they are and how they are worded).

- **arc**: before the first piece the Director plans the whole film at once: a beat a piece, each with the state the
  piece must END on (what its last frame shows), and the payoff the last piece reaches. Each piece is asked to
  complete its own beat; a check asks the eyes whether the piece shows its end state, and a miss becomes a note: the
  next piece must catch up first. The last piece must reach the payoff (a miss there is a retake). On the Spark smoke
  of phase 3 (2026-10-27) three pieces of "a lighthouse keeper's evening" were all "climbing the stairs" and the lamp
  was never lit: the Director, asked one piece at a time, kept the story where it was. An Infinity film is planned an
  arc at a time (``ARC_SPAN`` pieces), and a film made longer by voice gets the rest of its arc planned again.
- **location**: the kept keyframe of the first piece of a place (the opening, or a piece the Director declares a new
  scene) is the place's reference picture: later keyframes of the same place are drawn from it (an edit with it as a
  reference, beside the cast's portraits), and the eyes are asked whether a keyframe and a clip show the same building
  or place as that picture. On the phase 3 smoke every piece drew a new lighthouse and the setting question passed
  it, because it compared the picture with the Director's own setting line.
- **steers**: what the person said during the film ("make it rain", "faster", "new scene: the harbour at dawn"), from
  the next piece on: a Director note, a bible fact, a question to the eyes for the piece it first applies to.

Nothing here asks anyone to approve anything.
"""

from __future__ import annotations

import re
import time

ARC_SPAN = 4  # pieces an Infinity film's arc plans at a time
STEERS = 12  # steers kept on a run
STEER_KINDS = ("note", "scene", "pace")

ARC_SYSTEM = """You plan a short film that is generated one segment at a time by a text-to-video model. Plan the \
WHOLE arc before anything is filmed: one beat a segment, in order. Each beat is one concrete, visible step that a \
camera can film in the segment's few seconds, and each ends on a state the next one starts from. Answer with JSON only."""

WEATHER = re.compile(r"\b(rain\w*|snow\w*|storm\w*|fog\w*|mist\w*|wind\w*|thunder\w*|lightning|hail|sun\w*|night\w*|dawn|dusk|"
                     r"sunset|sunrise|dark\w*|bright\w*|light|lights|moon\w*|cloud\w*|weather)\b", re.I)


def clip(text, words: int) -> str:
    return " ".join(str(text or "").split()[:words])


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------- the arc ----------

def arc_prompt(seed_text: str, count: int, start: int = 1, story_so_far: str = "", payoff: str = "",
               steers: str = "") -> str:
    """What the Director reads to plan ``count`` beats from segment ``start`` on."""

    head = (f"{seed_text}\n\n" + (f"The story so far (segments 1 to {start - 1} are filmed):\n{story_so_far}\n\n" if story_so_far else "")
            + (f"{steers}\n" if steers else ""))
    ending = (f"Segment {start + count - 1} is the last one: it reaches the payoff." if not payoff else
              f"The earlier arc's payoff was: {payoff}. These segments carry the story on from there to a new payoff.")
    return (head + f"Plan segments {start} to {start + count - 1} ({count} segment{'s' if count != 1 else ''}). {ending}\n\n"
            "Rules:\n"
            "- Each beat is one concrete, physical step with a visible result, achievable in one short shot; no two beats "
            "restage the same action (three beats of climbing stairs is one beat).\n"
            "- The \"end_state\" of a beat is what its LAST frame shows: who is where, what has changed, in 25 words at most. "
            "It must be visibly different from the end state before it, and ONE frame from that segment's own camera must be "
            "able to show all of it: one place, one moment (not a lamp lit inside and the sea and cliffs outside at once).\n"
            "- The payoff is the brief's own promise, shown on screen (if the brief says the lamp is lit, the last end state "
            "shows the lamp lit). Get there by the last segment: pace the beats so that each moves a real step toward it.\n"
            "- Write only what a camera can record; no inner thoughts. Keep it suitable for a general audience.\n\n"
            'Answer: {"payoff": "<the payoff, 20 words max>", "beats": [{"segment": <number>, "beat": "<one line, 20 words max>", '
            '"end_state": "<25 words max>"}]}')


def arc_problem(data: dict, count: int) -> str:
    """Why an arc answer cannot be used ("" when it can)."""

    beats = data.get("beats")
    if not isinstance(beats, list) or len([b for b in beats if isinstance(b, dict)]) < count:
        return f'"beats" is a list of {count} objects, one a segment'
    for b in [b for b in beats if isinstance(b, dict)][:count]:
        if not str(b.get("beat") or "").strip() or not str(b.get("end_state") or "").strip():
            return 'every beat has "beat" and "end_state" (what its last frame shows)'
    if not str(data.get("payoff") or "").strip():
        return '"payoff" says what the film reaches at its end'
    ends = [str(b.get("end_state") or "").strip().lower() for b in beats if isinstance(b, dict)][:count]
    if len(set(ends)) < len(ends):
        return "two beats end on the same state: each beat must move the story one visible step on"
    return ""


def merge_arc(arc: dict | None, data: dict, start: int, count: int) -> dict:
    """The new beats (segments ``start`` .. ``start + count - 1``) into the run's arc."""

    arc = dict(arc or {"beats": []})
    kept = [b for b in arc.get("beats") or [] if b["index"] < start]
    new = [{"index": start + i, "beat": clip(b.get("beat"), 24), "end_state": clip(b.get("end_state"), 30)}
           for i, b in enumerate([b for b in data["beats"] if isinstance(b, dict)][:count])]
    arc["beats"] = kept + new
    arc["payoff"] = clip(data.get("payoff"), 24)
    arc["through"] = start + count - 1
    arc.setdefault("planned", []).append({"at": now(), "from": start, "to": start + count - 1})
    return arc


def beat_at(run: dict, index: int) -> dict | None:
    return next((b for b in (run.get("arc") or {}).get("beats") or [] if b["index"] == index), None)


def final_index(run: dict) -> int | None:
    """The last segment of a film of fixed length (None in Infinity: it has no last segment)."""

    s = run["settings"]
    if s.get("infinity"):
        return None
    return int(s.get("segments") or 0) or None


def is_final(run: dict, index: int) -> bool:
    return final_index(run) == index


def arc_needed(run: dict, index: int) -> int:
    """How many beats to plan before segment ``index`` (0 when the arc already covers it)."""

    if not run["settings"].get("arc", True):
        return 0
    if beat_at(run, index):
        return 0
    last = final_index(run)
    if last is None:
        return ARC_SPAN
    return max(0, last - index + 1)


def arc_text(run: dict, index: int) -> str:
    """The arc for the Director writing segment ``index``: the whole plan, this segment's beat and the state it must end
    on, the payoff, and a catch-up when the piece before did not reach its end state."""

    arc = run.get("arc") or {}
    beats = arc.get("beats") or []
    mine = beat_at(run, index)
    if not mine:
        return ""
    listed = "\n".join(f"  {b['index']}. {b['beat']} -> ends on: {b['end_state']}" for b in beats)
    text = (f"\nTHE FILM'S ARC (planned before filming; follow it, one beat a segment):\n{listed}\n"
            f"Payoff: {arc.get('payoff')}\n"
            f"This is segment {index}: it must COMPLETE beat {index} (\"{mine['beat']}\") within this one segment and END on: "
            f"{mine['end_state']}. Write the action so that this end state is reached on screen before the segment ends.")
    if is_final(run, index):
        text += f" This is the LAST segment: it must reach the payoff on screen: {arc.get('payoff')}."
    missed = run.get("behind") or []
    if missed:
        text += ("\nCATCH UP: the film is behind its arc. " + " ".join(
            f"Segment {m['index']} did not reach its end state ({m['end_state']})." for m in missed[-2:])
                 + " This segment must get there first, quickly, then complete its own beat.")
    return text + "\n"


def end_question(end_state: str, final: bool) -> dict:
    """The yes/no question a piece's last frames are asked: does it show the state its beat ends on."""

    return {"key": "s1", "category": "story", "text": f"By the end of these frames, is this shown: {clip(end_state, 30)}?",
            "expect": "yes", "major": True, "final": final}


# ---------- the place ----------

def anchor_of(run: dict, index: int) -> dict | None:
    """The reference picture of the place segment ``index`` is in: the anchor set by the last piece that opened a place
    (before ``index``), or None (a new place, or none yet)."""

    found = None
    for one in run.get("locations") or []:
        if one["from"] < index:
            found = one
    if found is None:
        return None
    seg = next((s for s in run.get("segments") or [] if s.get("index") == index), None)
    if seg and seg.get("scene_change") and index > found["from"]:
        return None  # the Director moved the story elsewhere: this piece opens a new place
    return found


def set_anchor(run: dict, seg: dict, picture: str) -> dict | None:
    """The kept keyframe of a piece that opens a place becomes the place's reference picture. Only the first piece of
    a place sets it (the opening, or a scene change); later pieces of the place keep the first one."""

    if not picture:
        return None
    places = run.setdefault("locations", [])
    opens = seg["index"] == 1 or bool(seg.get("scene_change")) or not places
    same = next((p for p in places if p["from"] == seg["index"]), None)
    if same is not None:  # the piece that opened the place was drawn again: its new keyframe is the place's picture
        same.update(image=picture, at=now())
        return same
    if not opens:
        return None
    one = {"from": seg["index"], "image": picture, "scene": clip(seg.get("scene"), 8),
           "setting": clip((seg.get("fields") or {}).get("setting"), 30), "at": now()}
    places.append(one)
    return one


LOCATION_TEXT = ("Is this the same building or place as in the reference picture of the place: the same structure, shape, "
                 "materials, colours and surroundings? A different camera angle, a closer view, or the inside of that same "
                 "building still counts as the same place.")


def location_question() -> dict:
    return {"key": "p1", "category": "location", "text": LOCATION_TEXT, "expect": "yes", "major": True}


def place_line(anchor: dict | None) -> str:
    if not anchor:
        return ""
    return (f"\nTHE PLACE IS FIXED: this segment is in the same place as segment {anchor['from']} ({anchor['scene']}: "
            f"{anchor['setting']}). Its keyframe is drawn from that segment's picture. Keep the same building and surroundings; "
            "only a segment with \"scene_change\": true moves to another place.\n")


# ---------- steers ----------

def add_steer(run: dict, text: str, kind: str = "note", at: int | None = None) -> dict:
    """A spoken steer, applied from the next segment that has not started rendering."""

    steer = {"text": clip(text, 40), "kind": kind if kind in STEER_KINDS else "note", "from": int(at or 1), "at": now(),
             "applied": False}
    run.setdefault("steers", []).append(steer)
    del run["steers"][:-STEERS]
    return steer


def steers_for(run: dict, index: int) -> list[dict]:
    return [s for s in run.get("steers") or [] if s["from"] <= index]


def steer_text(run: dict, index: int) -> str:
    """The person's steers for the Director: every one so far (they stay), the new ones said loudly."""

    out = []
    for s in steers_for(run, index)[-6:]:
        new = s["from"] == index
        if s["kind"] == "pace":
            out.append(("NEW: " if new else "") + f"the person asked for more pace ({s['text']}): faster, more physical action, "
                       "real-time speed, a quicker camera; pace kind \"action\" where the story allows.")
        elif s["kind"] == "scene":
            out.append(("NEW: " if new else "") + f"the person asked for a new scene: {s['text']}."
                       + (" This segment opens it: set \"scene_change\" to true and go there." if new else ""))
        else:
            out.append(("NEW: " if new else "") + f"the person asked: \"{s['text']}\"."
                       + (" Make it clearly visible in this segment, and keep it from now on." if new else " Keep it."))
    if not out:
        return ""
    return "\nTHE PERSON WATCHING SPOKE (their words steer the film from the segment they apply to; never ignore them):\n- " + "\n- ".join(out) + "\n"


def scene_steer(run: dict, index: int) -> dict | None:
    return next((s for s in run.get("steers") or [] if s["kind"] == "scene" and s["from"] == index), None)


def steer_question(steer: dict) -> dict:
    """The yes/no question for the piece a "note" steer first applies to."""

    return {"key": "u1", "category": "story", "text": f"Is this clearly visible in these frames: {clip(steer['text'], 20)}?",
            "expect": "yes", "major": True, "steer": True}


def steer_fact(steer: dict) -> tuple[str, str]:
    """(category, key) of the bible fact a steer becomes: the weather or the light, else a story fact."""

    if WEATHER.search(steer["text"]):
        return "lighting", "steer"
    return "story", "steer"
