# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, app/autopilot.py at 6cc6e67c5e138499ea2c17183eda261b710d1217 (with extract_json and mend_json from
# app/slate.py and the line check of app/qa.py's audio gate, same commit). Apache-2.0:
# LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE. Its scene graph and the
# Director's rules are in turn adapted from Video Rebirth's HyperFlow-Infinity (Apache-2.0,
# LICENSES/HyperFlow-Infinity-LICENSE); the idea is Echo-Infinity's.
# Changed: decoupled from the Studio's server, Slate and job queue. The render queue (submit / job_of / cancel) is the
# headless Studio's queue again, through gma/ap/service.py and gma/studio.py (the governor around each job); the Director is the app's brain (Bolt) on its Director lane with a
# bounded thinking_budget (``director``); the eyes are gma/ap/eyes.py (Gemma 4 12B, or Cosmos3-Edge) instead of
# cosmos.py and the QA gate's vision path; the QA gate's line check hears with Parakeet (Whisper only where Parakeet is
# not installed, gma/ap/ears.py) and its single identity verdict is left to the yes/no checks; run files live in the
# video panel's run folder (<video dir>/<run id>/) inside its run.json shell. Left out: the HTTP routes (the panel's
# own routes drive a run), the Slate seed, the folder migration, Ref2VA references (no Ref2VA on the local renderer),
# the Director on the vision server, and fight sequences (fight.py: no fight scenes in V1).
"""AutoPilot: open-ended, self-directing long-form video, inside the video panel.

A story seed (a world, where it starts, the characters with their looks, a visual style and the user's direction)
becomes a film that keeps going until it is done or stopped. Each turn of the loop:

  plan      the Director (the app's brain, Nemotron 3.5 Bolt, reasoning first within ``think_tokens``) reads the scene
            graph and the last segment's ending and writes the next segment as MiniMax H3's prompt fields; an answer
            that restages an earlier beat, repeats the segment before, keeps a scene past its budget, or names what
            must not be seen is turned back with the reason and asked again (``MAX_ATTEMPTS``)
  keyframe  a still a piece (on by default here: a keyframe per piece rather than last-frame chaining), checked by
            the eyes with yes/no questions before anything moves; a confirmed major problem draws it again
  render    this app's local renderer (FastH3 on this machine), the memory governor around it; the eyes are not
            resident during a render on a Spark (the governor's schedule evicts them; they load between renders)
  check     the line check (Parakeet), the watcher (the scene graph's delta, the state the segment ends in), the
            continuity checklist against the bible, and the dense yes/no checks in windows, voted on and looked at again
            (checks.py); a confirmed major problem renders the segment again at the next seed (``retakes``, two by
            default) with the Director rewriting the shot from the findings; a minor one becomes a Director note
  commit    the delta is merged into the scene graph; the kept take's check is written into the bible; a checkpoint
            (checkpoint.json, checkpoints/S0001.json) is written after every piece
  film      the film is joined again after every piece (``preview_every``, 1 here)

No step waits for anyone's approval. A restart (even a kill -9 mid-render) resumes from the checkpoint: finished
pieces are never rendered again.
"""

from __future__ import annotations

import base64
import copy
import difflib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gm_core import h3_prompt, lengths, negation
from gma import memguard  # the thermal guard's last stop (a clip it stopped is made again once)

from . import big_picture  # the film in passes: a draft, one consolidated fix plan, made again from its root, kept only if better
from . import checks  # the dense yes/no checks, decided in code, and the pace measured from the pixels
from . import continuity  # the continuity bible and what its checks mean
from . import judge as judge_mod  # d1-2B, the clip judge
from . import qc  # the self-check: start and end states, direction, contact and hard cuts, from what the frames show
from . import story  # the arc planned up front, the place's reference picture, the person's spoken steers

HOME: Path = Path()
FPS = 24
TICK = 1.0
LOG = 200
MAX_ATTEMPTS = 3  # a Director or watcher answer that does not hold up is asked again, this many tries in all
BRAIN_RETRY_S = 3.0  # a brain that failed to answer is asked once more after this long
STATE_WORDS, EVENTS_PER_SEGMENT = 12, 2
COMPACT_AT, FOLD = 22, 10  # above this many events the oldest FOLD become one chapter
RECENT_EVENTS = 8
SIMILAR_ACTION, SIMILAR_BEAT = 0.9, 0.85  # difflib ratios above which a new segment repeats an old one
LIMIT = 500  # segments a run may have when it is asked for a number (0 runs until it is stopped)
ACTIVE = ("running",)
CONTINUE_FADE, CUT_FADE = 1 / FPS, 0.25  # the joins of the film: the repeated frame dropped, and a dissolve on a new scene
FRAME = (448, 256)  # the eyes' frame size (cosmos.py's FRAME)
EYE_FRAMES = 7  # frames of a take the watcher and the checklist see, besides its last (cosmos.settings' default)
ENV = {**os.environ, "PATH": os.pathsep.join([os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])}

# the defaults hold the lessons of the first long run (the 60 s Viking film): fewer and longer pieces (10 s), a
# keyframe a piece, the film joined after every piece. The size is the local renderer's (864x480 live on a Spark).
DEFAULTS = {"style": "", "guidance": "", "preset": "small", "quality": "fh8", "width": None, "height": None, "seconds": 10,
            "loras": [], "segments": 3, "seed": 1, "max_scene_segments": 8, "beat_history": 12, "qa": True, "retakes": 2,
            "preview_every": 1, "styles": [], "keyframes": "on", "narration": False, "score": False, "score_db": -18.0, "identity": "auto", "ref_turbo": False, "music": "", "director": "brain",
            # every shot names a camera move, the light and colour, a music cue, its pace and a close detail where the
            # contact or the change happens (what Qwen3.8 writes unasked; a Director that writes plain, static shots needs it)
            "rich_shots": False,
            # multipass (big_picture.py's design, here): the whole film drafted with no per-piece retakes, reviewed as a
            # whole, one consolidated fix plan by the Director, then made again from the earliest piece it touches,
            # forward; kept only when the whole-film score improves ("do no harm"); fix_budget: regeneration passes
            "big_picture": False, "fix_budget": 1,
            "continuity": copy.deepcopy(continuity.DEFAULT),
            # the whole arc planned before the first piece (a beat and an end state a piece, the payoff), and a place's
            # first keyframe as the reference picture of its later keyframes (story.py)
            "arc": True, "location_anchor": True,
            # the eyes' requests of one piece at once (the watcher, the checklist, the windows): vLLM batches them
            "eyes_parallel": 4,
            # the Director reasons before it answers (Bolt's thinking switch), within this many tokens of reasoning
            "think": True, "think_tokens": 4096,
            # the dense yes/no checks (checks.py)
            "checks": copy.deepcopy(checks.DEFAULT),
            # Infinity: no fixed length; it ends on Stop, a time or length limit (0: none), or an ending the Director chooses.
            # Its film is joined and a checkpoint written after every piece, the last joins checked every film_every pieces
            "infinity": False, "max_minutes": 0, "max_film_seconds": 0, "film_every": 4, "film_window": 6, "story_every": 6,
            # memory first: a piece that does not fit is made shorter at the same size, never smaller; not under this
            "min_seconds": 4}
RETAKES = 4  # the most retakes a segment may be given
MEMORY_RETRIES = 3  # a piece the memory guard stopped is made again shorter, at the same size, this many times at most
INFINITY_RETRIES = 5  # an Infinity run that hits a problem waits and carries on, this many times in a row at most
INFINITY_WAIT_S = 60
ROOM_NOW = re.compile(r"keeps \d+ GB free as its floor|not enough memory|no room", re.I)  # a governor's "not now"
KEYFRAME_WAITS, KEYFRAME_WAIT_S = 3, 15
KEYFRAMES = ("auto", "on", "off")  # a picture a piece to open on: auto is on for a run with an outline
IDENTITY = ("auto", "frame", "references")  # how a new scene keeps who is who: Ref2VA pictures where installed, or words

# what the app hands in (attach); tests stand in for them
submit = lambda body: (_ for _ in ()).throw(RuntimeError("the renderer is not attached"))
job_of = lambda job_id: None
cancel = lambda job_id: None
director = lambda system, prompt, think: (_ for _ in ()).throw(RuntimeError("no Director brain is attached"))
eyes = None  # gma.ap.eyes.Eyes: look(system, frames, sheet, text) -> (answer, who)
judge = None  # gma.ap.judge.D1Judge: d1-2B judges every take (the default); None: the eyes' checks judge them
qc_eyes = None  # gma.ap.eyes.Eyes the self-check looks with (Gemma 4 12B), beside d1-2B's judging
director_warm = lambda: None  # loads the Director's brain ahead of its next question (a Spark evicts it for keyframes)
ears = None  # gma.ap.ears.Ears: ready() and line_check(clip, prompt)
ref2va = lambda: False  # no Ref2VA on the local renderer: a new scene opens on its keyframe or from text
ref2va_turbo = lambda: False
compose = h3_prompt.compose  # MiniMax H3's prompt form (the Studio's server.compose)


def compose_references(fields: dict, count: int) -> str:
    raise Failed("reference pictures to video (Ref2VA) are not on this app's renderer")
room = lambda params: None  # {"fits", "nearest": {"frames"}} for a piece's params, or None: no check here
on_change = lambda run: None  # every save: the panel's shell is written and its page told
finish = None  # the film's sound pass when the last piece is kept (gma/ap/service.py: narration and score, by the Studio)

runs: dict[str, dict] = {}
guard = threading.RLock()
starting = threading.Lock()


class Stopped(Exception):
    pass


class Failed(Exception):
    pass


class Bad(ValueError):
    """A seed or settings that cannot be used (the Studio's HTTP 400)."""


def attach(home: Path, **hooks) -> None:
    """The run folder's root and the app's pieces: submit, job_of, cancel, director, eyes, ears, room, on_change."""

    global HOME
    HOME = Path(home)
    HOME.mkdir(parents=True, exist_ok=True)
    for name, value in hooks.items():
        if name not in ("submit", "job_of", "cancel", "director", "director_warm", "eyes", "qc_eyes", "judge", "ears", "room", "on_change", "shell",
                        "finish") or value is None:
            continue
        globals()[name] = value


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def duration(clip: Path) -> float:
    done = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                          capture_output=True, text=True, env=ENV, timeout=60)
    return float(done.stdout.strip() or 0)


def extract_json(text: str):
    """The first complete JSON value in a reply that may carry fences or a preamble (slate.extract_json)."""

    cleaned = re.sub(r"```(?:json)?", "", text or "")
    cleaned = re.sub(r"<think>.*?</think>|^.*?</think>", "", cleaned, flags=re.S).strip()  # a reasoning model's thoughts
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth, quoted, escaped = 0, False, False
        for at in range(start, len(cleaned)):
            ch = cleaned[at]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = not quoted
            elif not quoted and ch == opener:
                depth += 1
            elif not quoted and ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:at + 1])
                    except ValueError:
                        mended = mend_json(cleaned[start:at + 1])
                        if mended is not None:
                            return mended
                        break
    raise ValueError("no valid JSON in the reply")


def mend_json(text: str):
    """The slips small models make in JSON, put right (trailing commas, Python's True/False/None, single quotes,
    // comments); None when it still does not parse."""

    fixed = re.sub(r"(?m)//[^\n\"]*$", "", text)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", re.sub(r"\bNone\b", "null", fixed)))
    for attempt in (fixed, re.sub(r"'([^'\n]*)'", r'"\1"', fixed)):
        try:
            return json.loads(attempt)
        except ValueError:
            continue
    return None

# ---------- the scene graph (adapted from HyperFlow-Infinity's memory.py, Apache-2.0) ----------

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AutoPilot scene graph",
    "description": "What the film has shown so far: who and what persists (nodes), what changed the story (events), and "
                   "older events folded into chapters. Written only from what the watcher saw, never from what was asked for.",
    "type": "object",
    "required": ["nodes", "events", "chapters"],
    "properties": {
        "nodes": {"type": "array", "items": {"type": "object", "required": ["id", "type", "label", "state"], "properties": {
            "id": {"type": "string", "description": "short unique id, reused for the same person, place or thing"},
            "type": {"enum": ["character", "place", "object"]},
            "label": {"type": "string", "description": "a name, three words at most"},
            "state": {"type": "string", "description": "how it is now, twelve words at most"}}}},
        "events": {"type": "array", "items": {"type": "object", "required": ["t", "subject", "action"], "properties": {
            "t": {"type": "string", "description": "MM:SS on the film's clock"},
            "subject": {"type": "string"}, "action": {"type": "string", "description": "a verb phrase, eight words at most"},
            "target": {"type": "string"}, "place": {"type": "string"}, "segment": {"type": "integer"}}}},
        "chapters": {"type": "array", "items": {"type": "object", "required": ["id", "title", "summary"], "properties": {
            "id": {"type": "string"}, "t_start": {"type": "string"}, "t_end": {"type": "string"},
            "title": {"type": "string"}, "summary": {"type": "string"},
            "carried_forward": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}}},
    },
}
EMPTY = {"graph": {"nodes": [], "events": [], "chapters": []}, "archive": [], "observed_through": 0}


def clip_words(text, limit: int) -> str:
    return " ".join(str(text or "").split()[:limit])


def mmss(seconds: float) -> str:
    value = int(round(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def resolve_id(new_id, known) -> str:
    """A proposed id mapped onto an existing node when it is plainly the same thing (``guard`` / ``guard_2``,
    ``ada`` / ``ada_older``, near spellings): vision models do not reliably reuse ids."""

    if not new_id or new_id in known:
        return new_id
    for old in known:
        if str(new_id).startswith(f"{old}_") or str(old).startswith(f"{new_id}_"):
            return old
    match = difflib.get_close_matches(str(new_id), [str(k) for k in known], n=1, cutoff=0.82)
    return match[0] if match else new_id


def merge(graph: dict, delta: dict, segment: int = 0) -> dict:
    """One watcher delta added to the graph: states replace, at most two events a segment, ids folded."""

    graph = copy.deepcopy(graph or EMPTY["graph"])
    graph.setdefault("chapters", [])
    by_id = {n.get("id"): n for n in graph["nodes"]}
    remap = {}
    for node in delta.get("add_nodes") or []:
        if not isinstance(node, dict) or not node.get("id"):
            continue
        raw = str(node["id"])
        target = resolve_id(raw, by_id.keys())
        remap[raw] = target
        state = clip_words(node.get("state"), STATE_WORDS)
        if target in by_id:
            if state:
                by_id[target]["state"] = state
        else:
            kind = node.get("type") if node.get("type") in ("character", "place", "object") else "character"
            graph["nodes"].append({"id": target, "type": kind, "label": clip_words(node.get("label"), 3) or target, "state": state})
            by_id[target] = graph["nodes"][-1]
    for change in delta.get("update_nodes") or []:
        if not isinstance(change, dict):
            continue
        node = by_id.get(remap.get(change.get("id"), resolve_id(change.get("id"), by_id.keys())))
        if node is not None and change.get("state"):
            node["state"] = clip_words(change["state"], STATE_WORDS)
    for event in [e for e in delta.get("add_events") or [] if isinstance(e, dict)][:EVENTS_PER_SEGMENT]:
        event = {k: event.get(k) for k in ("t", "subject", "action", "target", "place") if event.get(k)}
        for key in ("subject", "target", "place"):
            if event.get(key):
                event[key] = remap.get(event[key], resolve_id(event[key], by_id.keys()))
        event["action"] = clip_words(event.get("action"), 8)
        if segment:
            event["segment"] = segment
        if event["action"]:
            graph["events"].append(event)
    return graph


def fold(graph: dict, chapter: dict, folded: list) -> list:
    """The ``folded`` oldest events replaced by one chapter; the removed events are returned for the archive."""

    graph.setdefault("chapters", []).append({
        "id": f"ch{len(graph['chapters']) + 1}", "t_start": folded[0].get("t", ""), "t_end": folded[-1].get("t", ""),
        "title": clip_words(chapter.get("title"), 5), "summary": clip_words(chapter.get("summary"), 30),
        "carried_forward": [clip_words(x, 10) for x in (chapter.get("carried_forward") or [])[:3]]})
    graph["events"] = graph["events"][len(folded):]
    return folded


def as_text(graph: dict, recent: int = RECENT_EVENTS) -> str:
    """A bounded view for the Director: chapters, the cast as they are now, places and things, the last events.
    Bounded on purpose: a view that grew with the film grew the prompts with it."""

    graph, lines = graph or {}, []
    for ch in graph.get("chapters") or []:
        carried = "; ".join(ch.get("carried_forward") or [])
        lines.append(f"[{ch.get('t_start')}-{ch.get('t_end')}] {ch.get('title')}: {ch.get('summary')}" + (f" (still matters: {carried})" if carried else ""))
    nodes = graph.get("nodes") or []
    cast = [n for n in nodes if (n.get("type") or "character") == "character"]
    if cast:
        lines.append("Cast as they now are: " + "; ".join(f"{n.get('label')} -- {n.get('state')}" for n in cast))
    world = [n for n in nodes if (n.get("type") or "character") != "character"]
    if world:
        lines.append("Places and things in play: " + ", ".join(f"{n.get('label')} ({n.get('state')})" if n.get("state") else str(n.get("label")) for n in world))
    tail = (graph.get("events") or [])[-recent:]
    if tail:
        lines.append("Just happened: " + "; ".join(f"{e.get('t')} {e.get('subject')} {e.get('action')}" + (f" -> {e['target']}" if e.get("target") else "") for e in tail))
    return "\n".join(lines)


def scene_length(segments: list) -> int:
    """Segments since, and including, the last one that opened a scene."""

    n = 0
    for done in reversed(segments):
        n += 1
        if done.get("scene_change"):
            break
    return n


def signature(delta: dict):
    """A delta in a comparable form (timestamps left out); None when it carries no state to compare."""

    if not delta or delta.get("add_nodes") or not delta.get("add_events") or not delta.get("update_nodes"):
        return None
    norm = lambda value: " ".join(str(value or "").lower().split())
    events = sorted((norm(e.get("subject")), norm(e.get("action")), norm(e.get("target")), norm(e.get("place"))) for e in delta["add_events"] if isinstance(e, dict))
    states = sorted((norm(n.get("id")), norm(n.get("state"))) for n in delta["update_nodes"] if isinstance(n, dict))
    if any(not e[1] for e in events) or any(not s[1] for s in states):
        return None
    return [events, states]


def stagnation(segments: list):
    """Three segments in a row, in one scene, whose watcher saw the same events and states: the film is circling."""

    recent = segments[-3:]
    if len(recent) < 3 or len({s.get("scene") for s in recent}) != 1:
        return None
    marks = [signature((s.get("watch") or {}).get("delta")) for s in recent]
    if marks[0] is not None and all(m == marks[0] for m in marks):
        return {"segments": [s["index"] for s in recent], "reason": "three segments in a row show the same events and states"}
    return None


# ---------- the prompts (the Director's rules adapted from HyperFlow-Infinity's prompts.py, Apache-2.0) ----------

DIRECTOR_SYSTEM = """You are the director of a long-form film generated one short segment at a time by a \
text-to-video-and-audio model (MiniMax H3). Each segment is a few seconds long with its own synchronized sound.

Answer with a single JSON object and nothing else:
{
  "scene": "<short name of the scene this segment belongs to>",
  "scene_change": true | false,
  "beat": "<one short line: what this segment accomplishes in the story>",
  "new_characters": [{"name": "...", "look": "<concrete visual description: age, face, hair, clothing>"}],
  "era": "<empty, unless this segment opens the film or jumps to another time or era: then a short label like 'Boston, 1773'>",
  "era_rules": ["<with an era label only: what belongs to that era's material culture and what would be out of place, each 12 words max>"],
  "keyframe": "<only when the prompt asks for it: the still picture this segment opens on, 60 words max: who is where, the framing, the place, the light>",
  "shot": {
    "setting": "<the place, the time of day and the light, concrete and visual>",
    "camera": "<framing and camera movement>",
    "action": "<what happens, in temporal order: one legible, achievable action and its visible result>",
    "dialogue": [{"speaker": "<name>", "voice": "<how the voice sounds>", "text": "<a short spoken line>"}],
    "soundscape": "<the diegetic sound: ambience, impacts, movement>",
    "music": "<the score, or N/A>"
  },
  "pace": {"kind": "fight|action|chase|calm|dialogue", "beats": [{"from": <second>, "to": <second>, "speed": "fast|real|slow", "what": "<5 words>"}]},
  "qc": {"start": ["<object or person: its state at the FIRST frame, e.g. 'oil lamp: unlit, dark wick'>"],
         "end": ["<object or person: its state at the LAST frame, e.g. 'oil lamp: lit, flame inside the glass chimney'>"],
         "motion": "<who moves which way, relative to the camera and the place, e.g. 'Arthur climbs the stairs upward, away from the camera'; empty when nobody travels>",
         "wrong_motion": "<the opposite movement, the one that would be wrong, e.g. 'Arthur walks down the stairs toward the camera'; empty with no motion>",
         "contact": "<the one physical contact the action needs, e.g. 'the match flame touches the wick inside the glass chimney'; empty when none>",
         "start_place": "<where the person is at the first frame, inside or outside and which room, e.g. 'Arthur is inside the lantern room, the door behind him'>",
         "end_place": "<where the person is at the last frame, the same way>",
         "wrong_place": "<the place that would be wrong at the end, e.g. 'Arthur is outside on the gallery, beyond the door'>",
         "people": [{"who": "<a person in the shot, by look: 'an old man with short white hair in a brown wool sweater'>", "count": 1}]}
}

"scene_change" is false while the story stays in the same place with the same people: the segment then continues \
the previous one from its last frame, the action picking up exactly where it stopped, the same light and camera \
language. Set it to true when the story moves to another place or time: the segment then opens fresh. Prefer \
continuous scenes of several segments over rapid cutting, but move on once a scene's purpose is done.

Every segment moves the story one concrete, visible step. Restating the situation in other words is a wasted \
segment: "they are hunted", "they are cornered", "they hide", "they still hide" is one beat, not four. A new \
expression, another reaction, a camera change or restarting the same unfinished action is not a new step.

List a character in "new_characters" only the first time they appear, with a look concrete enough to repeat word \
for word forever after. Name characters by name in "action" and "setting".

Rules that matter:
- Keep it suitable for a general audience: nothing sexual, no graphic gore or lingering on injury; conflict and \
danger are shown without graphic detail. No real public figures. Children are never placed in harm.
- Write only what a camera and a microphone can record: no inner thoughts, backstory or commentary.
- Never ask for on-screen text, subtitles, logos or watermarks.
- Say what is there, never what is not: no "no X", "without X" or "X is absent". The video model has no negative \
prompt and what is named tends to appear.
- Dialogue is optional and short: about two words a second of the segment at most. Leave "dialogue" empty for a \
segment without speech.
- Continue from what the film actually showed (the watcher's report and the scene graph), not from what an earlier \
segment asked for. Previous beats are requests, not facts.
- The continuity bible is canonical: every person keeps their face, marks, wardrobe and its state and what they hold \
until the story changes it on screen; an action in progress continues or resolves; what has happened stays happened. \
Name only the details that matter in this shot (the musket in his right hand), and make every correction it asks \
for. A time-skip is a new scene with an "era" label.
- Write in English.
- One action a segment. Where the camera is and which way it looks is said in "camera", and the direction of any \
movement relative to it in "action" ("camera at the bottom of the stair looking up; he climbs away from it, upward"): \
one continuous shot, never a cut. An interaction names the exact point of contact ("the match flame touches the wick \
inside the glass chimney", "his hand closes on the door handle"). "qc" repeats these as checks: the state of the \
things that change at the first and at the last frame, the movement (and its wrong opposite) and the contact, each a \
short plain statement a single picture can show; each is checked on the frames. "start" and "end" hold one or two \
statements each, every one about ONE thing the action changes (the lamp, the door, where the person stands), lasting \
and visible in a still (never weather, lightning, sound or anything brief); "end" is what this shot's own action \
achieves. "motion" is one straight movement ("Elias climbs the stairs upward, away from the camera"), empty when \
nobody travels. Through a door, "motion" says which way ("Arthur steps in through the door, into the lantern \
room") and "wrong_motion" the other ("Arthur walks out through the door onto the gallery"). "people" lists each \
person in the shot once: they appear once, never twice.
- "scene" names the place the segment is in now: when the story moves into another room or place, give it that \
place's name.
- Pace. A fight, a chase or an action segment is mostly FAST and in real time: quick strikes, rapid exchanges, \
sharp impacts, a snappy camera (whip pans, quick handheld reframes). Slow motion is kept for one or two key moves \
only (the decisive blow, a signature technique), a third of the segment at most, and its timing is written into \
"action", like "0-6 s rapid real-time exchange of punches and blocks; 6-8 s slow motion on the centerline punch; \
8-10 s back to full speed as he drives forward". "pace.beats" lists those seconds with their speed. Outside the key \
move neither "action" nor "camera" says slow or slowly. A fight "action" opens with the strike already under way (its \
first words are the movement: "Eirik's fist is already driving into ..."), never with a stand-off or a look, and a fight \
segment ends in the middle of a movement, never a still pose: the next one starts from that frame. A calm or dialogue \
segment moves at natural real-time speed; its "beats" may be empty.

You run the film on your own: nobody approves a segment. You choose the story's next step and the camera for every \
segment yourself (framing, angle, movement: whatever shows the action best; there is no fixed camera mode). Favour \
action: each segment shows a physical action with a visible result. Keep the prompt short: "action" at most 40 words, \
"camera" at most 15, "setting" at most 30. The continuity bible is for you to keep things right, not to be copied out.

This is open-ended: there is no prescribed plot, act schedule, ending or total length. Decide the next short step \
from the evidence."""

WATCH_SYSTEM = ("You watch a film as it is generated, one short segment at a time, and keep its scene graph: you report only "
                "what each new segment changes, and anything that breaks continuity. Answer only with JSON.")

COMPACT_SYSTEM = "You compress a film's history into chapters. Answer only with JSON."


def seed_text(seed: dict, style: str) -> str:
    cast = "\n".join(f"- {c.get('name')}: {c.get('look') or 'as first seen'}" + (f" (segments {c.get('from') or 1} to {c.get('to') or 'the end'})" if c.get("from") or c.get("to") else "")
                     for c in seed.get("characters") or [] if c.get("name"))
    return "\n".join(bit for bit in (
        f"Inspiration (a starting point, not a plot to retell): {seed['inspiration']}" if seed.get("inspiration") else "",
        f"World: {seed.get('world') or 'open'}", f"Where the film starts: {seed.get('situation') or 'open'}",
        f"Characters:\n{cast}" if cast else "", f"Visual style: {style}" if style else "") if bit)


def outline_text(run: dict, index: int) -> str:
    """The user's ordered beats, when the seed has them: the whole list, and the one this segment carries."""

    outline = run["seed"].get("outline") or []
    if not outline:
        return ""
    listed = "\n".join(f"  {i}. {b['text']}" for i, b in enumerate(outline, start=1))
    beat = beat_of(run, index)
    if beat is None:
        return (f"\nThe user's outline ({len(outline)} beats) is done:\n{listed}\nFrom here on the film is open-ended: continue from "
                "where it ended.\n")
    return (f"\nThe user's outline, one beat a segment, in order:\n{listed}\nThis is segment {index}: it carries beat {index}"
            f" (\"{beat['text']}\"). Follow it, in your own concrete shot, continuing from what the film shows now."
            + (" It opens on a picture of its own, a new scene: set scene_change to true." if beat.get("image") else "") + "\n")


def direction(settings: dict, branch: dict | None, index: int) -> str:
    text = ""
    if settings.get("guidance"):
        text += f"\nThe user's creative direction: {settings['guidance']}\n"
    if branch and index > int(branch.get("after") or 0):
        text += (f"\nThe user asked for another way the story goes on after segment {branch['after']}:\n"
                 "<user_story_guidance>\n" + branch["guidance"] + "\n</user_story_guidance>\n"
                 "This is direction for what comes next, NOT something that already happened. Develop it from what the film "
                 "shows now; keep the shared past, who is who, what they hold and where they are. Get there through visible, "
                 "achievable actions: no teleporting, no restarting the opening, no pretending it already happened.\n")
    return text


def style_at(settings: dict, index: int) -> str:
    """The visual style of segment ``index``: the run's style (what every segment shares) and the scoped styles whose
    segments include it ({style, from, to}, like the cast sheets). On the 60 s Viking test film one style named the
    sea, the island and 1770s Boston at once, and Boston's streets came into the storm at sea: give each its span."""

    scoped = [str(x.get("style") or "").strip() for x in settings.get("styles") or []
              if isinstance(x, dict) and (x.get("from") or 1) <= index <= (x.get("to") or 10 ** 6)]
    return " ".join(bit for bit in (str(settings.get("style") or "").strip(), *scoped) if bit)


def first_prompt(run: dict, seconds: float) -> str:
    s = run["settings"]
    return (seed_text(run["seed"], style_at(s, 1)) + "\n" + outline_text(run, 1) + direction(s, run.get("branch"), 1)
            + f"\nWrite segment 1, about {seconds:.0f} seconds. It is the opening: establish the world, the look and the "
              "mood, and start the situation. Name its scene and set \"scene_change\" to true. Fill \"era\" and \"era_rules\" "
              "for the film's opening era.")


def next_prompt(run: dict, index: int, seconds: float, memory_text: str, note: str, bible_text: str = "") -> str:
    s, segments = run["settings"], committed(run)
    previous = segments[-1]
    beats = "\n".join(f"  {seg['index']}. {seg.get('beat', '')[:110]}" for seg in segments[-int(s.get("beat_history") or 12):]) or "  (none yet)"
    budget, length = int(s.get("max_scene_segments") or 8), scene_length(segments)
    pacing = f"The current scene is {previous.get('scene')!r} and has run {length} segment(s) of a {budget}-segment budget. "
    pacing += ("That budget is spent: set \"scene_change\" to true and move the story to a new place or situation grounded in what happened."
               if length >= budget else "Advance one new visible consequence without restaging what is done.")
    cast = "\n".join(f"- {name}: {look}" for name, look in cast_at(run, index).items())
    watched = previous.get("watch") or {}
    said = (previous.get("transcript") or "").strip()
    return (seed_text(run["seed"], style_at(s, index)) + "\n"
            + (f"\nCast -- repeat these looks word for word:\n{cast}\n" if cast else "")
            + f"\nBeats asked for so far (requests, not verified events; do not repeat them):\n{beats}\n"
            + f"\nYou are writing segment {index}, about {seconds:.0f} seconds. {pacing}\n"
            + f"\nThe previous segment ({previous.get('scene')}) was asked to show: {previous.get('fields', {}).get('action', '')}\n"
            + f"What the watcher saw it actually show: {watched.get('summary') or 'no report'}\n"
            + f"How it ends, on its last frame: {watched.get('ending') or 'no report'}\n"
            + (f"What was heard: {said}\n" if said else "")
            + (f"\nThe story so far (a running summary):\n{run['story_so_far']['text']}\n" if (run.get("story_so_far") or {}).get("text") else "")
            + (f"\nMemory of the film so far:\n{memory_text}\n" if memory_text else "")
            + (f"\nThe continuity bible (canonical, built from everything the film has shown; never contradict it):\n{bible_text}\n" if bible_text else "")
            + (f"\n{note}\n" if note else "")
            + outline_text(run, index) + direction(s, run.get("branch"), index)
            + (INFINITY_ASK if s.get("infinity") else "")
            + "\nContinue from what really came out, above all the state the last frame leaves things in. Use the memory to keep "
              "people, places and continuity consistent and to avoid repeating what already happened; do not describe the memory itself.")


INFINITY_ASK = ("\nThis film has no fixed length: it goes on while the story does. Add \"end\": true to your answer only when this "
                "segment brings the story to a real ending; otherwise leave it out.\n")


def watch_prompt(run: dict, seg: dict, view: dict, older: int, transcript: str) -> str:
    cast = "\n".join(f"- {name}: {look}" for name, look in cast_at(run, seg["index"]).items()) or "- (no pinned looks)"
    start = float(seg.get("start") or 0)
    return (f"The images are frames of segment {seg['index']} of the film, in time order, covering {mmss(start)} to "
            f"{mmss(start + float(seg.get('seconds') or 0))} on the film's clock; the last image is its final frame.\n"
            + (f"What is heard in it (speech to text, may be rough): {transcript}\n" if transcript else "What is heard in it: no speech.\n")
            + f"\nThe segment was ASKED to show (a request, not a fact): {seg.get('fields', {}).get('action', '')}\n"
            + f"\nThe cast's pinned looks:\n{cast}\n"
            + f"\nThe scene graph so far (every node and the {RECENT_EVENTS} latest events; {older} older events are kept but not shown):\n"
            + json.dumps(view, ensure_ascii=False, indent=1)
            + "\n\nReport ONLY what this segment adds or changes, as this JSON:\n"
            '{"add_nodes": [{"id": "<short unique id>", "type": "character|place|object", "label": "<3 words max>", "state": "<12 words max>"}], '
            '"update_nodes": [{"id": "<existing id>", "state": "<12 words max>"}], '
            '"add_events": [{"t": "<the time on the film\'s clock, like 01:07>", "subject": "<node id>", "action": "<verb phrase, 8 words max>", "target": "<node id or empty>", "place": "<node id>"}], '
            '"summary": "<25 words max: what actually happened>", '
            '"ending": "<40 words max: who and what is where on the final frame, poses, what they hold, the light>", '
            '"continuity": [{"kind": "identity|costume|object|place|other", "who": "<node id or name>", "note": "<15 words max>"}], '
            '"identity_drift": true | false}\n\n'
            "Rules that matter:\n"
            "- A node is a person, a place or a thing that persists. Never a node for an action, a shot or a moment. The "
            "same person older or the same crowd elsewhere keeps its node: reuse the id and update its state.\n"
            "- \"type\": character is only for people and animals; a vehicle, prop or effect is an object.\n"
            "- \"state\" replaces the old state: how it is NOW, twelve words.\n"
            "- At most TWO events, and only events that change the story: something gained or lost, a relationship turns, a "
            "place left for good, a goal met or broken. Glances, smiles, steps and camera moves are not events.\n"
            "- \"continuity\" lists only real problems you can see, never what stayed consistent: a face that changes into another person's, clothing that "
            "differs from the pinned look or from earlier frames, an object that vanishes or changes, a jump in place. Empty when none.\n"
            "- \"identity_drift\" is true only when a pinned character's face, age or build clearly becomes someone else's, "
            "or melts. Turning away, distance, motion blur or new lighting is not drift.\n"
            "- Describe what the frames show, not what was asked for; a voice that is only heard is not a node. Any list may be empty.")


def compact_prompt(events: list, chapters: list) -> str:
    lines = "\n".join(f"  {e.get('t', '--:--')} {e.get('subject', '')} {e.get('action', '')} {e.get('target', '')}".rstrip() for e in events)
    seen = "; ".join(c.get("title", "") for c in chapters) or "(none yet)"
    return (f"These are the events of one stretch of a film, oldest first:\n{lines}\n\nChapters already summarised: {seen}.\n\n"
            'Fold this stretch into a single chapter. Answer with:\n{"title": "<at most 5 words>", "summary": "<at most 30 words, '
            'what this stretch accomplished in the story>", "carried_forward": ["<at most 3 items, each at most 10 words: a '
            'consequence that still matters later>"]}\n\nWrite what changed, not what was shown. Anything left out is gone from the memory.')


def stagnation_note(found: dict) -> str:
    return ("The film is circling: " + found["reason"] + f" (segments {', '.join(map(str, found['segments']))}). The next segment MUST "
            "set scene_change to true, use a different scene name, and complete a visible change of situation chosen by you.")


# ---------- the brains ----------

def ask_text(run: dict, system: str, prompt: str) -> str:
    """The Director (the app's brain), reasoning first when the run says so (``think``) within ``think_tokens`` of
    reasoning (Bolt's thinking switch with TensorFold's thinking_budget: unbounded, it reasoned to the end and gave no
    answer); the reasoning is dropped from the answer."""

    s = run["settings"]
    think = int(s.get("think_tokens") or 0) if s.get("think", True) else 0
    return director(system, prompt, think)


def ask_vision(system: str, frames: list, sheet: Path, prompt: str, brain: dict | None = None) -> tuple[str, str]:
    """The eyes (gma/ap/eyes.py): every frame in one request, or the time-ordered sheet of them when the eyes take
    fewer pictures. ``brain`` is unused (the Studio's QA-gate fallback); kept so a stand-in has one signature."""

    if eyes is None:
        raise Failed("no eyes are set up for AutoPilot")
    return eyes.look(system, frames, sheet, prompt)


def ask_json(asker, check, prompt: str, what: str) -> tuple[dict, list[str]]:
    """A brain asked for JSON, again with the reason when the answer does not hold up: (answer, the reasons turned back)."""

    problems, asked, errors = [], prompt, 0
    for _ in range(MAX_ATTEMPTS):
        try:
            data = extract_json(asker(asked))
            problem = check(data) if isinstance(data, dict) else "the answer is not a JSON object"
        except ValueError as error:
            problem = str(error)
        except Exception as error:  # noqa: BLE001 - a brain that does not answer stops the run, to be resumed
            # once more after a breath: on the phase 4 Spark smoke Bolt failed one long reasoning round (HTTP 500,
            # "every admitted stream fails") and answered the next request
            errors += 1
            if errors > 1:
                raise Failed(f"the {what} did not answer: {str(error)[:300]}")
            problems.append(f"no answer: {str(error)[:120]}")
            time.sleep(BRAIN_RETRY_S)
            continue
        if not problem:
            return data, problems
        problems.append(problem)
        asked = prompt + f"\n\nYOUR LAST ANSWER COULD NOT BE USED: {problem}. Answer again with the JSON only."
    raise Failed(f"the {what}'s answer could not be used {MAX_ATTEMPTS} times: {problems[-1]}")


# ---------- the run's files ----------

def run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id or ""):
        raise Bad("bad run id")
    return HOME / run_id


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".tmp")
    scratch.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    scratch.replace(path)


def load(run_id: str) -> dict:
    """The AutoPilot run kept in the run's run.json shell (the video panel's run, its "autopilot" key)."""

    shell = read_json(run_dir(run_id) / "run.json")
    run = (shell or {}).get("autopilot")
    if not run:
        raise Bad("no such run")
    return run


def shell_of(run: dict) -> dict:
    """The panel's run.json shell around an AutoPilot run (the service puts its own panel fields in, ``shell``)."""

    return {"id": run["id"], "mode": "autopilot", "idea": run.get("name"), "state": run.get("state"), "autopilot": run}


shell = shell_of  # the service's: fn(run) -> the whole run.json


def save(run: dict) -> None:
    with guard:
        run["rev"] = int(run.get("rev") or 0) + 1
        run["updated"] = now()
        write_json(run_dir(run["id"]) / "run.json", shell(run))
    try:
        on_change(run)
    except Exception:  # noqa: BLE001 - the page's trouble never stops the film
        pass


def cooling(line: str) -> None:
    """The render gate waits for the GPU to cool: the line shows on every running film's status (not its log)."""

    for live in list(runs.values()):
        run = live.get("run") if isinstance(live, dict) else None
        if run and run.get("state") == "running":
            run["current"] = {**(run.get("current") or {}), "text": line, "stage": "cooling"}
            save(run)


def say(run: dict, line: str, **current) -> None:
    run.setdefault("log", []).append({"at": now(), "text": line})
    del run["log"][:-LOG]
    run["current"] = {"text": line, **current}
    save(run)


def memory_of(run: dict) -> dict:
    return read_json(run_dir(run["id"]) / "memory.json", copy.deepcopy(EMPTY))


def bible_of(run: dict) -> dict:
    """The continuity bible of the run; a run made before it existed gets one from its seed and cast."""

    found = read_json(run_dir(run["id"]) / "bible.json")
    return found if isinstance(found, dict) and found.get("version") else continuity.new(run["seed"], run.get("cast"))


def save_bible(run: dict, bible: dict, segment_id: str = "") -> None:
    write_json(run_dir(run["id"]) / "bible.json", bible)
    if segment_id:
        write_json(run_dir(run["id"]) / "bible" / f"{segment_id}.json", bible)


def committed(run: dict) -> list:
    return [s for s in run.get("segments") or [] if s.get("status") == "done"]


def relative(path: Path) -> str:
    return str(path.relative_to(HOME))


def on_disk(path: str | None) -> bool:
    if not path:
        return False
    target = (HOME / path).resolve()
    return HOME.resolve() in target.parents and target.is_file()


# ---------- the seed ----------

def span(value) -> tuple[int | None, int | None]:
    """A character's segments: "1-7", "8-", "-7" or "3" (both ends counted), or nothing for every segment."""

    text = str(value or "").replace(" ", "")
    found = re.fullmatch(r"(\d*)(-?)(\d*)", text)
    if not text or not found or not (found[1] or found[3]):
        return None, None
    first = int(found[1]) if found[1] else None
    last = int(found[3]) if found[3] else (None if found[2] else first)
    return first, last


def cast_at(run: dict, index: int) -> dict:
    """{name: look} of the cast in segment ``index``: a seed character only within its segments (a boy in 1-7 and the
    man he becomes from 8 are two sheets), and everyone the Director brought in."""

    seeded = {c["name"]: c for c in run["seed"].get("characters") or []}
    out = {}
    for name, look in (run.get("cast") or {}).items():
        c = seeded.get(name)
        if c and not ((c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6)):
            continue
        if look:
            out[name] = look
    return out


def beat_of(run: dict, index: int) -> dict | None:
    outline = (run.get("seed") or {}).get("outline") or []
    return outline[index - 1] if 0 < index <= len(outline) else None


def clean_seed(given: dict) -> dict:
    if not isinstance(given, dict):
        raise Bad("seed is an object: world, situation, characters")
    characters = []
    for i, c in enumerate(given.get("characters") or []):
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        image = str(c.get("image") or "").strip()
        if image and not on_disk(image):
            raise Bad(f"the picture of {c['name']} is not a file in the studio's folder: {image}")
        first, last = span(c.get("segments"))
        characters.append({"id": re.sub(r"[^a-z0-9]+", "_", str(c["name"]).lower()).strip("_") or f"c{i + 1}",
                           "name": str(c["name"]).strip()[:80], "look": str(c.get("look") or "").strip()[:600], "image": image or None,
                           **({"wardrobe": str(c["wardrobe"]).strip()[:300]} if str(c.get("wardrobe") or "").strip() else {}),
                           **({"role": "opponent"} if c.get("role") == "opponent" else {}),
                           **({"from": first} if first else {}), **({"to": last} if last else {})})
    outline = []
    for i, beat in enumerate((given.get("outline") or [])[:LIMIT]):
        beat = {"text": beat} if isinstance(beat, str) else beat if isinstance(beat, dict) else {}
        text, image = str(beat.get("text") or "").strip()[:600], str(beat.get("image") or "").strip()
        if image and not on_disk(image):
            raise Bad(f"the opening picture of beat {i + 1} is not a file in the studio's folder: {image}")
        if text or image:
            outline.append({"text": text, **({"image": image} if image else {})})
    rules = given.get("rules") or []
    rules = [line.strip() for line in (rules.splitlines() if isinstance(rules, str) else rules) if isinstance(line, str) and line.strip()][:8]
    seed = {"inspiration": str(given.get("inspiration") or "").strip()[:600], "world": str(given.get("world") or "").strip()[:1500],
            "situation": str(given.get("situation") or "").strip()[:1500], "characters": characters[:9],
            "opening_image": str(given.get("opening_image") or "").strip() or None, "slate": given.get("slate") or None,
            **({"outline": outline} if outline else {}), **({"rules": [r[:300] for r in rules]} if rules else {}),
            # a spoken request's opening still (gma/ap/service.py writes it with the seed): drawn before the Director plans
            **({"opening_keyframe": str(given["opening_keyframe"]).strip()[:600]} if str(given.get("opening_keyframe") or "").strip() else {})}
    if seed["opening_image"] and not on_disk(seed["opening_image"]):
        raise Bad("the opening picture is not a file in the studio's folder")
    if not (seed["world"] or seed["situation"] or seed["inspiration"]):
        raise Bad("give the story a seed: a world, where it starts, or an inspiration")
    return seed


def clean_settings(given: dict, base: dict | None = None) -> dict:
    s = copy.deepcopy(base or DEFAULTS)
    given = given if isinstance(given, dict) else {}
    for key in ("style", "guidance", "music"):
        if given.get(key) is not None:
            s[key] = str(given[key]).strip()[:2000]
    for key in ("preset", "quality"):
        if given.get(key):
            s[key] = str(given[key])
    if given.get("keyframe_model") is not None:  # the picture model an edited keyframe is drawn with ("" for the Studio's default)
        s["keyframe_model"] = str(given["keyframe_model"]).strip()[:40]
    for key, low, high in (("seconds", 1, 15), ("segments", 0, LIMIT), ("seed", 0, 2 ** 31 - 1), ("max_scene_segments", 2, 50),
                           ("beat_history", 3, 50), ("retakes", 0, RETAKES), ("preview_every", 1, 50), ("width", 256, 1536), ("height", 256, 1536),
                           ("think_tokens", 0, 32768), ("max_minutes", 0, 60 * 24 * 30), ("max_film_seconds", 0, 10 ** 6), ("film_every", 1, 100),
                           ("film_window", 2, 50), ("story_every", 1, 100), ("min_seconds", 1, 15), ("eyes_parallel", 1, 16),
                           ("fix_budget", 0, 2)):
        if given.get(key) not in (None, ""):
            try:
                s[key] = max(low, min(high, int(given[key])))
            except (TypeError, ValueError):
                raise Bad(f"{key} is a whole number from {low} to {high}")
    for key in ("qa", "ref_turbo", "think", "infinity", "arc", "location_anchor", "narration", "score", "rich_shots", "big_picture"):
        if given.get(key) is not None:
            s[key] = bool(given[key])
    if given.get("checks") is not None:
        try:
            s["checks"] = checks.options({"checks": {**(s.get("checks") or {}), **(given["checks"] if isinstance(given["checks"], dict) else {})}})
        except ValueError as error:
            raise Bad(str(error))
    if given.get("score_db") not in (None, ""):  # the score's level under the film (the Studio's soundtrack job)
        try:
            s["score_db"] = max(-30.0, min(0.0, float(given["score_db"])))
        except (TypeError, ValueError):
            raise Bad("score_db is a number from -30 to 0")
    if given.get("keyframes") in KEYFRAMES:
        s["keyframes"] = given["keyframes"]
    if given.get("identity") in IDENTITY:
        s["identity"] = given["identity"]
    if isinstance(given.get("styles"), list):
        styles = []
        for x in given["styles"][:12]:
            text = str((x or {}).get("style") or "").strip()[:1000] if isinstance(x, dict) else ""
            if text:
                first, last = span(x.get("segments")) if x.get("segments") not in (None, "") else (x.get("from"), x.get("to"))
                styles.append({"style": text, **({"from": int(first)} if first else {}), **({"to": int(last)} if last else {})})
        s["styles"] = styles
    if given.get("continuity") is not None:
        try:
            s["continuity"] = continuity.clean(given["continuity"], s.get("continuity"))
        except ValueError as error:
            raise Bad(str(error))
    return s


# ---------- runs (the Studio's routes, as plain calls: the panel's routes drive them) ----------

def create_run(body: dict, run_id: str | None = None) -> dict:
    """A new run: {name, seed: {world, situation, inspiration, characters: [{name, look}], outline}, settings}. Its
    folder is <HOME>/<run id>."""

    run_id = run_id or time.strftime("r%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    seed = clean_seed(body.get("seed") or {})
    settings = clean_settings(body.get("settings") or {})
    run = {"id": run_id, "name": (str(body.get("name") or "").strip() or "AutoPilot film")[:120], "created": now(),
           "state": "idle", "error": "", "seed": seed, "settings": settings, "segments": [], "branch": None, "film": None,
           "cast": {c["name"]: c["look"] for c in seed["characters"] if c.get("look")}, "log": [], "current": {}, "rev": 0}
    folder = run_dir(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "memory.json", copy.deepcopy(EMPTY))
    write_json(folder / "scene-graph.schema.json", SCHEMA)
    save_bible(run, continuity.new(seed, run["cast"]))
    say(run, "Created.")
    return run


def alive(run_id: str) -> bool:
    live = runs.get(run_id)
    return bool(live and live["thread"].is_alive())


def begin(run: dict) -> dict:
    """Start, or resume a paused, stopped or failed run where it stopped: in a thread of its own."""

    with starting:
        if alive(run["id"]):
            raise Bad("this run is already going")
        run["state"], run["error"], run["pause"] = "running", "", False
        for seg in run.get("segments") or []:  # a segment that failed is planned again on resume, at its own seed
            if seg.get("status") == "failed":
                # one that failed before its keyframe was kept draws its keyframe again, never a keyframe-less clip
                again = "keyframe" if keyframes_for(run, seg["index"], seg.get("scene_change")) and not on_disk(seg.get("keyframe")) else "planned"
                seg["status"], seg["job"], seg["error"] = again, None, ""
        say(run, "Started." if not committed(run) else f"Resumed after segment {len(committed(run))}.")
        stop = threading.Event()
        thread = threading.Thread(target=work, args=(run, stop), daemon=True, name="autopilot-" + run["id"])
        runs[run["id"]] = {"thread": thread, "stop": stop, "run": run}
        thread.start()
        return run


def pause(run_id: str) -> None:
    """Pause after the segment at work: it renders, is checked and joins the film; then the loop waits."""

    live = runs.get(run_id)
    if live and live["thread"].is_alive():
        live["run"]["pause"] = True
        say(live["run"], "Pausing after this segment.")


def stop_run(run_id: str) -> bool:
    """Stop now (the piece at work is picked up on resume). True when a live loop was told."""

    live = runs.get(run_id)
    if live and live["thread"].is_alive():
        live["stop"].set()
        return True
    return False


def branch(source: dict, body: dict) -> dict:
    """A new run that shares segments 1..``after`` of ``source`` (their clips stay where they are), with the memory as
    it was after segment ``after``, and goes on with new ``guidance``. {after, guidance, name?, segments?, seed?}"""

    run_id = source["id"]
    done = committed(source)
    try:
        after = int(body.get("after"))
    except (TypeError, ValueError):
        raise Bad("after is the number of the segment to branch from")
    if not 1 <= after <= len(done):
        raise Bad(f"branch from a committed segment, 1 to {len(done)}")
    guidance = str(body.get("guidance") or "").strip()
    if not guidance:
        raise Bad("give the branch its direction: what should happen from here")
    snapshot = read_json(run_dir(run_id) / "memory" / f"S{after:04d}.json")
    if not snapshot:
        raise Bad(f"segment {after} has no memory snapshot to branch from")
    new_id = time.strftime("r%m%d-%H%M%S-") + uuid.uuid4().hex[:5]
    settings = clean_settings({k: body[k] for k in ("segments", "seed") if body.get(k) not in (None, "")}, source["settings"])
    if body.get("segments") in (None, "") and settings["segments"]:
        settings["segments"] = after + max(1, settings["segments"] - after)
    elif settings["segments"]:
        settings["segments"] = after + settings["segments"]  # a number given is how many new segments
    shared = [{**copy.deepcopy(s), "shared": run_id} for s in done[:after]]
    run = {"id": new_id, "name": (str(body.get("name") or "").strip() or f"{source.get('name')} · branch at {after}")[:120],
           "created": now(), "state": "idle", "error": "", "seed": copy.deepcopy(source["seed"]), "settings": settings,
           "segments": shared, "branch": {"from": run_id, "after": after, "guidance": guidance[:2000]}, "film": None,
           "cast": copy.deepcopy(snapshot.get("cast") or source.get("cast") or {}), "log": [], "current": {}, "rev": 0}
    folder = run_dir(new_id)
    folder.mkdir(parents=True)
    memory = {key: copy.deepcopy(snapshot.get(key, EMPTY[key])) for key in EMPTY}
    write_json(folder / "memory.json", memory)
    for seg in shared:
        mark = read_json(run_dir(run_id) / "memory" / f"S{seg['index']:04d}.json")
        if mark:
            write_json(folder / "memory" / f"S{seg['index']:04d}.json", mark)
    write_json(folder / "scene-graph.schema.json", SCHEMA)
    bible = read_json(run_dir(run_id) / "bible" / f"S{after:04d}.json")
    save_bible(run, bible if isinstance(bible, dict) and bible.get("version") else continuity.new(run["seed"], run["cast"]))
    for seg in shared:
        mark = read_json(run_dir(run_id) / "bible" / f"S{seg['index']:04d}.json")
        if mark:
            write_json(folder / "bible" / f"S{seg['index']:04d}.json", mark)
    say(run, f"Branched from {source.get('name')} after segment {after}: {guidance[:200]}")
    return run

def resume_all() -> list[str]:
    """At the app's start: every run that was live when it stopped (a crash, a kill -9) carries on from its
    checkpoint, without anyone asking. Returns their ids."""

    picked = []
    for file in sorted(HOME.glob("*/run.json")):
        run = (read_json(file) or {}).get("autopilot")
        if run and run.get("state") in ACTIVE and not alive(run["id"]):
            try:
                say(run, "Picked up after a restart.")
                begin(run)
                picked.append(run["id"])
            except Bad:
                pass
    return picked


# ---------- the loop ----------

def wait(stop: threading.Event, seconds: float | None = None) -> None:
    if stop.wait(TICK if seconds is None else seconds):
        raise Stopped()


def work(run: dict, stop: threading.Event) -> None:
    misses, began = 0, time.time()
    while True:
        try:
            if step(run, stop, began):
                return
            misses = 0
        except Stopped:
            pending = next((s for s in run.get("segments") or [] if s.get("status") == "queued"), None)
            if pending and (job_of(pending.get("job") or "") or {}).get("status") == "queued":
                cancel(pending["job"])
                pending["status"], pending["job"] = "planned", None
            run["state"] = "paused"
            say(run, "Stopped. Resume to carry on from here: every finished segment and the memory are kept.")
            return
        except Exception as error:  # noqa: BLE001 - the reason goes to the page; resume tries again
            why = str(getattr(error, "detail", error))[:600]
            if run["settings"].get("infinity") and misses < INFINITY_RETRIES and not stop.is_set():
                # Infinity goes on by itself: the piece at work is planned again from its checkpoint after a pause
                misses += 1
                for seg in run.get("segments") or []:
                    if seg.get("status") == "failed":
                        seg["status"], seg["job"], seg["error"] = "planned", None, ""
                say(run, f"A problem: {why[:240]}. Infinity carries on from the last checkpoint in {INFINITY_WAIT_S} s "
                         f"(try {misses} of {INFINITY_RETRIES}).")
                if stop.wait(INFINITY_WAIT_S):
                    run["state"] = "paused"
                    say(run, "Stopped. Resume to carry on from here: every finished segment and the memory are kept.")
                    return
                continue
            run["state"], run["error"] = "failed", why
            say(run, "Stopped on a problem: " + run["error"])
            return


def finished(run: dict, began: float) -> str:
    """Why the run is complete ("" while it goes on): its number of segments, or, in Infinity, a time or length limit
    or an ending the Director chose."""

    s, done = run["settings"], committed(run)
    if not s.get("infinity"):
        limit = int(s.get("segments") or 0)
        return f"{limit} segments" if limit and len(done) >= limit else ""
    if done and done[-1].get("end"):
        return "the Director ended the story"
    seconds = sum(float(x.get("seconds") or 0) for x in done)
    if s.get("max_film_seconds") and seconds >= float(s["max_film_seconds"]):
        return f"the film reached {seconds:.0f} s of its {float(s['max_film_seconds']):.0f}"
    minutes = (float(run.get("worked_s") or 0) + time.time() - began) / 60
    if s.get("max_minutes") and minutes >= float(s["max_minutes"]):
        return f"{minutes:.0f} minutes of its {float(s['max_minutes']):.0f}"
    return ""


def step(run: dict, stop: threading.Event, began: float) -> bool:
    """One turn of the loop; True when the run has paused or is complete."""

    if stop.is_set():
        raise Stopped()
    take_redo(run)
    if run.get("pause"):
        run["state"], run["pause"] = "paused", False
        run["worked_s"] = float(run.get("worked_s") or 0) + time.time() - began
        if (run.get("film") or {}).get("segments") != len(committed(run)):
            publish(run)
        say(run, f"Paused after segment {len(committed(run))}. Resume to carry on.")
        return True
    why = finished(run, began)
    if why and run["settings"].get("big_picture") and not run["settings"].get("infinity"):
        if (run.get("film") or {}).get("segments") != len(committed(run)):
            publish(run)
        if big_picture.next_pass(run):
            return False  # the film is made again from the earliest piece its fix plan touches
    if why:
        if (run.get("film") or {}).get("segments") != len(committed(run)):
            publish(run)
        if continuity.options(run["settings"])["film_pass"] and len(committed(run)) >= 2 and not (run.get("film_check") or {}).get("segments") == len(committed(run)):
            film_check_safely(run)
        if finish is not None and run.get("film") and not run.get("film_final"):
            try:
                finish(run)  # the film's narration and score, by the headless Studio (gma/ap/sound.py)
            except Exception as error:  # noqa: BLE001 - the film stands without its sound pass
                say(run, "The film's sound pass failed: " + str(error)[:200])
        run["state"] = "done"
        run["worked_s"] = float(run.get("worked_s") or 0) + time.time() - began
        say(run, f"Done ({why}): {len(committed(run))} segments, {run['film']['seconds'] if run.get('film') else 0:.1f} s of film.")
        return True
    segment(run, stop)
    s, n = run["settings"], len(committed(run))
    if s.get("infinity") or n == 1 or n % int(s.get("preview_every") or 3) == 0:
        publish(run)  # Infinity: the film so far is joined after every piece, always playable
        checkpoint(run, committed(run)[-1])
    if s.get("infinity") and continuity.options(s)["film_pass"] and n >= 2 and n % max(1, int(s.get("film_every") or 4)) == 0:
        film_check_safely(run, int(s.get("film_window") or 6))
    if s.get("infinity") and n % max(1, int(s.get("story_every") or 6)) == 0:
        story_so_far(run)
    return False


# ---------- the person's voice during a run (gma/ap/service.py hands these over from the video agent) ----------

def start_opening(run: dict) -> None:
    """Segment 1's keyframe from the seed's ``opening_keyframe`` (written with the seed from the spoken request), sent
    to the renderer at once: the panel shows it while the Director plans the arc and the first piece."""

    text = str(run["seed"].get("opening_keyframe") or "").strip()
    if (run.get("big_picture") or {}).get("state") == "regenerating":
        return  # the fix plan's own opening is drawn: the Director writes it with the fix in hand
    if not text or run.get("opening") or run.get("segments") or not keyframes_for(run, 1):
        return
    s = run["settings"]
    looks = look_lines(run, {"setting": text, "action": ""}, 1)
    prompt = " ".join(bit for bit in (style_at(s, 1), looks, clip_words(text, 70)) if bit)
    seed = int(s.get("seed") or 0)
    params = {"prompt": prompt, "count": 1, "seed": seed, "preset": s["preset"], "autopilot": {"run": run["id"], "segment": 1, "keyframe": 1}}
    try:
        job = submit({"kind": "scout", "title": f"AutoPilot · {run['name']} · 1: opening keyframe", "params": params})
    except Exception as error:  # noqa: BLE001 - the Director's own keyframe is drawn instead
        say(run, "The opening keyframe could not be started: " + str(error)[:160], segment=1)
        return
    run["opening"] = {"job": job["id"], "prompt": prompt, "seed": seed, "at": now()}
    say(run, "Segment 1: its opening keyframe, drawn from the request while the Director plans.", stage="keyframe", segment=1, job=job["id"])


REDO_FIX = {"category": "story", "frame": 0, "evidence": "the person watching asked to redo this piece: film the same step again, better"}


def take_redo(run: dict, seg: dict | None = None) -> bool:
    """"Redo that" (``run["redo"]``), on the loop's own thread: the piece the person has seen is made again. A piece
    whose clip is rendering or made is taken again at the next seed; otherwise (the next piece is still being planned
    or keyframed) that one is dropped and the last committed piece is taken out of the film, its memory and bible put
    back as they were before it, and made again. True when something was redone."""

    if not run.pop("redo", None):
        return False
    seg = seg or next((x for x in run.get("segments") or [] if x.get("status") not in ("done",)), None)
    if seg is not None and (seg.get("takes") or seg["status"] in ("queued", "rendered")):
        if seg["status"] == "queued" and seg.get("job"):
            try:
                cancel(seg["job"])
            except Exception:  # noqa: BLE001
                pass
        seg.update(status="retake", fix=[dict(REDO_FIX)], job=None)
        say(run, f"Segment {seg['index']}: made again, as asked.", segment=seg["index"])
        return True
    done = committed(run)
    if not done:
        say(run, "Nothing has been shown yet to redo.")
        return False
    if seg is not None:
        if seg["status"] == "keyframe_queued" and seg.get("job"):
            try:
                cancel(seg["job"])
            except Exception:  # noqa: BLE001
                pass
        run["segments"].remove(seg)
        run.pop("opening", None)
    last = done[-1]
    folder, before = run_dir(run["id"]), last["index"] - 1
    memory = read_json(folder / "memory" / f"S{before:04d}.json") if before else None
    write_json(folder / "memory.json", {k: copy.deepcopy((memory or {}).get(k, EMPTY[k])) for k in EMPTY})
    bible = read_json(folder / "bible" / f"S{before:04d}.json") if before else None
    save_bible(run, bible if isinstance(bible, dict) and bible.get("version") else continuity.new(run["seed"], run.get("cast")))
    run["behind"] = [b for b in run.get("behind") or [] if b["index"] < last["index"]]
    last.update(status="retake", fix=[dict(REDO_FIX)], job=None)
    say(run, f"Segment {last['index']}: taken out of the film and made again, as asked.", segment=last["index"])
    return True


def steer(run_id: str, text: str, kind: str = "note") -> dict:
    """A spoken steer ("make it rain", "faster", "new scene: ..."), from the next piece that has not started rendering
    its clip. Returns the steer."""

    run = (runs.get(run_id) or {}).get("run") or load(run_id)
    segs = run.get("segments") or []
    at = next((x for x in segs if x.get("status") not in ("done",)), None)
    # a piece already planned keeps its plan (and one the Director is writing now): the steer is the next one's
    planning = at is None and (run.get("current") or {}).get("stage") == "director" and run.get("state") == "running"
    index = at["index"] + 1 if at else len(committed(run)) + (2 if planning else 1)
    one = story.add_steer(run, text, kind, index)
    if kind == "pace":
        run["settings"]["guidance"] = (str(run["settings"].get("guidance") or "") + " Faster, more physical action.").strip()[:2000]
    note(run, f"Heard: \"{one['text']}\" (from segment {index}).")
    return one


def note(run: dict, line: str) -> None:
    """A line in the run's log that leaves the stage at work as it is (the person's words arrive mid-render)."""

    run.setdefault("log", []).append({"at": now(), "text": line})
    del run["log"][:-LOG]
    save(run)


def extend(run_id: str, pieces: int = 1) -> dict:
    """"Keep going" / "twenty seconds more": the film is made longer by ``pieces`` (an Infinity film goes on); a run
    that is paused, stopped or done carries on. Its arc is planned on for the new pieces."""

    live = runs.get(run_id)
    run = live["run"] if live else load(run_id)
    s = run["settings"]
    unfinished = run.get("state") in ("paused", "failed", "stopped") and len(committed(run)) < int(s.get("segments") or 0)
    if not s.get("infinity") and not unfinished:  # a film stopped before its end just goes on: nothing is added
        s["segments"] = max(int(s.get("segments") or 0), len(committed(run))) + max(1, int(pieces))
    if run.get("arc"):
        run["arc"]["payoff_reached"] = run["arc"].get("payoff")
    if alive(run_id):
        run["pause"] = False
        note(run, f"Going on: {s['segments'] if not s.get('infinity') else 'no end'} segments.")
        return run
    return begin(run)


def redo(run_id: str) -> dict:
    """"Redo that": made again by the loop (now when it is at work, or on resume)."""

    live = runs.get(run_id)
    run = live["run"] if live else load(run_id)
    run["redo"] = {"at": now()}
    if alive(run_id):
        save(run)
        return run
    return begin(run)


def sharpen(run_id: str) -> dict:
    """"Go sharper": the next pieces at 1376x768 on the governor's sharp schedule (the brain is evicted for them)."""

    run = (runs.get(run_id) or {}).get("run") or load(run_id)
    run["settings"].update(preset="sharp")
    note(run, "The next pieces go sharper: 1376x768.")
    return run


def story_so_far(run: dict) -> None:
    """The running summary the Director reads in Infinity (bounded: the memory's chapters and the last beats folded
    into 120 words), written again every ``story_every`` pieces; a summary that cannot be made keeps the old one."""

    memory, done = memory_of(run), committed(run)
    beats = "\n".join(f"  {x['index']}. {x.get('beat', '')[:110]}" for x in done[-12:])
    prompt = ((f"The story so far, as last summarised:\n{run['story_so_far']['text']}\n\n" if (run.get("story_so_far") or {}).get("text") else "")
              + f"What the film remembers:\n{as_text(memory['graph'])}\n\nThe latest segments' beats:\n{beats}\n\n"
              'Write the story so far in 120 words at most: who wants what, what has happened that still matters, where things stand now. '
              'Answer: {"text": "<120 words max>"}')
    try:
        data, _ = ask_json(lambda text: ask_text(run, COMPACT_SYSTEM, text), lambda d: "" if str(d.get("text") or "").strip() else '"text" is the summary', prompt, "story summary")
        run["story_so_far"] = {"text": clip_words(data["text"], 130), "through": len(done), "at": now()}
        save(run)
    except Failed as error:
        say(run, "The story so far could not be summarised again: " + str(error)[:160])


def segment(run: dict, stop: threading.Event) -> None:
    """One segment from plan to commit; a segment left half-way (a stop, a restart) is picked up where it was."""

    seg = next((s for s in run["segments"] if s.get("status") not in ("done",)), None)
    if seg is None:
        index = len(committed(run)) + 1
        if index == 1:
            start_opening(run)  # a spoken request's opening still, drawn while the Director plans
        plan_arc(run, index)
        if take_redo(run):
            return  # "redo that" while the arc was planned: the piece before is made again first
        seg = plan(run)
    while seg["status"] != "done":
        if stop.is_set():
            # "stop now" while the Director planned (or a check ran): nothing new is sent to the renderer; the
            # segment keeps its status and is picked up on resume
            raise Stopped()
        if take_redo(run, seg):
            seg = next((s for s in run["segments"] if s.get("status") not in ("done",)), None)
            if seg is None:
                return
            continue
        if seg["status"] == "planned":
            render(run, seg)
        elif seg["status"] == "queued":
            follow(run, seg, stop)
        elif seg["status"] == "rendered":
            check(run, seg)
        elif seg["status"] == "keyframe":
            render_keyframe(run, seg)
        elif seg["status"] == "keyframe_queued":
            follow_keyframe(run, seg, stop)
        elif seg["status"] == "keyframe_ready":
            check_keyframe(run, seg)
        elif seg["status"] == "retake":
            fixing = bool(seg.get("fix"))
            if fixing:
                replan(run, seg)  # the Director rewrites the shot with the findings in hand
            seg["seed"] = int(seg["seed"]) + 1
            seg["status"] = "planned"  # the same opening picture: a retake varies the seed and the action's words only
            save(run)
        elif seg["status"] == "failed":
            raise Failed(f"segment {seg['index']} could not be rendered: {seg.get('error') or 'no reason'}")


# ---------- plan ----------

def frames_of(settings: dict) -> int:
    return lengths.nearest(float(settings.get("seconds") or 5))


def story_text(run: dict, index: int) -> str:
    """The arc (this segment's beat and the state it must end on, the payoff, a catch-up), the person's steers and the
    fixed place, for the Director writing segment ``index`` (story.py)."""

    place = story.place_line(story.anchor_of(run, index)) if run["settings"].get("location_anchor", True) and index > 1 else ""
    return story.arc_text(run, index) + place + story.steer_text(run, index)


def director_prompt(run: dict, index: int, seconds: float) -> tuple[str, dict | None]:
    """What the Director reads for segment ``index``, and the stagnation it must break (None when there is none)."""

    ask = KEYFRAME_ASK if keyframes_for(run, index) else ""
    opening = (run.get("opening") or {}).get("prompt") if index == 1 else None
    if opening:
        ask = OPENING_ASK.format(prompt=opening)
    if index == 1:
        bible = continuity.director_text(bible_of(run), 1, list(cast_at(run, 1)))
        return first_prompt(run, seconds) + story_text(run, 1) + f"\n\nThe continuity bible so far (canonical):\n{bible}" + ask, None
    done = committed(run)
    memory = memory_of(run)
    stuck = stagnation(done)
    bible = continuity.director_text(bible_of(run), index, list(cast_at(run, index)))
    return (next_prompt(run, index, seconds, as_text(memory["graph"]), stagnation_note(stuck) if stuck else "", bible)
            + story_text(run, index) + ask), stuck


def keyframes_for(run: dict, index: int, scene_change: bool | None = None) -> bool:
    """Whether segment ``index`` opens on a keyframe of its own: the run's choice (auto: when it has an outline), never
    for a beat that brings its own picture, and only where a scene opens (the film's first piece, a scene change): a
    piece that continues opens on the last frame of the piece before, kept, so the film physically goes on."""

    choice = run["settings"].get("keyframes", "auto")
    wanted = choice == "on" or (choice == "auto" and bool((run.get("seed") or {}).get("outline")))
    opens = index == 1 or bool(scene_change)
    return wanted and opens and not (beat_of(run, index) or {}).get("image")


# the video sizes of the presets AutoPilot offers (server.PRESETS): an edited keyframe is made at the video's own size
PRESET_SIZES = {"sharp": (1376, 768), "small": (864, 480), "p720": (1280, 736), "native": (1344, 768), "draft": (1344, 768)}

KEYFRAME_ASK = ("\nThis segment opens on a keyframe made first by the picture model: fill \"keyframe\" with that opening still "
                "(who is where, the framing, the place, the light; 60 words at most), and write the action from that picture on. "
                "The keyframe is the START of the shot: every object in its \"qc.start\" state, said plainly (\"the oil lamp unlit, "
                "its wick dark\"), and nothing of the outcome: no light, flame, open door or result the action only brings later. "
                "A fight's keyframe catches a blow already in motion (a fist or a weapon mid-swing, a body leaning into the "
                "strike), never a stand-off: the video model eases slowly out of a still pose.")


# the opening keyframe of a spoken request: drawn from the seed at once (it is in the panel while the Director writes)
OPENING_ASK = ("\nThis segment opens on a keyframe that is already being drawn from this description: {prompt}\nWrite the action "
               "from that picture on; \"keyframe\" may repeat it.")


RICH_SHOTS = """

Every shot is written to be filmed richly (each within its field's word limit):
- "camera" names a camera MOVE (a push-in, a tilt up with the climb, a slow dolly, a handheld follow), never only "static".
- "setting" names the light and its colour (cold blue lightning, a warm amber lamp glow, a bruised purple sky).
- "music" names a cue (an instrument, a tempo, a mood), never N/A.
- "pace.beats" gives the seconds of the shot with their speed, and a slow-motion moment where the key change happens.
- "camera" or "action" moves in close on the detail where the contact or the change of state happens (the hand on \
the latch, the key turning, the flame touching the wick)."""


def director_system(run: dict) -> str:
    return DIRECTOR_SYSTEM + (RICH_SHOTS if (run.get("settings") or {}).get("rich_shots") else "")


def director_asker(run: dict):
    return lambda text: ask_text(run, director_system(run), text)


def fields_of(shot: dict) -> dict:
    return {"setting": str(shot.get("setting") or ""), "camera": str(shot.get("camera") or ""), "action": str(shot.get("action") or ""),
            "dialogue": [d for d in shot.get("dialogue") or [] if isinstance(d, dict) and str(d.get("text") or "").strip()][:3],
            "soundscape": str(shot.get("soundscape") or ""), "music": str(shot.get("music") or "N/A")}


def plan(run: dict) -> dict:
    s, done = run["settings"], committed(run)
    index = len(done) + 1
    frames = frames_of(s)
    seconds = lengths.seconds_of(frames)
    run["state"] = "running"
    say(run, f"Segment {index}: the Director writes it.", stage="director", segment=index)
    bible = bible_of(run)
    continuity.arrive(bible, index, run["seed"])  # a cast sheet that starts here opens an era before the Director reads it
    save_bible(run, bible)
    prompt, stuck = director_prompt(run, index, seconds)
    before = done[-1] if done else None
    carried = list((before or {}).get("verified_end") or [])
    if (before or {}).get("described"):
        prompt += "\nWhat the last piece actually showed (the eyes, frame by frame): " + str(before["described"])[:600]
    if (before or {}).get("qc_notes"):
        prompt += ("\nNotes from its self-check (fix them in this piece where the story allows, never by jumping away "
                   "from its last frame): " + "; ".join(before["qc_notes"])[:600])
    prompt += big_picture.plan_text(run, index)
    if carried:
        prompt += ("\nThe last piece's last frame showed, verified on its frames: " + "; ".join(carried) + ". Unless this "
                   "segment changes scene, it opens exactly so: its \"qc.start\" and \"qc.start_place\" agree with it, and "
                   "nothing it left unlit, closed or in place is otherwise at the first frame.")

    def check_answer(data: dict) -> str:
        return director_problem(run, data, index, stuck)

    data, rejected = ask_json(director_asker(run), check_answer, prompt, "Director")
    shot = data["shot"]
    if index == 1 or (beat_of(run, index) or {}).get("image") or story.scene_steer(run, index):
        data["scene_change"] = True  # the opening, a beat that opens on a picture of its own, or a spoken "new scene"
    for person in data.get("new_characters") or []:
        if isinstance(person, dict) and person.get("name") and person.get("look"):
            run.setdefault("cast", {}).setdefault(str(person["name"]).strip(), str(person["look"]).strip())
    bible = bible_of(run)
    continuity.planned(bible, index, data, run.get("cast") or {})
    fields = fields_of(shot)
    continuity.voices(bible, index, fields["dialogue"])  # each speaker keeps the voice the bible first heard
    apply_steers(run, bible, index)
    save_bible(run, bible)
    seg = {"index": index, "id": f"S{index:04d}", "status": "planned", "scene": str(data["scene"]).strip()[:80],
           "scene_change": bool(data["scene_change"]), "beat": str(data["beat"]).strip()[:200], "fields": fields,
           "rejected": rejected, "seed": int(s.get("seed") or 0) + index - 1, "frames": frames, "seconds": seconds,
           "start": round(sum(float(x.get("seconds") or 0) for x in done), 3), "takes": [], "job": None, "output": None,
           "era": continuity.era_at(bible, index)["id"], "planned": now(), "pace": checks.pace_of(data.get("pace")),
           "qc": qc.clean_qc(data.get("qc")),
           "carried": carried if not data["scene_change"] else [],
           **({"end": True} if data.get("end") is True and s.get("infinity") else {})}
    if keyframes_for(run, index, seg["scene_change"]):
        seg.update(status="keyframe", keyframe_prompt=clip_words(data.get("keyframe"), 70), keyframes=[])
        opening = run.get("opening") if index == 1 else None
        if opening and opening.get("job"):
            # the opening keyframe drawn from the seed while the Director wrote: followed, then checked like any other
            seg.update(status="keyframe_queued", job=opening["job"], keyframe_prompt=opening["prompt"],
                       keyframes=[{"take": 1, "round": 0, "job": opening["job"], "prompt": opening["prompt"], "seed": opening["seed"],
                                   "path": None, "kind": "scout", "opening": True}])
    folder = run_dir(run["id"]) / "segments" / seg["id"]
    write_json(folder / "director.json", {"prompt": prompt, "answer": data, "rejected": rejected})
    run["segments"].append(seg)
    if rejected:
        say(run, f"Segment {index}: {len(rejected)} answer{'s' if len(rejected) != 1 else ''} of the Director turned back ({rejected[-1][:120]}).", segment=index)
    say(run, f"Segment {index}: {seg['beat']}", stage="planned", segment=index)
    checkpoint(run, next_seg=seg)
    return seg


def apply_steers(run: dict, bible: dict, index: int) -> None:
    """The person's steers that apply from segment ``index`` become bible facts (the weather and the light are the
    scene's, anything else a story fact); each is marked applied."""

    for steer in story.steers_for(run, index):
        if steer.get("applied") or steer["kind"] != "note":
            steer["applied"] = steer.get("applied") or steer["from"] <= index
            continue
        category, key = story.steer_fact(steer)
        if category == "lighting":
            continuity.establish(bible, "_scene", "lighting", key, steer["text"], index, "user", words=20)
        else:
            bible.setdefault("story", []).append({"t": "", "text": f"(the person asked) {steer['text']}", "segment": index, "subjects": []})
        steer.update(applied=True, applied_at=index)


def plan_arc(run: dict, index: int) -> None:
    """The arc for segment ``index`` on: the whole film at once (a fixed length), or the next ``story.ARC_SPAN`` pieces
    (Infinity, or a film made longer). A plan that cannot be made leaves the film to the Director's piece by piece."""

    count = story.arc_needed(run, index)
    if not count:
        return
    say(run, f"The Director plans the arc: segments {index} to {index + count - 1}.", stage="director", segment=index)
    seed = seed_text(run["seed"], style_at(run["settings"], index)) + outline_text(run, index) + direction(run["settings"], run.get("branch"), index)
    so_far = "; ".join(f"{x['index']}. {x.get('beat', '')}" for x in committed(run)[-8:])
    prompt = story.arc_prompt(seed, count, index, ((run.get("story_so_far") or {}).get("text") or so_far) if index > 1 else "",
                              (run.get("arc") or {}).get("payoff", "") if index > 1 else "", story.steer_text(run, index))
    try:
        data, rejected = ask_json(lambda text: ask_text(run, story.ARC_SYSTEM, text), lambda d: story.arc_problem(d, count), prompt, "arc")
    except Failed as error:
        say(run, "The arc could not be planned; the Director goes piece by piece: " + str(error)[:160], segment=index)
        return
    run["arc"] = story.merge_arc(run.get("arc"), data, index, count)
    write_json(run_dir(run["id"]) / f"arc-{index:04d}.json", {"prompt": prompt, "answer": data, "rejected": rejected})
    say(run, f"The arc: {run['arc']['payoff']}", stage="planned", segment=index)


def replan(run: dict, seg: dict) -> None:
    """A take that failed its continuity check written again by the Director, with each finding: the same story step
    in the same scene, the shot corrected. The next take renders it at a new seed."""

    index, take = seg["index"], len(seg["takes"])
    say(run, f"Segment {index}: the Director rewrites take {take} with the continuity findings.", stage="director", segment=index)
    prompt, stuck = director_prompt(run, index, float(seg["seconds"]))
    found = "\n".join(f"- {f['category']}" + (f" (frame {f['frame']})" if f.get("frame") else "") + f": {f['evidence']}" for f in seg.get("fix") or [])
    if any(f.get("category") == "pace" for f in seg.get("fix") or []):
        found += ("\nThe video model made it slow: write the fight as fast, real-time action (quick strikes, rapid exchanges, sharp "
                  "impacts, whip pans), name the speed plainly, and keep slow motion to the one key move with its seconds.")
    rules = rewrite_rules(seg.get("fix") or [])
    if rules:
        found += "\n" + rules
    prompt += (f"\n\nTHE TAKE YOU WROTE FOR SEGMENT {index} WAS RENDERED AND FAILED ITS CONTINUITY CHECK:\n{found}\n"
               f"Write segment {index} again: the same story step (\"{seg['beat']}\") in the same scene ({seg['scene']!r}), with the shot "
               "rewritten so that each of these comes out right; say the bible's details plainly where they matter. Keep \"scene\" and "
               "\"scene_change\" as they were.")

    def check_answer(data: dict) -> str:
        return director_problem(run, data, index, stuck if seg["scene_change"] else None)

    data, rejected = ask_json(director_asker(run), check_answer, prompt, "Director")
    fields = fields_of(data["shot"])
    bible = bible_of(run)
    continuity.voices(bible, index, fields["dialogue"])
    save_bible(run, bible)
    # the opening picture stays what it was: a retake never draws the anchor again
    seg.update(fields=fields, beat=str(data["beat"]).strip()[:200] or seg["beat"], rejected=(seg.get("rejected") or []) + rejected, pace=checks.pace_of(data.get("pace")),
               qc=qc.clean_qc(data.get("qc")) if isinstance(data.get("qc"), dict) else seg.get("qc") or qc.clean_qc(None),
               fixed=(seg.get("fixed") or []) + [{"take": take, "findings": seg.get("fix") or []}], fix=[])
    write_json(run_dir(run["id"]) / "segments" / seg["id"] / f"director-take{take + 1}.json", {"prompt": prompt, "answer": data, "rejected": rejected})




REWRITE = {
    "direction": ("REWRITE THE MOVEMENT, do not resample it: say the direction in BOTH \"action\" and \"camera\" with where the "
                  "camera stands, in capitals for the key word (\"Ewan climbs UP the spiral stairs toward the lantern room "
                  "above; camera below him looking UP\"), name where he starts and where he ends, and give him his back to "
                  "the camera if he moves away from it. The opposite movement is the failure: \"qc.wrong_motion\" names it."),
    "end place": ("REWRITE THE ENDING: say the room or spot he stops in, which side of the door, and the last thing the "
                  "camera sees."),
    "end state": "REWRITE THE RESULT: the action must reach the end state on screen, with seconds (\"by 8 s the wick burns\").",
    "contact": "REWRITE THE CONTACT: name both things and where they meet, and move the camera in close on it.",
    "people": "REWRITE THE STAGING: one person, one spot; no window, doorway or mirror behind them that could show a second copy.",
    "cut": "REWRITE AS ONE SHOT: one camera, one continuous move, no change of angle.",
    "seam": "REWRITE THE OPENING: the first second holds the last frame of the piece before, then moves from it.",
}


def rewrite_rules(findings: list) -> str:
    """For each failed check a rewrite rule (a retake rewrites the shot for its failure; it never resamples the text)."""

    seen, out = set(), []
    for f in findings:
        key = f.get("check") or f.get("category")
        if key in REWRITE and key not in seen:
            seen.add(key)
            out.append("- " + REWRITE[key])
    return "\n".join(out)


def retakes_now(run: dict) -> int:
    """The per-piece retake budget: none while a big-picture film drafts or regenerates (its fixes are the film's)."""

    s = run["settings"]
    return 0 if s.get("big_picture") else int(s.get("retakes") or 0)


def director_problem(run: dict, data: dict, index: int, stuck) -> str:
    """Why a Director answer cannot be used, or "" when it can."""

    shot = data.get("shot")
    if not isinstance(shot, dict) or not str(shot.get("action") or "").strip():
        return '"shot" is an object with at least "action"'
    if not str(data.get("scene") or "").strip() or not str(data.get("beat") or "").strip():
        return '"scene" and "beat" are short lines of text'
    if not isinstance(data.get("scene_change"), bool):
        return '"scene_change" is true or false'
    fields = {"scene": str(shot.get("setting") or ""), "camera": str(shot.get("camera") or ""), "action": str(shot.get("action") or ""),
              "soundscape": str(shot.get("soundscape") or ""), "dialogue": [d for d in shot.get("dialogue") or [] if isinstance(d, dict)]}
    for key, most in (("action", 60), ("camera", 25), ("scene", 45)):
        if len(fields[key].split()) > most:
            return f'"{"setting" if key == "scene" else key}" is {len(fields[key].split())} words: keep it to {most} at most, short and concrete'
    if keyframes_for(run, index, bool(data.get("scene_change"))) and not str(data.get("keyframe") or "").strip() and not (index == 1 and run.get("opening")):
        return '"keyframe" is the still picture this segment opens on'
    if data.get("end") is not None and not isinstance(data.get("end"), bool):
        return '"end" is true or false'
    paced = checks.pace_problem(checks.pace_of(data.get("pace")), fields["action"], fields["camera"], lengths.seconds_of(frames_of(run["settings"])))
    if paced:
        return paced
    named = negation.check_fields(fields)["negations"]
    if named:
        return (f"\"{named[0]['phrase']}\" names what must not be seen; the video model has no negative prompt and what is "
                "named tends to appear. Say what is there instead")
    if index == 1:
        return ""
    done = committed(run)
    previous = done[-1]
    action = fields["action"].strip().lower()
    if difflib.SequenceMatcher(None, action, str(previous["fields"].get("action") or "").strip().lower()).ratio() >= SIMILAR_ACTION:
        return "the action repeats the segment before almost word for word; move the story one visible step on"
    beat = str(data["beat"]).strip().lower()
    for seg in done[-int(run["settings"].get("beat_history") or 12):]:
        if difflib.SequenceMatcher(None, beat, str(seg.get("beat") or "").strip().lower()).ratio() >= SIMILAR_BEAT:
            return f"the beat restages segment {seg['index']} (\"{seg.get('beat')}\"); choose a materially different consequence"
    budget = int(run["settings"].get("max_scene_segments") or 8)
    if scene_length(done) >= budget and not data["scene_change"]:
        return f"the scene {previous.get('scene')!r} has used its {budget}-segment budget: set scene_change to true and move on"
    if stuck and (not data["scene_change"] or str(data["scene"]).strip().lower() == str(previous.get("scene") or "").strip().lower()):
        return "the film is circling: set scene_change to true with a different scene name"
    return ""


# ---------- render ----------

def pictured(run: dict, index: int) -> list[dict]:
    """The cast with a reference picture in segment ``index``."""

    active = cast_at(run, index)
    return [c for c in run["seed"].get("characters") or [] if c.get("image") and on_disk(c["image"])
            and (c["name"] in active or not c.get("look"))
            and (c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6)][:9]


def references_for(run: dict, index: int = 1) -> list[str]:
    return [c["image"] for c in pictured(run, index)]


def mode_of(run: dict, seg: dict) -> str:
    """How a segment opens: on the segment before (``previous``), the cast's pictures (``references``), the opening
    picture (``image``) or from text."""

    if seg.get("keyframe") and on_disk(seg["keyframe"]):
        return "keyframe"  # the piece's own keyframe, made and checked first
    identity = run["settings"].get("identity", "auto")
    pictures = bool(references_for(run, seg["index"])) and identity != "frame" and ref2va()
    if (beat_of(run, seg["index"]) or {}).get("image"):
        return "image"  # the user's own opening picture for this beat
    if seg["index"] > 1 and not seg["scene_change"]:
        return "previous"
    if pictures:
        return "references"
    if seg["index"] == 1 and run["seed"].get("opening_image"):
        return "image"
    return "text"


def wardrobe_at(run: dict, name: str, index: int) -> str:
    """The exact wardrobe line of ``name`` in segment ``index``: the seed sheet's for its segments (continuity.wardrobe_line)."""

    for c in run["seed"].get("characters") or []:
        if c.get("name") == name and (c.get("from") or 1) <= index <= (c.get("to") or 10 ** 6):
            return continuity.wardrobe_line(c)
    return ""


def look_lines(run: dict, fields: dict, index: int) -> str:
    """The pinned looks of the cast this segment names, said every time (the renderer has no memory), each with its
    exact wardrobe line again: "the same clothes in every shot"."""

    text = " ".join(str(fields.get(k) or "") for k in ("setting", "action")).lower()
    out = []
    for name, look in cast_at(run, index).items():
        if look and name.lower().split()[0] in text:
            out.append(f"{name} is {look.rstrip('.')}.")
            wear = wardrobe_at(run, name, index)
            if wear:
                out.append(f"{name} wears exactly this in every shot: {wear.rstrip('.')}.")
    return " ".join(out)


def prompt_of(run: dict, seg: dict, mode: str) -> str:
    s, f = run["settings"], dict(seg["fields"])
    pace = checks.pace_of(seg.get("pace"))
    paced = checks.pace_line(pace, float(seg.get("seconds") or 10))
    if paced:
        # a fight starts in motion: the speed is the first thing the action says, the plan's pace after it
        f["action"] = (checks.pace_opening(pace) + " " + f.get("action", "").strip() + " " + paced).strip()
    style = style_at(s, seg["index"])  # only this segment's era or setting: a style for every era bled Boston into the sea
    music = f.get("music") or "N/A"
    if s.get("music") and music.upper() == "N/A":
        music = s["music"]
    dialogue = [{"speaker": d.get("speaker") or "", "voice": d.get("voice") or "a natural voice", "text": str(d["text"]).strip(), "language": "English"}
                for d in f.get("dialogue") or []]
    if mode == "references":
        names = [c["name"] for c in pictured(run, seg["index"])]
        for line in dialogue:
            line["subject"] = names.index(line["speaker"]) + 1 if line["speaker"] in names else 1
        wear = " ".join(f"{n} wears exactly this in every shot: {wardrobe_at(run, n, seg['index']).rstrip('.')}." for n in names if wardrobe_at(run, n, seg["index"]))
        fields = {"scene": " ".join(bit for bit in (style, wear, f.get("setting", "")) if bit), "camera": f.get("camera", ""),
                  "action": f.get("action", ""), "dialogue": dialogue, "soundscape": f.get("soundscape", ""), "music": music,
                  "subjects": names, "people": True}
        return compose_references(fields, len(names))
    scene = " ".join(bit for bit in (style, look_lines(run, f, seg["index"]), f.get("setting", "")) if bit)
    fields = {"scene": scene, "camera": f.get("camera", ""), "action": f.get("action", ""), "dialogue": dialogue,
              "soundscape": f.get("soundscape", ""), "music": music}
    return compose(fields, mode in ("previous", "image", "keyframe"), frames=seg["frames"])


def render(run: dict, seg: dict) -> None:
    s = run["settings"]
    done = committed(run)
    mode = mode_of(run, seg)
    take = len(seg["takes"]) + 1
    output = f"{run['id']}/segments/{seg['id']}/take{take}.mp4"
    prompt = prompt_of(run, seg, mode)
    params = {"prompt": prompt, "preset": s["preset"], "frames": seg["frames"], "seed": int(seg["seed"]), "output": output,
              "qa": False,  # the gate runs here, with the watcher, once per take (check)
              "brain": s.get("brain") or {}, "autopilot": {"run": run["id"], "segment": seg["index"]}}
    if s["preset"] == "custom" and s.get("width") and s.get("height"):
        params.update(width=int(s["width"]), height=int(s["height"]))
    elif s["preset"] == "sharp":
        # "go sharper": 1376x768 on the governor's sharp schedule (the brain is evicted for it)
        params.update(width=PRESET_SIZES["sharp"][0], height=PRESET_SIZES["sharp"][1], schedule="sharp")
    if mode == "references":
        params["references"] = references_for(run, seg["index"])
        if s.get("ref_turbo") and ref2va_turbo():
            params["ref_turbo"] = True
    else:
        params["quality"] = s["quality"]
        if s.get("loras"):
            params["loras"] = s["loras"]
    if mode == "previous":
        params["first_frame_from"] = done[-1]["output"]
    if mode == "keyframe":
        params["image"] = seg["keyframe"]
    if mode == "image":
        params["image"] = (beat_of(run, seg["index"]) or {}).get("image") or run["seed"]["opening_image"]
    fitted = fitting(run, params)
    if fitted != params["frames"]:
        shorten(run, seg, "memory first: at this size only a shorter piece fits this machine now", fitted)
        params["frames"] = seg["frames"]
        params["prompt"] = prompt = prompt_of(run, seg, mode)
    while True:
        try:
            job = submit({"kind": "video", "title": f"AutoPilot · {run['name']} · {seg['index']}: {seg['beat'][:60]}", "params": params})
            break
        except Bad as error:
            shorter = re.search(r"What fits: (\d+) frames", str(error))
            if not shorter or not shorten(run, seg, "the queue's memory check: " + str(error)[:200], int(shorter[1])):
                raise Failed(f"the queue refused segment {seg['index']}: {error}") from None
            params["frames"] = seg["frames"]
            params["prompt"] = prompt = prompt_of(run, seg, mode)
    seg.update(status="queued", job=job["id"], mode=mode, prompt=prompt, output=output,
               settings={k: params.get(k) for k in ("preset", "quality", "frames", "width", "height", "loras", "ref_turbo") if params.get(k) is not None})
    say(run, f"Segment {seg['index']}: rendering" + (f", take {take} at seed {seg['seed']}" if take > 1 else "")
        + {"previous": ", from the last frame of the segment before", "references": ", from the cast's pictures (a new scene)",
           "image": ", from the opening picture", "keyframe": ", from its keyframe", "text": ", from text (a new scene)"}[mode] + ".", stage="render", segment=seg["index"], job=job["id"])


def follow(run: dict, seg: dict, stop: threading.Event) -> None:
    while True:
        job = job_of(seg.get("job") or "")
        if job is None:  # the app restarted: the clip is there, or the take is queued again
            seg["status"] = "rendered" if on_disk(seg.get("output")) else "planned"
            save(run)
            return
        status = job.get("status")
        if status == "done":
            seg["output"] = (job.get("outputs") or [seg["output"]])[0]
            seg["status"] = "rendered" if on_disk(seg["output"]) else "failed"
            seg["error"] = "" if seg["status"] == "rendered" else "the job finished without its clip"
            save(run)
            return
        if status == "cancelled":
            seg["status"], seg["job"] = "planned", None
            save(run)
            raise Stopped()
        if status == "failed":
            error = job.get("error") or "the render failed"
            stopped = memguard.LAST_STOP
            if stopped.get("kind") == "thermal" and time.time() - float(stopped.get("at") or 0) < 900 and int(seg.get("thermal_retries") or 0) < 1:
                # the thermal guard stopped this clip: once more, after the render gate has let the GPU cool
                seg["thermal_retries"] = int(seg.get("thermal_retries") or 0) + 1
                seg["status"], seg["job"] = "planned", None
                say(run, f"Segment {seg['index']}: stopped by the thermal guard ({stopped.get('why')}); made again once it has cooled.", segment=seg["index"])
                return
            if MEMORY_STOP.search(error) and shorten(run, seg, "the render was stopped for memory: " + clip_words(error, 40)):
                seg["status"], seg["job"] = "planned", None  # the same piece again, shorter, at the same size
                save(run)
                return
            seg["status"], seg["error"] = "failed", error
            save(run)
            return
        if run.get("redo"):
            return  # "redo that": the segment loop cancels this take and makes it again
        wait(stop)


# ---------- memory first ----------

# what a render that ran out of room says (spark_generate.py, memory_guard.py, the queue)
MEMORY_STOP = re.compile(r"memory|protect the machine|refused before anything was loaded|out of memory|\bOOM\b|NV_ERR_NO_MEMORY|killed", re.I)


def shortest(run: dict) -> int:
    return lengths.nearest(float(run["settings"].get("min_seconds") or DEFAULTS["min_seconds"]))


def fitting(run: dict, params: dict) -> int:
    """The longest length, up to the piece's own, that fits this machine's memory now at the piece's size (the memory
    guard's estimate with what is available now); the piece's own when it fits, or when nothing shorter does (the
    queue then says why)."""

    try:
        said = room(params)
    except Exception:  # noqa: BLE001 - no estimate: the queue and the render's own checks still stand
        said = None
    if not said or said.get("fits", True):
        return params["frames"]
    shorter = (said.get("nearest") or {}).get("frames")
    return shorter if shorter and shortest(run) <= shorter < params["frames"] else params["frames"]


def shorten(run: dict, seg: dict, why: str, frames: int | None = None) -> bool:
    """The piece made shorter at the same size: to ``frames``, or the next length the model takes (the memory guard
    stopped it). False when it is at the run's shortest already or has been shortened MEMORY_RETRIES times."""

    events = seg.setdefault("memory", [])
    if sum(1 for e in events if e.get("retry")) >= MEMORY_RETRIES:
        return False
    counts = [c for c in lengths.counts() if shortest(run) <= c < int(seg["frames"])]
    # a stop by the guard: a fifth shorter (one step of 17 frames is 0.7 s, too little to change the peak much)
    goal = frames or int(int(seg["frames"]) * 0.8)
    target = max([c for c in counts if c <= goal], default=None) or (min(counts) if counts and not frames else None)
    if not target:
        return False
    events.append({"at": now(), "from": seg["frames"], "to": target, "why": why[:300], "retry": not why.startswith("memory first")})
    ratio = lengths.seconds_of(target) / max(0.1, float(seg.get("seconds") or lengths.seconds_of(int(seg["frames"]))))
    for beat in (seg.get("pace") or {}).get("beats") or []:  # the pace plan's seconds, scaled to the new length
        beat["from"], beat["to"] = round(beat["from"] * ratio, 2), round(beat["to"] * ratio, 2)
    seg["frames"], seg["seconds"] = target, lengths.seconds_of(target)
    say(run, f"Segment {seg['index']}: {why[:160]}; now {lengths.seconds_of(target):.1f} s ({target} frames) at the same size.", segment=seg["index"])
    return True


# ---------- keyframes ----------

def render_keyframe(run: dict, seg: dict) -> None:
    """The piece's opening still: the Director's keyframe prompt, with this segment's style and the pinned looks of
    whoever it names, as one picture of the queue (a ``scout`` job of one) at the video's size."""

    s, n = run["settings"], len(seg.get("keyframes") or []) + 1
    round_ = len(seg.get("takes") or [])  # the keyframes of this video take (a rewritten shot gets new ones)
    looks = look_lines(run, {"setting": seg.get("keyframe_prompt", ""), "action": seg["fields"].get("action", "")}, seg["index"])
    prompt = " ".join(bit for bit in (style_at(s, seg["index"]), looks, seg.get("keyframe_prompt") or seg["fields"].get("setting", ""),
                                      qc.start_line(seg.get("qc") or {})) if bit)
    params = {"prompt": prompt, "count": 1, "seed": int(seg["seed"]) + 7919 * (n - 1), "preset": s["preset"], "autopilot": {"run": run["id"], "segment": seg["index"], "keyframe": n},
              **({"image_model": s["keyframe_model"]} if s.get("keyframe_model") else {})}
    if s["preset"] == "custom" and s.get("width") and s.get("height"):
        params.update(width=int(s["width"]) // 16 * 16, height=int(s["height"]) // 16 * 16)
    kind = "scout"
    named = " ".join((seg.get("keyframe_prompt") or "", seg["fields"].get("action", ""), seg["fields"].get("setting", ""))).lower()
    portraits = [c for c in pictured(run, seg["index"]) if c["name"].lower().split()[0] in named][:3]
    anchor = story.anchor_of(run, seg["index"]) if s.get("location_anchor", True) else None
    anchor = anchor if anchor and on_disk(anchor["image"]) else None
    size = PRESET_SIZES.get(s["preset"]) or ((int(s["width"]) // 16 * 16, int(s["height"]) // 16 * 16) if s.get("width") and s.get("height") else None)
    if portraits or anchor:
        # the place's first keyframe and the era's reference portraits are the pictures the keyframe is drawn beside:
        # the building, face, hair and clothes come from the pictures, not from words the picture model reads its own way
        kind = "edit"
        pictures = ([{"image": anchor["image"], "what": "the place"}] if anchor else []) + [{"image": c["image"], "what": c["name"]} for c in portraits]
        who = "; ".join(f"image {i + 1} is {one['what']}" for i, one in enumerate(pictures))
        keep = []
        if anchor:
            keep.append("Keep the place of image 1: the same building, its shape, materials, colours and surroundings (seen "
                        "from the angle this still needs, closer, or from inside it), in this still's own light and weather")
        if portraits:
            keep.append("Keep each person's face, hair, beard and clothes exactly as in their picture")
        params = {"image": pictures[0]["image"], "images": [one["image"] for one in pictures[1:]], "seed": params["seed"],
                  "prompt": f"{who}. {'. '.join(keep)}. Make this film still: " + prompt, "autopilot": params["autopilot"],
                  **({"image_model": s["keyframe_model"]} if s.get("keyframe_model") else {})}
        if size:
            params.update(width=size[0], height=size[1])
    elif size and s["preset"] != "small":
        params.update(width=size[0], height=size[1])  # "go sharper": the keyframe at the video's own size
    try:
        job = submit({"kind": kind, "title": f"AutoPilot · {run['name']} · {seg['index']}: keyframe {n}", "params": params})
    except Bad as error:
        raise Failed(f"the queue refused the keyframe of segment {seg['index']}: {error}") from None
    seg.setdefault("keyframes", []).append({"take": n, "round": round_, "job": job["id"], "prompt": prompt, "seed": params["seed"], "path": None,
                                            "kind": kind, **({"from": params["image"]} if kind == "edit" else {}),
                                            **({"anchor": anchor["from"]} if anchor else {})})
    seg.update(status="keyframe_queued", job=job["id"])
    say(run, f"Segment {seg['index']}: its keyframe" + (f", take {n}" if n > 1 else "") + ".", stage="keyframe", segment=seg["index"], job=job["id"])


def follow_keyframe(run: dict, seg: dict, stop: threading.Event) -> None:
    last = seg["keyframes"][-1]
    while True:
        job = job_of(last["job"])
        if job is None:  # the app restarted: made again
            seg["keyframes"].pop()
            seg["status"] = "keyframe"
            save(run)
            return
        if job.get("status") == "done":
            picture = next((o for o in job.get("outputs") or [] if str(o).endswith(".png")), None)
            picture = str(Path(picture).relative_to(HOME)) if picture and Path(picture).is_absolute() and HOME.resolve() in Path(picture).resolve().parents else picture
            if not on_disk(picture):
                seg["status"], seg["error"] = "failed", "the keyframe job finished without its picture"
            else:
                last["path"], seg["status"] = picture, "keyframe_ready"
            save(run)
            return
        if job.get("status") == "cancelled":
            seg["keyframes"].pop()
            seg["status"], seg["job"] = "keyframe", None
            save(run)
            raise Stopped()
        if job.get("status") == "failed":
            error = job.get("error") or "the keyframe failed"
            tries = seg.setdefault("keyframe_waits", 0)
            if ROOM_NOW.search(error) and tries < KEYFRAME_WAITS:
                # no room this moment (an engine still letting go): drawn again shortly, never a stopped film
                seg["keyframes"].pop()
                seg.update(status="keyframe", job=None, keyframe_waits=tries + 1)
                say(run, f"Segment {seg['index']}: no room for its keyframe yet; trying again shortly.", segment=seg["index"])
                wait(stop, KEYFRAME_WAIT_S)
                return
            seg["status"], seg["error"] = "failed", error
            save(run)
            return
        if run.get("redo"):
            return
        wait(stop)


KEYFRAME_CATEGORIES = ("identity", "wardrobe", "props", "setting", "era", "location", "lighting")


def check_keyframe(run: dict, seg: dict) -> None:
    """The keyframe checked by the eyes against the bible before anything moves: a major finding draws it again
    (the Director rewrites the keyframe prompt from it, at a new seed) while ``retakes`` allow; the take with the
    fewest problems opens the piece otherwise."""

    last = seg["keyframes"][-1]
    opts = continuity.options(run["settings"])
    categories = [c for c in continuity.checked(opts) if c in KEYFRAME_CATEGORIES]
    result = {"findings": [], "categories": categories}
    began = time.time()

    def checklist() -> dict:
        found = {}
        bible = bible_of(run)
        picture = HOME / last["path"]
        prompt = continuity.keyframe_prompt(bible, seg["index"], seg, last["prompt"], categories, list(cast_at(run, seg["index"])))
        try:
            data, _ = ask_json(lambda text: ask_vision(continuity.CHECK_SYSTEM, [("Frame 1, the keyframe", picture)], picture, text,
                                                       run["settings"].get("brain") or {})[0], continuity.check_problem, prompt, "keyframe check")
            checks = continuity.triage(data, 1, categories)
            for category, entry in checks.items():
                entry["category"] = category
                entry["action"] = continuity.act(entry, opts)
            found.update(checks=checks, findings=[e for e in checks.values() if e["action"] != "none"])
        except Failed as error:
            found["skipped"] = str(error)[:300]
        return found

    # the checklist and the yes/no questions at once: the eyes batch them
    with ThreadPoolExecutor(2) as pool:
        listed = pool.submit(checklist) if categories else None
        dense = dense_keyframe(run, seg, HOME / last["path"], run_dir(run["id"]) / "segments" / seg["id"])
        if listed is not None:
            result.update(listed.result())
    # the self-check: the keyframe shows the shot's start, never its outcome (a lamp already lit before the match)
    started = qc_keyframe(run, seg, HOME / last["path"], run_dir(run["id"]) / "segments" / seg["id"], last["take"])
    result["seconds"] = round(time.time() - began, 1)
    result["findings"] = (result.get("findings") or []) + dense["findings"] + started
    result["dense"] = {"stats": dense["stats"], "log": dense["log"]}
    add_stats(run, seg, dense["stats"])
    last["continuity"] = result
    # drawn again only for the self-check's confident, visible failure (a lamp already lit); the checklist's findings are notes
    majors = [f for f in result["findings"] if f.get("source") == "qc" and f.get("retake")]
    this_round = [k for k in seg["keyframes"] if k.get("round", 0) == last.get("round", 0)]
    if majors and len(this_round) <= retakes_now(run):
        rewrite_keyframe(run, seg, majors)
        seg["status"] = "keyframe"
        say(run, f"Segment {seg['index']}: keyframe {last['take']} failed ({'; '.join(f['category'] + ': ' + f['evidence'] for f in majors)[:200]}); drawn again.", segment=seg["index"])
        return
    kept = min(reversed(this_round), key=lambda k: (sum(1 for f in (k.get("continuity") or {}).get("findings") or [] if f["status"] == "major"),
                                                          len((k.get("continuity") or {}).get("findings") or [])))
    seg.update(keyframe=kept["path"], status="planned")
    if run["settings"].get("location_anchor", True) and story.set_anchor(run, seg, kept["path"]):
        say(run, f"Segment {seg['index']}: its keyframe is the reference picture of this place from now on.", segment=seg["index"])
    save(run)


def rewrite_keyframe(run: dict, seg: dict, findings: list) -> None:
    found = "\n".join(f"- {f['category']}: {f['evidence']}" for f in findings)
    prompt = (f"You wrote this keyframe prompt for segment {seg['index']} of a film: {seg.get('keyframe_prompt')}\n"
              f"The picture made from it failed its continuity check:\n{found}\n"
              f"The continuity bible (canonical):\n{continuity.director_text(bible_of(run), seg['index'], list(cast_at(run, seg['index'])))}\n"
              'Write the keyframe prompt again so that each of these comes out right. Answer: {"keyframe": "<60 words max>"}')
    try:
        data, _ = ask_json(director_asker(run), lambda d: "" if str(d.get("keyframe") or "").strip() else '"keyframe" is text', prompt, "Director")
        seg["keyframe_prompt"] = clip_words(data["keyframe"], 70)
    except Failed:
        pass  # the same prompt again, at the next keyframe seed


# ---------- check and watch ----------

def sample(clip: Path, folder: Path, count: int) -> tuple[list[tuple[str, Path]], Path, float]:
    """``count`` frames evenly through the clip and its last frame, 448x256 PNG, with captions; and one sheet of them."""

    length = duration(clip)
    folder.mkdir(parents=True, exist_ok=True)
    times = [round(i * max(length - 0.1, 0) / max(count - 1, 1), 2) for i in range(count)]
    frames = []
    for i, at in enumerate(times):
        target = folder / f"f{i + 1:02d}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(clip), "-frames:v", "1", "-vf",
                        f"scale={FRAME[0]}:{FRAME[1]}", str(target)], check=True, env=ENV, timeout=60)
        frames.append((f"Frame {i + 1}, at {at:.2f} s", target))
    last = folder / "last.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(clip), "-update", "1", "-vf",
                    f"scale={FRAME[0]}:{FRAME[1]}", str(last)], check=True, env=ENV, timeout=60)
    frames.append((f"Final frame, at {length:.2f} s", last))
    sheet = contact_sheet([p for _, p in frames], folder / "sheet.png")
    return frames, sheet, length


def contact_sheet(paths: list, sheet: Path, across: int = 4) -> Path:
    """The pictures in rows of ``across``, left to right then top to bottom, each scaled to one frame's size."""

    paths = list(paths)
    across = max(1, min(across, len(paths)))
    while len(paths) % across:
        paths.append(paths[-1])
    command = ["ffmpeg", "-v", "error", "-y"]
    for p in paths:
        command += ["-i", str(p)]
    rows = len(paths) // across
    size = f"scale={FRAME[0]}:{FRAME[1]}:force_original_aspect_ratio=decrease,pad={FRAME[0]}:{FRAME[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    graph = "".join(f"[{i}:v]{size}[p{i}];" for i in range(len(paths)))
    graph += "".join(f"{''.join(f'[p{r * across + c}]' for c in range(across))}" + (f"hstack={across}" if across > 1 else "copy") + f"[r{r}];" for r in range(rows))
    graph += "".join(f"[r{r}]" for r in range(rows)) + (f"vstack={rows}[out]" if rows > 1 else "copy[out]")
    subprocess.run(command + ["-filter_complex", graph, "-map", "[out]", "-frames:v", "1", str(sheet)], check=True, env=ENV, timeout=120)
    return sheet


def gate(clip: Path, prompt: str, sheet: Path | None = None, brain: dict | None = None) -> dict:
    """The QA gate's line check, heard with Parakeet (or Whisper where Parakeet is not installed): {"passed", "audio",
    "identity"}. The gate's single identity verdict is left to the watcher and the yes/no checks of every window."""

    audio = ears.line_check(clip, prompt) if ears is not None else {"result": "skipped", "note": "no ears are set up"}
    identity = {"result": "skipped", "note": "the identity is asked as yes/no questions in every window (the dense checks)"}
    return {"audio": audio, "identity": identity, "passed": audio.get("result") != "fail"}


def eyes_warm() -> None:
    """The eyes loaded ahead of their first question (a cold Gemma on a Spark takes about a minute)."""

    if eyes is not None and hasattr(eyes, "warm"):
        eyes.warm()


def ears_ready() -> dict:
    """What the ears can do here: {"runs", "transcript"} (qa.ready's shape)."""

    said = ears.ready() if ears is not None else {}
    return {"runs": bool(said.get("transcript")), "transcript": bool(said.get("transcript")), "by": said.get("by")}


def check_watch(data: dict) -> str:
    for key in ("add_nodes", "update_nodes", "add_events", "continuity"):
        if data.get(key) is not None and not isinstance(data.get(key), list):
            return f'"{key}" is a list'
    if not isinstance(data.get("identity_drift"), bool):
        # read leniently: "true"/"false" as words, and left out as false (the identity questions of every window and
        # the drift confirm look again). On the gate's Infinity run Cosmos3-Edge left it out three times in a row
        word = str(data.get("identity_drift") or "").strip().lower()
        data["identity_drift"] = word in ("true", "yes", "1")
    if not str(data.get("ending") or "").strip():
        return '"ending" says what the final frame shows'
    return ""


DRIFT_SYSTEM = "You compare people across frames of one video. Answer only with JSON."


def confirm_drift(run: dict, frames: list, watched: dict, index: int) -> tuple[bool, dict]:
    """A second, narrower look before identity drift fails a take: the first, middle and last frames, and one question
    (is it the same person). Measured on the first real run: Cosmos3-Edge's scene-graph answer called a courier who
    got on her vehicle "identity drift"; asked only about the face, hair and age it answers the question asked."""

    names = "; ".join(f"{n}: {look}" for n, look in cast_at(run, index).items()) or "the main character"
    pick = [frames[0], frames[len(frames) // 2], frames[-1]]
    prompt = (f"The three images are frames of one short video clip, in time order. The pinned looks: {names}.\n"
              "Look only at the main person's FACE, HAIR, AGE and BUILD. Clothing, props, vehicles, pose, distance, motion blur "
              "and lighting do not matter. Is it clearly the same person in all three frames where the person is visible?\n"
              'Answer: {"same": true | false, "why": "<15 words max>"}')
    try:
        data, _ = ask_json(lambda text: ask_vision(DRIFT_SYSTEM, pick, Path(pick[-1][1]), text, run["settings"].get("brain") or {})[0],
                           lambda d: "" if isinstance(d.get("same"), bool) else '"same" is true or false', prompt, "drift check")
    except Failed as error:
        return True, {"error": str(error)[:200]}  # the watcher's word stands when the check cannot be made
    return not data["same"], {"same": data["same"], "why": clip_words(data.get("why"), 20)}


def check(run: dict, seg: dict) -> None:
    """The take judged: the QA gate (Whisper line check, identity sheet), the watcher (scene-graph delta, drift) and
    the continuity check against the bible. A take that fails is rendered again while ``retakes`` allow: at the next
    seed, and, when the continuity check or the drift check found it, with the shot rewritten by the Director from the
    findings. When they run out the best take is kept, the reasons beside it."""

    if judge is not None:
        return check_d1(run, seg)
    s = run["settings"]
    clip = HOME / seg["output"]
    folder = run_dir(run["id"]) / "segments" / seg["id"]
    take = len(seg["takes"]) + 1
    say(run, f"Segment {seg['index']}: checking and watching take {take}.", stage="watch", segment=seg["index"])
    began = time.time()
    pool = ThreadPoolExecutor(max(3, int(s.get("eyes_parallel") or 1)))
    warming = pool.submit(eyes_warm)  # the eyes load (on a Spark they were out for the render) while the ears listen
    pool.submit(lambda: director_warm())  # and the Director's brain, out since the keyframe, for the next piece
    report = {"passed": None, "skipped": True, "note": "the QA gate is off for this run"}
    transcript = ""
    if s.get("qa", True) and ears_ready()["runs"]:
        try:
            report = gate(clip, seg.get("prompt") or "")
            transcript = ((report.get("audio") or {}).get("heard") or "").strip()
        except Exception as error:  # noqa: BLE001 - the clip is made; a gate that cannot run does not lose it
            report = {"passed": None, "error": str(error)[-300:]}
    elif ears_ready()["transcript"]:
        try:
            transcript = (ears.transcribe(clip) or "").strip()
        except Exception:  # noqa: BLE001
            transcript = ""
    frames, sheet, length = sample(clip, folder / f"frames-take{take}", EYE_FRAMES)
    memory = memory_of(run)
    graph = memory["graph"]
    view = {"nodes": graph["nodes"], "events": graph["events"][-RECENT_EVENTS:], "chapters": graph.get("chapters") or []}
    prompt = watch_prompt(run, {**seg, "seconds": length}, view, max(0, len(graph["events"]) - RECENT_EVENTS), transcript)
    who = []

    def asker(text: str) -> str:
        answer, by = ask_vision(WATCH_SYSTEM, frames, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    try:
        warming.result()
    except Exception:  # noqa: BLE001 - the first look loads them itself (and says why it cannot)
        pass
    # the watcher, the checklist and the dense windows at once: the eyes batch their requests
    say(run, f"Segment {seg['index']}: the watcher, the continuity checklist and the dense checks of take {take}.", stage="checks", segment=seg["index"])
    listed = pool.submit(check_continuity, run, seg, frames, sheet, transcript, clip, folder, take, length)
    densed = pool.submit(dense_check, run, seg, clip, folder, take, length, bible_of(run))
    try:
        watched, rejected = ask_json(asker, check_watch, prompt, "watcher")
        checked, dense = listed.result(), densed.result()
    finally:
        pool.shutdown(wait=True)
    watched = {"delta": {k: watched.get(k) or [] for k in ("add_nodes", "update_nodes", "add_events")},
               "summary": clip_words(watched.get("summary"), 40), "ending": clip_words(watched.get("ending"), 60),
               # a note that something stayed the same is not a problem (Cosmos3-Edge wrote such notes on the first real run)
               "continuity": [c for c in watched.get("continuity") or [] if isinstance(c, dict) and not continuity.BENIGN.search(str(c.get("note") or ""))][:6],
               "identity_drift": watched["identity_drift"], "by": who[-1] if who else "", "rejected": rejected,
               "sheet": relative(sheet), "frames": [relative(p) for _, p in frames]}
    if watched["identity_drift"]:
        watched["identity_drift"], watched["drift_check"] = confirm_drift(run, frames, watched, seg["index"])
    write_json(folder / f"watch-take{take}.json", watched)
    seg.setdefault("check_seconds", []).append(round(time.time() - began, 1))
    passed = set(dense.get("passed") or [])
    for found in checked.get("findings") or []:
        # the checklist's one verdict against the yes/no answers of every window: a major the windows never saw is a
        # correction for the Director, not a retake (the Viking v2 film's false retakes came from single verdicts)
        if found.get("status") == "major" and found["category"] in passed and found.get("source") != "rule":
            found.update(status="minor", demoted="the yes/no checks of every window saw no problem here")
            found["action"] = continuity.act(found, continuity.options(s))
    if watched["identity_drift"] and "identity" in passed:
        # the same for the watcher's drift (its confirm is three frames and one question): on the first live Infinity
        # run (2026-10-26) it rendered a piece again whose identity questions had passed in every window
        watched.update(identity_drift=False, drift_overruled="the identity questions passed in every window")
        write_json(folder / f"watch-take{take}.json", watched)
    checked.setdefault("findings", []).extend(dense["findings"])
    checked["dense"] = {k: dense.get(k) for k in ("stats", "pace", "windows", "passed", "filtered", "file", "by", "skipped", "arc") if dense.get(k) is not None}
    add_stats(run, seg, dense["stats"])
    reasons, fix = [], []
    audio, identity = report.get("audio") or {}, report.get("identity") or {}
    if audio.get("result") == "fail" and continuity.heard_alike(audio.get("wanted"), audio.get("heard")):
        # a name heard as a word that sounds the same ("Eirik!" as "Eric!") is a minor note, never a retake
        audio["alike"] = True
        if "audio" in (checked.get("categories") or []) or "audio" in continuity.options(s)["categories"]:
            checked.setdefault("findings", []).append({"category": "audio", "status": "minor", "verdict": "minor", "frame": 0, "source": "gate", "action": "note",
                                                       "subject": "", "evidence": f"the line \"{audio.get('wanted')}\" was heard as \"{audio.get('heard')}\": it sounds alike, kept"})
    if audio.get("result") == "fail" and not audio.get("alike"):
        reasons.append(f"audio gate: {audio.get('note', '')}")
    if identity.get("result") == "fail":
        reasons.append(f"identity gate: {identity.get('note', '')}")
        fix.append({"category": continuity.gate_category(identity.get("note")), "frame": 0, "evidence": clip_words(identity.get("note"), 30)})
    if watched["identity_drift"]:
        drift = "; ".join(c.get("note", "") for c in watched["continuity"] if c.get("kind") == "identity")[:200]
        reasons.append("the watcher saw identity drift: " + drift)
        fix.append({"category": "identity", "frame": 0, "evidence": drift or "the main character's face changes into someone else's"})
    for found in checked.get("findings") or []:
        if found.get("action") == "retake":
            reasons.append(f"continuity, {found['category']}: {found['evidence']}"[:240])
            fix.append(found)
    seg["takes"].append({"take": take, "seed": seg["seed"], "output": seg["output"], "job": seg.get("job"), "passed": not reasons,
                         "reasons": reasons, "qa": {k: report.get(k) for k in ("passed", "audio", "identity", "note", "error") if report.get(k) is not None},
                         "watch": watched, "continuity": checked, "transcript": transcript, "fields": copy.deepcopy(seg["fields"]),
                         "beat": seg["beat"], "prompt": seg.get("prompt"), "mode": seg.get("mode")})
    write_json(folder / f"continuity-take{take}.json", checked)
    if reasons and take <= retakes_now(run):
        seg["status"], seg["fix"] = "retake", fix
        say(run, f"Segment {seg['index']}: take {take} failed ({'; '.join(reasons)[:200]}); again at the next seed"
            + (", the shot rewritten from the findings." if fix else "."), segment=seg["index"])
        return
    kept = seg["takes"][-1]
    if reasons and len(seg["takes"]) > 1:
        kept = continuity.best(seg["takes"])
        if kept is not seg["takes"][-1]:
            seg.update(output=kept["output"], job=kept["job"], fields=kept["fields"], beat=kept["beat"], prompt=kept["prompt"], mode=kept["mode"], seed=kept["seed"])
        kept["reasons"] = kept["reasons"] + [f"no take passed: kept take {kept['take']} of {len(seg['takes'])}, the one with the fewest problems"]
    commit(run, seg, kept["watch"], kept["transcript"], kept["reasons"], kept["continuity"])


# ---------- the self-check (gma/ap/qc.py) ----------

def qc_ask(state: str, questions: dict, pictures: list) -> dict:
    """{key: calibrated p(yes)}: d1-2B (the judge, resident) asked about these pictures (gma/ap/qc.py)."""

    images = ["data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode() for p in pictures]
    probs, _, _ = judge.ask(state, questions, images)
    t = float(judge.spec.get("temperature", 1.0))
    return {k: judge_mod.calibrate(v, t) for k, v in probs.items()}


def looker():
    """The self-check's looker: the eyes when they answer with log-probabilities (Gemma 4 12B on vLLM), else d1-2B."""

    seeing = qc_eyes if qc_eyes is not None else eyes
    if seeing is not None and getattr(seeing, "engine", "") == "gemma":
        return qc.Looker(seeing)
    if seeing is not None and getattr(seeing, "engine", "") == "ghost":
        strict = (getattr(seeing, "spec", None) or {}).get("looker") == "strict"
        return (qc.StrictGhostLooker if strict else qc.GhostLooker)(seeing, HOME / ".qc-sheets")
    return qc.D1Looker(qc_ask) if judge is not None else None


def qc_keyframe(run: dict, seg: dict, picture: Path, folder: Path, take: int) -> list:
    """The opening keyframe's start states and people; one that shows the outcome already is drawn again (once)."""

    plan, look = seg.get("qc") or {}, looker()
    if not (plan.get("start") or plan.get("people")) or look is None:
        return []
    try:
        results = qc.keyframe_checks(look, plan, picture)
    except Exception as error:  # noqa: BLE001 - a check that cannot run never stops the piece; it is recorded
        write_json(folder / f"qc-keyframe{take}.json", {"error": str(error)[:300]})
        return []
    found = qc.failures(results, {})
    write_json(folder / f"qc-keyframe{take}.json", {"qc": plan, "checks": results, "seen": qc.seen(results),
                                                   "failed": [f["check"] for f in found]})
    if found:
        say(run, f"Segment {seg['index']}: self-check on keyframe {take}: "
            + "; ".join(f"{f['check']} ({'redraw' if f['retake'] else 'note'})" for f in found)[:160] + ".", segment=seg["index"])
    return found


def previous_last(run: dict, seg: dict, folder: Path) -> Path | None:
    """The kept last frame of the piece before (the frame this piece continues), at full size."""

    before = next((x for x in committed(run) if x["index"] == seg["index"] - 1), None)
    if seg.get("scene_change") or not before or not on_disk(before.get("output")):
        return None
    target = folder / "seam-before.png"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(HOME / before["output"]), "-update", "1",
                        str(target)], check=True, env=ENV, timeout=60, capture_output=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return target if target.is_file() else None


def qc_take(run: dict, seg: dict, clip: Path, folder: Path, take: int, frames: list, judge_cut: float | None = None) -> dict:
    """The take's self-check (gma/ap/qc.py): the seam into it, the describer's account, the shot's states, place,
    direction, contact and people, and hard cuts (numeric). Written to qc-take<N>.json; a failure is a retake only
    when it is confident and visible, otherwise a note for the next piece."""

    plan, look = seg.get("qc") or qc.clean_qc(None), looker()
    began = time.time()
    record = {"qc": plan, "cuts": qc.hard_cuts(clip), "checks": [], "looker": type(look).__name__ if look else None}
    if frames:
        before = previous_last(run, seg, folder)
        try:
            if before is not None:
                record["checks"].append(qc.seam(look, before, frames[0][1]))
            if look is not None:
                opens_on = before if before is not None else (HOME / seg["keyframe"] if seg.get("keyframe") else None)
                record["checks"] += qc.take_checks(look, plan, frames, seg.get("carried") or [], opens_on)
                three = [("First frame", frames[0][1]), ("Middle frame", frames[len(frames) // 2][1]), ("Last frame", frames[-1][1])]
                record["described"] = look.describe(three)
        except Exception as error:  # noqa: BLE001 - the numeric checks still stand
            record["error"] = str(error)[:300]
    qc.inherit(record["checks"])
    record["seen"] = qc.seen(record["checks"])
    record["findings"] = qc.failures(record["checks"], record["cuts"], judge_cut)
    for f in record["findings"]:
        f["action"] = "retake" if f["retake"] else "note"  # continuity.score counts the retakes when the best take is kept
    record["notes"] = qc.notes_of(record)
    record["seconds"] = round(time.time() - began, 1)
    write_json(folder / f"qc-take{take}.json", record)
    return record


# ---------- the d1-2B judge (gma/ap/judge.py) ----------

def planned_watch(run: dict, seg: dict) -> dict:
    """What the memory records of a take judged by d1 (which answers yes/no, it does not describe): the plan itself,
    the cast the shot names as nodes and its beat as the one event, said to be the plan and not a description."""

    fields = seg.get("fields") or {}
    text = " ".join(str(fields.get(k) or "") for k in ("setting", "action")).lower()
    named = [n for n in cast_at(run, seg["index"]) if n and n.lower().split()[0] in text][:3]
    action = clip_words(fields.get("action") or seg.get("beat"), 12)
    nodes = [{"id": re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") or "hero", "type": "character", "label": clip_words(n, 3), "state": action} for n in named]
    event = {"subject": nodes[0]["id"] if nodes else "", "action": clip_words(seg.get("beat"), 8)}
    return {"delta": {"add_nodes": nodes, "update_nodes": [], "add_events": [event] if event["action"] else []},
            "summary": clip_words(seg.get("beat"), 40), "ending": clip_words(fields.get("action"), 60), "continuity": [],
            "identity_drift": False, "by": "the plan (d1 judges, it does not describe)"}


def check_d1(run: dict, seg: dict) -> None:
    """The take judged by d1-2B alone (judge.py): the QA gate's line check as before, then eight frames and every
    question in one pass; a fault, or doubt about the person, the cut or the clothes, renders it again at the next
    seed with the shot rewritten from the findings, while ``retakes`` allow. The best take is kept when they run out.
    Every question's probability goes into the take's record and the run report (continuity-take<N>.json)."""

    s = run["settings"]
    clip = HOME / seg["output"]
    folder = run_dir(run["id"]) / "segments" / seg["id"]
    take = len(seg["takes"]) + 1
    say(run, f"Segment {seg['index']}: d1-2B judges take {take}.", stage="watch", segment=seg["index"])
    began = time.time()
    pool = ThreadPoolExecutor(2)
    pool.submit(lambda: director_warm())  # the Director's brain, out since the keyframe, for the next piece
    report = {"passed": None, "skipped": True, "note": "the QA gate is off for this run"}
    transcript = ""
    try:
        if s.get("qa", True) and ears_ready()["runs"]:
            try:
                report = gate(clip, seg.get("prompt") or "")
                transcript = ((report.get("audio") or {}).get("heard") or "").strip()
            except Exception as error:  # noqa: BLE001 - the clip is made; a gate that cannot run does not lose it
                report = {"passed": None, "error": str(error)[-300:]}
        frames, sheet, length = sample(clip, folder / f"frames-take{take}", EYE_FRAMES)  # the record's frames (joins, film pass)
        fields = seg.get("fields") or {}
        bible, index = bible_of(run), seg["index"]
        hero = next(iter(cast_at(run, index)), "")
        sid = continuity.subject_of(bible, hero, "character", create=False) if hero else None
        wear = next((f["value"] for f in continuity.current(bible, index, sid) if f["category"] == "wardrobe"), "") if sid else ""
        asked = judge_mod.context_questions(seg.get("beat"), fields.get("setting"), hero, wear or cast_at(run, index).get(hero, ""))
        verdict = judge.judge(clip, folder / f"judge-take{take}", length, asked, ENV)
    finally:
        pool.shutdown(wait=True)
    found = judge_mod.findings(verdict)
    judge_cut = ((verdict.get("questions") or {}).get("cut") or {}).get("p")
    checked_qc = qc_take(run, seg, clip, folder, take, frames, judge_cut)
    watched = {**planned_watch(run, seg), "sheet": relative(sheet), "frames": [relative(p) for _, p in frames]}
    if checked_qc.get("described"):
        # what the describer saw, not the plan: the record the next Director reads
        watched.update(summary=clip_words(checked_qc["described"], 80), seen=checked_qc["seen"], by="the eyes (the self-check's describer)")
    write_json(folder / f"watch-take{take}.json", watched)
    checked = {"by": verdict.get("by"), "categories": sorted({f["category"] for f in found + checked_qc["findings"]}),
               "findings": found + checked_qc["findings"], "judge": verdict,
               "qc": {k: checked_qc.get(k) for k in ("checks", "cuts", "seconds", "described", "notes", "looker")}}
    if verdict.get("error"):
        checked["skipped"] = "d1-2B could not judge it: " + verdict["error"]
    write_json(folder / f"continuity-take{take}.json", checked)
    seg.setdefault("check_seconds", []).append(round(time.time() - began, 1))
    add_stats(run, seg, {"requests": verdict.get("passes", 0), "questions": len(verdict.get("questions") or {}),
                         "failed_answers": len(found), "seconds": round((verdict.get("ms") or 0) / 1000, 2)})
    reasons, fix = [], []
    audio = report.get("audio") or {}
    if audio.get("result") == "fail" and continuity.heard_alike(audio.get("wanted"), audio.get("heard")):
        audio["alike"] = True
    if audio.get("result") == "fail" and not audio.get("alike"):
        reasons.append(f"audio gate: {audio.get('note', '')}")
    # a retake only for a confident, visible failure: d1-2B's faults on the person or a cut (not its doubts), and the
    # self-check's confident ones; everything else is a note for the Director's next piece
    sure = set(verdict.get("faults") or []) & {"person", "cut"}
    for f in found:
        if f.get("key") in sure:
            reasons.append(f"d1-2B, {f['category']}: {f['evidence']}"[:240])
            fix.append(f)
        else:
            f["action"] = "note"
    for f in checked_qc["findings"]:
        if f["retake"]:
            reasons.append(f"self-check, {f['check']}: {f['evidence']}"[:240])
            fix.append(f)
    notes = checked_qc["notes"] + [f"d1-2B, {f['category']}: {f['evidence']}" for f in found if f not in fix]
    seg["takes"].append({"take": take, "seed": seg["seed"], "output": seg["output"], "job": seg.get("job"), "passed": not reasons,
                         "reasons": reasons, "qa": {k: report.get(k) for k in ("passed", "audio", "identity", "note", "error") if report.get(k) is not None},
                         "watch": watched, "continuity": checked, "judge": verdict, "notes": notes, "transcript": transcript, "fields": copy.deepcopy(seg["fields"]),
                         "beat": seg["beat"], "prompt": seg.get("prompt"), "mode": seg.get("mode")})
    said = ", ".join(f"{k} {q['p']:.2f}" for k, q in (verdict.get("questions") or {}).items())
    say(run, f"Segment {seg['index']}: d1-2B on take {take}: " + (said or verdict.get("error", "no answer"))[:220]
        + (f" ({verdict.get('ms', 0):.0f} ms)" if verdict.get("ms") else "") + ".", segment=seg["index"])
    if reasons and take <= retakes_now(run):
        seg["status"], seg["fix"] = "retake", fix
        say(run, f"Segment {seg['index']}: take {take} failed ({'; '.join(reasons)[:200]}); again at the next seed"
            + (", the shot rewritten from the findings." if fix else "."), segment=seg["index"])
        return
    kept = seg["takes"][-1]
    if reasons and len(seg["takes"]) > 1:
        kept = continuity.best(seg["takes"])
        if kept is not seg["takes"][-1]:
            seg.update(output=kept["output"], job=kept["job"], fields=kept["fields"], beat=kept["beat"], prompt=kept["prompt"], mode=kept["mode"], seed=kept["seed"])
        kept["reasons"] = kept["reasons"] + [f"no take passed: kept take {kept['take']} of {len(seg['takes'])}, the one with the fewest problems"]
    seg["verified_end"] = qc.verified_end((kept.get("continuity") or {}).get("qc") or {})  # the next piece opens so
    seg["qc_notes"] = (kept.get("notes") or [])[:8]
    seg["described"] = ((kept.get("continuity") or {}).get("qc") or {}).get("described", "")
    seg["qc_checks"] = ((kept.get("continuity") or {}).get("qc") or {}).get("checks") or []
    commit(run, seg, kept["watch"], kept["transcript"], kept["reasons"], kept["continuity"])


# ---------- the continuity check ----------

def check_continuity(run: dict, seg: dict, frames: list, sheet: Path, transcript: str, clip: Path, folder: Path, take: int,
                     length: float) -> dict:
    """The take against the continuity bible: one checklist question to the eyes (the frames, the transcript, the
    bible's slice, a category a line), the answer triaged (continuity.triage), each major finding looked at again
    closer (second_look), and the 180-degree rule measured on a continuation. A check that cannot be made does not
    lose the take: it is marked skipped."""

    opts = continuity.options(run["settings"])
    categories = continuity.checked(opts)
    if not categories:
        return {"skipped": "no category is checked in this run's settings", "categories": [], "findings": []}
    bible = bible_of(run)
    index, names = seg["index"], list(cast_at(run, seg["index"]))
    continuation = seg.get("mode") == "previous"
    start = float(seg.get("start") or 0)
    prompt = continuity.check_prompt(bible, index, seg, len(frames), transcript, categories, f"{mmss(start)} to {mmss(start + length)}",
                                     names, continuation, style_at(run["settings"], index))
    who = []

    def asker(text: str) -> str:
        answer, by = ask_vision(continuity.CHECK_SYSTEM, frames, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    try:
        data, rejected = ask_json(asker, continuity.check_problem, prompt, "continuity check")
    except Failed as error:
        return {"skipped": str(error)[:300], "categories": categories, "findings": []}
    checks = continuity.triage(data, len(frames), categories)
    findings = []
    for category, entry in checks.items():
        entry["category"] = category
        if entry["status"] == "major":
            continuity.settle(entry, second_look(run, entry, frames, clip, folder, take, bible, index), opts)
        entry["action"] = continuity.act(entry, opts)
        if entry["action"] != "none":
            findings.append(entry)
    if continuation and "spatial" in categories:
        for entry in continuity.screen_flip(bible, index, data.get("positions")):
            entry["action"] = continuity.act(entry, opts)
            findings.append(entry)
    present = []
    for p in data.get("present") or []:
        try:
            n = int(p.get("frame") or 0) if isinstance(p, dict) else 0
        except (TypeError, ValueError):
            n = 0
        if isinstance(p, dict) and p.get("who"):
            present.append({"who": clip_words(p["who"], 4), "frame": relative(frames[n - 1][1]) if 1 <= n <= len(frames) else ""})
    return {"by": who[-1] if who else "", "strictness": opts["strictness"], "categories": categories, "checks": checks, "findings": findings,
            "facts": [f for f in data.get("facts") or [] if isinstance(f, dict)][:6], "present": present[:9],
            "positions": [p for p in data.get("positions") or [] if isinstance(p, dict)][:9],
            "open_actions": [a for a in data.get("open_actions") or [] if isinstance(a, dict)][:6],
            "light": clip_words(data.get("light"), 14), "rejected": rejected}


# ---------- the dense checks (checks.py) ----------

def window_asker(run: dict, folder: Path, who: list):
    """ask(system, pictures, text) on the vision brain, each request with a sheet of its own pictures for the QA gate's
    eyes when no vision server answers."""

    count = [0]
    lock = threading.Lock()

    def ask(system: str, pictures: list, text: str) -> str:
        with lock:  # the windows ask at once
            count[0] += 1
            mine = count[0]
        try:
            sheet = contact_sheet([p for _, p in pictures], folder / f"sheet-{mine:03d}.png", across=min(4, len(pictures)))
        except (subprocess.SubprocessError, OSError):
            sheet = Path(pictures[-1][1])
        answer, by = ask_vision(system, pictures, sheet, text, run["settings"].get("brain") or {})
        who.append(by)
        return answer
    return ask


def references(run: dict, bible: dict, index: int, seg: dict, folder: Path) -> list:
    """The reference picture of each person the shot names (their cast picture, else their clearest earlier frame)."""

    text = " ".join(str((seg.get("fields") or {}).get(k) or "") for k in ("setting", "action")).lower()
    out = []
    for name in [n for n in cast_at(run, index) if n.lower().split()[0] in text][:2]:
        sid = continuity.subject_of(bible, name, "character", create=False)
        ref = (bible["subjects"].get(sid) or {}).get("ref") if sid else None
        if ref and on_disk(ref):
            out.append((f"Reference picture of {name}", still(HOME / ref, folder / f"ref-{sid}.png", fit=(448, 448))))
    return out


def yes_no(run: dict, asker, pictures: list, header: str, asked: list, opts: dict, log: list, where: dict) -> tuple[list, dict]:
    """One window's questions, each failed major one voted on: the failed questions, as {question, answer}."""

    found = checks.ask_window(asker, pictures, header, asked, extract_json, log, where)
    failed = []
    for q in asked:
        answer = found.get(q["key"])
        if q["key"] == "pace" or not answer or answer == q["expect"]:  # the windows' pace answer is weighed in pace_verdict
            continue
        if q["major"] and opts["votes"] > 1:
            answer = checks.vote(asker, pictures, header, q, answer, opts["votes"], extract_json, log, {**where, "kind": "vote"})
            if not answer or answer == q["expect"]:
                continue
        failed.append({"question": q, "answer": answer})
    return failed, found


def finding_of(q: dict, answer: str, where: str, frame: int, pictures: list) -> dict:
    status = "major" if q["major"] else "minor"
    return {"category": q["category"], "status": status, "verdict": status, "frame": frame, "subject": "", "source": "yes/no",
            "evidence": clip_words(f"{where}: asked \"{q['text']}\", the answer was {answer}", 40), "_pictures": pictures}


def dense_check(run: dict, seg: dict, clip: Path, folder: Path, take: int, length: float, bible: dict) -> dict:
    """The take looked at in windows (checks.py): the bible's yes/no questions a window, the join with the piece before
    on a continuation, and the pace, measured and asked. Failed major questions are voted on, then looked at again
    (second_look). Returns {"findings", "passed" (categories asked and never failed), "log", "stats", "pace"}."""

    s = run["settings"]
    opts, copts = checks.options(s), continuity.options(s)
    categories = continuity.watched(copts)
    if not opts["dense"] or not categories:
        return {"skipped": "the dense checks are off for this run", "findings": [], "passed": [], "log": [], "stats": checks.summary([])}
    index = seg["index"]
    work = folder / f"windows-take{take}"
    work.mkdir(parents=True, exist_ok=True)
    who, log, failed = [], [], []
    asker = window_asker(run, work, who)
    asked_about, failed_about = set(), set()
    refs = references(run, bible, index, seg, work)
    place = place_reference(run, seg, work) if "location" in categories else None
    base = checks.questions(bible, index, seg, cast_at(run, index), categories, continuity)
    pace = checks.pace_of(seg.get("pace"))
    fast = opts["pace"] and "pace" in categories and pace["kind"] in checks.FAST_KINDS
    slow_answers = []
    spans = checks.windows(length, opts["every"], opts["frames"])

    def window(w: int, times: list) -> tuple | None:
        pics = [(f"Frame at {t:.2f} s", work / f"w{w:02d}-{i + 1}.png") for i, t in enumerate(times)]
        pics = [(c, p) for (c, p), t in zip(pics, times) if checks.grab(clip, t, p, FRAME, ENV)]
        if not pics:
            return None
        mid = sum(times) / len(times)
        asked = list(base) + ([checks.pace_question(checks.planned(pace, mid, length) == "slow")] if fast else [])
        # the place against its PICTURE (the first and the last window: the place is set by then, and kept)
        mine = refs + ([place] if place and w in (1, len(spans)) else [])
        if place and w in (1, len(spans)):
            asked.append(story.location_question())
        if not asked:
            return None
        header = (f"Window {w} of {len(spans)} of segment {index}: " + (f"the first {len(mine)} image(s) are reference pictures, then " if mine else "")
                  + f"{len(pics)} frames from {times[0]:.1f} to {times[-1]:.1f} s, in time order.")
        bad, found = yes_no(run, asker, mine + pics, header, asked, opts, log, {"kind": "window", "window": w, "at": times})
        return w, times, pics, asked, bad, found, mid

    # the windows at once: the eyes (vLLM) batch them
    with ThreadPoolExecutor(max(1, int(s.get("eyes_parallel") or 1))) as pool:
        done_windows = [r for r in pool.map(lambda args: window(*args), list(enumerate(spans, 1))) if r]
    for w, times, pics, asked, bad, found, mid in done_windows:
        if fast:
            slow_answers.append((mid, found.get("pace")))
        asked_about |= {q["category"] for q in asked if found.get(q["key"])}
        for b in bad:
            failed_about.add(b["question"]["category"])
            failed.append(finding_of(b["question"], b["answer"], f"window {w} ({times[0]:.1f}-{times[-1]:.1f} s)", len(pics) // 2 + 1, pics))
    arc_found = story_check(run, seg, clip, work, length, asker, opts, log)
    failed += arc_found["findings"]
    done = committed(run)
    previous = seg_last_frame(done[-1]) if done and seg.get("mode") == "previous" else None
    if opts["join"] and previous:
        first = work / "join-first.png"
        if checks.grab(clip, 0.0, first, FRAME, ENV):
            pics = [(f"Picture 1: the last frame of segment {done[-1]['index']}", HOME / previous), ("Picture 2: the first frame of this segment, at 0.00 s", first)]
            asked = checks.join_questions(categories)
            if asked:
                bad, found = yes_no(run, asker, pics, f"The join between segment {done[-1]['index']} and segment {index}: one continuous shot.", asked, opts, log,
                                    {"kind": "join", "window": 0, "at": [0.0]})
                asked_about |= {q["category"] for q in asked if found.get(q["key"])}
                for b in bad:
                    failed_about.add(b["question"]["category"])
                    failed.append(finding_of(b["question"], b["answer"], "the join with the segment before", 2, pics))
    paced = {}
    if fast:
        paced = checks.pace_verdict(pace, checks.motion(clip, ENV), slow_answers, length)
        asked_about.add("pace")
        if paced["status"] in ("major", "minor"):
            failed_about.add("pace")
            failed.append({"category": "pace", "status": paced["status"], "verdict": paced["status"], "frame": 0, "subject": "", "source": "measured",
                           "evidence": paced["evidence"] + ("; the film cuts that lead" if paced["status"] == "minor" else "")})
    # one finding a question (the first window it failed in, with how many windows saw it)
    findings, seen = [], {}
    for f in failed:
        key = (f["category"], f["evidence"].split(": asked", 1)[-1])
        if key in seen:
            seen[key]["windows"] = seen[key].get("windows", 1) + 1
            continue
        seen[key] = f
        findings.append(f)
    for f in findings:
        pictures = f.pop("_pictures", None)
        if f["status"] == "major" and f["category"] != "pace" and pictures:
            continuity.settle(f, second_look(run, f, pictures, clip, work, take, bible, index), copts)
        f["action"] = continuity.act(f, copts)
    log_file = folder / f"checks-take{take}.json"
    write_json(log_file, log)
    return {"findings": [f for f in findings if f["action"] != "none"], "filtered": [f for f in findings if f["action"] == "none"],
            "passed": sorted(asked_about - failed_about), "log": log, "stats": checks.summary(log), "pace": paced,
            "windows": len(spans), "by": who[-1] if who else "", "file": relative(log_file), "arc": arc_found["arc"]}


def story_check(run: dict, seg: dict, clip: Path, folder: Path, length: float, asker, opts: dict, log: list) -> dict:
    """The piece's last frames asked whether it reached the state its beat ends on (the arc), and whether the person's
    new steer shows. A missed end state is a note (the next piece catches up), on the last piece a retake (it must
    reach the payoff); a steer that does not show is a retake while retakes allow. Failed answers are voted on."""

    beat = story.beat_at(run, seg["index"])
    new = [x for x in story.steers_for(run, seg["index"]) if x["kind"] == "note" and x.get("applied_at") == seg["index"]]
    asked = ([story.end_question(beat["end_state"], story.is_final(run, seg["index"]))] if beat else []) + [story.steer_question(x) for x in new[-1:]]
    out = {"findings": [], "arc": None}
    if not asked or not run["settings"].get("arc", True) and not new:
        return out
    times = [round(max(0.0, length - d), 2) for d in (2.0, 1.0, 0.1)]
    pics = [(f"Frame at {t:.2f} s (near the end)", folder / f"end-{i + 1}.png") for i, t in enumerate(times)]
    pics = [(c, p) for (c, p), t in zip(pics, times) if checks.grab(clip, t, p, FRAME, ENV)]
    if not pics:
        return out
    bad, found = yes_no(run, asker, pics, f"The last seconds of segment {seg['index']}, in time order.", asked, opts, log,
                        {"kind": "story", "window": 0, "at": times})
    missed = {b["question"]["key"]: b["answer"] for b in bad}
    if beat:
        out["arc"] = {"end_state": beat["end_state"], "answer": found.get("s1"), "reached": "s1" not in missed and found.get("s1") == "yes",
                      "final": story.is_final(run, seg["index"])}
    for q in asked:
        if q["key"] not in missed:
            continue
        final = q.get("final")
        # a steer that does not show is made again once; after that it is a note the next piece carries (on the phase 4
        # smoke "make it rain" for a piece set in the lantern room failed twice: the arc kept the keeper indoors)
        status = "major" if final or (q.get("steer") and not seg.get("takes")) else "minor"
        what = (f"the last piece did not reach the payoff ({clip_words(beat['end_state'], 25)})" if final and q["key"] == "s1" else
                f"it did not end on its planned state ({clip_words(beat['end_state'], 25)})" if q["key"] == "s1" else
                f"what the person asked for does not show ({q['text']})")
        out["findings"].append({"category": "story", "status": status, "verdict": status, "frame": len(pics), "subject": "",
                                "source": "arc" if q["key"] == "s1" else "steer", "evidence": clip_words(f"{what}; the answer was {missed[q['key']]}", 40)})
    return out


def dense_keyframe(run: dict, seg: dict, picture: Path, folder: Path) -> dict:
    """The keyframe's yes/no questions (one still: who, what they wear, the place, the era), failed majors voted on."""

    s = run["settings"]
    opts, copts = checks.options(s), continuity.options(s)
    categories = [c for c in continuity.watched(copts) if c in KEYFRAME_CATEGORIES]
    if not (opts["dense"] and opts["keyframe"] and categories):
        return {"findings": [], "log": [], "stats": checks.summary([])}
    bible, index = bible_of(run), seg["index"]
    work = folder / f"keyframe-checks-{len(seg.get('keyframes') or [])}"
    work.mkdir(parents=True, exist_ok=True)
    asked = [q for q in checks.questions(bible, index, {**seg, "mode": "keyframe"}, cast_at(run, index), categories, continuity)
             # one still: "the same person in every frame" means something only against a reference picture
             if "between these frames" not in q["text"] and ("in every frame" not in q["text"] or "reference picture" in q["text"])]
    if "pace" in continuity.watched(copts) and checks.pace_of(seg.get("pace"))["kind"] in checks.FAST_KINDS:
        asked.append(checks.MID_MOTION)  # a fight opens on a blow in motion, not a stand-off
    place = place_reference(run, seg, work) if "location" in categories else None
    if place:
        asked.append(story.location_question())  # against the place's PICTURE, not the Director's own setting line
    if not asked:
        return {"findings": [], "log": [], "stats": checks.summary([])}
    refs = references(run, bible, index, seg, work) + ([place] if place else [])
    log = []
    bad, _ = yes_no(run, window_asker(run, work, []), refs + [("The keyframe, the still this segment opens on", picture)],
                    f"The keyframe of segment {index}" + (", after the reference picture(s)." if refs else "."), asked, opts, log, {"kind": "keyframe", "window": 0, "at": [0.0]})
    findings = []
    for b in bad:
        f = finding_of(b["question"], b["answer"], "the keyframe", 1, None)
        f.pop("_pictures", None)
        f["action"] = continuity.act(f, copts)
        findings.append(f)
    return {"findings": findings, "log": log, "stats": checks.summary(log)}


def place_reference(run: dict, seg: dict, folder: Path) -> tuple[str, Path] | None:
    """The place's reference picture for the eyes (the first keyframe of this place), when this piece has one."""

    if not run["settings"].get("location_anchor", True):
        return None
    anchor = story.anchor_of(run, seg["index"])
    if not anchor or not on_disk(anchor["image"]):
        return None
    return (f"Reference picture of the place (segment {anchor['from']}'s keyframe)",
            still(HOME / anchor["image"], folder / f"place-{anchor['from']}.png", fit=(448, 256)))


def add_stats(run: dict, seg: dict, stats: dict) -> None:
    """The checks' counts added to the segment's and the run's (what the page and the report show)."""

    for holder in (seg.setdefault("check_stats", {}), run.setdefault("check_stats", {})):
        for key in ("requests", "questions", "failed_answers", "votes", "seconds"):
            holder[key] = round(holder.get(key, 0) + (stats.get(key) or 0), 1)


def frame_time(caption: str) -> float | None:
    found = re.search(r"at ([0-9.]+) s", caption or "")
    return float(found[1]) if found else None


def still(source: Path, target: Path, at: float | None = None, box=None, fit=(896, 512)) -> Path:
    """One picture for a closer look: the clip's frame at ``at`` at full size (or a picture), cropped to ``box``
    (fractions of the picture) and scaled to fit ``fit``; the source itself when ffmpeg cannot make it."""

    chain = []
    if box:
        x0, y0, x1, y1 = box
        chain.append(f"crop=iw*{x1 - x0:.4f}:ih*{y1 - y0:.4f}:iw*{x0:.4f}:ih*{y0:.4f}")
        fit = (448, 448)
    chain.append(f"scale={fit[0]}:{fit[1]}:force_original_aspect_ratio=decrease")
    command = ["ffmpeg", "-v", "error", "-y"] + (["-ss", f"{at:.3f}"] if at is not None else []) + ["-i", str(source), "-frames:v", "1", "-vf", ",".join(chain), str(target)]
    try:
        subprocess.run(command, check=True, env=ENV, timeout=60, capture_output=True)
        return target if target.is_file() else source
    except (subprocess.SubprocessError, OSError):
        return source


def reference_for(run: dict, entry: dict, bible: dict, index: int, folder: Path, take: int) -> tuple[str, Path] | None:
    """What a second look compares with: the person's reference (their cast picture, else the clearest frame of the
    segment they were first seen in) for identity and wardrobe; the last frame of the segment before otherwise."""

    if entry["category"] in ("identity", "wardrobe"):
        who = entry.get("subject") or next(iter(cast_at(run, index)), "")
        sid = continuity.subject_of(bible, who, "character", create=False) if who else None
        ref = (bible["subjects"].get(sid) or {}).get("ref") if sid else None
        if ref and on_disk(ref):
            return (f"{continuity.name_of(bible, sid)} as established", still(HOME / ref, folder / f"ref-take{take}-{sid}.png", fit=(448, 448)))
    done = committed(run)
    if done and seg_last_frame(done[-1]):
        return (f"the last frame of segment {done[-1]['index']}, the segment before", HOME / seg_last_frame(done[-1]))
    return None


def seg_last_frame(seg: dict) -> str | None:
    frames = (seg.get("watch") or {}).get("frames") or []
    return frames[-1] if frames and on_disk(frames[-1]) else None


SECOND_SYSTEM = continuity.SECOND_SYSTEM


def second_look(run: dict, entry: dict, frames: list, clip: Path, folder: Path, take: int, bible: dict, index: int) -> dict:
    """A major finding looked at again before it counts: the frame it names, closer (cropped to where it is when the
    first look said), beside what it should match, and one question. Measured on the first real run: Cosmos3-Edge
    called a courier who got on her vehicle "identity drift"; asked only about the face it answered the question asked."""

    n = entry["frame"]
    caption, picture = frames[n - 1]
    closer = still(clip, folder / f"look-take{take}-{entry['category']}.png", frame_time(caption), entry.get("box"))
    pictures = [ref for ref in [reference_for(run, entry, bible, index, folder, take)] if ref]
    pictures.append((f"the new segment, frame {n}" + (", the part named, closer" if entry.get("box") else ""), closer))
    try:
        sheet = contact_sheet([p for _, p in pictures], folder / f"look-take{take}-{entry['category']}-sheet.png", across=len(pictures))
    except (subprocess.SubprocessError, OSError):
        sheet = closer
    prompt = continuity.second_prompt(bible, index, entry, pictures)
    try:
        data, _ = ask_json(lambda text: ask_vision(SECOND_SYSTEM, pictures, sheet, text, run["settings"].get("brain") or {})[0],
                           lambda d: "" if isinstance(d.get("confirmed"), bool) else '"confirmed" is true or false', prompt, "second look")
    except Failed as error:
        return {"error": str(error)[:200]}
    return {"confirmed": data["confirmed"], "why": clip_words(data.get("why"), 20), "pictures": [relative(p) if HOME.resolve() in p.resolve().parents else str(p.name) for _, p in pictures]}


def commit(run: dict, seg: dict, watched: dict, transcript: str, reasons: list, checked: dict | None = None) -> None:
    memory = memory_of(run)
    graph = merge(memory["graph"], watched["delta"], seg["index"])
    for event in graph["events"]:
        if event.get("segment") == seg["index"] and not re.fullmatch(r"\d{1,3}:\d{2}", str(event.get("t") or "")):
            event["t"] = mmss(float(seg["start"]))
    events = [copy.deepcopy(e) for e in graph["events"] if e.get("segment") == seg["index"]]
    archive = memory.get("archive") or []
    if len(graph["events"]) > COMPACT_AT:
        folded = copy.deepcopy(graph["events"][:FOLD])
        chapter, _ = ask_json(lambda text: ask_text(run, COMPACT_SYSTEM, text),
                              lambda d: "" if str(d.get("title") or "").strip() and str(d.get("summary") or "").strip() else '"title" and "summary" are text',
                              compact_prompt(folded, graph.get("chapters") or []), "memory's chapter fold")
        archive = archive + fold(graph, chapter, folded)  # only event detail folds: the bible keeps every canonical fact
    memory = {"graph": graph, "archive": archive, "observed_through": seg["index"]}
    folder = run_dir(run["id"])
    write_json(folder / "memory" / f"{seg['id']}.json", {**memory, "delta": watched["delta"], "cast": run.get("cast") or {},
                                                          "video": seg["output"]})
    write_json(folder / "memory.json", memory)
    bible = bible_of(run)
    checked = checked or {"skipped": "not checked", "findings": []}
    seen = continuity.absorb(bible, seg["index"], checked, events)
    save_bible(run, bible, seg["id"])
    summary = {k: checked.get(k) for k in ("by", "strictness", "categories", "checks", "findings", "skipped", "dense") if checked.get(k) is not None}
    seg.update(status="done", watch=watched, transcript=transcript, passed=not reasons, reasons=reasons, committed=now(),
               continuity={**summary, "present_frames": seen}, fix=[])
    arc = (checked.get("dense") or {}).get("arc")
    if arc and not arc.get("final"):
        if arc.get("reached"):
            run["behind"] = []  # caught up: this piece reached its own end state
        else:
            run.setdefault("behind", []).append({"index": seg["index"], "end_state": arc["end_state"]})
            del run["behind"][:-3]
    minors = [f for f in checked.get("findings") or [] if f.get("action") == "correct"]
    say(run, f"Segment {seg['index']} joins the film" + (f" (it did not pass: {'; '.join(reasons)[:160]})" if reasons else "")
        + (f"; {len(minors)} continuity correction{'s' if len(minors) != 1 else ''} for the next segment" if minors else "")
        + (f"; {seg['check_stats'].get('requests', 0):.0f} check requests, {seg['check_stats'].get('questions', 0):.0f} questions, "
           f"{seg['check_stats'].get('seconds', 0):.0f} s" if seg.get("check_stats") else "") + ".",
        stage="committed", segment=seg["index"])
    checkpoint(run, seg)


def checkpoint(run: dict, seg: dict | None = None, next_seg: dict | None = None) -> None:
    """What a run needs to go on after a crash, a stop by the memory guard or a restart, on disk after every piece
    (``checkpoint.json``, and ``checkpoints/S0001.json`` for each): the pieces kept with their seeds and lengths, the
    film so far, where the bible and the memory of that moment are, the story so far, and the next beat once planned.
    run.json carries the same state; resume_all picks up a run that was live, without anyone asking."""

    done = committed(run)
    point = {"at": now(), "run": run["id"], "segments": len(done), "seconds": round(sum(float(x.get("seconds") or 0) for x in done), 3),
             "pieces": [{"index": x["index"], "output": x.get("output"), "seed": x.get("seed"), "frames": x.get("frames"), "seconds": x.get("seconds"),
                         "passed": x.get("passed"), "scene": x.get("scene"), "beat": x.get("beat")} for x in done],
             "film": run.get("film"), "story_so_far": (run.get("story_so_far") or {}).get("text", ""),
             "bible": f"bible/{seg['id']}.json" if seg else "bible.json", "memory": f"memory/{seg['id']}.json" if seg else "memory.json",
             "check_stats": run.get("check_stats") or {},
             "next": {k: next_seg.get(k) for k in ("index", "scene", "beat", "fields", "pace", "seed", "frames")} if next_seg else None}
    write_json(run_dir(run["id"]) / "checkpoint.json", point)
    if seg:
        write_json(run_dir(run["id"]) / "checkpoints" / f"{seg['id']}.json", point)
    run["checkpoint"] = {"at": point["at"], "segments": point["segments"], "next": (next_seg or {}).get("beat")}


# ---------- the film-level pass ----------

def film_check_safely(run: dict, last: int = 0) -> None:
    """The film-level pass (over the last ``last`` segments, or all), whose failure is reported and never stops the run."""

    if eyes is None and judge is not None:
        # the film pass asks a describing model across pieces; d1-2B judges each take (person, cut, clothes) instead
        if not last:
            run["film_check"] = {"state": "skipped", "at": now(), "note": "no eyes: d1-2B judged every take; the film-level pass needs a describing vision model (autopilot.judge \"eyes\")"}
        return
    try:
        film_check(run, last)
    except Exception as error:  # noqa: BLE001 - a report only
        run["film_check"] = {"state": "failed", "at": now(), "error": str(error)[:300]}
        say(run, "The film's continuity pass could not be made: " + str(error)[:200])


def film_check(run: dict, last: int = 0) -> dict:
    """Cosmos samples across the whole film: each person's clearest frame of each segment they were seen in, an era
    at a time, asked whether any segment shows someone else (the man's face in segment 12 against segment 8), and one
    frame of each segment, asked for drift in places, props and era. A report only: nothing is rendered again."""

    bible, done = bible_of(run), committed(run)
    if last:
        done = done[-last:]  # an Infinity run's rolling pass: the last joins only
    brain = run["settings"].get("brain") or {}
    folder = run_dir(run["id"]) / ("film-check" + (f"-to{len(committed(run)):04d}" if last else ""))
    folder.mkdir(parents=True, exist_ok=True)
    report = {"state": "done", "at": now(), "segments": len(committed(run)), "identity": [], "world": [], "errors": [],
              **({"window": [done[0]["index"], done[-1]["index"]]} if last and done else {})}

    def ask(prompt: str, pictures: list, name: str, check) -> dict | None:
        try:
            sheet = contact_sheet([p for _, p in pictures], folder / f"{name}.png")
            data, _ = ask_json(lambda text: ask_vision(continuity.FILM_SYSTEM, pictures, sheet, text, brain)[0], check, prompt, "film check")
            return data
        except (Failed, subprocess.SubprocessError, OSError) as error:
            report["errors"].append(f"{name}: {str(error)[:200]}")
            return None

    for sid, eras in continuity.film_samples(bible, done).items():  # noqa: B007
        for era, picked in eras.items():
            picked = [(i, path) for i, path in picked if on_disk(path)]
            if len(picked) < 2:
                continue
            look = "; ".join(f["value"] for f in bible["facts"] if f["subject"] == sid and f["category"] == "identity" and f["era"] == era)
            pictures = [(f"Segment {i}", HOME / path) for i, path in picked]
            data = ask(continuity.film_identity_prompt(continuity.name_of(bible, sid), look, picked), pictures, f"identity-{sid}-era{era}",
                       lambda d: "" if isinstance(d.get("same"), bool) else '"same" is true or false')
            if data is not None:
                different = [int(x) for x in data.get("different") or [] if str(x).isdigit() and int(x) in {i for i, _ in picked}]
                report["identity"].append({"subject": continuity.name_of(bible, sid), "era": era, "segments": [i for i, _ in picked],
                                           "same": data["same"] and not different, "different": different, "why": clip_words(data.get("why"), 24)})
    frames = [(seg["index"], (seg.get("watch") or {}).get("frames") or []) for seg in done]
    frames = [(i, f[len(f) // 2]) for i, f in frames if f and on_disk(f[len(f) // 2])]
    if len(frames) > 8:
        step = (len(frames) - 1) / 7
        frames = [frames[round(k * step)] for k in range(8)]
    if len(frames) >= 2:
        data = ask(continuity.film_world_prompt(bible, frames), [(f"Segment {i}", HOME / path) for i, path in frames], "world",
                   lambda d: "" if isinstance(d.get("issues"), list) else '"issues" is a list')
        for issue in (data or {}).get("issues") or []:
            if isinstance(issue, dict) and str(issue.get("evidence") or "").strip() and not continuity.BENIGN.search(str(issue["evidence"])):
                report["world"].append({"category": str(issue.get("category") or "other")[:20], "evidence": clip_words(issue["evidence"], 24),
                                        "segments": [int(x) for x in issue.get("segments") or [] if str(x).isdigit()][:8]})
    write_json(run_dir(run["id"]) / ("film-check.json" if not last else f"film-check-to{len(committed(run)):04d}.json"), report)
    if not last:
        run["film_check"] = report
    if last:
        run.setdefault("film_checks", []).append({k: report.get(k) for k in ("at", "window", "segments")}
                                                 | {"drift": [r for r in report["identity"] if not r["same"]], "issues": report["world"]})
        del run["film_checks"][:-20]
    drift = [r for r in report["identity"] if not r["same"]]
    say(run, "The film's continuity pass: " + (f"{len(drift)} person/era with drift, " if drift else "no face drift, ")
        + f"{len(report['world'])} other issue{'s' if len(report['world']) != 1 else ''}" + (f", {len(report['errors'])} question(s) not answered" if report["errors"] else "") + ".",
        stage="film-check")
    return report


# ---------- the film ----------

def video_seconds(clip: Path) -> float:
    """The picture's own length: the container's is the longest stream's, and a film whose picture stopped early
    still reports its sound's length there."""

    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets",
                           "-of", "csv=p=0", str(clip)], capture_output=True, text=True, env=ENV, timeout=120)
    try:
        return int(done.stdout.strip().split(",")[0]) / FPS
    except ValueError:
        return duration(clip)


def has_audio(clip: Path) -> bool:
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(clip)],
                          capture_output=True, text=True, env=ENV, timeout=30)
    return bool(done.stdout.strip())


EASE_IN_MAX_S = 2  # the most of a fight piece's opening the film leaves out


def ease_in(seg: dict) -> float:
    """Seconds of a fight piece's opening the film leaves out: MiniMax H3 eases out of the still it starts from, and on
    the pace and wardrobe run (2026-10-26, 1376x768) 7 of 8 fight takes opened at 2 to 5 a second whatever the prompt
    said, the keyframe already mid-strike, before the exchange ran at 10 to 16. The leading seconds measured slow (under
    checks.SLOW) are cut, at most EASE_IN_MAX_S, when a fast second follows them; a piece the plan opens slowly keeps them."""

    pace = ((seg.get("continuity") or {}).get("dense") or {}).get("pace") or {}
    motion, plan = pace.get("motion") or [], checks.pace_of(seg.get("pace"))
    if plan["kind"] not in checks.FAST_KINDS or not motion:
        return 0.0
    lead = 0
    while lead < min(EASE_IN_MAX_S, len(motion) - 1) and motion[lead] < checks.SLOW and checks.planned(plan, lead + 0.5, len(motion)) == "fast":
        lead += 1
    return float(lead) if lead and any(m >= checks.SLOW for m in motion[lead:]) else 0.0


def publish(run: dict) -> dict | None:
    """Every committed segment joined into ``film.mp4``: a continuation cross-fades over the one frame it repeats,
    a new scene dissolves in; picture and sound fade together, so they stay in step."""

    parts = [s for s in committed(run) if on_disk(s.get("output"))]
    if not parts:
        return None
    folder = run_dir(run["id"])
    say(run, f"Joining the film: {len(parts)} segment{'s' if len(parts) != 1 else ''}.", stage="preview")
    size = None
    inputs, graph, lengths_ = [], [], []
    for i, seg in enumerate(parts):
        clip = HOME / seg["output"]
        length = duration(clip)
        cut = ease_in(seg)
        length = max(0.5, length - cut)
        lengths_.append(length)
        inputs += (["-ss", f"{cut:.3f}"] if cut else []) + ["-i", str(clip)]
        if size is None:
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(clip)],
                                   capture_output=True, text=True, env=ENV, timeout=30).stdout.strip().split(",")
            size = (int(probe[0]) // 2 * 2, int(probe[1]) // 2 * 2) if len(probe) == 2 and all(x.isdigit() for x in probe) else (864, 480)
        graph.append(f"[{i}:v]fps={FPS},scale={size[0]}:{size[1]},setsar=1,format=yuv420p,settb=AVTB[v{i}]")
        if has_audio(clip):
            graph.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{length:.4f}[a{i}]")
        else:
            graph.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{length:.4f},aformat=sample_fmts=fltp[a{i}]")
    vlast, alast, total = "[v0]", "[a0]", lengths_[0]
    for i in range(1, len(parts)):
        if parts[i].get("mode") == "previous" and not ease_in(parts[i]):  # with its ease-in cut it no longer repeats the frame
            # a continuation opens on the frame the one before ends on: that frame is dropped, and the sound cross-fades
            # over the same 1/24 s. (A one-frame xfade is not used: ffmpeg 6.1 drops the second clip with it.)
            fade = CONTINUE_FADE
            graph.append(f"[v{i}]trim=start_frame=1,setpts=PTS-STARTPTS[vt{i}]")
            graph.append(f"{vlast}[vt{i}]concat=n=2:v=1:a=0[vx{i}]")
        else:
            fade = min(CUT_FADE, lengths_[i] / 2, total / 2)
            graph.append(f"{vlast}[v{i}]xfade=transition=fade:duration={fade:.4f}:offset={total - fade:.4f}[vx{i}]")
        graph.append(f"{alast}[a{i}]acrossfade=d={fade:.4f}:c1=tri:c2=tri[ax{i}]")
        vlast, alast, total = f"[vx{i}]", f"[ax{i}]", total + lengths_[i] - fade
    target = folder / "film.mp4"
    scratch = folder / "film.part.mp4"
    command = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(graph), "-map", vlast, "-map", alast,
               "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", str(scratch)]
    done = subprocess.run(command, capture_output=True, text=True, env=ENV, timeout=3600)
    if done.returncode or not scratch.is_file():
        say(run, "The film could not be joined: " + (done.stderr or "").strip()[-300:])
        return None
    scratch.replace(target)
    run["film"] = {"path": relative(target), "segments": len(parts), "seconds": round(video_seconds(target), 2), "at": now()}
    save(run)
    return run["film"]
