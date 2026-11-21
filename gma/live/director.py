"""The live stage's director: a spoken or typed note becomes the scene the picture engine is given next.

The engine reads one description of the scene (who, what they do, where, the light, the camera) and makes the next
quarter second of picture from it. A note ("move closer to her", "make it night", "Mara, tell him why you're upset")
changes that description: the brain rewrites it, keeping every word the note does not touch, so the people and the
place hold. A note to one of the cast also gets that character's line, spoken in the character's voice.

    direction = await direct(brain, scene, note, cast)     # Direction(scene, who, line, ack)
    interim(scene, note)                                    # the scene with the note tacked on: shown while the brain writes

Before anything is filmed there is the rehearsal: the director plans the shot aloud and talks to the cast, and the brain
keeps the plan, the opening picture and the beats that follow "action", and answers as the character spoken to.

    plan = await rehearse(brain, scene, beats, note, cast)  # Plan(scene, beats, who, line, ack)
    command("And... action!")                               # "action"; "Cut!" -> "cut"; anything else -> None
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SYSTEM = """You keep the description of one live video scene for a director. A video model reads the description and \
films it, a quarter of a second at a time, so it must always say what is on screen NOW.

You get the current description, the cast, and the director's note. Reply with one JSON object and nothing else:
{"scene": "...", "who": null, "line": null, "ack": "..."}

Rules for "scene":
- One paragraph, 45 to 80 words, present tense, plain visual words: the camera first (shot size, movement), then who is \
there and what each is doing now, then the place and the light.
- Change only what the note asks. Copy everything else from the current description word for word, above all how each \
person looks and what they wear, so they stay the same people.
- A note about the camera ("closer", "pan left", "orbit them", "wide shot") changes only the camera words.
- A note that something happens ("she starts to run", "it begins to snow") is written as happening now.
- Never mention the director, the note, sound, music, text on screen or a cut.

"who" and "line": when the note is spoken to one of the cast, or asks someone to say or answer something, "who" is that \
character's name and "line" is what the character says, in character, one short sentence. The scene then shows that \
person speaking. Otherwise both are null.

"ack": two to five words the assistant says back to the director, like "Moving in." or "Night it is."."""


@dataclass
class Direction:
    scene: str
    who: str | None = None
    line: str | None = None
    ack: str = ""
    by: str = "brain"  # "brain", or "plain" when the brain's reply could not be read and the note was tacked on


def cast_of(text: str) -> list[dict]:
    """The cast of a scene's set-up: its lines of the form ``Name: how they look``. [{"name", "look"}]"""

    out = []
    for line in text.splitlines():
        found = re.match(r"\s*([A-Z][\w' -]{0,30}?)\s*:\s*(\S.*)$", line)
        if found and len(found.group(1).split()) <= 3:
            out.append({"name": found.group(1).strip(), "look": found.group(2).strip()})
    return out


CASTING = """A director typed the set-up of one film scene. Read it for the rehearsal. Reply with one JSON object and \
nothing else: {"cast": [{"name": "...", "look": "..."}], "scene": null, "beats": []}

"cast": the people (or creatures) who are on screen and could be spoken to, at most three, the most important first. \
"name": one word the director can call them by, capitalised: the word the set-up itself uses for them ("Warrior", \
"Witch", "Driver"), never a first name the set-up does not give. "look": how they look and what they wear, copied from \
the set-up's own words, up to 30 words, and whether it is a man or a woman when the set-up says so (he, him, she, her). \
A crowd, the dead and things are not cast. When the set-up named nobody on screen: [].

"scene" and "beats": only when the set-up tells a story, things that happen one after another. Then "scene" is the \
opening picture alone, what is on screen at the first moment before anything happens: one paragraph, 45 to 80 words, \
present tense, the camera first, then who is there (called by their cast name, with their look) and what they are doing, \
then the place, the light and the style, in the set-up's own words. And "beats" is what happens after that, in order: \
[{"s": seconds after the start, "note": one short direction as a director says it on set}], 4 to 6 seconds apart, at \
most 10, each one thing that can be seen, using the cast names. When the set-up is one picture with nothing happening \
after it: "scene" is null and "beats" is []."""


async def reading(brain, text: str, **settings) -> dict:
    """A set-up as the brain reads it: {"cast": those on screen, "scene": the opening picture or None, "beats": what
    happens after it}. Used when the set-up names nobody on a line of their own, or tells a whole story. Nothing
    (no cast, no scene, no beats) when the brain is away."""

    nothing = {"cast": [], "scene": None, "beats": []}
    try:
        reply = await brain.complete([{"role": "system", "content": CASTING}, {"role": "user", "content": text}],
                                     **{"max_tokens": 900, "temperature": 0.2, "timeout": 120, **settings})
        found = re.search(r"\{.*\}", reply or "", re.S)
        said = json.loads(found.group(0)) if found else {}
    except Exception:  # noqa: BLE001 - away or not the JSON asked for: the scene is taken as it was typed
        return nothing
    if not isinstance(said, dict):
        return nothing
    out, seen = [], set()
    for one in said.get("cast") if isinstance(said.get("cast"), list) else []:
        if not isinstance(one, dict):
            continue
        name = re.sub(r"[^\w' -]", "", str(one.get("name") or "")).strip()[:30]
        look = " ".join(str(one.get("look") or "").split())[:240]
        if name and look and name.lower() not in seen and len(name.split()) <= 3:
            seen.add(name.lower())
            out.append({"name": name[0].upper() + name[1:], "look": look})
    scene = said.get("scene")
    scene = " ".join(scene.split()) if isinstance(scene, str) and 8 <= len(scene.split()) <= 160 else None
    return {"cast": out[:3], "scene": scene, "beats": beats_of(said.get("beats")) if scene else []}


async def casting(brain, text: str, **settings) -> list[dict]:
    """The cast of a set-up that names nobody in ``Name: look`` lines, as the brain reads it from the prose."""

    return (await reading(brain, text, **settings))["cast"]


def opening(text: str) -> str:
    """The first description of a scene from what was typed: the cast's looks folded into the words about the scene."""

    cast = cast_of(text)
    names = {c["name"] for c in cast}
    rest = " ".join(line.strip() for line in text.splitlines()
                    if line.strip() and not any(re.match(rf"\s*{re.escape(n)}\s*:", line) for n in names))
    people = " ".join(f"{c['name']} is {c['look'].rstrip('.')}." for c in cast)
    return " ".join(part for part in (rest, people) if part).strip()


def interim(scene: str, note: str) -> str:
    """The scene with the note tacked on: what the engine is given at once, until the brain's rewrite is in."""

    note = note.strip().rstrip(".!")
    return f"{scene.rstrip()} Now: {note}." if note else scene


def read(reply: str, scene: str, note: str) -> Direction:
    """The brain's reply as a Direction; a reply that is not the JSON asked for falls back to the note tacked on."""

    found = re.search(r"\{.*\}", reply or "", re.S)
    try:
        said = json.loads(found.group(0)) if found else None
    except ValueError:
        said = None
    new = (said or {}).get("scene") if isinstance(said, dict) else None
    if not isinstance(new, str) or len(new.split()) < 8:
        return Direction(interim(scene, note), by="plain", ack="Okay.")
    who, line = said.get("who"), said.get("line")
    if not (isinstance(who, str) and isinstance(line, str) and who.strip() and line.strip()):
        who = line = None
    return Direction(" ".join(new.split()), who and who.strip(), line and line.strip(), str(said.get("ack") or "").strip()[:60])


async def direct(brain, scene: str, note: str, cast: list[dict] | None = None, **settings) -> Direction:
    """Ask the brain for the scene after ``note``. Any failure of the brain gives the plain direction, never an error:
    the stream goes on either way."""

    people = "; ".join(f"{c['name']}: {c['look']}" for c in cast or []) or "nobody is named"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Current description:\n{scene}\n\nCast: {people}\n\nDirector's note: {note}"}]
    try:
        reply = await brain.complete(messages, **{"max_tokens": 260, "temperature": 0.3, "timeout": 30, **settings})
    except Exception:  # noqa: BLE001 - a brain that is away, held for a render or slow: the note still lands
        return Direction(interim(scene, note), by="plain", ack="Okay.")
    return read(reply, scene, note)


REHEARSAL = """You are the first assistant director at the rehearsal of one live video scene. Nothing is being filmed \
yet: the director is planning the shot with you and talking to the cast. When the director says "action", a video model \
films the plan: it opens on the opening picture, and each beat is given to it at its time.

You get the opening picture, the cast, the beats planned so far, and what the director just said. Reply with one JSON \
object and nothing else:
{"scene": "...", "beats": [{"s": 4, "note": "..."}], "who": null, "line": null, "ack": "..."}

"scene": the opening picture, what is on screen at "action". One paragraph, 45 to 80 words, present tense, plain visual \
words: the camera first (shot size, movement), then who is there and what each is doing, then the place and the light. \
Change it only when the director changes how the scene opens (where people stand, how they look, the light, where the \
camera starts); otherwise copy it word for word.

"beats": the whole list of what happens after "action", in order. "s" is seconds after "action" (3 to 6 seconds apart \
unless the director gives times), "note" is one short direction as a director says it on set: "she walks toward the \
camera", "push in to a close-up", "Mara, tell him why you are upset". Add, change, move or remove beats as the director \
asks and keep the others as they are. At most 12 beats. A camera move and an action are separate beats.

"who" and "line": when the director speaks to one of the cast (a question, a motivation, how to play it), "who" is that \
character and "line" is the character's answer at rehearsal, in character, one or two short sentences. Otherwise null. \
An instruction to do something, even with the character's name on it ("Mara, smile a little", "Jon, walk to the door"), \
is not a question: it becomes a beat, and "who" and "line" stay null.

"ack": two to six words back to the director, like "Got it, she enters late."."""


@dataclass
class Plan:
    scene: str
    beats: list[dict]
    who: str | None = None
    line: str | None = None
    ack: str = ""
    by: str = "brain"  # "plain" when the brain's reply could not be read: the note became the next beat
    ignored: bool = False  # overheard talk that was no direction and no word to the cast: the plan is as it was


CUT_HEARD = {"cut", "cuts", "cat", "cot", "caught", "cup", "kat", "got", "gut", "but", "what", "hut", "cart", "cry", "cud", "kut", "cutt", "cuh", "could", "good"}  # a shouted "cut", as a recogniser hears it
CUT_SOUND = re.compile(r"[ckg](u|o|a|au|ar|oo)(t|tt|d|rt|ught)(s|y|ty)?")
ACTION_HEARD = {"action", "actions", "auction", "axion", "acton"}


def command(text: str, heard: bool = False, filming: bool = False) -> str | None:
    """The word on set that is an order, not a note: "action" or "cut" (said alone, or as "and action", "cut, cut").
    A word that was ``heard`` is taken more loosely: one short shout while ``filming`` is "cut" when it sounds like it,
    and "auction" at the table is "action"."""

    words = re.sub(r"[^a-z ]", " ", (text or "").lower()).split()
    words = [w for w in words if w not in ("and", "aaand", "okay", "ok", "everyone", "quiet", "please", "that", "s", "is", "a", "thats")]
    if words and len(words) <= 3 and all(w == "action" for w in words):
        return "action"
    if words and len(words) <= 3 and all(w == "cut" for w in words):
        return "cut"
    if heard and filming and words and len(words) <= 3 and any(w in CUT_HEARD or CUT_SOUND.fullmatch(w) for w in words):
        return "cut"  # measured with six synthetic voices: "Got there.", "God.", "Kart Karty.", "Cod Contra.", "End cut."
    if heard and words and len(words) <= 2:
        if not filming and words[0] in ACTION_HEARD:
            return "action"
    return None


AGAIN = {"again", "retake", "redo", "repeat", "rerun"}
FILM = {"render", "film", "print", "proceed", "shoot", "keep"}


def after_run(text: str) -> str | None:
    """What a short word means once a run-through was cut: "again" / "retake" plays it once more ("action"), "render it"
    / "film it" / "print it" / "proceed" films it ("film")."""

    words = [w for w in re.sub(r"[^a-z ]", " ", (text or "").lower()).split()
             if w not in ("and", "okay", "ok", "please", "it", "that", "this", "the", "take", "lets", "let", "s", "one", "now", "go", "do", "with", "rendering", "more", "time", "once", "run", "a")]
    said = set(re.sub(r"[^a-z ]", " ", (text or "").lower()).split())
    if len(said) > 6:
        return None
    if words and all(w in AGAIN for w in words) or said & {"retake"}:
        return "action"
    if words and all(w in FILM for w in words) or (said & {"rendering", "render"}) or said >= {"go", "ahead"}:
        return "film"
    return None


ASIDE = """You are {who}, a character in a film scene that is being run through in real time, before it is filmed. \
The director speaks to you in the middle of it. Answer at once, in character, as {who} would say it in the scene at this \
moment: one short sentence, at most 16 words, no stage directions. Reply with one JSON object and nothing else: \
{{"line": "..."}}{overheard}

The scene now: {scene}
The cast: {people}"""
NOT_FOR_US = """
The words were picked up by an open microphone. When they are plainly not said to you or about the scene (half a \
sentence, a remark to somebody else), reply with exactly {"ignore": true}."""


async def answer(brain, scene: str, cast: list[dict], who: str, text: str, overheard: bool = False, **settings) -> str | None:
    """What ``who`` says back when spoken to while the run-through plays; None when the words were not for them or the
    brain is away."""

    people = "; ".join(f"{c['name']}: {c['look']}" for c in cast or []) or who
    messages = [{"role": "system", "content": ASIDE.format(who=who, scene=scene, people=people, overheard=NOT_FOR_US if overheard else "")},
                {"role": "user", "content": text}]
    try:
        reply = await brain.complete(messages, **{"max_tokens": 80, "temperature": 0.5, "timeout": 15, **settings})
        found = re.search(r"\{.*\}", reply or "", re.S)
        said = json.loads(found.group(0)) if found else None
    except Exception:  # noqa: BLE001 - away or not the JSON asked for: nobody answers
        return None
    line = said.get("line") if isinstance(said, dict) and not said.get("ignore") else None
    return " ".join(line.split())[:200] if isinstance(line, str) and line.strip() else None


def echo(text: str, lines: list[str]) -> bool:
    """Heard words that are one of the cast's own lines coming back through the microphone: most of the words heard are
    words of one line just spoken."""

    heard = re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()
    if not heard:
        return False
    for line in lines:
        said = set(re.sub(r"[^a-z' ]", " ", (line or "").lower()).split())
        if said and sum(1 for w in heard if w in said) >= 0.6 * len(heard) and len(heard) >= min(3, len(said)):
            return True
    return False


def beats_of(given, limit: int = 12) -> list[dict]:
    """Beats as they are kept: [{"s": seconds after action, "note": the direction}], in order of time."""

    out = []
    for beat in given if isinstance(given, list) else []:
        try:
            s, note = float(beat["s"]), " ".join(str(beat["note"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        if note and 0 <= s <= 600:
            out.append({"s": round(s, 2), "note": note[:200]})
    return sorted(out, key=lambda b: b["s"])[:limit]


def plainly(scene: str, beats: list[dict], note: str) -> Plan:
    """The note as the next beat, four seconds after the last: what a rehearsal keeps when the brain is away."""

    return Plan(scene, beats_of([*beats, {"s": (beats[-1]["s"] if beats else 0) + 4, "note": note}]), by="plain", ack="Noted.")


OVERHEARD = """What follows was picked up by an open microphone in the room. The director is at the table talking with \
{who}: anything that is not a direction for the scene (a greeting, a question, small talk, "can you hear me") is said \
to {who}, so answer it as {who} in "who" and "line" and keep the scene and the beats exactly as they are. Only when it \
is plainly not for this room at all (half a sentence, a remark to someone else about something else) reply with exactly \
{{"ignore": true}} and nothing else.

"""
FILLER_HEARD = {"um", "uh", "er", "hmm", "well", "so", "yeah", "yes", "no", "okay", "ok", "right", "oh", "ah", "like", "thanks", "thank", "you", "the", "a", "i", "and"}


def chatter(text: str) -> bool:
    """Overheard words too thin to be a direction: one word, or nothing but filler."""

    words = re.sub(r"[^a-z' ]", " ", (text or "").lower()).split()
    return not words or all(w in FILLER_HEARD for w in words)


async def rehearse(brain, scene: str, beats: list[dict], note: str, cast: list[dict] | None = None, overheard: bool = False,
                   to: str | None = None, **settings) -> Plan:
    """The plan after what the director said at rehearsal. A brain that fails keeps the note as a beat (an
    ``overheard`` one, from the open microphone, is then left out)."""

    if overheard and chatter(note):
        return Plan(scene, beats, ignored=True)

    people = "; ".join(f"{c['name']}: {c['look']}" for c in cast or []) or "nobody is named"
    messages = [{"role": "system", "content": REHEARSAL},
                {"role": "user", "content": f"Opening picture:\n{scene}\n\nCast: {people}\n\nBeats so far: {json.dumps(beats)}\n\n"
                                            + (OVERHEARD.format(who=to or (cast[0]["name"] if cast else "the cast")) if overheard else "")
                                            + (f"The director is facing {to}: a question or a word to nobody by name is for {to}.\n\n" if to and len(cast or []) > 1 else "")
                                            + f"The director says: {note}"}]
    try:
        reply = await brain.complete(messages, **{"max_tokens": 700, "temperature": 0.3, "timeout": 45, **settings})
        found = re.search(r"\{.*\}", reply or "", re.S)
        said = json.loads(found.group(0)) if found else None
    except Exception:  # noqa: BLE001 - away, held, or not the JSON asked for
        said = None
    if overheard and (not isinstance(said, dict) or said.get("ignore")):
        return Plan(scene, beats, ignored=True)
    if not isinstance(said, dict) or not isinstance(said.get("beats"), list):
        return plainly(scene, beats, note)
    new = said.get("scene")
    new = " ".join(new.split()) if isinstance(new, str) and len(new.split()) >= 8 else scene
    who, line = said.get("who"), said.get("line")
    if not (isinstance(who, str) and isinstance(line, str) and who.strip() and line.strip()):
        who = line = None
    return Plan(new, beats_of(said["beats"]), who and who.strip(), line and line.strip(), str(said.get("ack") or "").strip()[:80])
