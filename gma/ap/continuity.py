# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, app/continuity.py at 6cc6e67c5e138499ea2c17183eda261b710d1217.
# Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: nothing; the code is unchanged.
"""The continuity bible: AutoPilot's recursive, memory-backed continuity (autopilot.py asks the questions, this file
keeps the answers and decides what they mean).

Every segment is checked against everything the film has established so far, not only the segment before. The bible
(``bible.json`` in the run's folder, ``bible/S0001.json`` after each segment) holds the canonical state:

  subjects   the people, places and things, with their names and a reference picture (a cast picture, else the
             clearest frame of their first segment)
  facts      one attribute each (who, category, key, value), versioned: a fact holds from one segment to another, and a
             new value closes the old one instead of overwriting it, so the bible can say what was true when
  eras       a time-skip or another era (a cast sheet that starts later, or the Director's ``era``): era-scoped facts
             (age, hair, build, wardrobe, props, era rules) close at an era change, film-scoped ones (face, eyes, scars
             and marks, voice, places, story) carry on
  scene      scene-scoped facts (who is where on screen, which way they move, actions in progress, light, music and
             ambience) close when a new scene starts
  story      what has happened (the watcher's events and the story facts it reports): never folded, never dropped
  corrections  the minor findings the Director is told to put right in the next segment

The scene graph's chapter fold (autopilot.commit) folds only event detail; nothing here is folded.

The check of a segment (autopilot.check_continuity) is one checklist question to the vision brain: a verdict a
category (ok, minor, major, not visible) with the frame and the evidence. ``triage`` turns that into findings: a
finding without evidence, or whose evidence says nothing is wrong, is dropped; a major one is looked at again, closer
(autopilot.second_look), and dropped when the second look does not see it. What a confirmed major does depends on the
run's strictness (``STRICTNESS``): a retake with the finding in the Director's hands, or a correction for the next
segment.
"""

from __future__ import annotations

import copy
import difflib
import re

CATEGORIES = ("identity", "wardrobe", "props", "location", "setting", "lighting", "era", "spatial", "action", "audio", "story", "pace")
# decided in code from measurements and yes/no answers (checks.py), never asked as a checklist verdict
MEASURED = ("pace",)
# what the frames can show; the audio is held in the prompts instead (each speaker's voice is the bible's, word for word)
WATCHED = tuple(c for c in CATEGORIES if c != "audio")
VERDICTS = ("ok", "minor", "major", "not_visible")
STRICTNESS = {
    # retake: the categories whose confirmed major finding renders the segment again; unsure: what a major finding
    # becomes when its second look cannot be made
    "lenient": {"retake": ("identity", "era"), "unsure": "drop"},
    # wardrobe is part of who someone is (the gate's Infinity film: a brown waistcoat became a dark coat with gold trim)
    "standard": {"retake": ("identity", "wardrobe", "era", "setting", "story", "action", "pace"), "unsure": "minor"},
    "strict": {"retake": WATCHED, "unsure": "major"},
}
DEFAULT = {"strictness": "standard", "categories": list(CATEGORIES), "film_pass": True}
QUESTIONS = {
    "identity": "Is each character the same person as established: face, hair, eye colour, age, build, scars and marks?",
    "wardrobe": "Is each character's clothing as established, in the same state (wet, torn, muddy) unless that changes on screen?",
    "props": "Do the characters still hold or carry what they had, and are objects where they were left?",
    "location": "Does the place keep its established layout and landmarks?",
    "setting": "Does the setting match what this segment asked for and its era: the place itself (no buildings or streets in a shot at sea, no ships in a city street) and nothing from another of the film's settings?",
    "lighting": "Do the time of day, the light and the weather match the scene so far, unless they change on screen?",
    "era": "Does everything visible belong to the era and setting rules: every object, garment, vehicle and building?",
    "spatial": "Are people where the segment before left them, on the same side of the frame, facing and moving the same way?",
    "action": "Does an action in progress at the end of the segment before continue here or visibly resolve, rather than vanish or restart?",
    "story": "Does anything shown contradict what has already happened (someone back who left, a broken thing whole again)?",
}
# a note that says nothing is wrong is not a finding (Cosmos3-Edge wrote such notes on the first real run)
# (and a note that opens with "remains", "maintains" or "persists": the 60 s Viking test film's watcher wrote three)
BENIGN = re.compile(r"(?<!in)consistent|unchanged|remains? the same|matches|as (established|expected|before)|no (visible )?(issue|problem|change)"
                    r"|nothing (wrong|changed)|^\s*(still )?(remains|maintains|persists|stays|keeps)\b", re.I)
# keys the vision brain says in many ways
KEYS = {"scar": "marks", "scars": "marks", "mark": "marks", "tattoo": "marks", "birthmark": "marks", "outfit": "wardrobe",
        "clothes": "wardrobe", "clothing": "wardrobe", "costume": "wardrobe", "dress": "wardrobe", "holding": "held",
        "holds": "held", "carries": "held", "carrying": "held", "prop": "held", "hairstyle": "hair", "eye": "eyes",
        "eye_colour": "eyes", "eye_color": "eyes", "body": "build", "time_of_day": "time"}
ERA_KEYS = ("age", "hair", "build", "look")  # identity that an era may change; face, eyes and marks carry on
LOOKED_ONCE = ("face", "hair", "eyes", "age", "build", "look", "marks")  # set once an era from what is seen: a later look is checked against it
STORY_LIMIT, DIRECTOR_STORY, CHECK_STORY = 10 ** 6, 14, 6
CORRECTIONS = 8


CLOTHES = re.compile(r"\b(coat|jacket|shirt|tunic|cloak|cape|dress|gown|skirt|trousers|breeches|pants|jeans|boots|shoes|belt|"
                     r"vest|waistcoat|hood|hat|helmet|armou?r|mail|apron|robe|scarf|gloves|sash|uniform|kilt|sandals|collar|"
                     r"jersey|sweater|hoodie|leggings|bodice|doublet|furs?|shield|visor)\b", re.I)


def wardrobe_line(character: dict) -> str:
    """A cast sheet's wardrobe as one short, exact line: its own ``wardrobe`` when the seed gives one, else the parts of
    its look that name clothes (and what is carried), in the sheet's own words. Said word for word in every keyframe and
    video prompt of its segments, and checked against in every window."""

    given = str(character.get("wardrobe") or "").strip()
    if given:
        return clip(given, 30)
    parts = [p.strip() for p in re.split(r"[,;]|\bwith\b|\band\b|\bwearing\b|\bin\b(?= (?:a|an|his|her|their)\b)", str(character.get("look") or "")) if p.strip()]
    return clip(", ".join(p for p in parts if CLOTHES.search(p)), 30)


def slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")


def clip(text, words: int) -> str:
    return " ".join(str(text or "").split()[:words])


def same(a, b) -> bool:
    a, b = " ".join(str(a or "").lower().split()), " ".join(str(b or "").lower().split())
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


def options(settings: dict) -> dict:
    """The run's continuity settings, the defaults for a run made before they existed."""

    given = (settings or {}).get("continuity") or {}
    out = copy.deepcopy(DEFAULT)
    if given.get("strictness") in STRICTNESS:
        out["strictness"] = given["strictness"]
    if isinstance(given.get("categories"), list):
        out["categories"] = [c for c in CATEGORIES if c in given["categories"]]
    if given.get("film_pass") is not None:
        out["film_pass"] = bool(given["film_pass"])
    return out


def clean(given, base: dict | None = None) -> dict:
    """The continuity settings from a request; an unknown strictness or category is a 400 at the caller."""

    out = options({"continuity": base or {}})
    if not isinstance(given, dict):
        return out
    if given.get("strictness") is not None:
        if given["strictness"] not in STRICTNESS:
            raise ValueError(f"strictness is one of {', '.join(STRICTNESS)}")
        out["strictness"] = given["strictness"]
    if given.get("categories") is not None:
        if not isinstance(given["categories"], list) or any(c not in CATEGORIES for c in given["categories"]):
            raise ValueError(f"categories is a list of {', '.join(CATEGORIES)}")
        out["categories"] = [c for c in CATEGORIES if c in given["categories"]]
    if given.get("film_pass") is not None:
        out["film_pass"] = bool(given["film_pass"])
    return out


def checked(opts: dict) -> list[str]:
    """The categories the checklist asks about (the measured ones are decided in code)."""
    return [c for c in opts["categories"] if c in WATCHED and c not in MEASURED]


def watched(opts: dict) -> list[str]:
    """Every category the run checks from the pictures, the measured ones too."""
    return [c for c in opts["categories"] if c in WATCHED]


# ---------- what was heard ----------

def sound_key(word: str) -> str:
    """American Soundex: Eirik and Eric are E620, so a name Whisper spells its own way still counts as said."""

    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"), "l": "4",
             **dict.fromkeys("mn", "5"), "r": "6"}
    out, last = word[0].upper(), codes.get(word[0], "")
    for ch in word[1:]:
        code = codes.get(ch, "")
        if code and code != last:
            out += code
        if ch not in "hw":
            last = code
    return (out + "000")[:4]


def heard_alike(wanted: str, heard: str) -> bool:
    """The line counted again with words that sound alike (the same Soundex key, or nearly the same spelling): the
    QA gate's audio check counts only exact words, and on the Viking test film it failed "Eirik!" heard as "Eric!"
    twice. The gate's own measure otherwise: two words of a line (one of a one-word line) and 60 % of them."""

    want = re.findall(r"[a-z0-9']+", str(wanted or "").lower())
    said = re.findall(r"[a-z0-9']+", str(heard or "").lower())
    if not want or not said:
        return False
    alike = lambda w, s: w == s or (len(w) > 2 and sound_key(w) == sound_key(s)) or difflib.SequenceMatcher(None, w, s).ratio() >= 0.75
    found = sum(1 for w in want if any(alike(w, s) for s in said))
    need = min(2, len(want))
    return len(said) >= need and found >= need and found >= 0.6 * len(want)


def gate_category(note: str) -> str:
    """The continuity category of a failed identity gate's note (on the Viking test film: a coat that turned from
    brown to red was the identity gate's reason)."""

    return "wardrobe" if re.search(r"wardrobe|cloth|coat|jacket|dress|shirt|tunic|cloak|costume|outfit", note or "", re.I) else "identity"


# ---------- the bible ----------

def scope(category: str, key: str) -> str:
    """How long a fact holds: ``scene`` (until a new scene), ``era`` (until a time-skip) or ``film``."""

    if category in ("spatial", "action", "lighting") or (category == "audio" and key != "voice"):
        return "scene"
    if category in ("wardrobe", "props", "era") or (category == "identity" and key in ERA_KEYS):
        return "era"
    return "film"


def covers(fact: dict, index: int) -> bool:
    return fact["from"] <= index and (fact.get("to") is None or fact["to"] >= index)


def era_at(bible: dict, index: int) -> dict:
    eras = [e for e in bible["eras"] if e["from"] <= index] or bible["eras"][:1]
    return eras[-1]


def new(seed: dict, cast: dict | None = None) -> dict:
    """The bible a run starts with: the seed's world and rules as the first era, each cast sheet's look for its
    segments (two sheets of one first name whose segments do not overlap are one person at two ages), and anyone the
    Director has brought in already."""

    bible = {"version": 1, "next": 1, "eras": [{"id": 1, "from": 1, "label": "", "rules": [clip(r, 30) for r in seed.get("rules") or [] if str(r).strip()][:8],
                                                   "world": clip(seed.get("world"), 80)}],
             "scene": {"name": "", "from": 1}, "subjects": {}, "facts": [], "story": [], "corrections": []}
    sheets = [c for c in seed.get("characters") or [] if c.get("name")]
    groups: dict[str, list] = {}
    for c in sheets:
        groups.setdefault(slug(str(c["name"]).split()[0]), []).append(c)
    for first, group in groups.items():
        spans = sorted(((c.get("from") or 1), (c.get("to") or 10 ** 6)) for c in group)
        apart = len(group) > 1 and all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))
        for c in group:
            sid = first if apart else slug(c["name"])
            subject = bible["subjects"].setdefault(sid, {"name": c["name"], "kind": "character", "names": [], "ref": None, "seen": []})
            subject["names"].append(c["name"])
            if c.get("image") and not subject["ref"]:
                subject["ref"] = c["image"]
            if c.get("look"):
                establish(bible, sid, "identity", "look", c["look"], c.get("from") or 1, "seed", until=c.get("to"), words=80)
            if wardrobe_line(c):  # canonical for its segments: what the eyes see is checked against it, never replaces it
                establish(bible, sid, "wardrobe", "wardrobe", wardrobe_line(c), c.get("from") or 1, "seed", until=c.get("to"), words=30)
    for name, look in (cast or {}).items():
        if look and subject_of(bible, name, "character", create=False) is None:
            sid = subject_of(bible, name, "character")
            establish(bible, sid, "identity", "look", look, 1, "director", words=80)
    return bible


def subject_of(bible: dict, who, kind: str = "character", create: bool = True) -> str | None:
    """The bible's id for a name the Director or the vision brain gave: its own id, one of its names, the same first
    name, or a near spelling; a new subject otherwise."""

    raw = str(who or "").strip()
    sid = slug(raw)
    if not sid:
        return None
    subjects = bible["subjects"]
    if sid in subjects:
        return sid
    for key, s in subjects.items():
        if any(slug(n) == sid for n in s.get("names") or []) or slug(s.get("name")) == sid:
            return key
    first = sid.split("_")[0]
    for key, s in subjects.items():
        if s.get("kind") == kind and len(first) > 2 and (key.split("_")[0] == first or any(slug(n).split("_")[0] == first for n in s.get("names") or [])):
            return key
    near = difflib.get_close_matches(sid, list(subjects), n=1, cutoff=0.82)
    if near:
        return near[0]
    if not create:
        return None
    subjects[sid] = {"name": clip(raw, 4), "kind": kind if kind in ("character", "place", "object") else "object", "names": [clip(raw, 4)], "ref": None, "seen": []}
    return sid


def establish(bible: dict, subject: str, category: str, key: str, value, index: int, source: str, *, until: int | None = None,
              once: bool = False, words: int = 16) -> dict | None:
    """A fact from segment ``index`` on. The same fact again changes nothing; a new value closes the old fact the
    segment before (its history stays) unless ``once`` (or the old one is the seed's sheet): then the old one stands."""

    key = KEYS.get(slug(key), slug(key))[:24] or category
    value = clip(value, words)
    if not subject or not value:
        return None
    live = [f for f in bible["facts"] if f["subject"] == subject and f["category"] == category and f["key"] == key and covers(f, index)]
    for fact in live:
        if same(fact["value"], value):
            return fact
        if once or fact["source"] == "seed":
            return None
        if fact["from"] >= index:
            fact.update(value=value, source=source)
            return fact
        fact["to"] = index - 1
    fact = {"id": bible["next"], "subject": subject, "category": category, "key": key, "value": value, "from": index,
            "to": until, "era": era_at(bible, index)["id"], "scope": scope(category, key), "source": source}
    bible["next"] += 1
    bible["facts"].append(fact)
    return fact


def close(bible: dict, index: int, scopes: tuple) -> None:
    """Facts of these scopes end before segment ``index``; what the eyes left for ``index`` itself (where people stood
    on the last frame before a cut) is voided."""

    for fact in bible["facts"]:
        if fact["scope"] in scopes and covers(fact, index) and (fact["from"] < index or fact["source"] == "watch"):
            fact["to"] = index - 1


def current(bible: dict, index: int, subject: str | None = None) -> list[dict]:
    return [f for f in bible["facts"] if covers(f, index) and (f.get("to") is None or f["to"] >= f["from"]) and (subject is None or f["subject"] == subject)]


def open_era(bible: dict, index: int, label: str = "", rules=None) -> dict:
    """A time-skip or another era from segment ``index``: era- and scene-scoped facts close; a sheet that starts here
    keeps its look (its fact begins at ``index``)."""

    era = era_at(bible, index)
    if era["from"] == index:
        if label:
            era["label"] = clip(label, 8)
        if rules:
            era["rules"] = [clip(r, 30) for r in rules if str(r).strip()][:8]
        return era
    close(bible, index, ("era", "scene"))
    era = {"id": len(bible["eras"]) + 1, "from": index, "label": clip(label, 8), "rules": [clip(r, 30) for r in rules or [] if str(r).strip()][:8],
           "world": era["world"]}
    bible["eras"].append(era)
    bible["scene"] = {"name": "", "from": index}
    return era


def arrive(bible: dict, index: int, seed: dict) -> None:
    """Before the Director writes segment ``index``: a cast sheet that starts here opens an era (the same person older)."""

    if index > 1 and any((c.get("from") or 1) == index for c in seed.get("characters") or [] if c.get("name")):
        open_era(bible, index, f"from segment {index}")


def planned(bible: dict, index: int, data: dict, cast: dict) -> None:
    """The Director's answer for segment ``index``: a new scene closes the scene's facts, a jump in time opens an era
    (only with a new scene), new characters get their look, and every speaker keeps one voice."""

    era, label = era_at(bible, index), clip(data.get("era"), 8)
    rules = [r for r in data.get("era_rules") or [] if isinstance(r, str)]
    if index == 1 or era["from"] == index:
        if label or rules:
            open_era(bible, index, label or era["label"], rules or era["rules"])
    elif label and data.get("scene_change") and not same(label, era["label"]) and difflib.SequenceMatcher(None, label.lower(), str(era["label"]).lower()).ratio() < 0.8:
        open_era(bible, index, label, rules)
    if data.get("scene_change") and bible["scene"]["from"] != index:
        close(bible, index, ("scene",))
    if data.get("scene_change") or not bible["scene"]["name"]:
        bible["scene"] = {"name": clip(data.get("scene"), 8), "from": index if data.get("scene_change") else bible["scene"]["from"]}
    for person in data.get("new_characters") or []:
        if isinstance(person, dict) and person.get("name") and person.get("look"):
            sid = subject_of(bible, person["name"], "character")
            if not any(f["key"] == "look" for f in current(bible, index, sid)):
                establish(bible, sid, "identity", "look", cast.get(str(person["name"]).strip()) or person["look"], index, "director", words=80)
    shot = data.get("shot") or {}
    if shot.get("music") and str(shot["music"]).strip().upper() != "N/A":
        establish(bible, "_scene", "audio", "music", shot["music"], index, "director")
    if shot.get("soundscape"):
        establish(bible, "_scene", "audio", "ambience", shot["soundscape"], index, "director")


def voices(bible: dict, index: int, dialogue: list) -> list:
    """Each speaker's voice as the bible first heard it described, word for word in every line after; a new speaker's
    voice becomes the bible's. The renderer has no memory of a voice: the description is all it gets."""

    for line in dialogue:
        sid = subject_of(bible, line.get("speaker"), "character")
        if not sid:
            continue
        held = next((f for f in current(bible, index, sid) if f["category"] == "audio" and f["key"] == "voice"), None)
        if held:
            line["voice"] = held["value"]
        elif str(line.get("voice") or "").strip():
            establish(bible, sid, "audio", "voice", line["voice"], index, "director", words=24)
    return dialogue


# ---------- what the bible says, for the Director and the eyes ----------

def in_play(bible: dict, index: int, names=()) -> list[str]:
    """The subjects that matter now: the cast of this segment, anyone seen in the last three segments, the places
    and things of this scene."""

    ids = {subject_of(bible, n, "character", create=False) for n in names} - {None}
    for sid, s in bible["subjects"].items():
        if any(index - 3 <= n < index for n in s.get("seen") or []):
            ids.add(sid)
    scene_from = bible["scene"]["from"]
    for f in current(bible, index):
        if f["subject"] != "_scene" and (f["from"] >= scene_from or bible["subjects"].get(f["subject"], {}).get("kind") == "character"):
            ids.add(f["subject"])
    return [sid for sid in bible["subjects"] if sid in ids]


def lines_of(bible: dict, index: int, sid: str, categories=None) -> str:
    facts = [f for f in current(bible, index, sid) if f["category"] not in ("spatial", "action") and (categories is None or f["category"] in categories)]
    return "; ".join(f"{f['key']}: {f['value']}" for f in facts)


def story_for(bible: dict, ids: list[str], latest: int) -> tuple[list[dict], int]:
    """The story facts that concern who and what is in play, and the latest ones; and how many more the bible holds."""

    story = bible["story"]
    keep = {i for i, s in enumerate(story) if set(s.get("subjects") or []) & set(ids)}
    keep |= set(range(max(0, len(story) - latest), len(story)))
    picked = [story[i] for i in sorted(keep)][-max(latest, DIRECTOR_STORY):]
    return picked, len(story) - len(picked)


def scene_lines(bible: dict, index: int) -> list[str]:
    out = []
    scene = [f for f in current(bible, index, "_scene")]
    light = "; ".join(f"{f['key']}: {f['value']}" for f in scene if f["category"] == "lighting")
    if light:
        out.append(f"Light, time and weather in this scene: {light}")
    sound = "; ".join(f"{f['key']}: {f['value']}" for f in scene if f["category"] == "audio")
    if sound:
        out.append(f"Music and ambience of this scene: {sound}")
    where = [f for f in current(bible, index) if f["category"] == "spatial"]
    if where:
        out.append("Where people were on the last frame: " + "; ".join(f"{name_of(bible, f['subject'])} {f['value']}" for f in where))
    doing = [f for f in current(bible, index) if f["category"] == "action"]
    if doing:
        out.append("Actions still in progress (continue or resolve them, never restart them): "
                   + "; ".join(f"{name_of(bible, f['subject'])} {f['value']} (since segment {f['from']})" for f in doing))
    return out


def name_of(bible: dict, sid: str) -> str:
    return (bible["subjects"].get(sid) or {}).get("name") or sid


def era_line(bible: dict, index: int) -> str:
    era = era_at(bible, index)
    bits = [f"Era {era['id']}" + (f", {era['label']}" if era["label"] else "") + f" (from segment {era['from']})"]
    if era.get("world"):
        bits.append(f"world: {era['world']}")
    if era["rules"]:
        bits.append("rules: " + "; ".join(era["rules"]))
    return " -- ".join(bits)


def director_text(bible: dict, index: int, names=()) -> str:
    """The bible's slice for the Director writing segment ``index``: the era and its rules, every canonical fact of who
    and what is in play, the scene's light and sound, where people are and what they are doing, the story facts that
    concern them, and the corrections asked for."""

    ids = in_play(bible, index, names)
    lines = [era_line(bible, index)]
    people = [(sid, lines_of(bible, index, sid)) for sid in ids if bible["subjects"][sid]["kind"] == "character"]
    people = [(sid, text) for sid, text in people if text]
    if people:
        lines.append("People (canonical: keep every detail; put the visible ones into the shot when they are in it):")
        lines += [f"- {name_of(bible, sid)}: {text}" for sid, text in people]
    things = [(sid, lines_of(bible, index, sid)) for sid in ids if bible["subjects"][sid]["kind"] != "character"]
    things = [(sid, text) for sid, text in things if text]
    if things:
        lines.append("Places and things: " + "; ".join(f"{name_of(bible, sid)} ({text})" for sid, text in things))
    lines += scene_lines(bible, index)
    story, more = story_for(bible, ids, 8)
    if story:
        lines.append("What has happened (never contradict it)" + (f"; {more} earlier facts are kept in the bible" if more else "") + ":")
        lines += [f"- {s.get('t') or ''} {s['text']}".replace("-  ", "- ") for s in story]
    fixes = bible.get("corrections") or []
    if fixes:
        lines.append("Continuity corrections from the last checks (make these right in this segment):")
        lines += [f"- {c['note']}" for c in fixes]
    return "\n".join(lines)


def check_text(bible: dict, index: int, names=(), continuation: bool = True) -> str:
    """The shorter slice the eyes check a segment against."""

    ids = in_play(bible, index, names)
    lines = [era_line(bible, index)]
    lines += [f"- {name_of(bible, sid)}: {text}" for sid in ids for text in [lines_of(bible, index, sid)] if text]
    if continuation:
        lines += scene_lines(bible, index)
    else:
        lines.append("This segment opens a new scene: positions, actions in progress and light start fresh.")
    story, _ = story_for(bible, ids, CHECK_STORY)
    if story:
        lines.append("What has happened: " + "; ".join(s["text"] for s in story[-CHECK_STORY:]))
    return "\n".join(lines)


# ---------- the check's answer ----------

CHECK_SYSTEM = ("You are the continuity supervisor of a film generated one short segment at a time. You compare the frames "
                "of a new segment with the film's continuity bible and report only problems you can see, with evidence. "
                "Answer only with JSON.")


def check_prompt(bible: dict, index: int, seg: dict, count: int, transcript: str, categories: list, clock: str,
                 names=(), continuation: bool = True, style: str = "") -> str:
    asked = (seg.get("fields") or {}).get("action", "")
    place = (seg.get("fields") or {}).get("setting", "")
    return (f"CONTINUITY CHECKLIST for segment {index}. The {count} images are its frames in time order, numbered 1 to {count} "
            f"by their captions (the final frame is number {count}), {clock} on the film's clock.\n"
            + (f"What is heard (speech to text, may be rough): {transcript}\n" if transcript else "What is heard: no speech.\n")
            + f"The segment was ASKED to show (a request, not a fact): {asked}\n"
            + (f"The setting it asked for: {place}" + (f" (scene: {seg['scene']})" if seg.get("scene") else "") + "\n" if place else "")
            + (f"Its visual style: {style}\n" if style else "")
            + "\nThe continuity bible, what the film has established before this segment (canonical):\n" + check_text(bible, index, names, continuation)
            + "\n\nCheck these categories against the bible and against the frames themselves:\n"
            + "\n".join(f"- {c}: {QUESTIONS[c]}" for c in categories)
            + "\n\nAnswer with this JSON:\n"
            '{"checks": {"<category>": {"verdict": "ok|minor|major|not_visible", "frame": <the number of the image that shows the problem, 0 when ok>, '
            '"evidence": "<what is wrong and where, 20 words max; empty when ok>", "subject": "<who or what; empty when ok>", '
            '"box": [x0, y0, x1, y1] <optional: where it is, as fractions 0 to 1 of the picture>}}, '
            '"facts": [{"category": "identity|wardrobe|props|location|lighting|story", "subject": "<name>", "key": "<face|hair|eyes|age|build|marks|wardrobe|state|held|layout|landmark|light|weather|time|...>", "value": "<12 words max>"}], '
            '"present": [{"who": "<name>", "frame": <the image with the clearest view of their face>}], '
            '"positions": [{"who": "<name>", "first": "left|center|right|absent", "last": "left|center|right|absent", "facing": "left|right|camera|away", "moving": "left|right|toward|away|still"}], '
            '"open_actions": [{"who": "<name>", "action": "<still in progress on the final frame, 8 words max>"}], '
            '"light": "<time of day, light and weather, 12 words max>"}\n\n'
            "Rules that matter:\n"
            "- Judge only what the frames show. A category whose subject is out of shot, turned away, too small, too dark or blurred is "
            "\"not_visible\", never a problem. A category the bible says nothing about is \"ok\" unless the frames contradict each other.\n"
            "- Every \"minor\" or \"major\" needs evidence: the number of the image where it shows and what exactly is wrong there. Without both it is \"ok\".\n"
            "- These are NOT continuity breaks: getting on or off a vehicle, turning, crouching, a new camera angle or distance, motion blur, "
            "a change of light, a coat taken off or a thing picked up or put down on screen, a person leaving the frame.\n"
            "- \"major\": a different person's face, age or build; an object, garment or vehicle from the wrong era; something that contradicts "
            "what has happened; an action in progress dropped or restarted; a held prop gone with no cause. \"minor\": a small detail off "
            "(a scar hidden or on the other cheek, a colour slightly different, a prop moved).\n"
            "- \"facts\": only clear details the bible lacks or that changed on screen, at most 6. \"present\": every character you can see. "
            "\"positions\" and \"open_actions\": as the frames show them. Any list may be empty.")


def keyframe_prompt(bible: dict, index: int, seg: dict, prompt: str, categories: list, names=()) -> str:
    """The keyframe's checklist: one still, the categories a picture can show, the same JSON as a segment's."""

    return (f"KEYFRAME CHECK for segment {index} of a film: the image (number 1) is the still this segment will open on, made from "
            f"this prompt: {prompt}\n"
            + f"The segment's setting: {(seg.get('fields') or {}).get('setting', '')} (scene: {seg.get('scene', '')})\n"
            + "\nThe continuity bible (canonical):\n" + check_text(bible, index, names, False)
            + "\n\nCheck these categories against the bible and the setting asked for:\n" + "\n".join(f"- {c}: {QUESTIONS[c]}" for c in categories)
            + '\n\nAnswer: {"checks": {"<category>": {"verdict": "ok|minor|major|not_visible", "frame": <1 when there is a problem, else 0>, '
            '"evidence": "<what is wrong, 20 words max; empty when ok>", "subject": "<who or what>"}}}\n'
            "A category whose subject is not in the picture is \"not_visible\". A problem needs evidence: what exactly is wrong.")


def check_problem(data: dict) -> str:
    checks = data.get("checks")
    if not isinstance(checks, (dict, list)):
        return '"checks" is an object with one entry a category'
    for key in ("facts", "present", "positions", "open_actions"):
        if data.get(key) is not None and not isinstance(data.get(key), list):
            return f'"{key}" is a list'
    return ""


def verdict_of(value) -> str:
    text = slug(value)
    if text in ("not_visible", "notvisible", "n_a", "na", "unknown", "unclear", "not_applicable"):
        return "not_visible"
    return text if text in VERDICTS else "ok"


def triage(data: dict, count: int, categories: list) -> dict:
    """{category: entry} from the eyes' answer: ``status`` ok, not_visible, no_evidence (a problem with no image number
    or no words), benign (words that say nothing is wrong), minor or major. Categories it left out are not_visible."""

    raw = data.get("checks")
    if isinstance(raw, list):  # [{"category": ..., "verdict": ...}] is read the same
        raw = {str(c.get("category")): c for c in raw if isinstance(c, dict)}
    out = {}
    for category in categories:
        entry = raw.get(category) if isinstance(raw, dict) else None
        entry = entry if isinstance(entry, dict) else ({"verdict": entry} if isinstance(entry, str) else {})
        verdict = verdict_of(entry.get("verdict")) if entry else "not_visible"
        try:
            frame = int(entry.get("frame") or 0)
        except (TypeError, ValueError):
            frame = 0
        evidence = clip(entry.get("evidence"), 25)
        status = verdict
        if verdict in ("minor", "major"):
            if not evidence or not 1 <= frame <= count:
                status = "no_evidence"
            elif BENIGN.search(evidence):
                status = "benign"
        box = entry.get("box")
        box = [float(x) for x in box] if isinstance(box, list) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box) else None
        if box and not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 and (box[2] - box[0]) * (box[3] - box[1]) >= 0.01):
            box = None
        out[category] = {"verdict": verdict, "status": status, "frame": frame, "evidence": evidence,
                         "subject": clip(entry.get("subject"), 4), **({"box": box} if box else {})}
    return out


def act(entry: dict, opts: dict) -> str:
    """What a finding does once it has been looked at again: ``retake``, ``correct`` (the Director is told) or ``none``."""

    status = entry["status"]
    if status == "major":
        return "retake" if entry["category"] in STRICTNESS[opts["strictness"]]["retake"] else "correct"
    if status == "minor":
        return "correct"
    return "none"


def settle(entry: dict, look: dict | None, opts: dict) -> dict:
    """A major finding after its second look: confirmed stays major, not seen is filtered (a false positive), and a
    look that could not be made follows the strictness (``unsure``)."""

    if entry["status"] != "major":
        return entry
    entry["second_look"] = look or {}
    if look and look.get("confirmed") is True:
        return entry
    if look and look.get("confirmed") is False:
        entry["status"] = "filtered"
        return entry
    unsure = STRICTNESS[opts["strictness"]]["unsure"]
    entry["status"] = {"drop": "filtered", "minor": "minor", "major": "major"}[unsure]
    return entry


SIDES = {"left": 0, "center": 1, "centre": 1, "middle": 1, "right": 2}


def screen_flip(bible: dict, index: int, positions: list) -> list[dict]:
    """The 180-degree rule, measured rather than asked: in a continuation, two people who swap sides of the frame
    between the last frame of the segment before and this one's first, or a move that reverses its screen direction."""

    before = {f["subject"]: f for f in current(bible, index) if f["category"] == "spatial" and f["key"] == "position"}
    moving_before = {f["subject"]: f["value"] for f in current(bible, index) if f["category"] == "spatial" and f["key"] == "moving"}
    now, moving_now = {}, {}
    for p in positions or []:
        if not isinstance(p, dict):
            continue
        sid = subject_of(bible, p.get("who"), "character", create=False)
        if sid and str(p.get("first") or "").lower() in SIDES:
            now[sid] = SIDES[str(p["first"]).lower()]
        if sid and str(p.get("moving") or "").lower() in ("left", "right"):
            moving_now[sid] = str(p["moving"]).lower()
    then = {sid: SIDES.get(str(f["value"]).split()[0].lower()) for sid, f in before.items()}
    out = []
    people = [sid for sid in now if then.get(sid) is not None]
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            was, is_ = then[a] - then[b], now[a] - now[b]
            if was and is_ and (was > 0) != (is_ > 0):
                out.append({"category": "spatial", "status": "minor", "verdict": "minor", "frame": 1, "source": "rule", "subject": name_of(bible, a),
                            "evidence": f"{name_of(bible, a)} and {name_of(bible, b)} swap sides of the frame across the join (the 180-degree line is crossed)"})
    for sid, way in moving_now.items():
        if moving_before.get(sid) in ("left", "right") and moving_before[sid] != way:
            out.append({"category": "spatial", "status": "minor", "verdict": "minor", "frame": 1, "source": "rule", "subject": name_of(bible, sid),
                        "evidence": f"{name_of(bible, sid)} moved screen {moving_before[sid]} and now moves screen {way}"})
    return out[:4]


def note_of(entry: dict) -> str:
    """A finding as a correction the Director can act on."""

    return clip(f"{entry['category']}: {entry.get('subject') + ', ' if entry.get('subject') else ''}{entry['evidence']}"
                f" -- keep it as the bible says", 40)


def score(take: dict) -> int:
    """How bad a take is: a failed gate, then confirmed major findings, then minor ones (lower is better)."""

    found = (take.get("continuity") or {}).get("findings") or []
    gate = sum(1 for r in take.get("reasons") or [] if str(r).startswith(("audio gate", "identity gate")))
    return (100 * (gate + int(bool((take.get("watch") or {}).get("identity_drift"))))
            + 10 * sum(1 for f in found if f.get("action") == "retake") + sum(1 for f in found if f.get("action") == "correct"))


def best(takes: list) -> dict:
    """The best take; the latest of equals (it carries the most corrections)."""

    return min(reversed(takes), key=score)


SECOND_SYSTEM = ("You are a careful continuity checker. You answer one question about the pictures, only from what you can see. "
                 "Answer only with JSON.")
FOCUS = {
    "identity": "Look only at the person's FACE, HAIR, AGE and BUILD. Clothing, props, vehicles, pose, distance, motion blur and light do not matter.",
    "wardrobe": "Look only at the clothing and its state (wet, torn, muddy).",
    "props": "Look only at the object named and who holds it.",
    "era": "Look only for an object, garment, vehicle or building that clearly does not belong to the era: {rules}.",
    "location": "Look only at the layout and the landmarks of the place.",
    "setting": "Look only at the place itself: is it the place this segment asked for, or does part of another setting show in it?",
    "lighting": "Look only at the time of day, the light and the weather.",
    "spatial": "Look only at where the people are in the frame and which way they face and move.",
    "action": "Look only at whether the action in progress continues or visibly resolves.",
    "story": "Look only at whether the picture contradicts what has happened.",
}


def second_prompt(bible: dict, index: int, entry: dict, pictures: list) -> str:
    """The narrower second look at a major finding: the one category, the established facts, and the reported problem."""

    sid = subject_of(bible, entry.get("subject"), "character", create=False) if entry.get("subject") else None
    facts = lines_of(bible, index, sid) if sid else ""
    rules = "; ".join(era_at(bible, index)["rules"]) or era_at(bible, index).get("world") or "the film's setting"
    shown = "; ".join(f"image {i + 1}: {caption}" for i, (caption, _) in enumerate(pictures))
    return (f"A FIRST LOOK REPORTED a continuity problem in segment {index} of a film ({entry['category']}, frame {entry['frame']}): "
            f"\"{entry['evidence']}\"\nThe images: {shown}.\n"
            + (f"What the film has established about {name_of(bible, sid)}: {facts}\n" if facts else "")
            + FOCUS.get(entry["category"], "Look closely.").format(rules=rules) + "\n"
            "Is that problem clearly there? If the subject is not clearly visible, or the difference comes from pose, distance, angle, "
            "light, motion blur, or a change that happens on screen, it is not.\n"
            'Answer: {"confirmed": true | false, "why": "<15 words max>"}')


# ---------- after a segment is kept ----------

def absorb(bible: dict, index: int, result: dict, events: list) -> dict:
    """The kept take's check written into the bible (the subjects seen, with their frames, are returned): what it established (identity once an era, everything else as
    its new state), who was seen (and the frame that shows them, the reference of a person without a picture), the
    positions and actions in progress at its end, the light, what happened (the watcher's events, never folded), and
    the corrections for the next segment."""

    seen = {}
    if not result or result.get("skipped"):
        for event in events:
            add_story(bible, index, event)
        return seen
    for fact in result.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        category = fact.get("category") if fact.get("category") in CATEGORIES else None
        if not category or not fact.get("value"):
            continue
        if category == "story":
            add_story(bible, index, {"text": clip(fact["value"], 16), "subjects": [subject_of(bible, fact.get("subject"), "object")] if fact.get("subject") else []})
            continue
        kind = {"location": "place", "setting": "place", "lighting": "place", "props": "object"}.get(category, "character")
        sid = "_scene" if category == "lighting" else subject_of(bible, fact.get("subject"), kind)
        key = KEYS.get(slug(fact.get("key")), slug(fact.get("key"))) or category
        if category == "identity":
            sheet = any(f["key"] == "look" and f["source"] in ("seed", "director") for f in current(bible, index, sid))
            if key == "look" or (sheet and key not in ("marks", "eyes")):
                continue  # a sheet's look is canonical; the eyes add only what it does not say (marks, eye colour)
            establish(bible, sid, category, key, fact["value"], index, "watch", once=key in LOOKED_ONCE)
        else:
            establish(bible, sid, category, key, fact["value"], index, "watch")
    if result.get("light"):
        establish(bible, "_scene", "lighting", "light", result["light"], index, "watch")
    for p in result.get("present") or []:
        sid = subject_of(bible, p.get("who"), "character") if isinstance(p, dict) else None
        if sid and sid != "_scene":
            seen[sid] = p.get("frame") or seen.get(sid)
    for sid, frame in seen.items():
        subject = bible["subjects"].get(sid)
        if subject is None:
            continue
        if index not in subject["seen"]:
            subject["seen"].append(index)
        if not subject.get("ref") and frame:
            subject["ref"] = frame
    # positions and actions as of the last frame replace the ones before (an action no longer in progress is resolved)
    for fact in current(bible, index):
        if fact["category"] in ("spatial", "action") and fact["from"] < index + 1:
            fact["to"] = index
    for p in result.get("positions") or []:
        if not isinstance(p, dict):
            continue
        sid = subject_of(bible, p.get("who"), "character", create=False)
        if not sid or str(p.get("last") or "").lower() not in SIDES:
            continue
        side = str(p["last"]).lower()
        establish(bible, sid, "spatial", "position", side + (f", facing {p['facing']}" if p.get("facing") else ""), index + 1, "watch")
        if str(p.get("moving") or "").lower() in ("left", "right", "toward", "away"):
            establish(bible, sid, "spatial", "moving", str(p["moving"]).lower(), index + 1, "watch")
    for a in result.get("open_actions") or []:
        if isinstance(a, dict) and a.get("action"):
            sid = subject_of(bible, a.get("who"), "character", create=False)
            if sid:
                establish(bible, sid, "action", "doing", clip(a["action"], 8), index + 1, "watch")
    for event in events:
        add_story(bible, index, event)
    # the corrections: this segment's findings replace earlier ones of the same kind and subject; a category that
    # checked ok is put right
    fixed = {c for c, e in (result.get("checks") or {}).items() if e.get("status") == "ok"}
    found = [f for f in result.get("findings") or [] if f.get("action") in ("correct", "retake")]
    keys = {(f["category"], slug(f.get("subject"))) for f in found}
    kept = [c for c in bible.get("corrections") or [] if c["category"] not in fixed and (c["category"], slug(c.get("subject"))) not in keys]
    kept += [{"segment": index, "category": f["category"], "subject": f.get("subject") or "", "note": note_of(f)} for f in found]
    bible["corrections"] = kept[-CORRECTIONS:]
    return seen


def add_story(bible: dict, index: int, event: dict) -> None:
    text = event.get("text") or " ".join(bit for bit in (str(event.get("subject") or ""), str(event.get("action") or ""),
                                                          f"-> {event['target']}" if event.get("target") else "") if bit)
    if not text.strip():
        return
    subjects = event.get("subjects")
    if subjects is None:
        subjects = [s for s in (subject_of(bible, event.get(k), "character", create=False) for k in ("subject", "target", "place")) if s]
    if any(s.get("segment") == index and same(s["text"], text) for s in bible["story"]):
        return
    bible["story"].append({"segment": index, "t": event.get("t") or "", "text": clip(text, 20), "subjects": [s for s in subjects if s]})
    del bible["story"][:-STORY_LIMIT]


# ---------- the film-level pass ----------

FILM_SYSTEM = "You compare frames from different segments of one film. Answer only with JSON."


def film_samples(bible: dict, segments: list, limit: int = 8) -> dict:
    """{subject: {era: [(segment, frame path)]}}: each person's clearest frame of each segment they were seen in,
    at most ``limit`` an era, spread through it."""

    out: dict = {}
    for seg in segments:
        seen = ((seg.get("continuity") or {}).get("present_frames") or {})
        for sid, frame in seen.items():
            if sid in bible["subjects"] and frame:
                out.setdefault(sid, {}).setdefault(era_at(bible, seg["index"])["id"], []).append((seg["index"], frame))
    for sid, eras in out.items():
        for era, found in eras.items():
            if len(found) > limit:
                step = (len(found) - 1) / (limit - 1)
                eras[era] = [found[round(i * step)] for i in range(limit)]
    return out


def film_identity_prompt(name: str, look: str, picked: list) -> str:
    listed = ", ".join(f"segment {i}" for i, _ in picked)
    return (f"SAME CHARACTER ACROSS THE FILM. The {len(picked)} images are {name} in different segments of one film, in order: {listed}. "
            + (f"The established look: {look}.\n" if look else "\n")
            + "Look only at the FACE, HAIR, AGE and BUILD. Clothing, pose, light, angle and distance do not matter. Which segments, "
            "if any, clearly show a different person from the others?\n"
            'Answer: {"same": true | false, "different": [<segment numbers>], "why": "<20 words max>"}')


def film_world_prompt(bible: dict, picked: list) -> str:
    eras = "; ".join(era_line(bible, e["from"]) for e in bible["eras"])
    listed = ", ".join(f"segment {i}" for i, _ in picked)
    return (f"ACROSS THE WHOLE FILM. The {len(picked)} images are one frame from each of these segments of one film, in order: {listed}.\n"
            f"The film's eras and their rules: {eras}\n"
            "Report drift between segments that should match: a place whose layout or landmarks change, an object or garment from the "
            "wrong era, a prop that changes, light that jumps within one scene. Only what the images show, with the segments.\n"
            'Answer: {"issues": [{"category": "era|location|props|wardrobe|lighting|other", "segments": [<numbers>], "evidence": "<20 words max>"}]}')
