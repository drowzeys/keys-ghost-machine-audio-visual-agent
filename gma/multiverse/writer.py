"""The Director's side of the multiverse: the opening shot, two or three genuinely different next directions at a node,
and a mix of two. The questions go to the profile's Director brain (``brains.director``, else the main brain) through
the brain's director lane (gma/lanes.py: it yields to every voice turn), the same brain and lane AutoPilot's Director
uses. With no brain, or no usable answer, a plain fallback writes the options from fixed direction kinds, so the tree
always moves (and development runs without a model)."""

from __future__ import annotations

import json
import random
import re

SYSTEM = ("You are the Director of a short concept film, made clip by clip with a video model. You write vivid, "
          "concrete, physical, photographable shots. No real, named people. Never say what must not be seen.")

OPENING = ("The concept to test: {idea}\n\nWrite the film's opening shot ({seconds} seconds). Reply with JSON only: "
           "{{\"label\": \"the shot in under nine plain words\", \"beat\": \"the story beat in under twelve words\", "
           "\"mood\": \"one or two words\", \"camera\": \"the camera's approach in a few words\", \"prompt\": \"two to "
           "four plain sentences: what the camera sees and hears, the subject, place and light, the movement of subject "
           "and camera\"}}")

OPTIONS = ("The concept being tested: {idea}\n\nThe path so far, one clip a line:\n{beats}\n\nThe last clip's prompt:\n"
           "{last}\n\nOffer {n} different directions for the next {seconds}-second clip, each opening on the last frame "
           "of the last clip. They must be genuinely different choices for the person testing the concept: a different "
           "story beat each, and also a different mood or camera approach; never two versions of the same idea. Same "
           "people, place and look unless a direction moves the story on.{avoid}{note}\n\nReply with JSON only: "
           "{{\"options\": [{{\"label\": \"what happens, under nine plain words, spoken aloud (\\\"he climbs to the "
           "lamp\\\")\", \"beat\": \"the story beat in under twelve words\", \"mood\": \"one or two words\", \"camera\": "
           "\"the camera's approach in a few words\", \"prompt\": \"two to four plain sentences: what the camera sees and "
           "hears, the movement of subject and camera\"}}]}}")

MIX = ("The concept being tested: {idea}\n\nThe path so far:\n{beats}\n\nTwo directions offered for the next clip:\n"
       "A: {a}\nB: {b}\n\nWrite ONE direction that combines them (the beat of one with the other's mood, camera or "
       "event), opening on the last frame of the last clip. Reply with JSON only: {{\"label\": \"under nine plain "
       "words\", \"beat\": \"under twelve words\", \"mood\": \"one or two words\", \"camera\": \"a few words\", "
       "\"prompt\": \"two to four plain sentences\"}}")

# the fallback's direction kinds: each a different beat, mood and camera, so its options are never near-duplicates
KINDS = [
    {"label": "the stakes rise", "beat": "something threatens what {who} wants", "mood": "tense", "camera": "slow push in",
     "prompt": "The danger grows: the light hardens, wind and sound rise, and {who} braces against it. The camera pushes in slowly."},
    {"label": "a hidden thing is revealed", "beat": "{who} discovers something unexpected", "mood": "mysterious",
     "camera": "low angle reveal", "prompt": "{who} notices something that was hidden until now and turns toward it. The camera "
     "tilts up from a low angle to reveal it."},
    {"label": "a quiet, human moment", "beat": "{who} pauses and the film breathes", "mood": "tender", "camera": "close-up, still",
     "prompt": "{who} stops, breathes and looks at what matters to them; warm light settles. The camera holds a still close-up."},
    {"label": "the world answers back", "beat": "something far away responds", "mood": "awe", "camera": "wide pull back",
     "prompt": "Far in the distance something answers: a light, a sound, a shape. The camera pulls back wide to show {who} small "
               "against it."},
    {"label": "a sudden turn of action", "beat": "{who} acts at once", "mood": "urgent", "camera": "handheld, fast",
     "prompt": "{who} moves fast, decided, crossing the space in a few strides. The camera follows handheld, close behind."},
    {"label": "a stranger arrives", "beat": "someone new enters", "mood": "uneasy", "camera": "over the shoulder",
     "prompt": "A stranger steps into view at the edge of the place; {who} turns to face them. The camera watches over {who}'s shoulder."},
    {"label": "the weather turns", "beat": "the place itself changes", "mood": "dramatic", "camera": "time-lapse sky",
     "prompt": "The sky changes fast above {who}: clouds race, the light swings from cold to warm. The camera holds wide as time speeds up."},
    {"label": "an old memory surfaces", "beat": "{who} remembers", "mood": "wistful", "camera": "soft focus drift",
     "prompt": "{who} holds something small and old and the light softens like a memory. The camera drifts in soft focus around them."},
    {"label": "a chase begins", "beat": "something gives chase", "mood": "thrilling", "camera": "tracking alongside",
     "prompt": "{who} runs and something follows close behind through the place. The camera tracks alongside at speed."},
    {"label": "a hard choice", "beat": "{who} must choose", "mood": "torn", "camera": "slow orbit",
     "prompt": "{who} stands between two ways forward, looking from one to the other. The camera orbits slowly around them."},
]

STOP = {"the", "a", "an", "and", "of", "to", "in", "on", "at", "his", "her", "their", "its", "with", "into", "from", "for",
        "is", "are", "as", "by", "it", "he", "she", "they", "them", "this", "that", "one", "up", "out", "over"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']{3,}", (text or "").lower()) if w not in STOP}


def similar(a: dict, b: dict) -> float:
    """How alike two directions are (0-1, Jaccard of their label and beat words)."""
    x, y = words(f"{a.get('label')} {a.get('beat')}"), words(f"{b.get('label')} {b.get('beat')}")
    return len(x & y) / max(1, len(x | y))


def distinct(plans: list[dict], avoid: list[dict] | None = None, limit: float = 0.5) -> list[dict]:
    """The plans with near-duplicates (of each other or of ``avoid``) left out."""
    kept: list[dict] = []
    for p in plans:
        if all(similar(p, q) < limit for q in kept + list(avoid or [])):
            kept.append(p)
    return kept


def parse_json(said: str):
    said = re.sub(r"<think>.*?</think>", "", said or "", flags=re.S)
    found = re.search(r"\{.*\}", said, re.S)
    try:
        return json.loads(found.group(0)) if found else None
    except ValueError:
        return None


def clean_plan(data) -> dict | None:
    if not isinstance(data, dict):
        return None
    plan = {k: re.sub(r"\s+", " ", str(data.get(k) or "")).strip().strip('"') for k in ("label", "beat", "mood", "camera", "prompt")}
    if len(plan["prompt"]) < 20 or not (plan["label"] or plan["beat"]):
        return None
    plan["label"] = plan["label"] or plan["beat"]
    plan["label"] = " ".join(plan["label"].split()[:12])
    return plan


def subject(idea: str) -> str:
    """Who the fallback's directions are about: the idea's first noun phrase, roughly ("a lighthouse keeper")."""
    found = re.search(r"\b(?:a|an|the)\s+((?:[a-z]+\s+){0,2}?[a-z]+)\b", (idea or "").lower())
    return ("the " + found.group(1)) if found else "the hero"


class Writer:
    """``brain`` has ``complete(messages, **settings)`` (gma/brain.py); ``lanes`` is gma.lanes.Lanes or None."""

    def __init__(self, brain=None, lanes=None, seconds: float = 5, think: int = 0) -> None:
        self.brain, self.lanes, self.seconds, self.think = brain, lanes, seconds, think
        self.asked = 0  # questions sent to the brain (the tests read it)

    async def ask(self, prompt: str, max_tokens: int = 900) -> str:
        if self.brain is None:
            return ""
        self.asked += 1
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        settings = {"max_tokens": max_tokens, "temperature": 0.9, "timeout": 300}
        if self.think:
            settings.update(chat_template_kwargs={"enable_thinking": True}, thinking_budget=self.think, max_tokens=self.think + max_tokens)
        try:
            if self.lanes is not None:
                return await self.lanes.director(lambda: self.brain.complete(messages, **settings), retries=30)
            return await self.brain.complete(messages, **settings)
        except Exception:  # noqa: BLE001 - no brain, held, yielded too often: the fallback writes it
            return ""

    async def opening(self, idea: str) -> dict:
        plan = clean_plan(parse_json(await self.ask(OPENING.format(idea=idea, seconds=self.seconds), 500)))
        if plan:
            return plan
        return {"label": "The opening", "beat": idea[:100], "mood": "", "camera": "establishing shot",
                "prompt": f"{idea}. An establishing shot: the place, the light and {subject(idea)}, the camera drifting slowly in."}

    async def options(self, idea: str, story: list[dict], n: int = 3, avoid: list[dict] | None = None, note: str = "",
                      seed: int = 0) -> list[dict]:
        """``n`` distinct directions from the end of ``story``; ``avoid`` are directions already offered here."""
        beats = "\n".join(f"{i + 1}. {s['beat'] or s['label']}" for i, s in enumerate(story[-8:]))
        avoid_line = ("\nAlready offered here (offer different ones): " + "; ".join(a.get("label") or "" for a in avoid)) if avoid else ""
        said = await self.ask(OPTIONS.format(idea=idea, beats=beats, last=story[-1]["prompt"], n=n, seconds=self.seconds,
                                             avoid=avoid_line, note=f"\nThe person asks: {note}" if note else ""))
        data = parse_json(said)
        plans = [p for p in (clean_plan(o) for o in ((data or {}).get("options") or [])) if p] if isinstance(data, dict) else []
        walked = list(avoid or []) + story[1:]  # never a direction already offered here, nor a beat already walked
        plans = distinct(plans, walked)[:n]
        if len(plans) < n:
            plans += distinct(self.fallback(idea, story, len(KINDS), seed + len(avoid or [])), plans + walked)[: n - len(plans)]
        return plans[:n]

    async def mix(self, idea: str, story: list[dict], a: dict, b: dict) -> dict:
        beats = "\n".join(f"{i + 1}. {s['beat'] or s['label']}" for i, s in enumerate(story[-8:]))
        line = lambda o: f"{o['label']} ({o['mood']}, {o['camera']}): {o['prompt']}"  # noqa: E731
        plan = clean_plan(parse_json(await self.ask(MIX.format(idea=idea, beats=beats, a=line(a), b=line(b)), 500)))
        if plan:
            return plan
        return {"label": f"{a['label']}, and {b['label']}"[:100], "beat": f"{a['beat']}; then {b['beat']}"[:160],
                "mood": b["mood"] or a["mood"], "camera": a["camera"] or b["camera"],
                "prompt": f"{a['prompt']} Then: {b['prompt']}"}

    def fallback(self, idea: str, story: list[dict], n: int, seed: int = 0) -> list[dict]:
        who = subject(idea)
        kinds = KINDS[:]
        random.Random(seed).shuffle(kinds)
        out = []
        for k in kinds[:n]:
            fill = lambda s: s.replace("{who}", who)  # noqa: E731
            prompt = fill(k["prompt"])
            out.append({"label": fill(k["label"]), "beat": fill(k["beat"]), "mood": k["mood"], "camera": k["camera"],
                        "prompt": prompt[0].upper() + prompt[1:] + f" ({idea[:120]})"})
        return out
