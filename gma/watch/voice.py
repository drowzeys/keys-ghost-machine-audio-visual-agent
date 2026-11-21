"""What the person says to the Ghost in invisible mode, found in code before any model sees the turn:

    "find the lighthouse" / "search for storms" / "show me the fox clips"   search the ledger, the page shows the matches
    "what's new" / "what just came in" / "latest"                           the newest creations, read in one line
    "what did hermes make"                                                   one agent's creations

The agent takes these words only while the invisible page is open (``service.seen_at``).
"""

from __future__ import annotations

import json
import re
import time

PAGE_S = 45
FIND = re.compile(r"\b(?:find|search(?: for)?|look for|look up|show me|where(?:'s| is| are)|pull up|open)\b\s+(?:me\s+)?(?:the\s+|a\s+|an\s+|all\s+(?:the\s+)?)?(.+)$")
NEW = re.compile(r"\bwhat(?:'s| is| has)? (?:new|just (?:came|come) in|landed|arrived)\b|\b(?:latest|newest|most recent)\b(?: (?:ones?|creations?|things?))?$")
AGENT = re.compile(r"\bwhat (?:did|has|have) (\w[\w .-]{0,30}?) (?:make|made|create|created|do|done|render|rendered)\b")
KIND = {"pictures": "image", "picture": "image", "images": "image", "image": "image", "photos": "image", "stills": "image",
        "clips": "video", "clip": "video", "videos": "video", "video": "video", "films": "video", "film": "video",
        "sounds": "audio", "sound": "audio", "audio": "audio", "music": "audio", "narration": "audio", "songs": "audio",
        "texts": "text", "text": "text", "ideas": "text", "notes": "text"}
FILLER = {"one", "ones", "creation", "creations", "thing", "things", "stuff", "that", "which", "with", "about", "of", "please",
          "from", "made", "by", "for", "me", "the", "a", "an", "all", "some", "any"}


def parse(text: str) -> tuple[str, dict] | None:
    low = re.sub(r"\s+", " ", (text or "").lower()).strip(" .!?")
    if not low:
        return None
    if NEW.search(low):
        return "new", {}
    found = AGENT.search(low)
    if found:
        return "agent", {"agent": found.group(1).strip()}
    found = FIND.search(low)
    if found:
        words, kind = [], ""
        for w in re.findall(r"[\w'-]+", found.group(1)):
            if w in KIND and not kind:
                kind = KIND[w]
            elif w not in FILLER:
                words.append(w)
        if words or kind:
            return "find", {"q": " ".join(words), "kind": kind}
    return None


class WatchAgent:
    kind = "video"  # the server routes a "video" agent's turn to handle() and speaks its one line alone
    always = True
    name = "watch"
    label = "Invisible mode"
    system = None
    ack = ("Looking.",)

    def __init__(self, service) -> None:
        self.service = service

    def page_open(self) -> bool:
        return time.monotonic() - self.service.seen_at < PAGE_S

    def wants(self, text: str) -> bool:
        return self.page_open() and parse(text) is not None

    async def run(self, prompt: str, progress, confirm=None) -> str:
        return (await self.handle(prompt, progress))[0]

    async def handle(self, text: str, progress) -> tuple[str, str]:
        act = parse(text)
        if act is None:
            return "I didn't catch what to look for.", ""
        what, args = act
        ledger = self.service.ledger
        if what == "new":
            found = ledger.search("", limit=3)
            q = ""
        elif what == "agent":
            agents = ledger.stats()["agents"]
            name = next((a for a in agents if args["agent"] in a.lower()), None)
            if not name:
                return f"Nothing from {args['agent']} yet.", ""
            found, q = ledger.search("", agent=name, limit=20), ""
        else:
            q = args["q"]
            found = ledger.search(q, kind=args["kind"], limit=40)
        progress("video", json.dumps({"action": "watch", "what": what, "q": q, "kind": args.get("kind", ""),
                                      "agent": args.get("agent", ""), "ids": [c["id"] for c in found]}))
        if not found:
            return (f"I found nothing for {q}." if q else "Nothing yet."), ""
        line = lambda c: (c.get("summary") or c.get("title") or c["kind"]).rstrip(".")
        if what == "new":
            return "Newest: " + "; then ".join(line(c) for c in found[:3]) + ".", ""
        many = len(found)
        return (f"{many} found. The best: {line(found[0])}." if many > 1 else f"Found it: {line(found[0])}."), \
            "\n".join(f"{c['id']} · {line(c)}" for c in found[:10])
