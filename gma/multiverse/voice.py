"""What the person says to the Ghost about the multiverse, found in code before any model sees the turn (as the video
panel's own requests are, voice_agent/video.py ``VideoAgent``), and the voice agent that answers it in one short line.

    "make a film about a lighthouse keeper in a storm"     a new film (its opening, then the first options)
    "take the left path" / "option B" / "the second one"   pick (left = the top branch, right = the bottom one)
    "the one where he opens the door"                      pick by the option's own words
    "go back" / "go back to the start" / "back to step 2"  the Ghost walks back, to branch from there
    "show me other options here" / "something else"        a new round of options at the Ghost
    "mix A with C"                                         a new option combining two
    "export this path"                                     the active timeline as one film
    "what are the options?"                                the Ghost reads them again

The agent takes these words only while the multiverse page is open (``service.seen_at``), so the classic page's video
panel keeps its own "make a video of ...".
"""

from __future__ import annotations

import re
import time

from .tree import Tree, TreeError
from .writer import words as content_words

LETTER = r"(?:option|path|branch|road|letter|number)?\s*\b([abc])\b"
NEW = re.compile(r"\b(?:make|start|create|test|try|shoot|begin|do|build)\b(?:\s+\w+){0,3}?\s+(?:film|video|movie|concept|story|short)s?\b"
                 r"(?:\s*[:,-]?\s*(about|of|where|with|in which|called|on)\b)?\s*(.*)$")
OTHERS = re.compile(r"\b(?:other|different|new|more|fresh)\s+(?:options|choices|paths|directions|ideas|branches)\b|\bsomething else\b|"
                    r"\bshow me (?:some )?others\b|\bnone of (?:these|those|them)\b")
MIX = re.compile(r"\b(?:mix|combine|blend|merge|cross)\s+(?:option\s+|path\s+)?([abc])\b\s*(?:with|and|plus|\+|&)\s*(?:option\s+|path\s+)?([abc])\b")
EXPORT = re.compile(r"\bexport\b|\b(?:render|make|save|join)\s+(?:this|the)\s+(?:path|timeline|whole film)\b")
BACK = re.compile(r"\b(?:go|step|jump|head|walk|take me)\s+back\b|\bback up\b|\bundo\b|\bgo to (?:the )?(?:start|beginning)\b")
BACK_TO = re.compile(r"\b(?:start|beginning|opening|first (?:clip|step|shot))\b|\b(?:step|clip|shot|node)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b")
READ = re.compile(r"\bwhat are (?:the|my|our) (?:options|choices|paths)\b|\b(?:read|tell|say) (?:me )?(?:the )?(?:options|choices|paths)\b|"
                  r"\bwhat can (?:i|we) (?:pick|choose)\b")
PICK_VERB = re.compile(r"\b(?:take|pick|choose|go|follow|select|want|like|let'?s|try|i'?ll have|do)\b")
ORDINAL = {"first": 0, "top": 0, "upper": 0, "left": 0, "second": 1, "middle": 1, "center": 1, "centre": 1, "third": 2,
           "bottom": -1, "lower": -1, "right": -1, "last": -1}
ORDINAL_RE = re.compile(r"\b(first|second|third|top|upper|bottom|lower|middle|center|centre|left|right|last)\b(?:\s+(?:one|path|option|branch|road|way))?")
WHERE = re.compile(r"\bthe one (?:where|with|that|in which|when)\b(.*)$")
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
PAGE_S = 45  # the page counts as open this long after it last asked for the film


def match_option(text: str, options: list[dict]) -> dict | None:
    """The option the words mean: a letter, a place ("left", "the second one"), or its own words."""
    low = text.lower().strip()
    if not options:
        return None
    by_letter = {o["letter"].lower(): o for o in options if o.get("letter")}
    found = re.search(r"\b(?:option|path|branch|road|letter)\s+([a-h])\b", low) or re.fullmatch(r"\W*([a-h])\W*", low) \
        or re.search(r"\b(?:take|pick|choose|select|go with|want|try)\s+([b-h])\b", low) \
        or re.search(r"\b(?:take|pick|choose|select|go with)\s+(a)\W*$", low)
    if found and found.group(1) in by_letter:
        return by_letter[found.group(1)]
    where = WHERE.search(low)
    if where or not ORDINAL_RE.search(low):
        query = content_words(where.group(1) if where else low) - {"take", "pick", "choose", "path", "option", "one", "want", "let's",
                                                                    "go", "follow", "where", "please", "branch", "road"}
        scored = sorted(((len(query & content_words(f"{o['label']} {o['beat']} {o['prompt']}")), i) for i, o in enumerate(options)), reverse=True)
        if scored and scored[0][0] >= 1 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return options[scored[0][1]]
        if where:
            return None
    found = ORDINAL_RE.search(low)
    if found:
        i = ORDINAL[found.group(1)]
        if i == 1 and found.group(1) in ("middle", "center", "centre") and len(options) % 2 == 0:
            return None
        return options[i] if -len(options) <= i < len(options) else None
    return None


def parse(text: str, tree: Tree | None) -> tuple[str, dict] | None:
    """(action, details) for the multiverse, or None when the words are not about it."""
    low = re.sub(r"\s+", " ", (text or "").lower()).strip(" .!?")
    if not low:
        return None
    found = NEW.search(low)
    if found:
        idea = (found.group(2) or "").strip(" ,.:-")
        fresh = re.search(r"\b(?:new|another|different)\b", low[: found.start(2)])
        if tree is None or fresh or found.group(1):
            return "new", {"idea": idea if (found.group(1) or tree is None or fresh) else ""}
    if tree is None:
        return None
    if (m := MIX.search(low)):
        return "mix", {"a": m.group(1).upper(), "b": m.group(2).upper()}
    if OTHERS.search(low):
        note = re.sub(OTHERS, "", low).strip(" ,.")
        return "others", {"note": note if len(note.split()) > 3 else ""}
    if EXPORT.search(low):
        return "export", {}
    if READ.search(low):
        return "read", {}
    if BACK.search(low):
        path = tree.active_path()
        target = None
        if (m := BACK_TO.search(low)):
            if m.group(1):
                n = NUMBERS.get(m.group(1)) or int(m.group(1))
                target = path[min(len(path), max(1, n)) - 1]
            else:
                target = path[0]
        elif len(path) > 1:
            target = path[-2]
        return "back", {"node": target}
    options = tree.options()
    if options and (PICK_VERB.search(low) or re.fullmatch(r"(?:option |path )?[a-h]", low) or WHERE.search(low) or ORDINAL_RE.search(low)):
        chosen = match_option(low, options)
        if chosen is not None:
            return "pick", {"node": chosen["id"]}
        if re.search(r"\b(?:path|option|branch|road|one)\b", low):
            return "unclear", {}
    return None


class MultiverseAgent:
    """The voice's handle on the multiverse (registered by routes.register ahead of the video agent). It answers its
    turn itself with one short line, as the video agent does (voice_agent/server.py ``video_turn``)."""

    kind = "video"  # the server routes a "video" agent's turn to handle() and speaks its line alone
    always = True
    name = "multiverse"
    label = "Multiverse"
    system = None
    ack = ("On it.",)

    def __init__(self, service) -> None:
        self.service = service
        self.asked_idea_at: float | None = None

    def page_open(self) -> bool:
        return time.monotonic() - self.service.seen_at < PAGE_S

    def tree(self) -> Tree | None:
        try:
            return self.service.store.load(self.service.focus) if self.service.focus else None
        except TreeError:
            return None

    def wants(self, text: str) -> bool:
        if not self.page_open():
            return False
        if self.asked_idea_at and time.monotonic() - self.asked_idea_at < 90 and len(text.split()) >= 3:
            return True
        return parse(text, self.tree()) is not None

    async def run(self, prompt: str, progress, confirm=None) -> str:
        return (await self.handle(prompt, progress))[0]

    async def handle(self, text: str, progress) -> tuple[str, str]:
        svc, tree = self.service, self.tree()
        if self.asked_idea_at and time.monotonic() - self.asked_idea_at < 90 and parse(text, tree) is None:
            self.asked_idea_at = None
            return await self.start(text, progress)
        act = parse(text, tree)
        if act is None:
            return "", ""
        action, d = act
        try:
            if action == "new":
                if not d["idea"]:
                    self.asked_idea_at = time.monotonic()
                    return ("What's the concept? Tell me in one line.",) * 2
                return await self.start(d["idea"], progress)
            if action == "pick":
                one = await svc.choose(tree.id, d["node"])
                progress("video", _event("pick", tree.id, one["id"]))
                return f"Taking path {one['letter']}: {one['label']}.", f"pick {one['letter']} ({one['id']})"
            if action == "unclear":
                return ("Which one? Say A, B or C, or left, middle or right.",) * 2
            if action == "back":
                if not d["node"]:
                    return ("We're at the start already.",) * 2
                one = await svc.goto(tree.id, d["node"])
                progress("video", _event("goto", tree.id, one["id"]))
                where = "the start" if one["parent"] is None else f"step {one['depth'] + 1}, {one['label']}"
                return f"Back to {where}. We can branch from here.", f"goto {one['id']}"
            if action == "others":
                svc.spawn(svc.request_options(tree.id, tree.ghost, d["note"], fresh=True))
                return ("Okay, other directions from here. One moment.",) * 2
            if action == "mix":
                by_letter = {o["letter"]: o for o in tree.options()}
                if d["a"] not in by_letter or d["b"] not in by_letter or d["a"] == d["b"]:
                    return ("I can only mix two of the options in front of us.",) * 2
                svc.spawn(svc.mix(tree.id, by_letter[d["a"]]["id"], by_letter[d["b"]]["id"]))
                return f"Okay, mixing {d['a']} with {d['b']}. It'll appear as a new path.", f"mix {d['a']}+{d['b']}"
            if action == "export":
                svc.spawn(_export(svc, tree.id, progress))
                return ("Exporting this path as one film. I'll put it up when it's joined.",) * 2
            if action == "read":
                options = tree.options()
                if not options:
                    return ("No paths yet: they come once this clip is made.",) * 2
                return (svc.options_line(options),) * 2
        except TreeError as error:
            return str(error).capitalize() + ".", str(error)
        return "", ""

    async def start(self, idea: str, progress) -> tuple[str, str]:
        tree = await self.service.create(idea)
        progress("video", _event("new", tree.id, tree.ghost))
        return "Starting the film. I'll make the opening, then show you the paths.", f"new film {tree.id}: {idea}"


def _event(action: str, pid: str, nid: str) -> str:
    import json

    return json.dumps({"action": "multiverse", "what": action, "project": pid, "node": nid})


async def _export(svc, pid: str, progress) -> None:
    try:
        await svc.export(pid)
        progress("video", _event("export", pid, svc.store.load(pid).ghost))
        svc.say("The film of this path is ready to download.", "happy")
    except TreeError as error:
        svc.say(f"I couldn't export it: {error}.", "concerned")
