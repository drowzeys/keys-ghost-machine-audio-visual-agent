"""d1-2B (LiquidAI's decision model) as an optional assist: quick calibrated answers where Boo's code is unsure.

d1 answers questions about a short state in one forward pass, with probabilities and no generated text, through
TensorFold's ``/v1/systemone`` route (``drowzeys/TensorFold`` branch ``d1-2b-cuda-wip`` until it lands upstream). Boo
asks it three things:

- **route** a turn the code router could not place (a camera, side, finger, memory, live-fact or tool request said in
  words the patterns miss), so the turn takes its fast path at once instead of a front-brain reply and a hand-over;
- **permission**: whether a reply to "may it run this?" means once, always or no, when the words patterns miss it;
- **clips** (``Checker`` in video.py, where the video panel is): yes/no checks on a clip's frames.

Everything here is optional (the profile's ``d1`` section) and bounded: a call that is slower than ``timeout_ms``, a
server that is down, or an answer under ``threshold`` (``thresholds`` per answer; "task" 0.85 unless set) gives
``None``, and Boo does what it did without d1.
"""

from __future__ import annotations

import asyncio
import time

import httpx

# what each route means, as d1 reads it; only the routes this Boo can act on are offered (video and studio where those
# back agents are in the profile)
ROUTES = {
    "video": "Asks Boo to make, render or generate a video, clip or film",
    "studio": "Asks about Imagination Studio: the render queue, shots, takes, films, rendering or revising a shot",
    "camera": "Asks what the camera sees: the person, their clothes, objects or the room",
    "side": "Asks about left or right: which hand, which side something is on",
    "fingers": "Asks how many fingers are held up or which hand is raised",
    "fact": "Asks for a live fact: news, weather, scores, prices, who holds an office now",
    "task": "Asks for real work with tools: search the web deeply, write or edit files, run commands, send something",
    "memory": "Asks what happened earlier or what Boo saw before",
    "chat": "Ordinary conversation, opinions, jokes, or talk about videos without asking for one",
}

PERMISSION = {
    "allow_once": "Yes, go ahead this time",
    "allow_session": "Yes, and don't ask again for the rest of this task",
    "deny": "No, don't do it, stop or wait",
    "other": "Not an answer to the question: something else entirely",
}


class D1:
    """A d1-2B server (``url``) asked with a deadline. ``threshold`` is the least confidence acted on."""

    def __init__(self, spec: dict, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = spec["url"].rstrip("/")
        self.path = spec.get("path", "/v1/systemone")
        self.threshold = float(spec.get("threshold", 0.8))
        # a stricter bar where a wrong answer costs most: a chat sent to a back agent waits half a minute for tools
        self.thresholds = {"task": 0.85, **{k: float(v) for k, v in spec.get("thresholds", {}).items()}}
        self.timeout_s = float(spec.get("timeout_ms", 60)) / 1000
        self.clip_timeout_s = float(spec.get("clip_timeout_ms", 2000)) / 1000
        self.route_on = spec.get("route", True)
        self.permission_on = spec.get("permission", True)
        key = spec.get("key")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(10, connect=1), headers=headers, transport=transport)
        self.last_ms: float | None = None  # how long the last call took (or None when it gave up)

    async def ask(self, state: str, questions: dict, images: list[str] | None = None,
                  timeout_s: float | None = None) -> dict | None:
        """The answers to ``questions`` about ``state`` (and ``images``, data URLs), or None if d1 is slow or down."""

        body = {"state": state, "questions": questions}
        if images:
            body["images"] = images
        start = time.perf_counter()
        try:
            r = await asyncio.wait_for(self.client.post(self.url + self.path, json=body), timeout_s or self.timeout_s)
            r.raise_for_status()
            answers = r.json()["answers"]
        except (asyncio.TimeoutError, httpx.HTTPError, ValueError, KeyError, TypeError):
            self.last_ms = None
            return None
        self.last_ms = (time.perf_counter() - start) * 1000
        return answers

    async def choose(self, state: str, instructions: str, criteria: dict) -> tuple[str, float] | None:
        answers = await self.ask(state, {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})
        try:
            got = answers["q"]
            choice, confidence = got["choice"], float(got.get("confidence", got["probabilities"][got["choice"]]))
        except (TypeError, KeyError, ValueError):
            return None
        return (choice, confidence) if choice in criteria and confidence >= self.thresholds.get(choice, self.threshold) else None

    async def route(self, text: str, offered: set[str]) -> str | None:
        """Where a turn the code could not place should go: one of ``offered`` (and never "chat"), or None."""

        if not self.route_on or len(offered) < 2:
            return None
        criteria = {name: ROUTES[name] for name in ROUTES if name in offered or name == "chat"}
        got = await self.choose(text, "Which kind of request is this spoken turn?", criteria)
        return got[0] if got and got[0] != "chat" else None

    async def permission(self, text: str) -> str | None:
        """``allow_once``, ``allow_session`` or ``deny`` for a reply to a tool-permission question, or None."""

        if not self.permission_on:
            return None
        got = await self.choose(f'Asked "May it go ahead?", the person said: "{text}"',
                                "What does this reply to the permission question mean?", PERMISSION)
        return got[0] if got and got[0] != "other" else None

    async def close(self) -> None:
        await self.client.aclose()
