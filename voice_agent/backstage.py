"""The back agents: what the front voice hands a turn to when it needs more than a quick answer.

Two kinds, chosen in the profile's "agents":
  direct  any OpenAI-compatible chat model (Redsnow 5.3 on two Sparks); fast, so its reply streams into the same turn.
  acp     any agent that speaks ACP (Agent Client Protocol: JSON-RPC 2.0, one JSON message per line on stdio) - Hermes,
          Claude Code, Codex, Qwen Code. It can run tools for a minute, so it works in the background and reports when
          done. One warm process and session per agent, like Eyes of Kai's Hermes client (tonyd2wild/eyes-of-kai).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import AsyncIterator, Callable

from .engines import Brain

SAFE_KINDS = {"read", "search", "think", "fetch"}  # tool kinds that look and change nothing: never asked about

SPOKEN = ("Your reply is read aloud by a voice: answer in at most four short spoken sentences, plain words, no lists, "
          "no markdown, no code blocks, no emoji.")


def speakable(text: str) -> str:
    """Agent output as something a voice can read: no markdown, code, links or tables."""

    text = re.sub(r"\[\w+(?: \w+)*\]", " ", text)  # the front brain's mood and hand-off tags, copied from the conversation
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"^\s*[#>|*\-+]+\s*", "", text, flags=re.M)
    text = re.sub(r"[*_~|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Direct:
    kind = "direct"

    def __init__(self, name: str, spec: dict) -> None:
        self.name, self.label = name, spec.get("label", name)
        self.system = spec.get("system", SPOKEN)
        self.brain = Brain(spec)

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        async for piece in self.brain.stream(messages):
            yield piece


class Acp:
    """One warm ACP agent process; prompts run one at a time."""

    kind = "acp"

    def __init__(self, name: str, spec: dict) -> None:
        self.name, self.label = name, spec.get("label", name)
        self.command = spec["command"]
        self.cwd = spec.get("cwd", "~")
        self.allow = spec.get("permissions", "allow_once")  # what to answer a tool permission request, or "deny"
        self.proc: asyncio.subprocess.Process | None = None
        self.session: str | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.updates: asyncio.Queue | None = None
        self.next_id = 1
        self.lock = asyncio.Lock()
        self.reader: asyncio.Task | None = None
        self.confirm: Callable | None = None  # async (title, kind) -> "allow_once" | "allow_session" | "deny"
        self.session_allowed = False

    async def _send(self, message: dict) -> None:
        self.proc.stdin.write((json.dumps(message) + "\n").encode())
        await self.proc.stdin.drain()

    async def _call(self, method: str, params: dict, timeout: float | None = 60) -> dict:
        rid, self.next_id = self.next_id, self.next_id + 1
        self.pending[rid] = asyncio.get_running_loop().create_future()
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        reply = await asyncio.wait_for(self.pending[rid], timeout)
        if "error" in reply:
            raise RuntimeError(f"{self.name}: {reply['error'].get('message', reply['error'])}")
        return reply.get("result", {})

    async def _read(self) -> None:
        while line := await self.proc.stdout.readline():
            try:
                message = json.loads(line)
            except ValueError:
                continue  # not protocol (stdout noise)
            if "id" in message and message.get("method") is None:
                future = self.pending.pop(message["id"], None)
                if future and not future.done():
                    future.set_result(message)
            elif message.get("method") == "session/request_permission":
                # answered in its own task: asking the person can take a minute, and the reader must keep reading
                asyncio.create_task(self._permit(message))
            elif message.get("method") == "session/update" and self.updates:
                update = message["params"]["update"]
                kind = update.get("sessionUpdate")
                if kind == "agent_message_chunk":
                    await self.updates.put(("text", update.get("content", {}).get("text", "")))
                elif kind == "tool_call":
                    await self.updates.put(("tool", update.get("title") or update.get("kind") or "working"))
        for future in self.pending.values():  # the process ended
            if not future.done():
                future.set_result({"error": {"message": "the agent process exited"}})
        self.pending.clear()

    async def _permit(self, message: dict) -> None:
        """Answers a tool permission request: by the profile's rule, or by asking the person ("permissions": "ask")."""

        options = message["params"].get("options", [])
        call = message["params"].get("toolCall") or {}
        title, kind = call.get("title") or "a tool", call.get("kind") or ""
        choice = self.allow
        if self.allow == "ask":
            if kind in SAFE_KINDS or self.session_allowed:
                choice = "allow_once"
            elif self.confirm:
                choice = await self.confirm(title, kind)
            else:
                choice = "deny"
        if choice == "allow_session":
            self.session_allowed = True
        pick = next((o for o in options if o["optionId"] == choice), None) or \
            next((o for o in options if o["optionId"].startswith("deny")), options[0] if options else None)
        await self._send({"jsonrpc": "2.0", "id": message["id"],
                          "result": {"outcome": {"outcome": "selected", "optionId": pick["optionId"]}}})
        if self.updates:
            await self.updates.put(("permission", f"{pick['optionId']}: {title}"))

    async def start(self) -> None:
        if self.proc and self.proc.returncode is None and self.session:
            return
        self.proc = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, limit=2**24)
        self.reader = asyncio.create_task(self._read())
        await self._call("initialize", {"protocolVersion": 1, "clientInfo": {"name": "voice-agent", "version": "0.1"},
                                        "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False},
                                                               "terminal": False}})
        cwd = os.path.expanduser(self.cwd) if self.command[0] != "ssh" else self.cwd
        self.session = (await self._call("session/new", {"cwd": cwd, "mcpServers": []}))["sessionId"]

    async def run(self, prompt: str, progress: Callable[[str, str], None], confirm: Callable | None = None) -> str:
        """The agent's whole reply to ``prompt``; ``progress(kind, text)`` hears its tool steps as they happen, and
        ``confirm(title, kind)`` decides tools when the profile says "ask"."""

        async with self.lock:
            self.confirm = confirm
            try:
                await self.start()
            except Exception:
                self.session = None
                raise
            text = await self._turn(f"{prompt}\n\n({SPOKEN})", progress)
            if stalled(text):  # it promised to look instead of looking: the promise would be the last thing heard
                text = await self._turn(NUDGE, progress)
            return speakable(text)

    async def _turn(self, words: str, progress: Callable[[str, str], None]) -> str:
        self.updates = asyncio.Queue()
        call = asyncio.create_task(self._call("session/prompt", {
            "sessionId": self.session, "prompt": [{"type": "text", "text": words}]}, None))
        text = ""
        try:
            while not call.done() or not self.updates.empty():
                getter = asyncio.create_task(self.updates.get())
                done, _ = await asyncio.wait({getter, call}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    kind, value = getter.result()
                    if kind == "text":
                        text += value
                    else:
                        progress(kind, value)
                else:
                    getter.cancel()
            await call
        except asyncio.CancelledError:
            if self.session:
                await self._send({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": self.session}})
            raise
        finally:
            self.updates = None
        return text


NUDGE = ("That reply only said you would look into it, and the user hears nothing after it. Use your tools now (web "
         f"search for anything current) and give the actual answer. ({SPOKEN})")


def stalled(text: str) -> bool:
    """A reply that is only a promise to go and do it ("On it, I'm checking who's president")."""

    low = text.lower()
    return bool(text.strip()) and len(text) < 200 and bool(PROMISE.search(low) or ON_IT.search(low)) and "?" not in text


def plain_tool(title: str, kind: str) -> str:
    """A tool request as a few spoken words."""

    low = title.lower()
    if "execute_code" in low or kind == "execute" or "terminal" in low:
        return "run a command on the computer"
    if kind in ("edit", "delete", "move") or "write" in low or "patch" in low:
        return "change some files"
    if "send" in low or "message" in low or "telegram" in low:
        return "send a message"
    if "browser" in low or "web" in low:
        return "use the web browser"
    return "use " + re.sub(r"[_:]+", " ", title.split(":")[0]).strip()[:40]


YES = re.compile(r"\b(yes|yeah|yep|sure|go ahead|do it|okay|ok|allow|approve|fine|go for it|please do)\b")
ALWAYS = re.compile(r"\b(always|everything|all of them|for the rest|don't ask)\b")
NO = re.compile(r"\b(no|nope|don't|do not|stop|cancel|deny|wait|not now|never mind)\b")


def answer(text: str) -> str | None:
    """A spoken reply to "should I go ahead?": allow_once, allow_session, deny, or None if it is not an answer."""

    low = re.sub(r"\b(don'?t|do not) (ask|check)( me)?( again)?", " always ", text.lower())
    if NO.search(low):
        return "deny"
    if YES.search(low):
        return "allow_session" if ALWAYS.search(low) else "allow_once"
    return None


PROMISE = re.compile(r"\b(i'?ll|i will|let me|i'?m going to|i am going to|i'?m|i am|give me a (sec|second|moment) to)\s+"
                     r"(go |quickly |now |just )?(check|checking|look|looking|find|finding|search|searching|run|running|"
                     r"render|rendering|make|making|create|creating|send|sending|ask|asking|get|getting|pull|pulling|"
                     r"open|opening|start|starting|fetch|fetching|research|researching|write|writing)\b")


ON_IT = re.compile(r"\b(on it|working on it|on the case|one (sec|second|moment)|give me a (sec|second|moment)|"
                   r"hang on|hold on|bear with me|right away)\b")


# the front brain saying it can't: exactly what the back agent is for (live facts, the computer, the web)
CANNOT = re.compile(r"\b((don'?t|do not|can'?t|cannot|unable to|not able to)\s+(have\s+)?(access|browse|check|look|see|search|"
                    r"reach|connect|verify|get|find|confirm|know)|no (access|way to)|real[- ]time|up[- ]to[- ]date|"
                    r"as of my (last|knowledge|training)|knowledge cut ?off|latest information|current events|hermes|"
                    r"back agent)\b")


def promised(text: str) -> bool:
    """Whether the front brain promised to do something, or said it can't (only a back agent can), without handing
    over."""

    low = text.lower()
    return bool(PROMISE.search(low) or ON_IT.search(low) or CANNOT.search(low))


# what speech recognition makes of an agent's name
SOUNDS_LIKE = {"hermes": ("hermès", "her mess", "hermis", "hermies", "hermas", "herms", "air mess", "her miss", "herpes",
                          "harmes", "hermos", "hermez", "her maze")}


# questions whose answer changes after a model's training: the front brain would answer from stale memory
LIVE = re.compile(r"\b(president|prime minister|chancellor|king|pope|ceo of|news|headlines?|weather|forecast|temperature "
                  r"outside|stock|share price|price of|bitcoin|exchange rate|score|who won|election|latest|recent(ly)?|"
                  r"this (week|month|year)'?s?|currently happening|trending|released (today|this))\b")


def asked_for(text: str, agent, live: bool = True) -> bool:
    """Whether the person asked for the back agent by name or for deep thought ("ask Hermes", "think hard"), or (with
    ``live``) for a fact that changes after training."""

    low = text.lower()
    names = {agent.name.lower(), agent.label.lower(), "the agent", "your agent", *SOUNDS_LIKE.get(agent.name, ())}
    return any(f"{verb} {n}" in low for verb in ("ask", "tell", "have", "get") for n in names) or \
        any(w in low for w in ("think hard", "think deeply", "deep think", "in the background")) or \
        (live and bool(LIVE.search(low)))


class Ack:
    """Stands in for the front brain when a turn goes straight to a working back agent: a short line, no guessing."""

    LINES = ("Let me check.", "On it, one moment.", "Let me find out.", "Checking now.", "Give me a moment to look.")
    turn = 0

    async def stream(self, messages: list[dict], image_png: str | None = None) -> AsyncIterator[str]:
        Ack.turn += 1
        yield self.LINES[Ack.turn % len(self.LINES)]


def load(profile: dict) -> dict:
    agents = {}
    for name, spec in profile.get("agents", {}).items():
        agents[name] = (Acp if spec.get("kind") == "acp" else Direct)(name, spec)
    return agents
