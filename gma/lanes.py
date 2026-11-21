"""Brain lanes: one brain, two kinds of work. Voice turns go first; the Director (AutoPilot's planning, the video
planner, any background question) runs only while nobody is talking, and yields the moment someone starts to speak or
barges in: its request is cancelled and asked again once it is quiet.

    lanes = Lanes(busy=engines.busy)
    async with lanes.voice():                 # a voice turn (the conversation's)
        ...
    text = await lanes.director(lambda: brain.complete(messages))   # background work

``preempt(why)`` is called at speech start and at barge-in; the voice() context calls it too.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Awaitable, Callable


class Yielded(Exception):
    """A director request was cancelled for a voice turn too many times; the caller decides what to do."""


class Lanes:
    def __init__(self, busy: Callable[[], bool] | None = None, quiet_s: float = 0.4, retries: int = 20,
                 wait_s: float = 120.0) -> None:
        self.busy = busy or (lambda: False)
        self.quiet_s = quiet_s  # the conversation stays quiet this long before the director goes on
        self.retries = retries
        self.wait_s = wait_s
        self.voice_turns = 0
        self.director_tasks: set[asyncio.Task] = set()
        self.preempted = 0  # director requests cut for a voice turn, all told
        self.last_voice = 0.0
        self.last_why = ""

    def talking(self) -> bool:
        return self.voice_turns > 0 or bool(self.busy())

    def preempt(self, why: str = "barge-in") -> int:
        """Cancels every director request in flight (it is asked again once quiet). Returns how many."""

        self.last_voice, self.last_why = time.monotonic(), why
        cut = 0
        for task in list(self.director_tasks):
            if not task.done():
                task.cancel()
                cut += 1
        self.preempted += cut
        return cut

    @contextlib.asynccontextmanager
    async def voice(self):
        self.voice_turns += 1
        self.preempt("voice turn")
        try:
            yield
        finally:
            self.voice_turns -= 1
            self.last_voice = time.monotonic()

    async def quiet(self, deadline: float) -> bool:
        """Waits for the conversation to be quiet for quiet_s; False at the deadline."""

        while time.monotonic() < deadline:
            if not self.talking() and time.monotonic() - self.last_voice >= self.quiet_s:
                return True
            await asyncio.sleep(0.05)
        return False

    async def director(self, call: Callable[[], Awaitable], retries: int | None = None):
        """``call()`` on the director lane: once quiet, as a task the voice lane may cancel; asked again after each
        cancel, up to ``retries`` times (then Yielded). Cancelling the caller cancels it for good."""

        deadline = time.monotonic() + self.wait_s
        for _ in range((self.retries if retries is None else retries) + 1):
            if not await self.quiet(deadline):
                raise Yielded("the conversation did not go quiet")
            task = asyncio.ensure_future(call())
            self.director_tasks.add(task)
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                me = asyncio.current_task()
                if me is not None and me.cancelling():  # the caller itself was cancelled
                    task.cancel()
                    raise
                if not task.cancelled():
                    raise
                continue  # cut for a voice turn: once quiet, again
            finally:
                self.director_tasks.discard(task)
        raise Yielded("cut for voice turns too many times")

    def status(self) -> dict:
        return {"voice_turns": self.voice_turns, "director_in_flight": len(self.director_tasks),
                "preempted": self.preempted, "last_preempt": self.last_why or None}
