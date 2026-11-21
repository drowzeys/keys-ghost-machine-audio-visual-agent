"""The open microphone of the live stage: the page sends what it hears all the time (16 kHz mono, 16-bit), the voice
agent's speech detector (Silero VAD, voice_agent/engines.py Vad) finds where a sentence begins and ends, and each
sentence is recognised and taken as a direction, so the director talks to the set without holding a button.

    ear = Ear(vad, hear)                 # hear: async (audio float32) -> None, called for one sentence at a time
    await ear.audio(pcm16_bytes)         # as the page sends it

While the page plays sound of its own that the microphone may pick up (a filmed part), it says so and only an order
("cut", "action") is taken from what was heard (gma/live/routes.py).
"""

from __future__ import annotations

import asyncio

import numpy as np

RATE, WINDOW = 16_000, 512  # the detector's window: 32 ms


class Ear:
    def __init__(self, vad, hear, start_p: float = 0.5, end_p: float = 0.35, start_windows: int = 4, end_windows: int = 22,
                 preroll: int = 10, min_s: float = 0.35, max_s: float = 20.0, speech=None) -> None:
        self.vad, self.hear, self.speech_cb = vad, hear, speech
        self.start_p, self.end_p, self.start_windows, self.end_windows = start_p, end_p, start_windows, end_windows
        self.preroll, self.min_n, self.max_n = preroll, int(min_s * RATE), int(max_s * RATE)
        self.pending = np.zeros(0, dtype=np.float32)
        self.recent: list[np.ndarray] = []
        self.speech: list[np.ndarray] = []
        self.in_speech, self.voiced, self.silent = False, 0, 0
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.worker = asyncio.create_task(self._work())

    async def audio(self, data: bytes) -> None:
        self.pending = np.concatenate([self.pending, np.frombuffer(data[: len(data) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768])
        while len(self.pending) >= WINDOW:
            window, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            await self.window(window)

    def drop(self) -> None:
        """What was being heard is forgotten (the page spoke over it)."""

        self.pending = np.zeros(0, dtype=np.float32)
        self.recent, self.speech, self.in_speech, self.voiced, self.silent = [], [], False, 0, 0
        self.vad.reset()

    async def window(self, window: np.ndarray) -> None:
        p = self.vad(window)
        if not self.in_speech:
            self.recent = (self.recent + [window])[-max(self.preroll, self.start_windows):]
            self.voiced = self.voiced + 1 if p >= self.start_p else 0
            if self.voiced >= self.start_windows:
                self.in_speech, self.silent, self.speech = True, 0, list(self.recent)
                if self.speech_cb:
                    await self.speech_cb(True)
            return
        self.speech.append(window)
        self.silent = self.silent + 1 if p < self.end_p else 0
        if self.silent >= self.end_windows or len(self.speech) * WINDOW >= self.max_n:
            audio = np.concatenate(self.speech)
            self.in_speech, self.voiced, self.speech, self.recent = False, 0, [], []
            if self.speech_cb:
                await self.speech_cb(False)
            if len(audio) - self.silent * WINDOW >= self.min_n and not self.queue.full():
                self.queue.put_nowait(audio)

    async def _work(self) -> None:
        while True:
            audio = await self.queue.get()
            try:
                await self.hear(audio)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one sentence lost; the microphone stays open
                pass

    def close(self) -> None:
        self.worker.cancel()
