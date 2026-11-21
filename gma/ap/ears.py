# The line check is ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman),
# the AutoPilot branch, app/qa.py at 6cc6e67c5e138499ea2c17183eda261b710d1217 (words, lines, audio_gate: Sam
# Wasserman's QA gate). Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: the transcript is Parakeet's (the app's own recognizer, or Parakeet loaded here) instead of faster-whisper;
# Whisper only where Parakeet is not installed, run in another Python (an install's venv) as a child process.
"""AutoPilot's ears: what a take says, and whether it says the line its prompt asked for.

    "autopilot": {"ears": {"kind": "auto", "whisper_python": "~/.local/opt/tensorfold-studio/venv/bin/python",
                           "whisper_model": "large-v3-turbo"}}

``kind``: ``parakeet`` (the app's recognizer, handed in as ``hear``, or Parakeet loaded here from the profile's
"asr"), ``whisper`` (faster-whisper on the CPU, in this Python or in ``whisper_python``), or ``auto``: Parakeet when it
is there, else Whisper. A machine with neither skips the line check (a skipped gate never fails a take).
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import tempfile
from pathlib import Path

ENV = {**os.environ, "PATH": os.pathsep.join([os.environ.get("PATH", ""), "/opt/homebrew/bin", "/usr/local/bin"])}
WHISPER = r"""
import sys, wave
import numpy as np
from faster_whisper import WhisperModel
with wave.open(sys.argv[1]) as h:
    a = np.frombuffer(h.readframes(h.getnframes()), np.int16).astype(np.float32) / 32768.0
m = WhisperModel(sys.argv[2], device="cpu", compute_type="int8")
parts, _ = m.transcribe(a, beam_size=5)
print(" ".join(p.text.strip() for p in parts))
"""


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def lines(prompt: str) -> list[str]:
    """The spoken lines a MiniMax H3 prompt asks for: <d>[English] the words</d>."""

    return [re.sub(r"^\[[^\]]*\]\s*", "", found).strip() for found in re.findall(r"<d>(.*?)</d>", prompt or "", re.S)]


def samples(clip: Path, folder: Path):
    """The clip's sound, 16 kHz mono: (float32 array, wav path)."""

    import wave

    import numpy as np

    sound = folder / "ears.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                    str(sound)], check=True, env=ENV, timeout=120)
    with wave.open(str(sound)) as handle:
        audio = np.frombuffer(handle.readframes(handle.getnframes()), np.int16).astype(np.float32) / 32768.0
    return audio, sound


class Ears:
    def __init__(self, spec: dict | None = None, hear=None, asr: dict | None = None) -> None:
        self.spec = dict(spec or {})
        self.kind = self.spec.get("kind", "auto")
        self.hear = hear  # the app's recognizer: fn(float32 16 kHz audio) -> text, on its own thread
        self.asr = asr  # the profile's "asr" section, to load Parakeet here when no recognizer is handed in
        self.whisper_python = os.path.expanduser(self.spec.get("whisper_python") or "")
        self.whisper_model = self.spec.get("whisper_model", "large-v3-turbo")
        self.loaded = None

    def parakeet(self):
        """fn(audio) -> text with Parakeet, or None when it is not installed here."""

        if self.hear is not None:
            return self.hear
        if self.loaded is not None:
            return self.loaded
        if not self.asr or "parakeet" not in str(self.asr.get("model", "")).lower():
            return None
        need = {"nemo": "nemo", "parakeet-mlx": "parakeet_mlx"}.get(self.asr.get("kind"))
        if not need or importlib.util.find_spec(need) is None:
            return None
        from voice_agent.engines import Recognizer

        self.loaded = Recognizer(self.asr)
        return self.loaded

    def whisper_here(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def whisper_there(self) -> bool:
        return bool(self.whisper_python) and os.access(self.whisper_python, os.X_OK)

    def ready(self) -> dict:
        """{"transcript": bool, "by": "parakeet" | "whisper" | None}"""

        if self.kind in ("auto", "parakeet"):
            try:
                if self.parakeet() is not None:
                    return {"transcript": True, "by": "parakeet"}
            except Exception:  # noqa: BLE001 - a recognizer that cannot load: Whisper, if allowed
                pass
        if self.kind in ("auto", "whisper") and (self.whisper_here() or self.whisper_there()):
            return {"transcript": True, "by": "whisper"}
        return {"transcript": False, "by": None}

    def transcribe(self, clip: Path) -> str | None:
        by = self.ready()["by"]
        if by is None:
            return None
        with tempfile.TemporaryDirectory() as scratch:
            audio, sound = samples(Path(clip), Path(scratch))
            if by == "parakeet":
                return (self.parakeet()(audio) or "").strip()
            if self.whisper_here():
                from faster_whisper import WhisperModel

                model = WhisperModel(self.whisper_model, device="cpu", compute_type="int8")
                parts, _ = model.transcribe(audio, beam_size=5)
                return " ".join(p.text.strip() for p in parts)
            done = subprocess.run(["nice", self.whisper_python, "-I", "-c", WHISPER, str(sound), self.whisper_model],
                                  capture_output=True, text=True, env={**ENV, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=900)
            if done.returncode:
                raise RuntimeError((done.stderr or "whisper failed").strip()[-300:])
            return done.stdout.strip()

    def line_check(self, clip: Path, prompt: str) -> dict:
        """The QA gate's audio coherence: the line the prompt asked for, heard in the take (qa.audio_gate)."""

        wanted = [w for line in lines(prompt) for w in words(line)]
        by = self.ready()["by"]
        try:
            heard = self.transcribe(clip)
        except Exception as error:  # noqa: BLE001 - a broken tool is a skipped gate, not a failed clip
            return {"result": "skipped", "note": f"the transcript failed: {str(error)[-200:]}", "by": by}
        if heard is None:
            return {"result": "skipped", "note": "no Parakeet or Whisper on this machine", "by": None}
        said = words(heard)
        if not wanted:
            return {"result": "pass", "heard": heard, "by": by,
                    "note": "the prompt has no spoken line" + (": the clip is not silent of speech" if len(said) >= 4 else "")}
        found = sum(1 for w in wanted if w in said)
        need = min(2, len(wanted))  # a one-word line cannot hold two words
        ok = len(said) >= need and found >= need and found >= 0.6 * len(wanted)
        return {"result": "pass" if ok else "fail", "heard": heard, "wanted": " ".join(lines(prompt)), "by": by,
                "note": f"{found} of the line's {len(wanted)} words heard"}
