# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman),
# app/lengths.py at 912374e1b6111fdab5d55947cc695c9c303f83b7. Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Unchanged except this header.
"""How long a clip can be: MiniMax H3 and FastH3 take 17n + 5 frames at 24 frames a second, nothing in between.

The app used to offer four lengths (5, 8, 10 and 15 seconds: 124, 192, 243 and 362 frames). Every count from 22
frames (0.92 s) to 362 (15.08 s) is one the model takes, 21 lengths in all, and a job may now ask for any of them:
by `frames`, or by `seconds`, which is brought to the nearest count. 5, 8, 10 and 15 seconds are the same four counts
as before. The seconds a count gives are exact: frames / 24.

The longest clip is the model's own 15 seconds on every preset of both builds. On a DGX Spark that was measured
at the largest canvas (SPARK_LONGEST in server.py: 362 frames at 1376x768 with a first and a last frame left 29.7 GB
of 121 free on TensorFold's int8 engine); the Mac's limits are the ones it had.
"""

from __future__ import annotations

FPS = 24
STEP, BASE = 17, 5
SHORTEST, LONGEST = STEP + BASE, 21 * STEP + BASE  # 22 and 362 frames
# 5, 8, 10 and 15 seconds as the app has always rendered them
CLASSIC = {5: 124, 8: 192, 10: 243, 15: 362}


def valid(frames: int) -> bool:
    return SHORTEST <= frames <= LONGEST and (frames - BASE) % STEP == 0


def counts(longest: int = LONGEST) -> list[int]:
    """Every frame count the model takes, shortest first."""

    return list(range(SHORTEST, min(longest, LONGEST) + 1, STEP))


def nearest(seconds: float, longest: int = LONGEST) -> int:
    """The frame count closest to ``seconds`` of video, within what the model (and ``longest``) takes."""

    return min(counts(longest), key=lambda frames: (abs(frames - seconds * FPS), frames))


def seconds_of(frames: int) -> float:
    return frames / FPS


def says(frames: int) -> str:
    """'243 frames, 10.125 s': the exact length, as the page writes it."""

    return f"{frames} frames, {frames / FPS:.3f}".rstrip("0").rstrip(".") + " s"


def of(params: dict, longest: int = LONGEST) -> int:
    """The frame count a job asks for: ``frames`` (one the model takes, or ValueError with the reason) or ``seconds``
    (any number: the nearest count)."""

    if params.get("frames") not in (None, ""):
        try:
            frames = int(params["frames"])
        except (TypeError, ValueError):
            raise ValueError("frames is a whole number")
        if not valid(frames):
            below, above = nearest((frames - STEP / 2) / FPS), nearest((frames + STEP / 2) / FPS)
            near = " or ".join(str(n) for n in sorted({below, above}))
            raise ValueError(f"a clip is 17n + 5 frames long, from {SHORTEST} to {LONGEST} ({SHORTEST / FPS:.2f} to {LONGEST / FPS:.2f} s): "
                             f"{frames} is not one of them, the nearest is {near}")
    else:
        try:
            seconds = float(params.get("seconds", 5))
        except (TypeError, ValueError):
            raise ValueError("seconds is a number")
        if not SHORTEST / FPS - 0.5 <= seconds <= LONGEST / FPS + 0.5:
            raise ValueError(f"a clip is {SHORTEST / FPS:.2f} to {LONGEST / FPS:.2f} seconds long ({SHORTEST} to {LONGEST} frames), got {seconds:g} s")
        frames = nearest(seconds)
    if frames > longest:
        raise ValueError(f"at this size a clip is at most {longest} frames ({longest / FPS:.2f} s) on this machine, got {frames}")
    return frames
