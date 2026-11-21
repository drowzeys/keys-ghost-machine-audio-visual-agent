"""The Neon grade: a colour table (a 3D LUT, ``.cube``) that ffmpeg's ``lut3d`` lays over a clip before its HDR finish.

The look is the owner's reference clip (2026-11-17: a rider over a city at dusk, graded to two colours): what is dark or
grey goes to cobalt blue, what is bright or warm goes to hot red and magenta, and nothing stays grey. A colour that is
strong already keeps its own hue (a green light stays green) and is made stronger; warm hues turn toward red.

The table is made here, in plain Python (no numpy), once a version: ``cube(path)`` writes it and returns the path.
``grade(rgb)`` is one colour through it, 0..1 in and out, display-referred (the clip's own 8-bit values).
"""

from __future__ import annotations

import colorsys
from pathlib import Path

VERSION = 1
SIZE = 33
# what a grey of this brightness becomes: black, cobalt, violet, hot pink-red, red-orange, a warm white
RAMP = [(0.00, (0.00, 0.00, 0.05)), (0.22, (0.03, 0.08, 0.62)), (0.42, (0.22, 0.12, 0.86)), (0.60, (0.92, 0.10, 0.52)),
        (0.78, (1.00, 0.16, 0.20)), (0.92, (1.00, 0.52, 0.22)), (1.00, (1.00, 0.94, 0.86))]
DUO = 0.85  # how far a grey goes to the ramp (a strong colour goes less far: it keeps its hue)
SATURATION = 1.7  # a colour's own saturation, multiplied
CONTRAST = 0.55  # 0: the clip's own tones; 1: a full S curve
RED = 0.6  # warm hues (yellow to orange) turn this far toward red


def _ramp(level: float) -> tuple[float, float, float]:
    for (a, low), (b, high) in zip(RAMP, RAMP[1:]):
        if level <= b:
            t = (level - a) / (b - a)
            return tuple(x + (y - x) * t for x, y in zip(low, high))
    return RAMP[-1][1]


def _luma(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def grade(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (min(1.0, max(0.0, x)) for x in rgb)
    level = _luma(r ** 2.2, g ** 2.2, b ** 2.2) ** (1 / 2.2)  # brightness as the eye reads it
    curved = level * level * (3 - 2 * level)
    target = level + (curved - level) * CONTRAST
    hue, sat, value = colorsys.rgb_to_hsv(r, g, b)
    if 0.0 < hue < 1 / 6:  # orange and yellow turn toward red
        hue *= 1 - RED
    own = colorsys.hsv_to_rgb(hue, min(1.0, sat * SATURATION), value)
    two = _ramp(target)
    weight = DUO * (1 - sat) ** 1.5
    mixed = [o * (1 - weight) + t * weight for o, t in zip(own, two)]
    # the brightness the curve asked for, as far as the colour can carry it (a pure blue is never as bright as a white)
    now = _luma(*(x ** 2.2 for x in mixed)) ** (1 / 2.2)
    wanted = target * (1 - weight) + _luma(*(x ** 2.2 for x in two)) ** (1 / 2.2) * weight
    gain = min(wanted / now, 1 / max(mixed)) if now > 1e-4 and max(mixed) > 0 else 1.0
    return tuple(min(1.0, max(0.0, x * gain)) for x in mixed)


def cube(path: str | Path) -> Path:
    """The table as a ``.cube`` file (red changes fastest), written once: an existing file of this version is kept."""

    path = Path(path)
    head = f"# keys Ghost Studio Neon grade v{VERSION}\n"
    try:
        if path.read_text().startswith(head):
            return path
    except OSError:
        pass
    step = 1 / (SIZE - 1)
    rows = [head, f"LUT_3D_SIZE {SIZE}\n"]
    for b in range(SIZE):
        for g in range(SIZE):
            for r in range(SIZE):
                rows.append("%.5f %.5f %.5f\n" % grade((r * step, g * step, b * step)))
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text("".join(rows))
    part.replace(path)
    return path
