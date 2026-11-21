"""A stand-in renderer for developing the multiverse without any video model: keyframes are scenes drawn with Pillow
from the prompt's words (sky, sea, a lighthouse, a figure, a storm, a ship's light, a lit window ...), and clips are
made with ffmpeg: the opening frame drifts (a slow zoom) and dissolves into the scene of the clip's prompt, so a clip
really does open on the frame it was given and ends somewhere new. Every picture is generated; no photographs, no
recordings. The ``Renderer`` interface is voice_agent/video.py's, so the multiverse service cannot tell it apart from
the real one (``room`` always says yes; ``delay_s`` makes a render take a little while, so the page's progress shows).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from voice_agent.video import Progress, Refused, Renderer, find_ffmpeg

PALETTES = {  # (sky top, sky bottom, sea, light)
    "night": ((8, 10, 30), (34, 40, 84), (10, 18, 40), (255, 236, 170)),
    "storm": ((20, 22, 28), (64, 70, 82), (24, 32, 40), (255, 244, 200)),
    "dawn": ((40, 44, 96), (240, 150, 120), (60, 70, 110), (255, 220, 160)),
    "golden": ((70, 90, 160), (250, 190, 110), (60, 90, 120), (255, 230, 170)),
    "day": ((70, 130, 200), (190, 220, 240), (40, 90, 140), (255, 255, 230)),
    "tender": ((60, 40, 70), (230, 160, 140), (70, 60, 90), (255, 210, 170)),
    "awe": ((10, 20, 50), (60, 110, 160), (14, 30, 60), (200, 240, 255)),
    "mysterious": ((14, 24, 30), (40, 80, 90), (12, 30, 34), (160, 255, 220)),
    "urgent": ((50, 14, 20), (160, 60, 50), (30, 20, 30), (255, 200, 140)),
}
WORDS = {
    "storm": r"storm rain thunder lightning wind gale tempest breaks shatter danger".split(),
    "night": r"night dark moon stars midnight".split(),
    "dawn": r"dawn sunrise morning".split(),
    "golden": r"sunset dusk golden evening".split(),
    "tender": r"tender quiet pause breathes warm".split(),
    "awe": r"answers far distance awe wide".split(),
    "mysterious": r"hidden reveal discovers mysterious secret strange".split(),
    "urgent": r"urgent fast runs action sudden".split(),
}


def has(text: str, *words: str) -> bool:
    low = f" {text.lower()} "
    return any(f" {w}" in low for w in words)


def palette_for(text: str) -> tuple:
    low = text.lower()
    best, score = "night", 0
    for name, words in WORDS.items():
        s = sum(low.count(w) for w in words)
        if s > score:
            best, score = name, s
    return PALETTES[best]


def gradient(size, top, bottom) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size)
    px = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(1, h - 1)
        px.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return img


def scene(prompt: str, seed: int, size=(864, 480)) -> Image.Image:
    """A generated still for a prompt: the same seed and words give the same picture."""
    w, h = size
    rnd = random.Random(int(hashlib.sha1(f"{seed}:{prompt}".encode()).hexdigest()[:8], 16))
    sky_top, sky_bottom, sea, light = palette_for(prompt)
    img = gradient(size, sky_top, sky_bottom)
    d = ImageDraw.Draw(img, "RGBA")
    horizon = int(h * (0.58 + rnd.uniform(-0.05, 0.05)))
    if sky_top[2] < 60 or has(prompt, "night", "star", "moon"):
        for _ in range(140):
            x, y = rnd.randrange(w), rnd.randrange(horizon)
            d.point((x, y), fill=(255, 255, 255, rnd.randrange(80, 255)))
    if has(prompt, "moon", "night"):
        mx, my = rnd.randrange(int(w * 0.55), int(w * 0.9)), rnd.randrange(int(h * 0.08), int(h * 0.25))
        d.ellipse([mx - 22, my - 22, mx + 22, my + 22], fill=(240, 236, 214, 235))
    # the sea, with bands of waves
    d.rectangle([0, horizon, w, h], fill=sea)
    for i in range(26):
        y = horizon + int((h - horizon) * (i / 26) ** 1.4)
        d.line([(0, y), (w, y + rnd.randint(-2, 2))], fill=tuple(min(255, c + 22) for c in sea) + (90,), width=1)
    # a lighthouse (or a tower) on a rock, when the prompt has one; else a far shore
    if has(prompt, "lighthouse", "tower", "lamp", "keeper"):
        lx = int(w * rnd.uniform(0.18, 0.36))
        d.polygon([(lx - 70, h), (lx - 40, horizon - 10), (lx + 60, horizon - 6), (lx + 110, h)], fill=(18, 18, 22))
        top, base = int(h * 0.16), horizon - 8
        d.polygon([(lx - 18, base), (lx - 11, top), (lx + 11, top), (lx + 18, base)], fill=(225, 222, 214))
        for k in range(3):
            y = top + (base - top) * (k + 1) // 4
            d.rectangle([lx - 17, y, lx + 17, y + 12], fill=(170, 40, 40))
        d.rectangle([lx - 14, top - 26, lx + 14, top], fill=light + (255,) if has(prompt, "lamp", "light", "beam", "climb") else (60, 60, 70))
        d.polygon([(lx - 18, top - 26), (lx, top - 42), (lx + 18, top - 26)], fill=(30, 30, 36))
        if has(prompt, "lamp", "light", "beam", "climb", "answers"):
            ang = rnd.uniform(-0.25, 0.25)
            for spread, alpha in ((0.09, 40), (0.05, 70)):
                far = w * 1.2
                d.polygon([(lx, top - 13), (lx + far * math.cos(ang - spread), top - 13 + far * math.sin(ang - spread)),
                           (lx + far * math.cos(ang + spread), top - 13 + far * math.sin(ang + spread))], fill=light + (alpha,))
        if has(prompt, "window", "room", "door", "inside"):
            d.rectangle([lx - 5, top + 40, lx + 5, top + 56], fill=light + (255,))
    else:
        d.polygon([(0, horizon), (w * 0.3, horizon - 26), (w * 0.55, horizon - 8), (w, horizon - 30), (w, horizon)], fill=(20, 22, 30))
    if has(prompt, "ship", "boat", "vessel"):
        sx, sy = int(w * rnd.uniform(0.62, 0.85)), horizon + 6
        d.polygon([(sx - 40, sy), (sx + 40, sy), (sx + 28, sy + 12), (sx - 28, sy + 12)], fill=(12, 12, 16))
        d.line([(sx, sy), (sx, sy - 44)], fill=(12, 12, 16), width=3)
        d.ellipse([sx - 5, sy - 52, sx + 5, sy - 42], fill=light + (255,))
        d.ellipse([sx - 16, sy - 63, sx + 16, sy - 31], fill=light + (60,))
    if has(prompt, "keeper", "he ", "she ", "figure", "man", "woman", "hero", "person", "climbs", "child"):
        fx, fy = int(w * rnd.uniform(0.45, 0.6)), int(h * 0.92)
        d.ellipse([fx - 9, fy - 92, fx + 9, fy - 74], fill=(14, 14, 18))
        d.polygon([(fx - 16, fy), (fx - 12, fy - 72), (fx + 12, fy - 72), (fx + 16, fy)], fill=(14, 14, 18))
        if has(prompt, "lantern", "light", "lamp"):
            d.ellipse([fx + 14, fy - 52, fx + 26, fy - 40], fill=light + (255,))
            d.ellipse([fx + 4, fy - 62, fx + 36, fy - 30], fill=light + (50,))
    if has(prompt, "storm", "rain", "thunder", "lightning", "breaks"):
        for _ in range(420):
            x, y = rnd.randrange(w), rnd.randrange(h)
            d.line([(x, y), (x - 6, y + 18)], fill=(200, 210, 230, 70), width=1)
        if has(prompt, "lightning", "thunder", "breaks"):
            x, y = rnd.randrange(int(w * 0.4), w), 0
            pts = [(x, y)]
            while y < horizon:
                x, y = x + rnd.randint(-28, 28), y + rnd.randint(20, 46)
                pts.append((x, y))
            d.line(pts, fill=(240, 240, 255, 230), width=3)
    if has(prompt, "fog", "mist"):
        d.rectangle([0, int(h * 0.35), w, h], fill=(200, 210, 220, 60))
    return img.filter(ImageFilter.GaussianBlur(0.6))


class StubRenderer(Renderer):
    name = "stub"
    label = "stub renderer (Pillow scenes + ffmpeg, no video model)"
    governed = False

    def __init__(self, root: str | Path, delay_s: float = 1.5, ffmpeg: str | None = None) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.delay_s = float(delay_s)
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.jobs: dict[str, dict] = {}
        self.rooms = 0  # how often room was asked (the tests read it)

    async def room(self, clip: dict) -> Refused | None:
        self.rooms += 1
        if any(j["state"] == "rendering" for j in self.jobs.values()):
            return Refused("This machine is already rendering a clip.")
        return None

    def new(self, kind: str, **more) -> str:
        handle = f"{kind[0]}{uuid.uuid4().hex[:10]}"
        folder = self.root / handle
        folder.mkdir(parents=True, exist_ok=True)
        self.jobs[handle] = {"kind": kind, "state": "rendering", "at": 0, "of": 8, "output": None, "error": None,
                             "folder": folder, "began": time.monotonic(), **more}
        return handle

    async def keyframe(self, prompt: str, seed: int, references: list[str] | None = None, size: tuple[int, int] | None = None) -> str:
        w, h = size or (864, 480)
        if references and not 512 * 512 <= w * h <= 1536 * 1536:  # the real edit's rule (FLUX3-Klein beside a reference)
            raise Refused(f"an edit is made at 512x512 to 1536x1536 pixels in all, and {w}x{h} is outside that")
        handle = self.new("keyframe")
        self.jobs[handle]["task"] = asyncio.create_task(self.make_keyframe(handle, prompt, seed, references, size or (864, 480)))
        return handle

    async def make_keyframe(self, handle, prompt, seed, references, size) -> None:
        job = self.jobs[handle]
        try:
            await self.tick(job, self.delay_s * 0.5)

            def draw() -> str:
                img = scene(prompt, seed, tuple(size))
                if references and Path(references[0]).is_file():  # drawn beside the frame before: it shows through
                    ref = Image.open(references[0]).convert("RGB").resize(img.size)
                    img = Image.blend(ref, img, 0.7)
                out = job["folder"] / "keyframe.jpg"
                img.save(out, quality=90)
                return str(out)
            job["output"] = await asyncio.to_thread(draw)
            job["state"] = "done"
        except Exception as error:  # noqa: BLE001
            job["state"], job["error"] = "failed", str(error)

    async def submit(self, clip: dict) -> str:
        handle = self.new("video", clip=dict(clip))
        self.jobs[handle]["task"] = asyncio.create_task(self.make_clip(handle))
        return handle

    async def make_clip(self, handle) -> None:
        job = self.jobs[handle]
        clip = job["clip"]
        try:
            if not self.ffmpeg:
                raise RuntimeError("the stub renderer needs ffmpeg")
            w, h = int(clip.get("width") or 864) // 2 * 2, int(clip.get("height") or 480) // 2 * 2
            seconds = max(1.0, float(clip.get("seconds") or 3))
            folder = job["folder"]
            start, end = folder / "start.png", folder / "end.png"

            def stills() -> None:
                first = clip.get("image_file") or clip.get("image")
                img = Image.open(first).convert("RGB").resize((w, h)) if first and Path(first).is_file() else scene(clip["prompt"], clip.get("seed", 0), (w, h))
                img.save(start)
                scene(clip["prompt"], int(clip.get("seed", 0)) + 1, (w, h)).save(end)
            await asyncio.to_thread(stills)
            await self.tick(job, self.delay_s)
            out = folder / "clip.mp4"
            fade = min(seconds * 0.6, max(0.5, seconds - 0.8))
            frames = int(seconds * 24)
            # each still drifts in (a slow zoom), and the opening dissolves into the clip's own scene
            zoom = lambda: f"zoompan=z='min(zoom+0.0009,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps=24"  # noqa: E731
            args = [self.ffmpeg, "-v", "error", "-y", "-loop", "1", "-t", f"{seconds}", "-i", str(start), "-loop", "1", "-t", f"{seconds}",
                    "-i", str(end), "-filter_complex",
                    f"[0:v]scale={w * 2}:{h * 2},{zoom()},trim=duration={seconds},setpts=PTS-STARTPTS[a];"
                    f"[1:v]scale={w * 2}:{h * 2},{zoom()},trim=duration={seconds},setpts=PTS-STARTPTS[b];"
                    f"[a][b]xfade=transition=fade:duration={fade:.2f}:offset={seconds - fade - 0.3:.2f},format=yuv420p[v]",
                    "-map", "[v]", "-t", f"{seconds}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-movflags", "+faststart", str(out)]
            done = await asyncio.to_thread(subprocess.run, args, capture_output=True, timeout=300)
            if done.returncode != 0 or not out.is_file():
                raise RuntimeError("ffmpeg: " + done.stderr.decode(errors="replace")[-300:])
            job["output"], job["state"], job["at"] = str(out), "done", job["of"]
            job["timings"] = {"render_s": round(time.monotonic() - job["began"], 2), "renderer": "stub"}
        except Exception as error:  # noqa: BLE001
            job["state"], job["error"] = "failed", str(error)

    async def tick(self, job: dict, seconds: float) -> None:
        steps = 8
        for i in range(steps):
            job["at"] = i
            await asyncio.sleep(seconds / steps)

    async def poll(self, handle: str) -> Progress:
        job = self.jobs.get(handle)
        if job is None:
            return Progress("failed", error="no such job")
        return Progress(job["state"], job["at"], job["of"], "denoising" if job["state"] == "rendering" else job["state"],
                        None, job["output"], job["error"], timings=job.get("timings"))

    async def picture(self, ref: str) -> bytes | None:
        path = Path(ref)
        return path.read_bytes() if path.is_file() else None

    async def fetch(self, output: str, dest: Path) -> None:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, output, dest)
