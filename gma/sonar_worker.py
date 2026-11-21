"""The Sonar look's picture (gma/sonar.py plans it, this draws it), as its own process:

    python gma/sonar_worker.py PLAN.json

The clip is shown dark and black-and-white. At each pulse a wave leaves the subject and travels through the scene's
space, over the surfaces it meets: the wave's place at a pixel is that pixel's distance from the subject in metres, read
from a depth estimate of every frame (Depth Anything V2, metric, small), never a circle drawn on the picture. Behind the
front the picture turns to its colour copy; an early pulse's colour dies back, the last one stays. The front is a line of
light with two ripples behind it, contour lines a hand apart that hug each surface for a moment, and the outlines of
objects (where depth jumps) lit as it passes.

PLAN.json: {"source", "colours": [sdr copy, hdr copy], "outs": [sdr out, hdr out], "pulses": [s], "speed": m/s,
"far": m, "dist": m, "at": [x, y] | null, "model": name, "depth": file.npy | null, "floor_mb", "ffmpeg"}. The outs
have no sound. numpy for the picture; torch and transformers for the depth (not needed when "depth" names a file).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
SMALL = (688, 384)  # the depth's own size: half a 1376x768 clip
GLOW = np.array([0.45, 0.92, 1.0], np.float32)
RINGS = ((0.0, 1.0, 0.10), (0.45, 0.45, 0.07), (0.9, 0.22, 0.07))  # the front and two ripples: metres behind, gain, width
CONTOUR_M = 0.22
FOCAL = 0.75  # of the picture's width: a moderate wide lens
HOLD = 1.3  # s an early pulse's colour holds (each later one 0.7 s longer); the last one stays


def available_mb() -> int:
    try:
        with open("/proc/meminfo") as lines:
            return int(next(row for row in lines if row.startswith("MemAvailable")).split()[1]) // 1024
    except (OSError, StopIteration):
        return 1 << 30  # not Linux: no reading, no guard


def probe(ffmpeg: str, path: str) -> tuple[int, int, str]:
    tool = str(Path(ffmpeg).with_name("ffprobe")) if Path(ffmpeg).parent != Path(".") else "ffprobe"
    done = subprocess.run([tool, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate",
                           "-of", "json", path], capture_output=True, text=True, timeout=60)
    one = (json.loads(done.stdout or "{}").get("streams") or [{}])[0]
    if done.returncode or not one.get("width"):
        raise SystemExit(f"cannot read a picture from {Path(path).name}")
    return int(one["width"]), int(one["height"]), str(one.get("avg_frame_rate") or "24/1")


def reader(ffmpeg: str, path: str, width: int, height: int, hdr: bool) -> subprocess.Popen:
    matrix = "bt2020" if hdr else "bt709"
    return subprocess.Popen([ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-vf",
                             f"scale={width}:{height}:in_color_matrix={matrix}:in_range=tv:flags=bicubic", "-f", "rawvideo",
                             "-pix_fmt", "rgb48le" if hdr else "rgb24", "-"], stdout=subprocess.PIPE)


def writer(ffmpeg: str, path: str, width: int, height: int, rate: str, hdr: bool) -> subprocess.Popen:
    head = [ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le" if hdr else "rgb24", "-s", f"{width}x{height}",
            "-r", rate, "-i", "-"]
    if hdr:  # real HDR as the look's own copies are: BT.2020 / HLG, 10-bit HEVC
        tail = ["-vf", "scale=out_color_matrix=bt2020:out_range=tv:flags=lanczos,format=yuv420p10le", "-c:v", "libx265", "-preset",
                "medium", "-crf", "18", "-tag:v", "hvc1", "-x265-params",
                "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:log-level=error",
                "-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-color_range", "tv"]
    else:
        tail = ["-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset",
                "medium", "-crf", "17", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
    return subprocess.Popen(head + tail + ["-f", "mp4", path], stdin=subprocess.PIPE)


def estimate(ffmpeg: str, source: str, model: str, floor_mb: int, out: Path) -> np.ndarray:
    """Metres to every pixel of every frame, small, kept in a file (a film is long): [frames, 384, 688] float16."""

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as error:
        raise SystemExit(f"the Sonar look needs torch and transformers for its depth estimate ({error})")
    width, height = SMALL
    device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model)
    net = AutoModelForDepthEstimation.from_pretrained(model).to(device).eval()
    frames = subprocess.Popen([ffmpeg, "-v", "error", "-i", source, "-map", "0:v:0", "-vf", f"scale={width}:{height}", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    made = []
    with open(out, "wb") as file:
        while True:
            raw = frames.stdout.read(width * height * 3)
            if len(raw) < width * height * 3:
                break
            if available_mb() < floor_mb:
                frames.kill()
                raise SystemExit(f"memory is low (under {floor_mb} MB free): the Sonar look stopped")
            with torch.no_grad():
                seen = processor(images=np.frombuffer(raw, np.uint8).reshape(height, width, 3), return_tensors="pt").to(device)
                depth = torch.nn.functional.interpolate(net(**seen).predicted_depth[None], size=(height, width), mode="bicubic",
                                                        align_corners=False)[0, 0]
            file.write(depth.float().cpu().numpy().astype(np.float16).tobytes())
            made.append(1)
    frames.wait()
    if not made:
        raise SystemExit("no frames to estimate depth from")
    return np.memmap(out, np.float16, "r", shape=(len(made), height, width))


def subject(depth: np.ndarray, around: tuple | None = None) -> tuple[float, float, float]:
    """(x, y, metres) of the subject in a small depth frame, x and y as fractions of the picture: the middle of the nearest thing in
    the middle of the picture; or, once it is known (``around``), the nearest thing close to where it was, so a
    bus that passes nearer does not take the pulses over."""

    height, width = depth.shape
    top, bottom, left, right = int(height * 0.12), int(height * 0.68), int(width * 0.36), int(width * 0.64)
    if around is not None:  # followed from side to side only: lower down the ground is always nearer, and would take it
        middle = min(max(around[0], 0.2), 0.8)
        left, right = int(width * (middle - 0.09)), int(width * (middle + 0.09))
    window = depth[top:max(bottom, top + 2), left:max(right, left + 2)]
    floor = 0.0 if around is None else around[2] * 0.7  # what passes much nearer than the subject was is not the subject
    kept = window[window >= floor]
    if kept.size < 8:
        return tuple(around[:3])
    near = float(np.percentile(kept, 10))
    ys, xs = np.nonzero((window <= near * 1.12) & (window >= floor))
    if len(xs) < 8:
        return (left + right) / 2 / width, (top + bottom) / 2 / height, near
    return (left + float(np.median(xs))) / width, (top + float(np.median(ys))) / height, near


def follow(was: tuple | None, now: tuple[float, float, float]) -> tuple:
    """The subject's (x, y, metres, x where first found), eased from frame to frame and kept within reach of where it was
    first found: the pulses leave from one place, whatever passes by."""

    if was is None:
        return (*now, now[0])
    home = was[3]
    x = min(max(0.85 * was[0] + 0.15 * now[0], home - 0.1), home + 0.1)
    return x, 0.85 * was[1] + 0.15 * now[1], 0.85 * was[2] + 0.15 * now[2], home


class Wave:
    """The pulses over one clip: ``at(n, depth)`` is (reveal, light) for frame ``n``, each [height, width] 0..1 and up."""

    def __init__(self, width: int, height: int, fps: float, pulses: list[float], speed: float, far: float, dist: float,
                 at: list[float] | None = None) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.pulses, self.speed, self.far, self.dist, self.fixed = list(pulses), speed, far, dist, at
        self.holds = [HOLD + 0.7 * i for i in range(len(pulses) - 1)] + [1e9]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        focal = FOCAL * width
        self.rx, self.ry = (xx - width / 2) / focal, (yy - height / 2) / focal  # a pixel's point in space is (rx, ry, 1) * depth
        self.scale: float | None = None
        self.smooth: np.ndarray | None = None
        self.source: np.ndarray | None = None
        self.where: tuple | None = None

    def full(self, depth: np.ndarray) -> np.ndarray:
        rows = np.linspace(0, depth.shape[0] - 1, self.height).round().astype(int)
        cols = np.linspace(0, depth.shape[1] - 1, self.width).round().astype(int)
        return depth[rows][:, cols]

    def at(self, n: int, depth: np.ndarray, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        t = n / self.fps
        small = depth.astype(np.float32)
        x, y, near = subject(small, self.where)
        self.where = follow(self.where, (x, y, near))
        x, y, near = self.where[:3]
        if self.fixed:
            x, y = self.fixed
            hy, hx = int(y * (small.shape[0] - 1)), int(x * (small.shape[1] - 1))
            near = float(np.median(small[max(hy - 6, 0):hy + 7, max(hx - 6, 0):hx + 7]))
        if self.scale is None:  # the estimate's metres are loose: the subject is put `dist` metres away, once
            self.scale = self.dist / max(near, 1e-3)
        small = np.clip(small * self.scale, 0.3, self.far)
        self.smooth = small if self.smooth is None else 0.5 * self.smooth + 0.5 * small  # steadier from frame to frame
        z = self.full(self.smooth)
        hz = float(np.clip(near * self.scale, 0.3, self.far))
        px, py = (x - 0.5) * self.width / (FOCAL * self.width), (y * self.height - self.height / 2) / (FOCAL * self.width)
        here = np.array([px * hz, py * hz, hz], np.float32)
        self.source = here if self.source is None else 0.85 * self.source + 0.15 * here
        d = np.sqrt((self.rx * z - self.source[0]) ** 2 + (self.ry * z - self.source[1]) ** 2 + (z - self.source[2]) ** 2)
        jump = np.zeros_like(z)  # an object's outline: where depth jumps
        jump[:, 2:-2] += np.abs(z[:, 4:] - z[:, :-4])
        jump[2:-2, :] += np.abs(z[4:, :] - z[:-4, :])
        lines = np.clip(jump / np.maximum(z, 0.5) * 2.5, 0, 1)
        if edges is not None:
            lines = np.clip(lines + edges, 0, 1)
        reveal, light = np.zeros_like(z), np.zeros_like(z)
        for start, hold in zip(self.pulses, self.holds):
            if t < start:
                continue
            reach = self.speed * (t - start)
            passed = np.clip((reach - d) / 0.15, 0, 1)
            since = np.maximum(t - start - d / self.speed, 0)
            reveal = np.maximum(reveal, passed * np.exp(-since / hold))
            for back, gain, wide in RINGS:
                light += gain * np.exp(-((d - reach + back) / wide) ** 2)
            light += 0.5 * passed * np.exp(-since / 0.5) * (np.cos(d * (2 * np.pi / CONTOUR_M)) > 0.86)
            light += 0.35 * passed * np.exp(-since / 0.25)
        done = self.pulses[-1] + (self.far * 1.25) / self.speed + 0.3  # the last wave is past everything: all colour, whatever the depth said
        if t > done:
            reveal = np.maximum(reveal, min((t - done) / 0.5, 1.0))
        return reveal, light * (0.35 + 1.6 * lines)


def draw(colour: np.ndarray, reveal: np.ndarray, light: np.ndarray, hdr: bool) -> np.ndarray:
    """One frame, 0..1: the colour copy where the wave has been, a dark blue-grey of it elsewhere, the wave's light over both."""

    luma = colour @ (np.array([0.2627, 0.678, 0.0593], np.float32) if hdr else np.array([0.2126, 0.7152, 0.0722], np.float32))
    grey = (np.power(luma, 1.3) * (0.55 if hdr else 0.62))[..., None] * np.array([0.94, 0.98, 1.06], np.float32)
    frame = grey * (1 - reveal[..., None]) + colour * reveal[..., None]
    return np.clip(frame + light[..., None] * GLOW * (0.8 if hdr else 1.0), 0, 1)


def edges_of(colour: np.ndarray) -> np.ndarray:
    luma = colour @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    out = np.zeros_like(luma)
    out[:, 1:-1] += np.abs(luma[:, 2:] - luma[:, :-2])
    out[1:-1, :] += np.abs(luma[2:, :] - luma[:-2, :])
    return np.clip(out * 6, 0, 1)


def main(plan: dict) -> int:
    ffmpeg = plan.get("ffmpeg") or "ffmpeg"
    width, height, rate = probe(ffmpeg, plan["source"])
    top, _, under = rate.partition("/")
    fps = float(top) / float(under or 1) or 24.0
    with tempfile.TemporaryDirectory(prefix="ghost-sonar-") as scratch:
        if plan.get("depth"):
            depth = np.load(plan["depth"], mmap_mode="r")
        else:
            depth = estimate(ffmpeg, plan["source"], plan.get("model") or MODEL, int(plan.get("floor_mb", 6000)), Path(scratch) / "depth.f16")
        wave = Wave(width, height, fps, plan["pulses"], float(plan["speed"]), float(plan["far"]), float(plan["dist"]), plan.get("at"))
        kinds = (False, True)  # the SDR copy, the HDR copy
        ins = [reader(ffmpeg, path, width, height, hdr) for path, hdr in zip(plan["colours"], kinds)]
        outs = [writer(ffmpeg, path, width, height, rate, hdr) for path, hdr in zip(plan["outs"], kinds)]
        sizes = [width * height * 3, width * height * 6]
        n = 0
        while True:
            raws = [one.stdout.read(size) for one, size in zip(ins, sizes)]
            if any(len(raw) < size for raw, size in zip(raws, sizes)):
                break
            sdr = np.frombuffer(raws[0], np.uint8).reshape(height, width, 3).astype(np.float32) / 255
            high = np.frombuffer(raws[1], np.uint16).reshape(height, width, 3).astype(np.float32) / 65535
            reveal, light = wave.at(n, depth[min(n, len(depth) - 1)], edges_of(sdr))
            outs[0].stdin.write((draw(sdr, reveal, light, False) * 255 + 0.5).astype(np.uint8).tobytes())
            outs[1].stdin.write((draw(high, reveal, light, True) * 65535 + 0.5).astype(np.uint16).tobytes())
            n += 1
        for one in ins:
            one.kill()
        codes = []
        for one in outs:
            one.stdin.close()
            codes.append(one.wait())
        del depth
    if not n or any(codes):
        raise SystemExit("ffmpeg could not write the Sonar look's picture")
    print(json.dumps({"frames": n, "fps": fps}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(json.loads(Path(sys.argv[1]).read_text())))
