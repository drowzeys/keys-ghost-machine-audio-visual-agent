"""Ghost Studio's clips: every one made by the FP4 clip worker (gma/fp4_worker.py), FastH3 V2 on FastVideo's FP4
sparse-attention recipe, and by nothing else.

- ``ClipWorker``: the worker's HTTP API (``/health``, ``/v1/clips``, ``/v1/clips/{id}``), synchronous, a timeout a call;
- ``need_mb``: what a clip asks of the machine at its peak, the worker's whole process, measured on a DGX Spark;
- ``last_frame``: the picture a clip that opens on the clip before it starts from (ffmpeg, the clip's last frame).

The worker is a managed engine of the registry (the profile's ``clips`` engine, role ``clips``): it loads for a clip
and stays between takes; a keyframe's or the film's sound's schedule, which does not keep it, releases it. A clip
that the worker cannot make fails with the worker's reason: there is no other route.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ENGINE = "fp4"  # the profile's engine name for the worker (its role is "clips")
ROLE = "clips"
# the worker's process at its peak (MB), measured on the DGX Spark (2026-11-07; FastH3 V2 NVFP4-Consumer with the AdaLN table, VSA
# FP4 0.8, eager VAE): what MemAvailable fell by from before its load to the lowest point of a first-frame clip, by
# (width, height, frames): 55.6 GB at 864x480x243 (a 139-151 s clip), 66.9 GB at 1376x768x243 (481 s). Shorter clips
# take the next longer measured figure.
NEED_MB = {(864, 480, 243): 56320, (1376, 768, 243): 67584}
ERROR_MB = 4096


def need_mb(width: int, height: int, frames: int) -> int:
    """The nearest measured peak at this size (the largest at a size never measured)."""

    same = {f: mb for (w, h, f), mb in NEED_MB.items() if (w, h) == (width, height)}
    if not same:
        return max(NEED_MB.values())
    return same[min(same, key=lambda f: (f < frames, abs(f - frames)))]


class ClipError(RuntimeError):
    pass


class ClipWorker:
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

    def call(self, method: str, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                return json.loads(answer.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                said = json.loads(error.read() or b"{}").get("error") or error.reason
            except ValueError:
                said = error.reason
            raise ClipError(f"the FP4 clip worker refused: {said}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise ClipError(f"the FP4 clip worker is not answering ({error})") from None

    def health(self) -> dict:
        return self.call("GET", "/health")

    def submit(self, body: dict) -> dict:
        return self.call("POST", "/v1/clips", body)

    def clip(self, handle: str) -> dict:
        return self.call("GET", f"/v1/clips/{handle}")


def last_frame(video: str | Path, dest: Path, ffmpeg: str = "ffmpeg") -> Path:
    """The last frame of ``video`` as a PNG (a clip that continues the one before opens on it)."""

    dest.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-sseof", "-0.25", "-i", str(video), "-update", "1",
                           "-frames:v", "1", str(dest)], capture_output=True, text=True, timeout=60)
    if done.returncode != 0 or not dest.is_file():
        raise ClipError(f"no last frame from {video}: {done.stderr.strip()[-200:]}")
    return dest
