"""Makes the FP4 clip worker's AdaLN table for first-frame clips, once, on the machine that will use it.

FastVideo's FastH3 V2 release ships the AdaLN modulation only for text-to-video (``transformer/adaln_tables.pt``, two
rows a timestep). A clip that opens on a picture adds a third row, so its entries are computed here from the release's
own projections with FastVideo's ``FASTVIDEO_H3_ADALN_DUMP``: the worker is started without a table, renders one short
first-frame clip, and what it dumped is joined with the shipped table. With the joined table the worker never loads
the 26 GB of projections again (docs/ghost-studio-notes.md, "The FP4 clip worker").

    fv-venv/bin/python tools/adaln_table.py --model MODEL_DIR --out TABLE.pt [--port 8429] [--cache DIR]

Needs about 90 GB of free memory for a few minutes (the worker with the projections resident). Nothing is downloaded.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def call(url: str, body: dict | None = None, timeout: float = 30) -> dict:
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def joined(shipped: dict, dumped: dict) -> dict:
    """Every block's entries from both tables (the dump wins where both have one)."""

    return {index: {**shipped.get(index, {}), **dumped.get(index, {})} for index in sorted(set(shipped) | set(dumped))}


def rows(table: dict) -> set[int]:
    return {ast.literal_eval(key)[1][0] for key in table[min(table)]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the FastVideo-FastH3-8-Step-V2-NVFP4-Consumer folder")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8429)
    parser.add_argument("--cache", default="~/ghost-studio/cache")
    parser.add_argument("--timeout", type=float, default=1500)
    args = parser.parse_args()
    import torch
    from PIL import Image

    model, out = Path(args.model).expanduser(), Path(args.out).expanduser()
    shipped_file = model / "transformer" / "adaln_tables.pt"
    if not shipped_file.is_file():
        print(f"no {shipped_file}: is this the FastH3 V2 NVFP4 Consumer release?", file=sys.stderr)
        return 2
    work = Path(tempfile.mkdtemp(prefix="adaln-"))
    dump, first, clip = work / "dump.pt", work / "first.png", work / "clip.mp4"
    Image.new("RGB", (864, 480), (96, 112, 128)).save(first)
    env = {**os.environ, "FASTVIDEO_H3_ADALN_CACHE": "1", "FASTVIDEO_H3_ADALN_DUMP": str(dump), "HF_HUB_OFFLINE": "1"}
    env.pop("FASTVIDEO_H3_ADALN_TABLE", None)
    worker = subprocess.Popen([sys.executable, "-I", str(HERE.parent / "gma" / "fp4_worker.py"), "--model", str(model),
                               "--port", str(args.port), "--cache", str(Path(args.cache).expanduser())],
                              env=env, start_new_session=True)
    base, began = f"http://127.0.0.1:{args.port}", time.monotonic()
    try:
        while True:  # the model loads, projections and all
            if worker.poll() is not None:
                print("the clip worker stopped while loading", file=sys.stderr)
                return 1
            try:
                if call(base + "/health", timeout=5).get("loaded"):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                pass
            if time.monotonic() - began > args.timeout:
                print("the clip worker did not load in time", file=sys.stderr)
                return 1
            time.sleep(3)
        print(f"worker loaded in {time.monotonic() - began:.0f} s; rendering one first-frame clip", flush=True)
        job = call(base + "/v1/clips", {"prompt": "A calm grey sea under an overcast sky, the camera holds still.", "image": str(first),
                                        "width": 864, "height": 480, "frames": 124, "seed": 1, "output": str(clip)})["id"]
        while True:
            state = call(f"{base}/v1/clips/{job}")
            if state.get("status") == "done":
                break
            if state.get("status") == "failed" or worker.poll() is not None:
                print(f"the clip failed: {state.get('error')}", file=sys.stderr)
                return 1
            if time.monotonic() - began > args.timeout:
                print("the clip did not finish in time", file=sys.stderr)
                return 1
            time.sleep(3)
    finally:
        try:
            os.killpg(worker.pid, signal.SIGTERM)
            worker.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if not dump.is_file():
        print("the worker wrote no table (FASTVIDEO_H3_ADALN_DUMP)", file=sys.stderr)
        return 1
    table = joined(torch.load(shipped_file, map_location="cpu"), torch.load(dump, map_location="cpu"))
    if rows(table) != {2, 3}:
        print(f"the joined table has rows {sorted(rows(table))}, not text-to-video's 2 and a first frame's 3", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, out)
    print(f"{out}: {len(table)} blocks, {len(table[min(table)])} entries a block ({out.stat().st_size / 1e6:.0f} MB)")
    for left in (dump, Path(str(dump) + ".inputs"), first, clip):
        left.unlink(missing_ok=True)
    work.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
