"""Per-second machine samples during a film (tools/fp4_soak.py's sampler, fsynced, one file a minute): memory,
pressure, GPU temperature, power, SM clock, throttle reasons, the hottest zone and the FP4 worker's RSS.

    python tools/sample_machine.py --out ~/ghost-studio/out/bp3-samples     (stop: kill its pid)
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp4_soak import Sampler  # noqa: E402


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True)
    out = Path(a.parse_args().out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state: dict = {"stage": "film"}
    Sampler(out, state).start()
    while True:
        found = subprocess.run(["pgrep", "-f", "fp4_worker.py --model"], capture_output=True, text=True).stdout.split()
        state["pid"] = int(found[0]) if found else None
        time.sleep(5)


if __name__ == "__main__":
    main()
