"""How long the set's steps take, for the page's progress track: when each step began and ended, and what a step of
its kind took the last times (kept in a small file beside the run, so a restart still knows).

    clock = Clock(run / "usual.json")
    clock.plan([("still", "still", "Reference shot"), ("idle", "clip", "Mara listening")])
    clock.begin("still"); ...; clock.end("still")
    clock.state()   # [{"key", "label", "state": waiting|running|done|failed, "elapsed_s", "expect_s"}]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

USUAL = {"still": 50.0, "clip": 85.0, "blocking": 10.0, "lips": 25.0, "face": 38.0, "sharp": 260.0}  # seconds, measured 2026-11-19 on a Spark
LEARNED = ("still", "clip", "face", "sharp")  # the steps this clock times itself
COLD = 70.0  # what the first clip after the clip model was away takes on top (its load)


class Clock:
    def __init__(self, file: Path | None = None) -> None:
        self.file, self.usual = file, dict(USUAL)
        self.steps: dict[str, dict] = {}
        try:
            kept = json.loads(Path(file).read_text()) if file else {}
            self.usual.update({k: float(v) for k, v in kept.items() if k in LEARNED and 1 <= float(v) <= 3600})
        except (OSError, ValueError, TypeError):
            pass

    def plan(self, steps: list[tuple[str, str, str]], keep: tuple[str, ...] = ()) -> None:
        """The steps ahead, in order; those named in ``keep`` stay as they are when they are there already."""

        old = self.steps
        self.steps = {key: old[key] if key in keep and key in old else {"kind": kind, "label": label, "state": "waiting", "began": None, "took": None, "extra": 0.0}
                      for key, kind, label in steps}

    def begin(self, key: str, extra: float = 0.0) -> None:
        step = self.steps.get(key)
        if step is not None and step["state"] != "running":
            step.update(state="running", began=time.monotonic(), took=None, extra=float(extra))

    def end(self, key: str, failed: bool = False, learn: bool = True) -> None:
        step = self.steps.get(key)
        if step is None or step["state"] in ("done", "failed"):
            return
        took = time.monotonic() - step["began"] if step["began"] is not None else None
        step.update(state="failed" if failed else "done", took=took)
        if took is not None and not failed and learn and not step["extra"] and took >= 1:
            self.usual[step["kind"]] = round(0.5 * self.usual.get(step["kind"], took) + 0.5 * took, 1)
            try:
                if self.file:
                    Path(self.file).write_text(json.dumps(self.usual))
            except OSError:
                pass

    def seen(self, key: str, state: str) -> None:
        """A step somebody else runs, as it is now: its times are taken from when the change was seen."""

        step = self.steps.get(key)
        if step is None or step["state"] == state:
            return
        if state == "running":
            self.begin(key)
        elif state in ("done", "failed"):
            self.end(key, failed=state == "failed", learn=False)  # seen late, between two words to the page: no measure
        else:
            step.update(state="waiting", began=None, took=None)

    def drop(self, key: str) -> None:
        """A running step that was interrupted waits again."""

        step = self.steps.get(key)
        if step is not None and step["state"] == "running":
            step.update(state="waiting", began=None, took=None)

    def state(self) -> list[dict]:
        now, out = time.monotonic(), []
        for key, step in self.steps.items():
            elapsed = step["took"] if step["took"] is not None else now - step["began"] if step["began"] is not None else 0.0
            out.append({"key": key, "label": step["label"], "state": step["state"], "elapsed_s": round(elapsed, 1),
                        "expect_s": round(self.usual.get(step["kind"], 60.0) + step["extra"], 1)})
        return out
