"""The film's sound pass (gma/ap/sound.py) for a film AutoPilot already made: the Director writes the narration, the
app's governor enters its "audio" schedule (POST /api/governor/phase), the headless Studio makes the Drama 3 take and
the MiniMax Music 3 score and lays the score under the film, and the governor goes back to talking.

    python tools/film_sound.py --run-dir <video dir>/<run id> --app http://127.0.0.1:8196 \
        --studio http://127.0.0.1:7890 --brain http://127.0.0.1:8092/v1 --model qwen38-a [--no-narration] [--no-score]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gma.ap import sound  # noqa: E402
from gma.studio import StudioClient  # noqa: E402


def post(url: str, body: dict, timeout: float = 900) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return json.loads(answer.read() or b"{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--app", default="http://127.0.0.1:8196")
    parser.add_argument("--studio", default="http://127.0.0.1:7890")
    parser.add_argument("--brain", default="http://127.0.0.1:8092/v1")
    parser.add_argument("--model", default="qwen38-a")
    parser.add_argument("--no-narration", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    folder = Path(args.run_dir).expanduser().resolve()
    shell = json.loads((folder / "run.json").read_text())
    run = shell.get("autopilot") or shell
    run["settings"].update(narration=not args.no_narration, score=not args.no_score)
    began = time.time()

    def director(system, prompt, think):
        said = post(args.brain.rstrip("/") + "/chat/completions", {
            "model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}})
        return said["choices"][0]["message"]["content"]

    text = sound.narration_text(run, director, float(run["film"]["seconds"])) if not args.no_narration else ""
    log = []
    said = lambda _run, line, **_k: (log.append(line), print(line, flush=True))  # noqa: E731
    post(args.app + "/api/governor/phase", {"phase": "rendering", "schedule": "audio"})
    timings: dict = {}
    try:
        record = sound.finish_film(run, StudioClient(args.studio), lambda *_: text, said, folder.parent, timings)
    finally:
        post(args.app + "/api/governor/phase", {"phase": "talking"})
    out = {"record": record, "timings": timings, "log": log, "narration": text, "total_s": round(time.time() - began, 1)}
    print(json.dumps(out, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=1))
    return 0 if record else 1


if __name__ == "__main__":
    raise SystemExit(main())
