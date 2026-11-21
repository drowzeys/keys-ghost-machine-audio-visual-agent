"""keys Ghost Machine Studio for development: the whole app (voice_agent/server.py) with stand-ins where the big
models would be, so the page, the voice loop and the tree can be exercised on any machine.

    python -m gma.multiverse.dev --port 8296 [--host 0.0.0.0] [--seed-demo] [--asr parakeet|none] [--voice-url URL]

- the renderer: gma/multiverse/stub.py (generated Pillow scenes, ffmpeg clips; no video model)
- the writer: the plain fallback directions (no Director model)
- the front brain: a scripted stand-in served by this same app at /stub/v1 (a few fixed lines, with mood tags)
- the ears: Parakeet TDT 0.6B v3 (parakeet-mlx, ~2.5 GB, in process) by default on a Mac, or none (typing only)
- the voice: a Kokoro-82M server at --voice-url (synthetic voices only)
- the films: --dir (default ~/studio/multiverse-dev/films); --seed-demo writes a lighthouse demo tree, four levels deep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

LINES = {
    "hello": "[happy] I'm the Ghost. Tell me a concept for a film and I'll shoot the opening, then show you the paths.",
    "film": "[excited] Say make a film about, and then your idea, and I'll start walking it.",
    "default": "[curious] I'm the development stand-in for the brain, so I only know the film words: pick a path, go back, "
               "other options, mix two, or export this path.",
}


def stub_line(messages: list[dict]) -> str:
    last = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "") or ""
    low = str(last).lower()
    if any(w in low for w in ("hello", "hi ", "hey", "who are you")) or low.strip() in ("hi", "hey"):
        return LINES["hello"]
    if any(w in low for w in ("film", "video", "movie", "concept")):
        return LINES["film"]
    return LINES["default"]


def add_stub_brain(app) -> None:
    @app.post("/stub/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        line = stub_line(body.get("messages") or [])
        if not body.get("stream"):
            return JSONResponse({"choices": [{"message": {"role": "assistant", "content": line}}]})

        async def stream():
            for i, word in enumerate(line.split(" ")):
                chunk = {"choices": [{"delta": {"content": (" " if i else "") + word}}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream")


def profile_for(args) -> dict:
    port = args.port
    asr = {"kind": "parakeet-mlx", "model": "mlx-community/parakeet-tdt-0.6b-v3"} if args.asr == "parakeet" else {"kind": "none", "model": "none"}
    return {
        "name": "multiverse-dev", "platform": "mac" if platform.system() == "Darwin" else "spark",
        "assistant_name": "Ghost",
        "asr": asr,
        "brain": {"url": f"http://127.0.0.1:{port}/stub/v1", "model": "ghost-stub",
                  "system": "You are the Ghost, the guide of a film studio. Start every reply with your mood in brackets.",
                  "extra": {"max_tokens": 120}},
        "voice": {"url": args.voice_url, "model": "kokoro-82m", "sample_rate": 24000, "chunks": {"first_words": 5},
                  "voice": args.voice, "keep_warm": False},
        "voice_engine": "full",
        "turn": {"end_silence_ms": 500},
        # the voice server is adopted (never started or stopped here); the stand-in brain is this app itself, unsupervised
        "engines": {"kokoro": {"role": "voice", "mode": "adopt", "port": int(args.voice_url.rsplit(":", 1)[1].split("/")[0]), "resident_gb": 1}},
        "lookup": False,
        "supervisor": {"home": str(Path(args.dir).expanduser() / "_supervisor"), "monitor_s": 60},
        "multiverse": {"dir": str(Path(args.dir).expanduser()), "renderer": "stub", "writer": "plain",
                       "stub_delay_s": args.delay, "draft": {"width": 480, "height": 272, "seconds": 3},
                       "full": {"width": 864, "height": 480, "seconds": 4}},
        "watch": {"dir": str(Path(args.dir).expanduser().parent / "watch"), "scan_s": 5},  # invisible mode, /invisible
    }


DEMO = {
    "idea": "a lighthouse keeper in a storm at night",
    "opening": {"label": "The lighthouse in the storm", "beat": "a lone keeper watches the storm", "mood": "brooding",
                "camera": "slow push in from the sea",
                "prompt": "A lighthouse on a black rock at night, storm rain slanting across the beam; a keeper figure in the lit window. Thunder rolls."},
    "levels": [
        [{"label": "he climbs to the lamp", "beat": "the keeper climbs the tower", "mood": "resolute", "camera": "follows up the stairs",
          "prompt": "The keeper climbs the spiral stairs toward the lamp, lantern light swinging, the storm rain beating the tower at night."},
         {"label": "the storm breaks the window", "beat": "the storm smashes in", "mood": "urgent", "camera": "handheld, close",
          "prompt": "Lightning; the storm breaks the window and rain floods the keeper's room, the figure thrown back from the glass."},
         {"label": "a ship's light answers", "beat": "a ship signals from the dark", "mood": "awe", "camera": "wide pull back",
          "prompt": "Far out on the night sea a ship's light blinks back at the lighthouse beam; the keeper figure turns to watch."}],
        [{"label": "the lamp blazes to life", "beat": "the beam sweeps the sea", "mood": "triumphant", "camera": "low angle on the lamp",
          "prompt": "The keeper lights the lamp; the lighthouse beam blazes across the storm and the night sea, rain glittering in it."},
         {"label": "the lamp is dark, the keeper waits", "beat": "the light has failed", "mood": "dread", "camera": "still, from outside",
          "prompt": "The lamp stays dark in the storm; only the keeper's small lantern moves in the tower window at night."}],
        [{"label": "the ship turns toward the light", "beat": "the beam guides the ship home", "mood": "hope", "camera": "over the keeper's shoulder",
          "prompt": "A ship's light turns toward the lighthouse beam across the night sea as the storm eases; the keeper watches from the gallery."},
         {"label": "the beam finds something in the fog", "beat": "a shape in the fog", "mood": "mysterious", "camera": "slow tilt along the beam",
          "prompt": "The lighthouse beam sweeps through fog and mist over the night sea and catches a strange shape out on the water."},
         {"label": "dawn breaks over a calm sea", "beat": "the storm is over", "mood": "tender", "camera": "wide, still",
          "prompt": "Dawn sunrise over a calm sea; the lighthouse stands quiet, the keeper figure at the gallery rail, the storm gone."}],
    ],
}


async def seed_demo(service) -> str:
    """A lighthouse film four levels deep: the opening, three paths (B taken), two paths after B (A taken), three
    paths after that (drafts), the Ghost on B-A. Rendered through the service's own queue (the stub renderer)."""
    from .tree import Tree

    service.settings["auto_options"] = False  # the demo writes its own options
    tree = Tree.new(DEMO["idea"], {**DEMO["opening"], "seed": 11}, settings={k: service.settings[k] for k in ("options", "draft", "full")})
    service.save(tree)
    service.enqueue(tree.id, tree.ghost, "full")
    await until(lambda: tree.get(tree.ghost)["quality"] == "full")
    at = tree.ghost
    for depth, plans in enumerate(DEMO["levels"]):
        added = tree.add_options(at, [dict(p, seed=100 * (depth + 1) + i) for i, p in enumerate(plans)])
        service.save(tree)
        for o in added:
            service.enqueue(tree.id, o["id"], "draft")
        await until(lambda: all(tree.get(o["id"])["quality"] for o in added))
        if depth == len(DEMO["levels"]) - 1:
            break
        pick = added[2] if depth == 0 else added[0]  # the ship's light answers, then the lamp blazes
        await service.choose(tree.id, pick["id"])
        await until(lambda: tree.get(pick["id"])["quality"] == "full")
        at = pick["id"]
    # a mix at the last step (the roads not taken at each level stay, dimmed)
    last = tree.options()
    mixed = await service.mix(tree.id, last[0]["id"], last[2]["id"])
    await until(lambda: tree.get(mixed["id"])["quality"] == "draft")
    service.settings["auto_options"] = True
    service.focus = tree.id
    return tree.id


async def until(check, limit: float = 600) -> None:
    end = time.monotonic() + limit
    while not check():
        if time.monotonic() > end:
            raise TimeoutError("the demo's render took too long")
        await asyncio.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8296)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--dir", default="~/studio/multiverse-dev/films")
    parser.add_argument("--asr", choices=["parakeet", "none"], default="parakeet" if platform.system() == "Darwin" else "none")
    parser.add_argument("--voice-url", default="http://127.0.0.1:8197/v1")
    parser.add_argument("--voice", default="bf_emma")
    parser.add_argument("--delay", type=float, default=1.5, help="seconds a stub render pretends to take")
    parser.add_argument("--seed-demo", action="store_true")
    args = parser.parse_args()
    from voice_agent import server

    if args.asr == "none":
        server.Recognizer = lambda spec: (lambda audio: "")  # typing only
    profile = profile_for(args)
    app = server.build(profile)
    add_stub_brain(app)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    srv = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", ws_max_size=2**24))

    async def run() -> None:
        serving = asyncio.get_running_loop().create_task(srv.serve())
        while not srv.started and not serving.done():
            await asyncio.sleep(0.1)
        service = app.state.multiverse
        if args.seed_demo and not any(p["idea"] == DEMO["idea"] for p in service.store.list()):
            pid = await seed_demo(service)
            print(time.strftime("%H:%M:%S"), json.dumps({"type": "demo", "project": pid}), flush=True)
        await serving

    asyncio.run(run())


if __name__ == "__main__":
    main()
