"""The multiverse's HTTP side, registered on the app by one hook in voice_agent/server.py ``build``:

    GET  /                                       the multiverse page (the Ghost first); the classic page: /classic
    GET  /multiverse/{file}                      its static files (voice_agent/static/multiverse/)
    GET  /api/multiverse/projects                the films, newest first
    POST /api/multiverse/projects                {"idea": "..."}: a new film (its opening, then its first options)
    GET  /api/multiverse/projects/{p}            the tree: nodes, the Ghost, the active path, each node's state, the queue
    GET  /api/multiverse/projects/{p}/events     the same, as server-sent events, each time it changes
    POST /api/multiverse/projects/{p}/options    {"node"?, "note"?, "fresh"?}: options at a node (the Ghost's)
    POST /api/multiverse/projects/{p}/choose     {"node"}: the Ghost takes that option (its full clip is rendered)
    POST /api/multiverse/projects/{p}/goto       {"node"}: the Ghost walks back to a node, to branch from there
    POST /api/multiverse/projects/{p}/mix        {"a", "b"}: a new option mixing two at the same step
    POST /api/multiverse/projects/{p}/retry      {"node"}: a failed render again
    POST /api/multiverse/projects/{p}/export     the active path as one film; GET .../film.mp4 downloads it
    GET  /api/multiverse/projects/{p}/files/{node}/{name}   keyframe, draft, clip, poster, first and last frames

The profile's "multiverse" section turns it on: {"dir": ..., "renderer": "app" | "stub", "writer": "brain" | "plain",
"looker": "numbers" | "autopilot", "landing": true, "options": 3, "draft": {...}, "full": {...}}.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .service import MultiverseService
from .tree import Store, TreeError
from .voice import MultiverseAgent
from .writer import Writer

STATIC = Path(__file__).resolve().parents[2] / "voice_agent" / "static"
WORKING = {"planning", "keyframe", "loading", "rendering", "checking", "stopping"}
FILES = re.compile(r"^(keyframe\.(?:jpg|png)|draft\.mp4|clip\.mp4|poster\.jpg|first\.jpg|last\.jpg)$")
TYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".js": "text/javascript", ".mjs": "text/javascript",
         ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml"}


def make_service(engines, profile: dict) -> MultiverseService:
    spec = dict(profile.get("multiverse") or {})
    video = getattr(engines, "video", None)
    root = spec.get("dir") or (str(Path(video.root) / "multiverse") if video else "~/ghost-studio/home/ghost/multiverse")
    store = Store(root)
    if spec.get("renderer") == "stub":
        from .stub import StubRenderer

        renderer = StubRenderer(Path(store.root) / "_stub", delay_s=float(spec.get("stub_delay_s", 1.5)))
    else:  # the app's own renderer: keyframes from the headless Studio, clips from the FP4 worker, under the governor
        renderer = getattr(getattr(engines, "supervisor", None), "renderer", None) or (video.renderer if video else None)
        if renderer is None:
            raise SystemExit("multiverse: this profile has no renderer (its \"video\" section) - or set multiverse.renderer to \"stub\"")
    brains = getattr(engines, "brains", None)
    brain = None if spec.get("writer") == "plain" or brains is None else (brains.director or brains.main)
    writer = Writer(brain, getattr(engines, "lanes", None), seconds=float((spec.get("full") or {}).get("seconds", 5)),
                    think=int(spec.get("think_tokens", 0)))
    looker = None
    if spec.get("looker") == "autopilot" and video is not None and getattr(video, "autopilot", None) is not None:
        eyes = getattr(video.autopilot, "eyes", None)
        if eyes is not None:
            from gma.ap.qc import Looker

            looker = Looker(eyes)

    def external_busy() -> bool:  # the video panel or AutoPilot at work: their render goes first, never beside ours
        run = getattr(video, "run", None) if video else None
        return bool(run and run.get("state") in WORKING)

    def say(text: str, mood: str = "happy") -> None:  # spoken in the conversation used last, once nobody is talking
        talks = getattr(engines, "talks", None)
        if not talks:
            return
        talk = max(talks, key=lambda t: t.active_at)
        old = getattr(talk, "news", None)
        if old is not None and not old.done():
            old.cancel()
        talk.news = asyncio.get_running_loop().create_task(talk.when_quiet(lambda: talk.say(text, talk.name, mood, "says", remember=False)))

    return MultiverseService(store, renderer, writer, spec, external_busy=external_busy, say=say, looker=looker)


def register(app: FastAPI, engines, profile: dict) -> MultiverseService:
    """Adds the multiverse to the app (before the app's own "/" so the Ghost's page is the landing page)."""
    service = make_service(engines, profile)
    spec = profile.get("multiverse") or {}
    agents = getattr(engines, "agents", None)
    if isinstance(agents, dict):  # ahead of the video agent: its words are claimed first while the page is open
        rest = dict(agents)
        agents.clear()
        agents["multiverse"] = MultiverseAgent(service)
        agents.update(rest)
        agents["multiverse"].busy = getattr(engines, "busy", lambda: False)
    app.state.multiverse = service

    def bad(error: Exception, code: int = 400):
        return JSONResponse({"error": str(error)}, status_code=code)

    async def body_of(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    if spec.get("landing", True):
        @app.get("/")
        def landing():
            return FileResponse(STATIC / "multiverse" / "index.html", headers={"Cache-Control": "no-store"})

        @app.get("/classic")
        def classic():
            return FileResponse(STATIC / "index.html")

    @app.get("/multiverse/{name:path}")
    def static(name: str):
        path = (STATIC / "multiverse" / name).resolve()
        if not str(path).startswith(str((STATIC / "multiverse").resolve())) or not path.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(path, media_type=TYPES.get(path.suffix), headers={"Cache-Control": "no-cache"})

    @app.get("/api/multiverse/projects")
    async def projects():
        service.resume()  # renders cut off by a restart go back on the queue (the first time the page asks)
        service.seen_at = time.monotonic()
        return JSONResponse({"projects": service.store.list(), "focus": service.focus})

    @app.post("/api/multiverse/projects")
    async def create(request: Request):
        body = await body_of(request)
        try:
            tree = await service.create(str(body.get("idea") or ""), body.get("title"))
        except TreeError as error:
            return bad(error)
        return JSONResponse(service.view(tree.id))

    @app.get("/api/multiverse/projects/{pid}")
    def project(pid: str):
        try:
            view = service.view(pid)
        except TreeError as error:
            return bad(error, 404)
        service.focus, service.seen_at = pid, time.monotonic()
        return JSONResponse(view)

    @app.get("/api/multiverse/projects/{pid}/events")
    async def events(pid: str, request: Request):
        try:
            service.view(pid)
        except TreeError as error:
            return bad(error, 404)
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        service.listeners.setdefault(pid, set()).add(queue)
        service.focus = pid

        async def stream():
            try:
                yield f"data: {json.dumps(service.view(pid))}\n\n"
                while not await request.is_disconnected():
                    service.seen_at = time.monotonic()
                    try:
                        await asyncio.wait_for(queue.get(), 15)
                    except asyncio.TimeoutError:
                        yield ": still here\n\n"
                        continue
                    await asyncio.sleep(0.2)  # a burst of changes is sent once
                    yield f"data: {json.dumps(service.view(pid))}\n\n"
            finally:
                service.listeners.get(pid, set()).discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    def acting(name: str, call):
        async def handler(pid: str, request: Request):
            body = await body_of(request)
            try:
                result = await call(pid, body)
            except TreeError as error:
                return bad(error, 409)
            except KeyError as error:
                return bad(TreeError(f"missing {error}"))
            service.focus, service.seen_at = pid, time.monotonic()
            return JSONResponse({"ok": True, "result": result, "tree": service.view(pid)})
        app.post(f"/api/multiverse/projects/{{pid}}/{name}", name=f"multiverse_{name}")(handler)

    async def options(pid, body):
        if body.get("fresh"):
            service.spawn(service.request_options(pid, body.get("node"), str(body.get("note") or ""), fresh=True))
            return "writing"
        return [o["id"] for o in await service.request_options(pid, body.get("node"), str(body.get("note") or ""))]

    async def choose(pid, body):
        return (await service.choose(pid, body["node"]))["id"]

    async def goto(pid, body):
        return (await service.goto(pid, body["node"]))["id"]

    async def mix(pid, body):
        return (await service.mix(pid, body["a"], body["b"]))["id"]

    async def retry(pid, body):
        return (await service.retry(pid, body["node"]))["id"]

    async def export(pid, body):
        return (await service.export(pid)).name

    for name, call in (("options", options), ("choose", choose), ("goto", goto), ("mix", mix), ("retry", retry), ("export", export)):
        acting(name, call)

    @app.get("/api/multiverse/projects/{pid}/film.mp4")
    def film(pid: str, hdr: bool = False):
        try:
            made = service.store.load(pid).data.get("film")
            path = service.store.folder(pid) / made["file"] if made else None
        except TreeError:
            path = None
        if not path or not path.is_file():
            return JSONResponse({"error": "export the path first"}, status_code=404)
        if getattr(engines, "look", None):  # the film in the look set now (gma/look.py), once that copy is made
            path = engines.look.shown(path, hdr)
        return FileResponse(path, media_type="video/mp4", filename=f"multiverse-{pid}.mp4")

    @app.get("/api/multiverse/projects/{pid}/files/{nid}/{name}")
    def node_file(pid: str, nid: str, name: str):
        if not FILES.match(name):
            return JSONResponse({"error": "no such file"}, status_code=404)
        try:
            path = service.store.node_folder(pid, nid) / name
        except TreeError:
            return JSONResponse({"error": "no such file"}, status_code=404)
        if not path.is_file():
            return JSONResponse({"error": "no such file"}, status_code=404)
        return FileResponse(path, media_type=TYPES.get(path.suffix), headers={"Cache-Control": "max-age=3600"})

    return service
