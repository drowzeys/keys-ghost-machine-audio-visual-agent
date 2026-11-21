"""Invisible mode's HTTP side (keys Ghost Multiverse Studio), registered by one hook in voice_agent/server.py ``build``:

    GET  /invisible                              the page: the Ghost above a timeline of everything the agents made
    GET  /invisible/{file}                       its static files (voice_agent/static/invisible/)
    GET  /api/watch/creations?q=&kind=&agent=&limit=   search (best match first), or the newest
    GET  /api/watch/timeline?limit=              the newest creations, oldest first, and the status
    GET  /api/watch/creations/{id}               one creation, with its parents and children
    GET  /api/watch/creations/{id}/file          its image, clip, sound
    GET  /api/watch/creations/{id}/thumb         a small picture of an image or a clip
    GET  /api/watch/events                       every change, as server-sent events
    GET  /api/watch/status                       counts, agents, what the Ghost is describing
    POST /api/watch/creations                    an agent's creation (JSON: agent, kind, title, prompt or text,
                                                 parents, path under an allowed folder, ...)
    POST /api/watch/upload?agent=&name=&title=&kind=&parents=a,b&prompt=   the file itself as the request body

The profile's "watch" section turns it on: {"dir", "multiverse", "runs", "studios": [...], "allow": [...],
"describe": true, "scan_s": 15}.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from voice_agent.video import find_ffmpeg

from .describer import Describer
from .ledger import Ledger
from .service import WatchError, WatchService
from .voice import WatchAgent

STATIC = Path(__file__).resolve().parents[2] / "voice_agent" / "static"
TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif", ".wav": "audio/wav", ".mp3": "audio/mpeg",
         ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".js": "text/javascript", ".mjs": "text/javascript",
         ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml", ".txt": "text/plain", ".md": "text/plain"}


def make_service(engines, profile: dict, multiverse=None) -> WatchService:
    spec = dict(profile.get("watch") or {})
    video = getattr(engines, "video", None)
    runs = Path(video.root) if video is not None and getattr(video, "root", None) else None  # the video section's runs folder
    home = runs.parent if runs is not None else Path("~/ghost-studio/home/ghost").expanduser()
    mv = profile.get("multiverse") or {}
    spec.setdefault("multiverse", mv.get("dir") or str(home / "multiverse"))
    spec.setdefault("runs", str(runs or home / "video"))
    root = Path(spec.get("dir") or home / "watch").expanduser()
    ffmpeg = find_ffmpeg(spec.get("ffmpeg"))
    describer = None
    if spec.get("describe", True):
        brains = getattr(engines, "brains", None)
        brain = None if brains is None else (brains.director or brains.main)
        ap = getattr(video, "autopilot", None) if video is not None else None
        eyes = getattr(ap, "eyes", None)
        if eyes is None and spec.get("eyes", True):  # AutoPilot keeps no eyes when a judge checks its clips: the profile's own
            from gma.ap import eyes as ap_eyes

            registry = getattr(getattr(engines, "supervisor", None), "registry", None)
            eyes = ap_eyes.make(profile, registry)
        describer = Describer(brain, getattr(engines, "lanes", None), eyes if spec.get("eyes", True) else None,
                              getattr(ap, "ears", None), ffmpeg)

    def busy() -> bool:  # renders and the Director's writing go first; the Ghost describes in the quiet between
        lanes = getattr(engines, "lanes", None)
        if lanes is not None and lanes.talking():
            return True
        if multiverse is not None and (multiverse.current or multiverse.queued or multiverse.writing):
            return True
        run = getattr(video, "run", None) if video is not None else None
        return bool(run and run.get("state") in ("planning", "keyframe", "loading", "rendering", "checking", "stopping"))

    return WatchService(Ledger(root / "ledger.sqlite"), spec, describer, busy, ffmpeg)


def register(app: FastAPI, engines, profile: dict) -> WatchService:
    service = make_service(engines, profile, getattr(app.state, "multiverse", None))
    agents = getattr(engines, "agents", None)
    if isinstance(agents, dict):  # first: its words ("find the lighthouse") are claimed only while its page is open
        rest = dict(agents)
        agents.clear()
        agents["watch"] = WatchAgent(service)
        agents.update(rest)
    app.state.watch = service

    def bad(error: Exception | str, code: int = 400):
        return JSONResponse({"error": str(error)}, status_code=code)

    @app.get("/invisible")
    def page():
        return FileResponse(STATIC / "invisible" / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/invisible/{name:path}")
    def static(name: str):
        path = (STATIC / "invisible" / name).resolve()
        if not str(path).startswith(str((STATIC / "invisible").resolve())) or not path.is_file():
            return bad("not found", 404)
        return FileResponse(path, media_type=TYPES.get(path.suffix), headers={"Cache-Control": "no-cache"})

    def seen() -> None:
        service.start()
        service.seen_at = time.monotonic()

    @app.get("/api/watch/status")
    async def status():
        service.start()
        return service.status()

    @app.get("/api/watch/timeline")
    async def timeline(limit: int = 3000):
        seen()
        return {"creations": [service.public(c) for c in service.ledger.all(max(1, min(limit, 20000)))], "status": service.status()}

    @app.get("/api/watch/creations")
    async def search(q: str = "", kind: str = "", agent: str = "", limit: int = 60):
        seen()
        return {"creations": service.search(q, kind, agent, max(1, min(limit, 500))), "q": q}

    @app.get("/api/watch/creations/{cid}")
    def detail(cid: str):
        out = service.detail(cid)
        return out if out else bad("no such creation", 404)

    @app.get("/api/watch/creations/{cid}/file")
    def media(cid: str):
        path = service.media(cid)
        if path is None:
            return bad("no file", 404)
        return FileResponse(path, media_type=TYPES.get(path.suffix.lower()))

    @app.get("/api/watch/creations/{cid}/thumb")
    async def thumb(cid: str):
        path = await asyncio.to_thread(service.thumb, cid)
        if path is None:
            return bad("no picture", 404)
        return FileResponse(path, media_type=TYPES.get(path.suffix.lower()), headers={"Cache-Control": "max-age=300"})

    @app.post("/api/watch/creations")
    async def add(request: Request):
        service.start()
        try:
            body = await request.json()
        except ValueError:
            return bad("send JSON")
        if not isinstance(body, dict):
            return bad("send a JSON object")
        try:
            return service.add(body)
        except WatchError as error:
            return bad(error)

    @app.post("/api/watch/upload")
    async def upload(request: Request):
        service.start()
        q = request.query_params
        fields = {k: q.get(k) for k in ("agent", "kind", "title", "prompt", "ref", "summary", "description") if q.get(k)}
        if q.get("parents"):
            fields["parents"] = [p for p in q.get("parents").split(",") if p]
        data = await request.body()
        try:
            return await asyncio.to_thread(service.upload, q.get("name") or "creation", data, fields)
        except WatchError as error:
            return bad(error)

    @app.get("/api/watch/events")
    async def events(request: Request):
        seen()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        service.subs.add(queue)

        async def stream():
            try:
                yield f"data: {json.dumps({'type': 'status', 'creation': None, 'status': service.status()})}\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        service.seen_at = time.monotonic()
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                service.subs.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return service
