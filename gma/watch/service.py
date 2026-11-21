"""Invisible mode's engine: the Ghost watches what agents make, keeps the ledger, and describes each creation.

- every ``scan_s`` it reads the sources it watches by itself (gma/watch/sources.py) into the ledger;
- agents add their own creations through the open API (``add`` with a file path under an allowed folder, or ``upload``);
- when no render is running and nobody is talking, it describes the newest creation it has not described yet;
- the page hears every change as a server-sent event.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from . import sources
from .ledger import KINDS, Ledger, kind_of

SAFE = re.compile(r"[^\w.-]+")


class WatchError(ValueError):
    pass


class WatchService:
    def __init__(self, ledger: Ledger, spec: dict, describer=None, busy: Callable[[], bool] | None = None,
                 ffmpeg: str | None = None) -> None:
        self.ledger, self.spec, self.describer = ledger, dict(spec), describer
        self.busy = busy or (lambda: False)
        self.ffmpeg = ffmpeg
        self.dir = ledger.path.parent
        self.files = self.dir / "files"
        self.thumbs = self.dir / "thumbs"
        self.files.mkdir(parents=True, exist_ok=True)
        self.thumbs.mkdir(parents=True, exist_ok=True)
        roots = [spec.get("multiverse"), spec.get("runs"), *[s.get("library") for s in spec.get("studios") or []],
                 *(spec.get("allow") or []), str(self.files)]
        self.roots = [Path(r).expanduser().resolve() for r in roots if r]
        self.subs: set[asyncio.Queue] = set()
        self.task: asyncio.Task | None = None
        self.seen_at = 0.0
        self.describing: str | None = None
        self.scanned_at = 0.0
        self.last_error = ""

    # ------------------------------------------------------------------------------------------------- the loop
    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.get_running_loop().create_task(self.loop())

    async def loop(self) -> None:
        scan_s = float(self.spec.get("scan_s", 15))
        while True:
            try:
                await self.scan()
                deadline = time.monotonic() + scan_s
                while time.monotonic() < deadline:
                    if not await self.describe_one():
                        await asyncio.sleep(min(3.0, max(0.2, deadline - time.monotonic())))
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - one bad source never stops the watch
                self.last_error = f"{type(error).__name__}: {error}"[:300]
                await asyncio.sleep(scan_s)

    async def scan(self) -> int:
        found = await asyncio.to_thread(sources.scan, self.spec, self.ledger)
        changed = 0
        for r in found:
            try:
                c, new = self.ledger.put(r["ref"], kind=r["kind"], agent=r["agent"], source=r["source"], at=r.get("at"),
                                         title=r.get("title", ""), prompt=r.get("prompt", ""), parents=r.get("parents"),
                                         file=r.get("file", ""), meta=r.get("meta"))
            except ValueError:
                continue
            if new:
                changed += 1
                self.emit("put", c)
        self.scanned_at = time.time()
        return changed

    async def describe_one(self) -> bool:
        """Describes one creation if the machine is free; False when there was nothing to do (or no room)."""
        if self.describer is None or self.busy():
            return False
        todo = self.ledger.pending(1)
        if not todo:
            return False
        c = todo[0]
        self.describing = c["id"]
        self.emit("status", None)
        try:
            parents = [p for p in (self.ledger.get(pid) for pid in c["parents"]) if p]
            words, by = await self.describer.describe(c, parents)
            done = self.ledger.describe(c["id"], words["summary"], words["description"], words["tags"],
                                        status="described", error="")
            if done:
                done["meta"] = {**done["meta"], "described_by": by}
                self.ledger.put(done["ref"], kind=done["kind"], agent=done["agent"], source=done["source"], at=done["at"],
                                title=done["title"], prompt=done["prompt"], parents=done["parents"], file=done["file"],
                                meta=done["meta"])
                self.emit("put", self.ledger.get(c["id"]))
        except Exception as error:  # noqa: BLE001 - one creation that cannot be described is left with its own words
            self.ledger.describe(c["id"], c.get("title") or c["kind"], c.get("prompt") or "", [], status="plain",
                                 error=f"{type(error).__name__}: {error}")
        finally:
            self.describing = None
            self.emit("status", None)
        return True

    # ------------------------------------------------------------------------------------------------- the page
    def public(self, c: dict | None) -> dict | None:
        if c is None:
            return None
        out = {k: v for k, v in c.items() if k != "file"}
        out["meta"] = {k: v for k, v in (c.get("meta") or {}).items() if k != "poster"}
        out["media"] = f"/api/watch/creations/{c['id']}/file" if c.get("file") else None
        out["thumb"] = f"/api/watch/creations/{c['id']}/thumb" if c["kind"] in ("image", "video") and c.get("file") else None
        return out

    def status(self) -> dict:
        return {**self.ledger.stats(), "describing": self.describing, "scanned_at": self.scanned_at,
                "error": self.last_error, "busy": bool(self.busy()),
                "describer": self.describer.abilities() if hasattr(self.describer, "abilities") else None}

    def emit(self, kind: str, c: dict | None) -> None:
        event = {"type": kind, "creation": self.public(c), "status": self.status()}
        for q in list(self.subs):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ------------------------------------------------------------------------------------------------- the open API
    def allowed(self, path: Path) -> bool:
        try:
            real = path.expanduser().resolve()
        except OSError:
            return False
        return any(real == r or r in real.parents for r in self.roots)

    def add(self, body: dict) -> dict:
        """An agent's creation: {"agent", "kind"?, "title"?, "prompt"?, "text"?, "parents"?: [ids or refs], "ref"?,
        "path"?: a file under an allowed folder, "at"?, "summary"?, "description"?, "tags"?, "meta"?}."""
        agent = str(body.get("agent") or "").strip()
        if not agent:
            raise WatchError("say which agent made it: \"agent\"")
        path = str(body.get("path") or "").strip()
        if path:
            p = Path(path).expanduser()
            if not p.is_file():
                raise WatchError(f"no file at {path}")
            if not self.allowed(p):
                raise WatchError("that file is outside the folders the Ghost may read; upload it instead "
                                 "(POST /api/watch/upload) or add its folder to watch.allow")
            path = str(p.resolve())
        kind = str(body.get("kind") or (kind_of(path) if path else "text"))
        if kind not in KINDS:
            raise WatchError(f"kind must be one of {', '.join(KINDS)}")
        if kind != "text" and not path:
            raise WatchError(f"a {kind} needs its file: \"path\" (or upload it)")
        text = str(body.get("text") or "")
        prompt = str(body.get("prompt") or text)
        if kind == "text" and not (prompt or body.get("title")):
            raise WatchError("a text needs its words: \"text\"")
        ref = str(body.get("ref") or f"api:{agent}:{uuid.uuid4().hex[:12]}")
        parents = body.get("parents") or []
        if not isinstance(parents, list):
            parents = [parents]
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        c, _ = self.ledger.put(ref, kind=kind, agent=agent, source="api", at=body.get("at"), title=str(body.get("title") or ""),
                               prompt=prompt, parents=[str(p) for p in parents], file=path, meta=meta,
                               summary=str(body.get("summary") or ""), description=str(body.get("description") or ""),
                               tags=body.get("tags") if isinstance(body.get("tags"), list) else None)
        self.emit("put", c)
        return self.public(c)

    def upload(self, name: str, data: bytes, fields: dict) -> dict:
        if not data:
            raise WatchError("the upload is empty")
        limit = int(self.spec.get("upload_mb", 512)) * 1024 * 1024
        if len(data) > limit:
            raise WatchError(f"the upload is over {limit // (1024 * 1024)} MB")
        clean = SAFE.sub("-", Path(name or "creation").name).strip("-.") or "creation"
        day = time.strftime("%Y%m%d")
        folder = self.files / day
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{uuid.uuid4().hex[:8]}-{clean}"
        target.write_bytes(data)
        return self.add({**fields, "path": str(target)})

    def media(self, cid: str) -> Path | None:
        c = self.ledger.get(cid)
        if not c or not c.get("file"):
            return None
        p = Path(c["file"])
        return p if p.is_file() and self.allowed(p) else None

    def thumb(self, cid: str) -> Path | None:
        """A small picture of an image or a clip (made once, from its poster when it has one)."""
        c = self.ledger.get(cid)
        if not c or c["kind"] not in ("image", "video"):
            return None
        out = self.thumbs / f"{cid}.jpg"
        source = self.media(cid)
        if source is None:
            return None
        if out.is_file() and out.stat().st_mtime >= source.stat().st_mtime:
            return out
        poster = Path((c.get("meta") or {}).get("poster") or "")
        start = poster if poster.is_file() and self.allowed(poster) else source
        if not self.ffmpeg:
            return start if start.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else None
        args = [self.ffmpeg, "-v", "error", "-y"]
        if start.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v"):
            args += ["-ss", "0.6"]
        args += ["-i", str(start), "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(out)]
        subprocess.run(args, capture_output=True, timeout=60)
        return out if out.is_file() else None

    def search(self, q: str = "", kind: str = "", agent: str = "", limit: int = 60) -> list[dict]:
        return [self.public(c) for c in self.ledger.search(q, kind, agent, limit)]

    def detail(self, cid: str) -> dict | None:
        c = self.ledger.get(cid)
        if not c:
            return None
        out = self.public(c)
        out["parent_list"] = [self.public(p) for p in (self.ledger.get(i) for i in c["parents"]) if p]
        out["child_list"] = [self.public(k) for k in self.ledger.children(cid)]
        return out
