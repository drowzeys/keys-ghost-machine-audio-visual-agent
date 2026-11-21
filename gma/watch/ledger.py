"""The ledger of creations the Ghost watches over in invisible mode: every image, clip, sound and text an agent made,
when, by whom, from what (its parents: a fork is a creation with a parent that already had a child), and the Ghost's
own summary and description of it. SQLite with an FTS5 index, so any creation can be found by its words.

A creation's id comes from its ``ref`` (its source's own name for it, e.g. ``mv:p1:n2`` or ``api:hermes:42``), so
watching a source again updates the same creation, and a parent can be named before it is seen.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

KINDS = ("image", "video", "audio", "text")
ID = re.compile(r"^c[0-9a-f]{12}$")
FIELDS = ("id", "ref", "at", "kind", "agent", "source", "title", "prompt", "parents", "file", "summary", "description",
          "tags", "status", "error", "meta", "added")
SCHEMA = """
create table if not exists creations (
  id text primary key, ref text unique not null, at real not null, kind text not null, agent text not null,
  source text not null, title text default '', prompt text default '', parents text default '[]', file text default '',
  summary text default '', description text default '', tags text default '', status text default 'new',
  error text default '', meta text default '{}', added real not null);
create index if not exists creations_at on creations(at);
create virtual table if not exists search using fts5(id unindexed, title, prompt, summary, description, agent, tags, kind,
  tokenize = 'porter unicode61');
"""


def id_of(ref: str) -> str:
    return "c" + hashlib.sha1(ref.encode()).hexdigest()[:12]


def parent_id(value: str) -> str:
    """A parent named by its id, or by its ref."""
    value = str(value)
    return value if ID.match(value) else id_of(value)


def kind_of(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return "image"
    if ext in (".mp4", ".mov", ".webm", ".mkv", ".m4v"):
        return "video"
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"):
        return "audio"
    return "text"


def query_of(text: str) -> str:
    """Plain words as an FTS5 query: every word must match, each as a prefix ("light" finds "lighthouse")."""
    words = re.findall(r"\w+", text.lower())
    return " AND ".join(f'"{w}"*' for w in words[:12])


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("pragma journal_mode = wal")

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        out = {k: r[k] for k in FIELDS}
        out["parents"] = json.loads(out["parents"] or "[]")
        out["meta"] = json.loads(out["meta"] or "{}")
        out["tags"] = [t for t in (out["tags"] or "").split(",") if t]
        return out

    def _index(self, c: dict) -> None:
        self.db.execute("delete from search where id = ?", (c["id"],))
        self.db.execute("insert into search (id, title, prompt, summary, description, agent, tags, kind) values (?,?,?,?,?,?,?,?)",
                        (c["id"], c["title"], c["prompt"], c["summary"], c["description"], c["agent"], " ".join(c["tags"]), c["kind"]))

    def put(self, ref: str, *, kind: str, agent: str, source: str, at: float | None = None, title: str = "",
            prompt: str = "", parents: list | None = None, file: str = "", meta: dict | None = None,
            summary: str = "", description: str = "", tags: list | None = None) -> tuple[dict, bool]:
        """Adds or updates the creation ``ref``: (the creation, whether it changed). A new file (or new words) makes
        the Ghost look at it again; an unchanged one keeps its description."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        if not ref or not agent:
            raise ValueError("a creation needs a ref and an agent")
        cid = id_of(ref)
        pids = []
        for p in parents or []:
            pid = parent_id(p)
            if pid != cid and pid not in pids:
                pids.append(pid)
        with self.lock:
            old = self.row(self.db.execute("select * from creations where id = ?", (cid,)).fetchone())
            now = time.time()
            c = dict(old or {"id": cid, "ref": ref, "added": now, "summary": "", "description": "", "tags": [], "status": "new",
                             "error": "", "meta": {}})
            new = {"at": float(at if at is not None else (old or {}).get("at") or now), "kind": kind, "agent": agent[:60],
                   "source": source[:40], "title": (title or "")[:300], "prompt": (prompt or "")[:4000], "parents": pids,
                   "file": file or ""}
            fresh = old is None or any(old[k] != new[k] for k in ("file", "prompt", "title", "kind"))
            changed = old is None or any(old[k] != new[k] for k in new) or (meta is not None and meta != old["meta"])
            c.update(new)
            if meta is not None:
                c["meta"] = meta
            if summary or description:  # the agent described it itself
                c.update(summary=summary[:200], description=description[:3000], tags=[str(t)[:30] for t in (tags or [])][:8],
                         status="described")
                changed = True
            elif fresh:
                c["status"] = "new"
            if changed:
                self.db.execute(
                    "insert or replace into creations (id, ref, at, kind, agent, source, title, prompt, parents, file, summary,"
                    " description, tags, status, error, meta, added) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (c["id"], c["ref"], c["at"], c["kind"], c["agent"], c["source"], c["title"], c["prompt"],
                     json.dumps(c["parents"]), c["file"], c["summary"], c["description"], ",".join(c["tags"]), c["status"],
                     c["error"], json.dumps(c["meta"]), c["added"]))
                self._index(c)
                self.db.commit()
            return c, changed

    def describe(self, cid: str, summary: str, description: str, tags: list, status: str = "described", error: str = "") -> dict | None:
        with self.lock:
            self.db.execute("update creations set summary = ?, description = ?, tags = ?, status = ?, error = ? where id = ?",
                            (summary[:200], description[:3000], ",".join(str(t)[:30] for t in tags[:8]), status, error[:300], cid))
            c = self.row(self.db.execute("select * from creations where id = ?", (cid,)).fetchone())
            if c:
                self._index(c)
            self.db.commit()
            return c

    def get(self, cid: str) -> dict | None:
        with self.lock:
            return self.row(self.db.execute("select * from creations where id = ?", (cid,)).fetchone())

    def by_file(self, file: str) -> dict | None:
        with self.lock:
            return self.row(self.db.execute("select * from creations where file = ? order by at limit 1", (file,)).fetchone())

    def children(self, cid: str) -> list[dict]:
        with self.lock:
            rows = self.db.execute("select * from creations where parents like ? order by at", (f'%"{cid}"%',)).fetchall()
        return [self.row(r) for r in rows]

    def pending(self, limit: int = 1) -> list[dict]:
        """Creations the Ghost has not described yet, newest first (what just landed matters most)."""
        with self.lock:
            rows = self.db.execute("select * from creations where status = 'new' order by at desc limit ?", (limit,)).fetchall()
        return [self.row(r) for r in rows]

    def all(self, limit: int = 3000) -> list[dict]:
        """The newest ``limit`` creations, oldest first (the timeline's order)."""
        with self.lock:
            rows = self.db.execute("select * from (select * from creations order by at desc limit ?) order by at", (limit,)).fetchall()
        return [self.row(r) for r in rows]

    def search(self, text: str = "", kind: str = "", agent: str = "", limit: int = 60) -> list[dict]:
        """The creations whose words match (best first), or the newest; a kind and an agent narrow it."""
        where, args = [], []
        if kind:
            where.append("c.kind = ?"); args.append(kind)
        if agent:
            where.append("c.agent = ?"); args.append(agent)
        q = query_of(text)
        with self.lock:
            if q:
                sql = ("select c.* from search s join creations c on c.id = s.id where search match ?"
                       + "".join(f" and {w}" for w in where) + " order by bm25(search, 0, 4, 1, 3, 2, 2, 2, 1) limit ?")
                rows = self.db.execute(sql, [q, *args, limit]).fetchall()
            else:
                sql = "select * from creations c" + (" where " + " and ".join(where) if where else "") + " order by at desc limit ?"
                rows = self.db.execute(sql, [*args, limit]).fetchall()
        return [self.row(r) for r in rows]

    def stats(self) -> dict:
        with self.lock:
            total = self.db.execute("select count(*) from creations").fetchone()[0]
            agents = {r[0]: r[1] for r in self.db.execute("select agent, count(*) from creations group by agent order by 2 desc")}
            kinds = {r[0]: r[1] for r in self.db.execute("select kind, count(*) from creations group by kind")}
            waiting = self.db.execute("select count(*) from creations where status = 'new'").fetchone()[0]
        return {"total": total, "agents": agents, "kinds": kinds, "waiting": waiting}
