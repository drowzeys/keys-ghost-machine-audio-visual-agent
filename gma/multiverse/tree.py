"""The multiverse tree: nodes are clips, the Ghost stands on one of them, and the path from the root to the Ghost is the
active timeline. Plain dicts, so a project is one JSON file (``tree.json``) beside its node folders:

    <root>/<project>/tree.json
    <root>/<project>/nodes/<node>/keyframe.jpg | draft.mp4 | clip.mp4 | poster.jpg | last.jpg | first.jpg

A node:
    id, parent, children (ids, in the order they were written), depth
    letter ("A", "B", "C" within its round of options), round (1 for the first options at the parent, 2 after
    "show me other options", ...), label (the one line the Ghost speaks), beat, prompt, mood, camera, seed
    status: planned | waiting | keyframe | drafting | draft | rendering | done | failed
    quality: None | "draft" | "full"   (what its best clip is)
    taken: on a timeline the person walked (chosen at least once)
    mixed_from: [ids] when it was written as a mix of two options
    files: {name: file name in its folder}, cost: {keyframe_s, draft_s, full_s, ...}, qc: {...}, error

What the page draws for each node (``states``): "ghost" (where the Ghost stands), "active" (on the active timeline),
"option" (the newest options ahead of the Ghost, the possible futures), "dimmed" (every road not taken, and options of
an earlier round).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

LETTERS = "ABCDEFGH"
STATUSES = ("planned", "waiting", "keyframe", "drafting", "draft", "rendering", "done", "failed")
ID = re.compile(r"^[a-z0-9]{4,24}$")


class TreeError(ValueError):
    """A request that does not fit the tree (no such node, an option that is not ahead of the Ghost, ...)."""


def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:10]


def node(parent: dict | None, plan: dict, **more) -> dict:
    return {"id": new_id("n"), "parent": parent["id"] if parent else None, "children": [],
            "depth": parent["depth"] + 1 if parent else 0, "letter": None, "round": 0,
            "label": str(plan.get("label") or plan.get("beat") or "")[:120], "beat": str(plan.get("beat") or "")[:160],
            "prompt": str(plan.get("prompt") or ""), "mood": str(plan.get("mood") or "")[:60],
            "camera": str(plan.get("camera") or "")[:80], "seed": int(plan.get("seed") or 0), "status": "planned",
            "quality": None, "taken": False, "mixed_from": [], "files": {}, "cost": {}, "qc": {}, "error": None,
            "progress": None, "created": time.time(), "updated": time.time(), **more}


class Tree:
    """One project's tree. Every change goes through a method here, so the rules live in one place."""

    def __init__(self, data: dict) -> None:
        self.data = data

    # ----- making one -----

    @classmethod
    def new(cls, idea: str, opening: dict, settings: dict | None = None, title: str | None = None) -> "Tree":
        idea = re.sub(r"\s+", " ", str(idea or "")).strip()
        if not idea:
            raise TreeError("a film needs an idea")
        root = node(None, opening, taken=True)
        root["label"] = root["label"] or "The opening"
        now = time.time()
        return cls({"id": new_id("p"), "title": (title or idea)[:80], "idea": idea[:600], "created": now, "updated": now,
                    "root": root["id"], "ghost": root["id"], "nodes": {root["id"]: root}, "settings": dict(settings or {}),
                    "history": [{"at": now, "action": "start", "node": root["id"]}], "film": None})

    # ----- reading -----

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def ghost(self) -> str:
        return self.data["ghost"]

    def get(self, nid: str) -> dict:
        found = self.data["nodes"].get(nid)
        if found is None:
            raise TreeError(f"no node {nid!r} in this film")
        return found

    def parent(self, nid: str) -> dict | None:
        p = self.get(nid)["parent"]
        return self.get(p) if p else None

    def path_to(self, nid: str) -> list[str]:
        out, at = [], self.get(nid)
        while at is not None:
            out.append(at["id"])
            at = self.get(at["parent"]) if at["parent"] else None
        return out[::-1]

    def active_path(self) -> list[str]:
        """The active timeline: the root to where the Ghost stands."""
        return self.path_to(self.ghost)

    def round_of(self, nid: str) -> int:
        """The newest round of options written at this node (0: none yet)."""
        return max((self.get(c)["round"] for c in self.get(nid)["children"]), default=0)

    def options(self, nid: str | None = None, every_round: bool = False) -> list[dict]:
        """The options ahead of a node (the Ghost's by default): its newest round, in letter order."""
        at = self.get(nid or self.ghost)
        newest = self.round_of(at["id"])
        kids = [self.get(c) for c in at["children"]]
        return kids if every_round else [k for k in kids if k["round"] == newest]

    def states(self) -> dict[str, str]:
        """What the page draws each node as: ghost, active, option or dimmed (see the module's docstring)."""
        active = set(self.active_path())
        ahead = {o["id"] for o in self.options()}
        out = {}
        for nid in self.data["nodes"]:
            out[nid] = ("ghost" if nid == self.ghost else "active" if nid in active
                        else "option" if nid in ahead else "dimmed")
        return out

    def story(self, nid: str | None = None) -> list[dict]:
        """The beats along the path to a node (the Ghost's by default), for the Director."""
        return [{"label": n["label"], "beat": n["beat"], "prompt": n["prompt"]} for n in map(self.get, self.path_to(nid or self.ghost))]

    # ----- changing it -----

    def touch(self, action: str, nid: str, **more) -> None:
        now = time.time()
        self.data["updated"] = now
        self.data["history"].append({"at": now, "action": action, "node": nid, **more})
        del self.data["history"][:-500]

    def add_options(self, nid: str, plans: list[dict], mixed_from: list[str] | None = None, same_round: bool = False) -> list[dict]:
        """Writes a round of options at a node: letters go on from the round's (a mix joins the newest round)."""
        at = self.get(nid)
        newest = self.round_of(nid)
        rnd = newest if (same_round and newest) else newest + 1
        used = {self.get(c)["letter"] for c in at["children"] if self.get(c)["round"] == rnd}
        added = []
        for plan in plans:
            letter = next((x for x in LETTERS if x not in used), None)
            if letter is None:
                break
            used.add(letter)
            one = node(at, plan, letter=letter, round=rnd, mixed_from=list(mixed_from or []))
            self.data["nodes"][one["id"]] = one
            at["children"].append(one["id"])
            added.append(one)
        self.touch("options", nid, round=rnd, added=[a["id"] for a in added])
        return added

    def choose(self, nid: str) -> dict:
        """The Ghost steps onto an option ahead of it (or any option in the tree: the Ghost walks to its parent's
        timeline first). Returns the node; the caller renders it at full quality when it is not yet."""
        one = self.get(nid)
        if one["parent"] is None:
            raise TreeError("the opening is not an option; go back to it instead")
        one["taken"] = True
        self.data["ghost"] = nid
        self.touch("choose", nid)
        return one

    def goto(self, nid: str) -> dict:
        """The Ghost walks back (or forward) to a node already on a walked timeline, to branch from there."""
        one = self.get(nid)
        if not one["taken"]:
            raise TreeError("that road was never taken; choose it instead")
        self.data["ghost"] = nid
        self.touch("goto", nid)
        return one

    def set(self, nid: str, **fields) -> dict:
        one = self.get(nid)
        if "status" in fields and fields["status"] not in STATUSES:
            raise TreeError(f"no status {fields['status']!r}")
        one.update(fields)
        one["updated"] = time.time()
        self.data["updated"] = one["updated"]
        return one

    # ----- what the page gets -----

    def snapshot(self) -> dict:
        return {**self.data, "active": self.active_path(), "states": self.states(),
                "ahead": [o["id"] for o in self.options()]}


class Store:
    """The projects on disk, one folder each; writes are atomic (a temp file renamed over tree.json)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache: dict[str, Tree] = {}

    def folder(self, pid: str) -> Path:
        if not ID.match(pid or ""):
            raise TreeError("no such film")
        return self.root / pid

    def node_folder(self, pid: str, nid: str) -> Path:
        if not ID.match(nid or ""):
            raise TreeError("no such node")
        path = self.folder(pid) / "nodes" / nid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, tree: Tree) -> None:
        with self.lock:
            folder = self.folder(tree.id)
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / f".tree.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(tree.data, indent=1))
            os.replace(tmp, folder / "tree.json")
            self.cache[tree.id] = tree

    def load(self, pid: str) -> Tree:
        with self.lock:
            if pid in self.cache:
                return self.cache[pid]
            path = self.folder(pid) / "tree.json"
            if not path.is_file():
                raise TreeError("no such film")
            tree = Tree(json.loads(path.read_text()))
            self.cache[pid] = tree
            return tree

    def list(self) -> list[dict]:
        out = []
        for path in self.root.glob("*/tree.json"):
            try:
                tree = self.load(path.parent.name)
            except (TreeError, ValueError, OSError):
                continue
            root = tree.get(tree.data["root"])
            out.append({"id": tree.id, "title": tree.data["title"], "idea": tree.data["idea"], "updated": tree.data["updated"],
                        "nodes": len(tree.data["nodes"]), "depth": tree.get(tree.ghost)["depth"],
                        "poster": root["files"].get("poster") and f"{root['id']}/{root['files']['poster']}"})
        return sorted(out, key=lambda p: -p["updated"])
