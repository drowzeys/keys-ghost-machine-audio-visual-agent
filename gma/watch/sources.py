"""What the Ghost watches by itself in invisible mode: the creations the app's own makers leave on disk, read without
asking them anything. Each scanner returns plain records for Ledger.put (``ref``, ``kind``, ``agent``, ``source``,
``at``, ``title``, ``prompt``, ``parents`` as refs, ``file``, ``meta``).

- **multiverse**: each film's idea, then every node with a clip, draft or keyframe (a node's parent is the clip before it;
  its siblings are the forks), and each exported film.
- **runs**: the video folder's runs. An AutoPilot run is its idea, every take of every piece (a retake is a fork of the
  piece before it), the joined film, its narration and the finished film; a video panel run is its idea and its clips.
- **studio**: an Imagination Studio's finished jobs (its outputs under its library folder), skipping jobs by title
  (the app's own keyframes are already in the films above).
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
from pathlib import Path

from .ledger import kind_of


def when(value, fallback: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value).timestamp()
        except ValueError:
            pass
    return fallback


def mtime(path: Path, fallback: float = 0.0) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return fallback


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def multiverse(root: str | Path) -> list[dict]:
    root, out = Path(root).expanduser(), []
    if not root.is_dir():
        return out
    for tree_file in sorted(root.glob("*/tree.json")):
        tree = read_json(tree_file)
        if not tree or not isinstance(tree.get("nodes"), dict):
            continue
        pid, folder = tree.get("id") or tree_file.parent.name, tree_file.parent
        created = when(tree.get("created"), mtime(tree_file))
        idea_ref = f"mv:{pid}"
        out.append({"ref": idea_ref, "kind": "text", "agent": "you", "source": "multiverse", "at": created - 0.001,
                    "title": tree.get("title") or tree.get("idea") or pid, "prompt": tree.get("idea") or "",
                    "parents": [], "file": "", "meta": {"film": pid, "open": f"/#{pid}"}})
        for nid, n in tree["nodes"].items():
            files = n.get("files") or {}
            name = files.get("clip") or files.get("draft") or files.get("keyframe")
            if not name:
                continue
            path = folder / "nodes" / nid / name
            if not path.is_file():
                continue
            poster = files.get("poster") or files.get("first")
            kind = kind_of(name)
            label = n.get("label") or n.get("beat") or nid
            out.append({"ref": f"mv:{pid}:{nid}", "kind": kind, "agent": "Multiverse Director", "source": "multiverse",
                        "at": when(n.get("at") or n.get("created"), mtime(path, created)),
                        "title": label[:1].upper() + label[1:], "prompt": n.get("prompt") or "",
                        "parents": [f"mv:{pid}:{n['parent']}"] if n.get("parent") else [idea_ref], "file": str(path),
                        "meta": {"film": pid, "node": nid, "quality": n.get("quality"), "letter": n.get("letter"),
                                 "taken": bool(n.get("taken")), "mood": n.get("mood"), "camera": n.get("camera"),
                                 "mixed_from": n.get("mixed_from") or [], "open": f"/#{pid}",
                                 "poster": str(folder / "nodes" / nid / poster) if poster else ""}})
        for film in sorted(folder.glob("film-*.mp4")):
            if ".look-" in film.name:  # a copy of a film in a look (gma/look.py), not a film of its own
                continue
            nid = film.stem[len("film-"):]
            out.append({"ref": f"mv:{pid}:film:{nid}", "kind": "video", "agent": "Multiverse export", "source": "multiverse",
                        "at": mtime(film), "title": f"Film of the path to “{(tree['nodes'].get(nid) or {}).get('label', nid)}”",
                        "prompt": tree.get("idea") or "", "parents": [f"mv:{pid}:{nid}"], "file": str(film),
                        "meta": {"film": pid, "open": f"/#{pid}"}})
    return out


def runs(root: str | Path) -> list[dict]:
    root, out = Path(root).expanduser(), []
    if not root.is_dir():
        return out
    for run_file in sorted(root.glob("*/run.json")):
        run = read_json(run_file)
        if not run:
            continue
        rid, folder = run.get("id") or run_file.parent.name, run_file.parent
        created = when(run.get("created"), mtime(run_file))
        ap = run.get("autopilot") if isinstance(run.get("autopilot"), dict) else None
        agent = "AutoPilot" if ap else "Video panel"
        idea_ref = f"run:{rid}"
        out.append({"ref": idea_ref, "kind": "text", "agent": "you", "source": "autopilot" if ap else "video",
                    "at": created - 0.001, "title": (ap or {}).get("name") or run.get("idea") or rid,
                    "prompt": run.get("idea") or "", "parents": [], "file": "", "meta": {"run": rid, "mode": run.get("mode")}})
        last = idea_ref
        if ap:
            for seg in ap.get("segments") or []:
                sid, chosen = seg.get("id") or f"S{seg.get('index', 0):04d}", seg.get("output")
                parent, kept = last, None
                for take in seg.get("takes") or []:
                    rel = take.get("output")
                    path = root / rel if rel else None
                    if not path or not path.is_file():
                        continue
                    ref = f"run:{rid}:{sid}:t{take.get('take')}"
                    reasons = take.get("reasons") or []
                    out.append({"ref": ref, "kind": "video", "agent": "AutoPilot", "source": "autopilot", "at": mtime(path, created),
                                "title": (seg.get("beat") or sid) + (f" (take {take.get('take')})" if take.get("take", 1) > 1 else ""),
                                "prompt": seg.get("prompt") or "", "parents": [parent], "file": str(path),
                                "meta": {"run": rid, "piece": seg.get("index"), "take": take.get("take"),
                                         "passed": take.get("passed"), "kept": rel == chosen,
                                         "reasons": [str(r)[:200] for r in reasons][:4], "scene": seg.get("scene"),
                                         "poster": str(path.with_name(f"frames-take{take.get('take')}") / "f01.png")}})
                    if rel == chosen or kept is None:
                        kept = ref
                last = kept or last
            film, final = ap.get("film") or {}, ap.get("film_final") or {}
            if film.get("path") and (root / film["path"]).is_file():
                out.append({"ref": f"run:{rid}:film", "kind": "video", "agent": "AutoPilot", "source": "autopilot",
                            "at": mtime(root / film["path"]), "title": f"{ap.get('name') or 'The film'}, joined",
                            "prompt": run.get("idea") or "", "parents": [last], "file": str(root / film["path"]),
                            "meta": {"run": rid, "pieces": film.get("segments"), "seconds": film.get("seconds")}})
                narration = final.get("narration") or {}
                parts = [f"run:{rid}:film"]
                if narration.get("file") and (folder / narration["file"]).is_file():
                    out.append({"ref": f"run:{rid}:narration", "kind": "audio", "agent": "AutoPilot · Drama 3", "source": "autopilot",
                                "at": mtime(folder / narration["file"]), "title": "The narration",
                                "prompt": narration.get("text") or "", "parents": [f"run:{rid}:film"],
                                "file": str(folder / narration["file"]), "meta": {"run": rid}})
                    parts.append(f"run:{rid}:narration")
                if final.get("path") and (root / final["path"]).is_file():
                    score = (final.get("score") or {}).get("caption") or ""
                    out.append({"ref": f"run:{rid}:final", "kind": "video", "agent": "AutoPilot", "source": "autopilot",
                                "at": mtime(root / final["path"]), "title": f"{ap.get('name') or 'The film'}, finished",
                                "prompt": ("Score: " + score) if score else run.get("idea") or "", "parents": parts,
                                "file": str(root / final["path"]), "meta": {"run": rid, "seconds": final.get("seconds")}})
        else:
            for clip in run.get("clips") or []:
                i = clip.get("index", 0)
                path = (root / clip["output"]) if clip.get("output") else folder / f"clip{i:03d}.mp4"
                if not path.is_file():
                    path = folder / f"clip{i:03d}.mp4"
                if not path.is_file():
                    continue
                ref = f"run:{rid}:clip{i}"
                out.append({"ref": ref, "kind": "video", "agent": "Video panel", "source": "video", "at": mtime(path, created),
                            "title": clip.get("beat") or f"Clip {i + 1}", "prompt": clip.get("prompt") or "", "parents": [last],
                            "file": str(path), "meta": {"run": rid, "clip": i,
                                                        "poster": str(folder / f"clip{i:03d}.jpg")}})
                last = ref
    return out


def studio(spec: dict, ledger=None) -> list[dict]:
    """An Imagination Studio's finished jobs: {"url", "library" (its home folder), "name", "skip_titles": [...]}."""
    url, library = str(spec.get("url") or "").rstrip("/"), Path(str(spec.get("library") or "")).expanduser()
    if not url or not library.is_dir():
        return []
    try:
        with urllib.request.urlopen(url + "/api/jobs", timeout=4) as reply:
            jobs = json.loads(reply.read())
    except (OSError, ValueError):
        return []
    jobs = jobs if isinstance(jobs, list) else jobs.get("jobs") or []
    skip = tuple(spec.get("skip_titles") or ())
    name, out = spec.get("name") or "Imagination Studio", []
    for job in jobs:
        if not isinstance(job, dict) or job.get("status") != "done":
            continue
        if skip and str(job.get("title") or "").startswith(skip):
            continue
        source = job.get("source")
        parents = []
        if isinstance(source, str) and source and ledger is not None:
            made = ledger.by_file(str(library / source))
            if made:
                parents.append(made["id"])
        for i, rel in enumerate(job.get("outputs") or []):
            path = library / rel
            if not path.is_file():
                continue
            out.append({"ref": f"studio:{name}:{job.get('id')}:{i}", "kind": kind_of(rel), "agent": f"{name} · {job.get('kind')}",
                        "source": "studio", "at": when(job.get("finished"), mtime(path)), "title": job.get("title") or job.get("kind") or "",
                        "prompt": "", "parents": parents, "file": str(path), "meta": {"job": job.get("id"), "kind": job.get("kind")}})
    return out


def scan(spec: dict, ledger=None) -> list[dict]:
    """Everything the profile's "watch" section points at."""
    out = []
    if spec.get("multiverse"):
        out += multiverse(spec["multiverse"])
    if spec.get("runs"):
        out += runs(spec["runs"])
    for s in spec.get("studios") or []:
        out += studio(s, ledger)
    return out
