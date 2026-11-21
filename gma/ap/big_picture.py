"""The big-picture Director: a film made in passes instead of fixed piece by piece.

Fixing each piece as it is made (a retake here, a redrawn keyframe there) breaks the film's flow: every regeneration
moves its seams, the next piece inherits a different last frame, and patches pile up (the user, 2026-11-08: "the more
the fixes found and done the more disjointed the video will be"). With ``settings.big_picture``:

1. **Draft.** The whole film is made with no per-piece retakes, each piece chained on the last frame of the one before
   (autopilot.retakes_now is 0); the self-check still runs on every piece and records what it found.
2. **Review.** The looker describes the whole film (the first and last frame of every piece on one sheet), and one
   Director call sees every finding together with the arc, the shots, what the looker saw and every seam. It writes
   ONE consolidated fix plan: related findings merged into fewer, larger fixes; each fixed at its root, in the earliest
   piece it starts in (a wrong direction in piece 1 is fixed in piece 1, not patched later); cosmetic notes dropped
   when their fix would cost continuity, each with its reason; at most MAX_FIXES fixes.
3. **Regenerate.** The film is made again from the earliest piece the plan touches, forward, in order, so every seam
   is chained again (never one piece patched in the middle between mismatched neighbours). A piece the plan does not
   touch is asked to keep its draft as closely as it can.
4. **Keep or reject.** The new film is kept only when its whole-film score (story fidelity and seam continuity) beats
   the draft's and its weakest seam is no weaker than the draft's; otherwise the draft is put back ("do no harm").
   ``settings.fix_budget`` passes at most (1 by default, 2 at most); a rejected pass ends the passes.

Every plan is logged in the run folder (``fix-plan-pass<N>.json``: the findings, the merged fixes, what was dropped and
why, the scores and the decision); ``run["big_picture"]`` keeps the same record."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

MAX_FIXES = 3
STORY = ("end state", "end place", "direction", "contact")  # what the piece was meant to show
FIDELITY_WEIGHT = 0.7

FIX_SYSTEM = """You are the director of a short film that has been drafted in full. You see the whole film at once: \
its arc, every piece's shot, what a vision model saw in each piece, every problem its checks found and every seam \
between pieces. You write ONE consolidated fix plan for a second pass that makes the film again from the earliest \
piece the plan touches, forward, each piece opening on the last frame of the one before.

Think about the film as one continuous flow, not a list of defects:
- Merge related findings into one larger fix (a wrong direction and a wrong end place in the same piece are one fix: \
the movement). A problem that starts in an earlier piece is fixed THERE, at its root, never patched in a later piece.
- Every regeneration changes the seams after it, so fewer, more decisive fixes are better than many small ones. Drop \
cosmetic findings (a look, a minor prop, a doubtful answer of the checker) whose fix would cost continuity, and say why.
- A finding the frames clearly do not support (the checker's description contradicts it) is dropped as a false alarm.
- At most 3 fixes. Each fix says exactly what the shot must do differently, concretely enough to film: the camera's \
place and move, the direction of movement in plain words ("climbs UP the stairs, away from the camera below"), where \
the person starts and stops, and the visible result with its seconds. A fix for a piece that opens on a keyframe (the \
film's first piece) also gives that keyframe's new description when the opening picture is part of the problem.
- If nothing is worth the risk of regenerating, return no fixes.

Answer with one JSON object and nothing else:
{"overview": "<40 words: what the film does well and what breaks its flow>",
 "fixes": [{"piece": <number>, "merges": ["F1", "F3"], "root_cause": "<15 words>", "change": "<50 words: what this piece's shot must do differently>", "keyframe": "<only for a piece that opens on a keyframe and needs a new one: 60 words, else empty>"}],
 "dropped": [{"finding": "F2", "why": "<15 words>"}]}
Every finding id appears once: in one fix's "merges" or in "dropped"."""


def ap():
    from . import autopilot
    return autopilot


def state(run: dict) -> dict:
    return run.setdefault("big_picture", {"pass": 0, "state": "draft", "passes": []})


# ---------- the review ----------

def findings_of(segs: list) -> list[dict]:
    """Every failed self-check of the film's kept pieces, numbered F1, F2, ... (the seam and opening checks included)."""

    out = []
    for seg in segs:
        for c in seg.get("qc_checks") or []:
            if not c.get("passed"):
                out.append({"id": f"F{len(out) + 1}", "piece": seg["index"], "check": c["check"], "state": c.get("state", ""),
                            "where": c.get("frame", ""), "p": c.get("p"), "confident": bool(c.get("confident"))})
    return out


def seams_of(segs: list) -> list[dict]:
    return [{"into": seg["index"], "corr": c.get("corr"), "p": c.get("p")} for seg in segs for c in seg.get("qc_checks") or []
            if c.get("check") == "seam"]


def score_of(segs: list) -> dict:
    """The whole-film score: story fidelity (the share of each piece's story checks that passed, averaged) and seam
    continuity (the seams' mean layout correlation); ``low``: the weakest seam."""

    rates = []
    for seg in segs:
        story = [c for c in seg.get("qc_checks") or [] if c.get("check") in STORY]
        rates.append(sum(1 for c in story if c.get("passed")) / len(story) if story else 1.0)
    fidelity = sum(rates) / len(rates) if rates else 0.0
    corrs = [s["corr"] for s in seams_of(segs) if s.get("corr") is not None]
    continuity = sum(corrs) / len(corrs) if corrs else 1.0
    return {"fidelity": round(fidelity, 3), "continuity": round(continuity, 3), "low": round(min(corrs), 3) if corrs else None,
            "score": round(FIDELITY_WEIGHT * fidelity + (1 - FIDELITY_WEIGHT) * continuity, 3), "pieces": [round(r, 3) for r in rates]}


def edge_frames(run: dict, segs: list, folder: Path) -> list[tuple[str, Path]]:
    """The first and the last frame of every piece, in film order (the looker's whole-film sheet)."""

    a = ap()
    out = []
    folder.mkdir(parents=True, exist_ok=True)
    for seg in segs:
        clip = a.HOME / seg["output"]
        for label, args in (("first", ["-ss", "0"]), ("last", ["-sseof", "-0.1"])):
            target = folder / f"{seg['id']}-{label}.png"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", *args, "-i", str(clip), "-frames:v", "1", "-update", "1", str(target)],
                               check=True, capture_output=True, timeout=60, env=a.ENV)
                out.append((f"piece {seg['index']} {label}", target))
            except (subprocess.SubprocessError, OSError):
                pass
    return out


def review(run: dict, n: int) -> dict:
    a = ap()
    segs = a.committed(run)
    folder = a.run_dir(run["id"]) / f"review-pass{n}"
    found = findings_of(segs)
    film_seen = ""
    look = a.looker()
    if look is not None:
        try:
            film_seen = look.describe(edge_frames(run, segs, folder)) or ""
        except Exception as error:  # noqa: BLE001 - the plan is written from the pieces' own records
            film_seen = f"(the whole-film look failed: {str(error)[:120]})"
    return {"findings": found, "seams": seams_of(segs), "score": score_of(segs), "film_seen": film_seen,
            "pieces": [{"index": s["index"], "beat": s.get("beat"), "action": (s.get("fields") or {}).get("action"),
                        "camera": (s.get("fields") or {}).get("camera"), "keyframe": s.get("keyframe_prompt", ""),
                        "seen": s.get("described", "")} for s in segs]}


def plan_prompt(run: dict, seen: dict) -> str:
    arc = run.get("arc") or {}
    beats = "\n".join(f"  {b['index']}. {b['beat']} -> ends on: {b['end_state']}" for b in arc.get("beats") or [])
    pieces = []
    for p in seen["pieces"]:
        mine = [f for f in seen["findings"] if f["piece"] == p["index"]]
        seam = next((s for s in seen["seams"] if s["into"] == p["index"]), None)
        pieces.append(
            f"PIECE {p['index']}: {p['beat']}\n  action: {p['action']}\n  camera: {p['camera']}"
            + (f"\n  opening keyframe: {p['keyframe']}" if p.get("keyframe") else "")
            + f"\n  what the looker saw: {str(p['seen'])[:500]}"
            + (f"\n  seam into it: layout correlation {seam['corr']} (1.0 = the same picture; under 0.5 a jump), looker {seam['p']}" if seam else "")
            + "\n  findings:" + ("".join(f"\n    {f['id']} {f['check']} ({'sure' if f['confident'] else 'doubtful'}, p {f['p']}): "
                                         f"{f['state']} [{f['where']}]" for f in mine) or " none"))
    return (f"THE FILM'S ARC:\n{beats}\nPayoff: {arc.get('payoff', '')}\n\n" + "\n\n".join(pieces)
            + f"\n\nTHE WHOLE FILM, first and last frame of every piece, as the looker saw it: {seen['film_seen'][:700]}\n"
            + f"\nWhole-film score now: story {seen['score']['fidelity']}, seams {seen['score']['continuity']}.\n"
            + "Write the consolidated fix plan.")


def plan_problem(data: dict, seen: dict, count: int) -> str:
    fixes, dropped = data.get("fixes"), data.get("dropped")
    if not isinstance(fixes, list) or not isinstance(dropped, list):
        return '"fixes" and "dropped" are lists'
    if len(fixes) > MAX_FIXES:
        return f"{len(fixes)} fixes: merge them into {MAX_FIXES} at most"
    ids = {f["id"] for f in seen["findings"]}
    covered = []
    for fix in fixes:
        if not isinstance(fix, dict) or not isinstance(fix.get("piece"), int) or not 1 <= fix["piece"] <= count:
            return f'each fix names its "piece", a number from 1 to {count}'
        if not str(fix.get("change") or "").strip():
            return 'each fix says its "change"'
        covered += [str(x) for x in fix.get("merges") or []]
    covered += [str((d or {}).get("finding")) for d in dropped if isinstance(d, dict)]
    missing = sorted(ids - set(covered), key=lambda x: int(x[1:]))
    if missing:
        return f"findings {', '.join(missing)} are in no fix and not dropped: merge each into a fix or drop it with its reason"
    return ""


# ---------- regenerate, keep or reject ----------

def move(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(source), str(target))


def stash(run: dict, start: int, into: Path) -> dict:
    """Pieces ``start`` on, out of the film into ``into`` (their folders, memory, bible and checkpoints), with the
    film and the run's own records of them; the memory and the bible put back as they were before piece ``start``."""

    a = ap()
    folder = a.run_dir(run["id"])
    gone = [s for s in run["segments"] if s["index"] >= start]
    for seg in gone:
        for part in ("segments", "memory", "bible", "checkpoints"):
            name = seg["id"] if part == "segments" else f"{seg['id']}.json"
            move(folder / part / name, into / part / name)
    for name in ("film.mp4", "memory.json", "bible.json"):
        if (folder / name).exists():
            into.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, into / name)
    kept = {"segments": copy.deepcopy(gone), "film": copy.deepcopy(run.get("film")), "locations": copy.deepcopy(run.get("locations")),
            "behind": copy.deepcopy(run.get("behind")), "opening": copy.deepcopy(run.get("opening")),
            "steers": copy.deepcopy(run.get("steers"))}
    run["segments"] = [s for s in run["segments"] if s["index"] < start]
    before = start - 1
    memory = a.read_json(folder / "memory" / f"S{before:04d}.json") if before else None
    a.write_json(folder / "memory.json", {k: copy.deepcopy((memory or {}).get(k, a.EMPTY[k])) for k in a.EMPTY})
    bible = a.read_json(folder / "bible" / f"S{before:04d}.json") if before else None
    a.save_bible(run, bible if isinstance(bible, dict) and bible.get("version") else a.continuity.new(run["seed"], run.get("cast")))
    run["behind"] = [b for b in run.get("behind") or [] if b["index"] < start]
    run["locations"] = [p for p in run.get("locations") or [] if p["from"] < start]
    for steer in run.get("steers") or []:
        if steer.get("applied") and steer.get("applied_at", steer.get("from", 0)) >= start:
            steer["applied"] = False
    if start == 1:
        run.pop("opening", None)
    return kept


def restore(run: dict, kept: dict, start: int, folder_from: Path, folder_rejected: Path) -> None:
    """The draft put back (the regenerated pieces moved aside to ``folder_rejected``)."""

    a = ap()
    folder = a.run_dir(run["id"])
    for seg in [s for s in run["segments"] if s["index"] >= start]:
        for part in ("segments", "memory", "bible", "checkpoints"):
            name = seg["id"] if part == "segments" else f"{seg['id']}.json"
            move(folder / part / name, folder_rejected / part / name)
    if (folder / "film.mp4").exists():
        folder_rejected.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "film.mp4", folder_rejected / "film.mp4")
    for seg in kept["segments"]:
        for part in ("segments", "memory", "bible", "checkpoints"):
            name = seg["id"] if part == "segments" else f"{seg['id']}.json"
            move(folder_from / part / name, folder / part / name)
    for name in ("film.mp4", "memory.json", "bible.json"):
        if (folder_from / name).exists():
            shutil.copy2(folder_from / name, folder / name)
    run["segments"] = [s for s in run["segments"] if s["index"] < start] + kept["segments"]
    for key in ("film", "locations", "behind", "opening", "steers"):
        if kept.get(key) is not None:
            run[key] = kept[key]


def next_pass(run: dict) -> bool:
    """At the end of a pass: score a regenerated film against the draft (keep or put the draft back), then review the
    film and plan the next pass while the budget allows. True when the film is to be made again from a piece on."""

    a = ap()
    bp = state(run)
    folder = a.run_dir(run["id"])
    if bp.get("state") == "regenerating":
        decide(run)
    budget = int(run["settings"].get("fix_budget") or 0)
    if bp.get("state") == "settled" or bp["pass"] >= budget:
        bp["state"] = "settled"
        a.save(run)
        return False
    n = bp["pass"] + 1
    a.say(run, f"Big picture, pass {n}: the looker reviews the whole film; the Director plans its fixes as one.", stage="director")
    seen = review(run, n)
    count = len(a.committed(run))
    prompt = plan_prompt(run, seen)
    try:
        data, rejected = a.ask_json(lambda text: a.ask_text(run, FIX_SYSTEM, text), lambda d: plan_problem(d, seen, count), prompt, "fix plan")
    except a.Failed as error:
        bp.update(state="settled")
        a.write_json(folder / f"fix-plan-pass{n}.json", {"review": seen, "prompt": prompt, "error": str(error)[:300]})
        a.say(run, "Big picture: no fix plan could be written; the draft stands: " + str(error)[:160])
        return False
    fixes = [f for f in data["fixes"] if isinstance(f, dict)]
    record = {"pass": n, "review": seen, "prompt": prompt, "plan": data, "rejected_answers": rejected, "draft_score": seen["score"]}
    if not fixes:
        bp.update(state="settled")
        record["decision"] = "no fixes: the draft stands"
        a.write_json(folder / f"fix-plan-pass{n}.json", record)
        bp["passes"].append({k: record[k] for k in ("pass", "plan", "draft_score", "decision")})
        a.say(run, "Big picture: the Director keeps the draft as it is (no fix worth its cost to continuity).")
        return False
    start = min(int(f["piece"]) for f in fixes)
    kept = stash(run, start, folder / "drafts" / f"pass{n}")
    record["start"] = start
    a.write_json(folder / f"fix-plan-pass{n}.json", record)
    bp.update(**{"pass": n, "state": "regenerating", "start": start, "fixes": fixes, "draft": kept,
                 "draft_score": seen["score"], "draft_pieces": [s for s in kept["segments"]]})
    merged = sum(len(f.get("merges") or []) for f in fixes)
    a.say(run, f"Big picture, pass {n}: {len(seen['findings'])} findings -> {len(fixes)} fixes ({merged} merged, "
               f"{len(data['dropped'])} dropped); the film is made again from segment {start} on.", stage="planned")
    a.save(run)
    return True


def decide(run: dict) -> None:
    a = ap()
    bp = state(run)
    folder = a.run_dir(run["id"])
    n, start, kept = bp["pass"], bp["start"], bp["draft"]
    earlier = [s for s in a.committed(run) if s["index"] < start]
    draft = score_of(earlier + kept["segments"])
    new = score_of(a.committed(run))
    weaker = draft["low"] is not None and new["low"] is not None and new["low"] < draft["low"]
    keep = new["score"] > draft["score"] and not weaker
    why = (f"kept: score {new['score']} > {draft['score']}" + (f", weakest seam {new['low']} >= {draft['low']}" if new["low"] is not None else "")
           if keep else f"draft put back: score {new['score']} vs {draft['score']}" + (f", weakest seam {new['low']} < {draft['low']}" if weaker else ""))
    if not keep:
        restore(run, kept, start, folder / "drafts" / f"pass{n}", folder / "drafts" / f"pass{n}-rejected")
        a.publish(run)
    record = a.read_json(folder / f"fix-plan-pass{n}.json") or {}
    record.update(new_score=new, draft_score=draft, decision=why)
    a.write_json(folder / f"fix-plan-pass{n}.json", record)
    bp["passes"].append({"pass": n, "start": start, "fixes": bp.get("fixes"), "draft_score": draft, "new_score": new, "decision": why})
    bp.pop("draft", None)
    bp.pop("draft_pieces", None)
    bp["state"] = "reviewed" if keep else "settled"
    a.say(run, f"Big picture, pass {n}: {why}.")
    a.save(run)


def plan_text(run: dict, index: int) -> str:
    """What the Director reads while the film is made again: this piece's fix, or the draft to keep."""

    bp = run.get("big_picture") or {}
    if bp.get("state") != "regenerating" or index < int(bp.get("start") or 0):
        return ""
    mine = [f for f in bp.get("fixes") or [] if int(f.get("piece") or 0) == index]
    draft = next((s for s in (bp.get("draft") or {}).get("segments") or [] if s["index"] == index), None)
    text = (f"\n\nTHE BIG-PICTURE PASS: the whole film was drafted and reviewed; it is made again from segment {bp['start']} on, "
            "in order, each piece opening on the last frame of the new piece before.")
    for fix in mine:
        text += (f"\nTHE FIX FOR THIS SEGMENT (root cause: {fix.get('root_cause', '')}): {fix['change']}"
                 + (f"\nIts opening keyframe now: {fix['keyframe']}" if str(fix.get("keyframe") or "").strip() else "")
                 + "\nWrite the shot so that this fix shows on screen; keep everything else of the draft.")
    if draft:
        f = draft.get("fields") or {}
        text += (f"\nThe draft of this segment (keep what the fix does not change, so the film flows as before): beat "
                 f"\"{draft.get('beat')}\"; action: {f.get('action')}; camera: {f.get('camera')}; setting: {f.get('setting')}.")
    return text
