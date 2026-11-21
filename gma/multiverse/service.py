"""The multiverse's work: write options with the Director, render them one at a time, check the cut, export a path.

Renders go through the app's own renderer (gma/studio.py ``HeadlessRenderer`` on a real profile): ``room`` is the
memory governor's admission for the clip (it compares the clip's estimated peak with what is free, releases idle
models first, and says no rather than start what will not fit), and every keyframe and clip runs inside the governor's
session (``begin`` / ``enter`` / ``leave`` in the renderer). This service adds no guard of its own: it only keeps to
the rule that a render is never started beside another one. One worker takes one job at a time from a priority queue
(the picked branch's full render before any draft), waits while the video panel or AutoPilot is rendering, and asks
``room`` again (every ``room_retry_s``) while the governor says there is no room.

A draft (an option's cheap preview): the keyframe (drawn beside the parent's last frame where the renderer can), then,
unless ``draft.clip`` is off, a short low-res clip opening on the parent's last frame. A full render (the picked
branch): the clip at full size and length, opening on the parent's last frame (the opening shot: on its keyframe).
After each clip: its poster, first and last frames (ffmpeg), and the cut checked against the parent's last frame
(gma/ap/qc.py ``seam_numbers``; ``seam`` with a looker when one is given).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from voice_agent.video import Progress, Refused, find_ffmpeg

from .tree import Store, Tree, TreeError
from .writer import Writer

DEFAULTS = {
    "options": 3,
    "auto_options": True,  # options are written at a node as soon as its full clip is made
    "draft": {"width": 640, "height": 352, "seconds": 5, "clip": True, "keyframe": True},  # H3 makes 5-15 s only
    "full": {"width": 864, "height": 480, "seconds": 5},
    "room_retry_s": 5.0,
    "room_tries": 360,  # half an hour of "no room yet" before a render is given up
    "poll_s": 1.0,
    "seam_corr": None,  # under this layout correlation the cut is a jump (None: gma/ap/qc.py's SEAM_CORR)
}


EDIT_PIXELS = (512 * 512, 1536 * 1536)  # a keyframe drawn beside a reference picture (an edit) is made in this many pixels


def keyframe_size(width: int, height: int, edit: bool) -> tuple[int, int]:
    """The keyframe's size: the clip's own, except that an edit (beside the parent's last frame) is scaled, at the
    same shape, into the edit's pixel range (a 640x352 draft's keyframe is drawn at 704x384), in steps of 16."""
    if not edit:
        return width, height
    low, high = EDIT_PIXELS
    scale = 1.0
    if width * height < low:
        scale = (low / (width * height)) ** 0.5
    elif width * height > high:
        scale = (high / (width * height)) ** 0.5
    step = 16
    w, h = width * scale, height * scale
    if scale >= 1:
        w, h = -(-int(w) // step) * step, -(-int(h) // step) * step
        while w * h < low:
            w, h = w + step, h
    else:
        w, h = max(step, int(w) // step * step), max(step, int(h) // step * step)
    return int(w), int(h)


def merged(spec: dict | None) -> dict:
    out = {**DEFAULTS, **{k: v for k, v in (spec or {}).items() if k not in ("draft", "full")}}
    out["draft"] = {**DEFAULTS["draft"], **((spec or {}).get("draft") or {})}
    out["full"] = {**DEFAULTS["full"], **((spec or {}).get("full") or {})}
    return out


def sniff(data: bytes) -> str:
    return "png" if data[:4] == b"\x89PNG" else "jpg"


class MultiverseService:
    def __init__(self, store: Store, renderer, writer: Writer | None = None, settings: dict | None = None,
                 external_busy: Callable[[], bool] | None = None, say: Callable[[str, str], None] | None = None,
                 looker=None, ffmpeg: str | None = None) -> None:
        self.store, self.renderer = store, renderer
        self.writer = writer or Writer()
        self.settings = merged(settings)
        self.external_busy = external_busy or (lambda: False)
        self.say = say or (lambda text, mood="happy": None)
        self.looker = looker
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.queue: asyncio.PriorityQueue | None = None
        self.seq = itertools.count()
        self.queued: dict[tuple[str, str, str], int] = {}  # (pid, nid, kind) -> seq, for the jobs not yet taken
        self.current: dict | None = None
        self.worker: asyncio.Task | None = None
        self.tasks: set[asyncio.Task] = set()
        self.listeners: dict[str, set[asyncio.Queue]] = {}
        self.writing: dict[str, str] = {}  # pid -> the node whose options are being written
        self.means: dict[str, list[float]] = {"keyframe": [], "draft": [], "full": []}
        self.focus: str | None = None  # the film the page shows (the voice's "this film")
        self.seen_at = 0.0  # when the page last asked for a film (the voice takes multiverse words only while it is open)
        self.resumed = False

    # ----- the page's view -----

    def view(self, pid: str) -> dict:
        tree = self.store.load(pid)
        cur = self.current if self.current and self.current["pid"] == pid else None
        return {**tree.snapshot(), "writing": self.writing.get(pid),
                "queue": [{"node": k[1], "kind": k[2]} for k, _ in sorted(self.queued.items(), key=lambda kv: kv[1]) if k[0] == pid],
                "working": {"node": cur["nid"], "kind": cur["kind"]} if cur else None,
                "estimates": self.estimates(), "settings": {k: self.settings[k] for k in ("options", "draft", "full")},
                "renderer": getattr(self.renderer, "label", None) or getattr(self.renderer, "name", "renderer")}

    def estimates(self) -> dict:
        """The mean seconds a keyframe, a draft and a full clip took here (None until one was made)."""
        return {k: round(sum(v[-10:]) / len(v[-10:]), 1) if v else None for k, v in self.means.items()}

    def changed(self, pid: str) -> None:
        for q in list(self.listeners.get(pid, ())):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(time.monotonic())

    def save(self, tree: Tree) -> None:
        self.store.save(tree)
        self.changed(tree.id)

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    # ----- what the person asks -----

    async def create(self, idea: str, title: str | None = None) -> Tree:
        """A new film: the tree with its opening planned at once; the Director writes the opening and it is rendered
        in the background (the voice answers in a second)."""
        tree = Tree.new(idea, {"label": "Writing the opening…", "prompt": idea, "seed": int(time.time()) % 100000},
                        settings={k: self.settings[k] for k in ("options", "draft", "full")}, title=title)
        self.save(tree)
        self.focus = tree.id
        self.spawn(self._open(tree.id))
        return tree

    async def _open(self, pid: str) -> None:
        tree = self.store.load(pid)
        root = tree.get(tree.data["root"])
        self.writing[pid] = root["id"]
        self.changed(pid)
        try:
            plan = await self.writer.opening(tree.data["idea"])
        finally:
            self.writing.pop(pid, None)
        tree.set(root["id"], **{k: plan[k] for k in ("label", "beat", "prompt", "mood", "camera")})
        self.save(tree)
        self.enqueue(pid, root["id"], "full")

    async def request_options(self, pid: str, nid: str | None = None, note: str = "", fresh: bool = False) -> list[dict]:
        """Writes a round of options at a node (the Ghost's by default) and queues their drafts. ``fresh`` (the
        person's "show me other options") writes a new round even when one exists; the old one stays, dimmed."""
        tree = self.store.load(pid)
        nid = nid or tree.ghost
        at = tree.get(nid)
        if at["quality"] != "full":
            raise TreeError("that clip is not made yet; its options come once it is")
        if self.writing.get(pid) == nid:
            return []
        old = tree.options(nid)
        if old and not fresh:
            return old
        self.writing[pid] = nid
        self.changed(pid)
        try:
            avoid = tree.options(nid, every_round=True)
            plans = await self.writer.options(tree.data["idea"], tree.story(nid), int(self.settings["options"]), avoid, note,
                                              seed=len(tree.data["nodes"]))
        finally:
            self.writing.pop(pid, None)
        for i, p in enumerate(plans):
            p["seed"] = (at["seed"] + 101 * (len(at["children"]) + i + 1)) % 1_000_000
        if fresh:  # the old round's drafts not yet made are dropped: the person asked for others
            for o in old:
                self.drop(pid, o["id"], "draft")
        added = tree.add_options(nid, plans)
        self.save(tree)
        for o in added:
            self.enqueue(pid, o["id"], "draft")
        if nid == tree.ghost and added:
            self.say(self.options_line(added), "curious")
        return added

    @staticmethod
    def options_line(options: list[dict]) -> str:
        parts = [f"{o['letter']}: {o['label']}" for o in options]
        return "Here are the paths. " + ". ".join(parts) + ". Which way?"

    async def choose(self, pid: str, nid: str) -> dict:
        tree = self.store.load(pid)
        one = tree.choose(nid)
        for sib in tree.options(one["parent"], every_round=True):  # the roads not taken: no more drafts for them
            if sib["id"] != nid:
                self.drop(pid, sib["id"], "draft")
        self.save(tree)
        self.drop(pid, nid, "draft")
        if one["quality"] != "full":
            self.enqueue(pid, nid, "full")
        elif self.settings["auto_options"] and not tree.options(nid):
            self.spawn(self.request_options(pid, nid))
        return one

    async def goto(self, pid: str, nid: str) -> dict:
        tree = self.store.load(pid)
        one = tree.goto(nid)
        self.save(tree)
        if one["quality"] == "full" and self.settings["auto_options"] and not tree.options(nid) and self.writing.get(pid) != nid:
            self.spawn(self.request_options(pid, nid))
        else:  # options here whose drafts were dropped when a sibling was taken: drafted now, the Ghost is back
            self.draft_missing(tree, nid)
        return one

    def draft_missing(self, tree: Tree, nid: str) -> int:
        """Queues the draft of every option at ``nid`` that was never drafted (dropped when a sibling was taken)."""
        count = 0
        for opt in tree.options(nid):
            if opt["status"] == "planned" and opt["quality"] is None and not opt["taken"]:
                self.enqueue(tree.id, opt["id"], "draft")
                count += 1
        return count

    async def mix(self, pid: str, a: str, b: str) -> dict:
        tree = self.store.load(pid)
        x, y = tree.get(a), tree.get(b)
        if not x["parent"] or x["parent"] != y["parent"]:
            raise TreeError("only two options at the same step can be mixed")
        plan = await self.writer.mix(tree.data["idea"], tree.story(x["parent"]), x, y)
        plan["seed"] = (x["seed"] + y["seed"] + 7) % 1_000_000
        added = tree.add_options(x["parent"], [plan], mixed_from=[a, b], same_round=True)
        if not added:
            raise TreeError("there is no room for another option at this step")
        self.save(tree)
        self.enqueue(pid, added[0]["id"], "draft")
        return added[0]

    async def retry(self, pid: str, nid: str) -> dict:
        tree = self.store.load(pid)
        one = tree.get(nid)
        if one["status"] == "planned" and one["quality"] is None:  # never drafted (dropped when a sibling was taken)
            self.enqueue(pid, nid, "full" if one["taken"] else "draft")
            return one
        if one["status"] != "failed":
            return one
        tree.set(nid, status="planned" if one["quality"] is None else ("draft" if one["quality"] == "draft" else "done"), error=None)
        self.save(tree)
        self.enqueue(pid, nid, "full" if one["taken"] else "draft")
        return one

    async def export(self, pid: str) -> Path:
        """The active timeline joined into one film (each clip at its best: full, else its draft), re-encoded to one
        size so drafts and full clips join."""
        tree = self.store.load(pid)
        files = []
        for nid in tree.active_path():
            n = tree.get(nid)
            name = n["files"].get("clip") or n["files"].get("draft")
            if name:
                files.append(self.store.node_folder(pid, nid) / name)
        files = [f for f in files if f.is_file()]
        if not files:
            raise TreeError("no clip on this path is made yet")
        if not self.ffmpeg:
            raise TreeError("exporting needs ffmpeg on this machine")
        out = self.store.folder(pid) / f"film-{tree.ghost}.mp4"
        w, h = int(self.settings["full"]["width"]), int(self.settings["full"]["height"])
        probes = [await asyncio.to_thread(self.probe, f) for f in files]
        args = [self.ffmpeg, "-v", "error", "-y"]
        for f in files:
            args += ["-i", str(f)]
        parts = []
        for i, (has_audio, seconds) in enumerate(probes):
            parts.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                         f"setsar=1,fps=24[v{i}];")
            # each clip's own sound (H3 makes it with the picture); a clip without any gets silence of its length
            # (the sound is cut to the picture's length, so a clip's sound never runs into the next picture)
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{seconds:.3f},apad,"
                         f"atrim=0:{seconds:.3f},asetpts=PTS-STARTPTS[a{i}];" if has_audio else
                         f"anullsrc=r=48000:cl=stereo,atrim=0:{seconds:.3f},asetpts=PTS-STARTPTS[a{i}];")
        joined = "".join(f"[v{i}][a{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=1[outv][outa]"
        args += ["-filter_complex", "".join(parts) + joined, "-map", "[outv]", "-map", "[outa]",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        done = await asyncio.to_thread(subprocess.run, args, capture_output=True, timeout=900)
        if done.returncode != 0 or not out.is_file():
            raise TreeError("ffmpeg could not join the path: " + done.stderr.decode(errors="replace")[-200:])
        tree.data["film"] = {"file": out.name, "path": tree.active_path(), "at": time.time()}
        self.save(tree)
        return out

    def probe(self, clip: Path) -> tuple[bool, float]:
        """(has a sound track, the picture's length in seconds) of a clip, with ffprobe beside ffmpeg."""
        probe = str(Path(self.ffmpeg).with_name("ffprobe"))
        if not Path(probe).is_file():
            probe = shutil.which("ffprobe") or "ffprobe"
        got = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_type:format=duration", "-of", "json",
                              str(clip)], capture_output=True, text=True, timeout=60)
        info = json.loads(got.stdout or "{}")
        kinds = {st.get("codec_type") for st in info.get("streams", [])}
        return "audio" in kinds, float((info.get("format") or {}).get("duration") or 0) or 5.0

    # ----- the queue -----

    def resume(self) -> int:
        """Once per start (the queue lives in memory): every render cut off by a restart goes back on the queue - a node
        left keyframe/drafting/rendering/waiting, and an option still planned. Taken nodes get their full clip, the rest
        their draft. Returns how many were queued again."""
        if self.resumed:
            return 0
        self.resumed = True
        count = 0
        for entry in self.store.list():
            try:
                tree = self.store.load(entry["id"])
            except TreeError:
                continue
            touched = False
            for nid, one in tree.data["nodes"].items():
                cut = one["status"] in ("waiting", "keyframe", "drafting", "rendering")
                # an option still planned, ahead of the Ghost (not one dropped when a sibling was taken: those are
                # drafted when the Ghost goes back there, or by the page's "Draft it")
                pending = one["status"] == "planned" and one["quality"] is None and one["parent"] == tree.ghost
                if not (cut or pending) or not one.get("prompt"):
                    continue
                if one["label"].startswith("Writing"):
                    continue  # its plan was never written
                if cut:
                    tree.set(nid, status="planned", progress=None)
                    touched = True
                kind = "full" if one["taken"] and one["quality"] != "full" else "draft"
                if kind == "draft" and one["quality"] in ("draft", "full"):
                    continue
                self.enqueue(tree.id, nid, kind)
                count += 1
            if touched:
                self.save(tree)
        return count

    def enqueue(self, pid: str, nid: str, kind: str) -> None:
        if self.queue is None:
            self.queue = asyncio.PriorityQueue()
        key = (pid, nid, kind)
        if key in self.queued or (self.current and (self.current["pid"], self.current["nid"], self.current["kind"]) == key):
            return
        seq = next(self.seq)
        self.queued[key] = seq
        self.queue.put_nowait((0 if kind == "full" else 1, seq, {"pid": pid, "nid": nid, "kind": kind}))
        if self.worker is None or self.worker.done():
            self.worker = asyncio.get_running_loop().create_task(self.work())
        self.changed(pid)

    def drop(self, pid: str, nid: str, kind: str) -> None:
        """A queued job forgotten (the worker skips it); a job at work goes on."""
        if self.queued.pop((pid, nid, kind), None) is not None:
            self.changed(pid)

    async def work(self) -> None:
        while True:
            _, seq, job = await self.queue.get()
            key = (job["pid"], job["nid"], job["kind"])
            if self.queued.get(key) != seq:
                continue  # dropped (or queued again later)
            del self.queued[key]
            self.current = job
            try:
                await self.render(job)
            except asyncio.CancelledError:
                raise
            except (Refused, TreeError, OSError, RuntimeError, ValueError) as error:
                self.failed(job, str(error))
            except Exception as error:  # noqa: BLE001 - one render's trouble never stops the queue
                self.failed(job, f"{type(error).__name__}: {error}")
            finally:
                self.current = None
                self.changed(job["pid"])

    def failed(self, job: dict, why: str) -> None:
        try:
            tree = self.store.load(job["pid"])
            tree.set(job["nid"], status="failed", error=why[:400], progress=None)
            self.save(tree)
        except TreeError:
            pass

    async def follow(self, tree: Tree, nid: str, handle: str, stage: str) -> Progress:
        began = time.monotonic()
        while True:
            got = await self.renderer.poll(handle)
            if got.state == "done":
                if not got.output:
                    raise Refused("The render finished without a file.")
                return got
            if got.state in ("failed", "cancelled"):
                raise Refused(f"The {stage} {got.state}" + (f": {str(got.error)[:200]}" if got.error else "."))
            tree.set(nid, progress={"stage": got.stage or stage, "at": got.at, "of": got.of, "eta_s": got.eta_s,
                                    "elapsed_s": round(time.monotonic() - began, 1)})
            self.changed(tree.id)
            await asyncio.sleep(float(self.settings["poll_s"]))

    async def admitted(self, tree: Tree, nid: str, clip: dict) -> None:
        """Waits for the video panel to be idle and for the governor to say the clip fits (``renderer.room``)."""
        tries = 0
        while True:
            if self.external_busy():
                why = "waiting for the video panel's render to finish"
            else:
                refusal = await self.renderer.room(clip)
                if refusal is None:
                    return
                why = str(refusal)
            tries += 1
            if tries > int(self.settings["room_tries"]):
                raise Refused(f"No room to render after waiting: {why}")
            if tree.get(nid)["status"] != "waiting" or tree.get(nid)["error"] != why:
                tree.set(nid, status="waiting", error=why)
                self.save(tree)
            await asyncio.sleep(float(self.settings["room_retry_s"]))

    async def render(self, job: dict) -> None:
        pid, nid, kind = job["pid"], job["nid"], job["kind"]
        tree = self.store.load(pid)
        one = tree.get(nid)
        if (kind == "draft" and one["quality"] in ("draft", "full")) or (kind == "full" and one["quality"] == "full"):
            return
        folder = self.store.node_folder(pid, nid)
        parent = tree.parent(nid)
        start = None
        if parent is not None:
            last = parent["files"].get("last")
            start = self.store.node_folder(pid, parent["id"]) / last if last else None
            if start is None or not start.is_file():
                raise TreeError("the clip before this one is not made yet")
        size = self.settings["draft" if kind == "draft" else "full"]
        began = time.monotonic()
        clip = {"prompt": one["prompt"], "seed": one["seed"], "seconds": float(size["seconds"]), "width": int(size["width"]),
                "height": int(size["height"]), "title": f"Multiverse · {one['letter'] or 'opening'} · {one['label'][:40]}"}
        # the keyframe: every draft's preview still (beside the parent's last frame), and the opening's first frame
        needs_key = (kind == "draft" and size.get("keyframe", True) and "keyframe" not in one["files"]) or \
                    (parent is None and "keyframe" not in one["files"])
        if needs_key:
            await self.admitted(tree, nid, clip)
            tree.set(nid, status="keyframe", error=None, progress={"stage": "keyframe"})
            self.save(tree)
            t0 = time.monotonic()
            handle = await self.keyframe(one["prompt"], one["seed"], [str(start)] if start else None,
                                         keyframe_size(int(size["width"]), int(size["height"]), start is not None))
            done = await self.follow(tree, nid, handle, "keyframe")
            picture = await self.renderer.picture(done.output)
            if not picture:
                raise Refused("The keyframe was made but could not be read.")
            name = f"keyframe.{sniff(picture)}"
            (folder / name).write_bytes(picture)
            tree.set(nid, files={**one["files"], "keyframe": name},
                     cost={**one["cost"], "keyframe_s": round(time.monotonic() - t0, 1)})
            self.means["keyframe"].append(time.monotonic() - t0)
            self.save(tree)
        if kind == "draft" and not size.get("clip", True):
            tree.set(nid, status="draft", quality="draft", progress=None, error=None)
            self.save(tree)
            return
        if parent is None:
            start = folder / one["files"]["keyframe"]
        clip["image_file"] = str(start)
        await self.admitted(tree, nid, clip)
        tree.set(nid, status="drafting" if kind == "draft" else "rendering", error=None,
                 progress={"stage": "queued"})
        self.save(tree)
        t0 = time.monotonic()
        handle = await self.renderer.submit(clip)
        done = await self.follow(tree, nid, handle, "clip")
        name = "draft.mp4" if kind == "draft" else "clip.mp4"
        await self.renderer.fetch(done.output, folder / name)
        render_s = round(time.monotonic() - t0, 1)
        files = {**tree.get(nid)["files"], ("draft" if kind == "draft" else "clip"): name}
        files.update(await asyncio.to_thread(self.frames, folder / name, folder))
        qc = dict(tree.get(nid)["qc"])
        if start is not None and parent is not None and (folder / "first.jpg").is_file():
            qc[f"seam_{kind}"] = await asyncio.to_thread(self.seam, start, folder / "first.jpg")
        cost = {**tree.get(nid)["cost"], f"{kind}_s": round(time.monotonic() - began, 1), f"{kind}_render_s": render_s,
                f"{kind}_spec": f"{clip['width']}x{clip['height']} · {clip['seconds']:g} s"}
        if done.timings:
            cost[f"{kind}_timings"] = done.timings
        self.means[kind].append(render_s)
        tree.set(nid, files=files, qc=qc, cost=cost, progress=None, error=None,
                 status="draft" if kind == "draft" else "done", quality="draft" if kind == "draft" else "full")
        self.save(tree)
        if kind == "full" and self.settings["auto_options"] and nid == tree.ghost and not tree.options(nid):
            self.spawn(self.request_options(pid, nid))

    async def keyframe(self, prompt: str, seed: int, references: list[str] | None, size: tuple[int, int]) -> str:
        """The renderer's keyframe, beside the reference picture when it can take one (HeadlessRenderer and the stub
        can; the base Renderer takes a prompt and a seed only)."""
        try:
            return await self.renderer.keyframe(prompt, seed, references=references, size=size)
        except TypeError:
            return await self.renderer.keyframe(prompt, seed)

    def frames(self, clip: Path, folder: Path) -> dict:
        """poster.jpg (a small still near the start), first.jpg and last.jpg (the next clip opens on it)."""
        if not self.ffmpeg:
            return {}
        out = {}
        for args, name in ((["-ss", "0.3", "-i", str(clip), "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4"], "poster.jpg"),
                           (["-i", str(clip), "-frames:v", "1", "-q:v", "2"], "first.jpg"),
                           (["-sseof", "-0.25", "-i", str(clip), "-frames:v", "1", "-update", "1", "-q:v", "2"], "last.jpg")):
            subprocess.run([self.ffmpeg, "-v", "error", "-y", *args, str(folder / name)], capture_output=True, timeout=120)
            if (folder / name).is_file():
                out[name.split(".")[0]] = name
        return out

    def seam(self, before: Path, first: Path) -> dict:
        """The cut into a clip, against its parent's last frame: gma/ap/qc.py's numbers (layout correlation and
        brightness gap), and the looker's yes/no when one is set (``qc.seam``)."""
        from gma.ap import qc

        try:
            if self.looker is not None:
                r = qc.seam(self.looker, before, first, self.ffmpeg or "ffmpeg")
                return {"corr": r.get("corr"), "luma_gap": r.get("luma_gap"), "p": r.get("p"), "ok": bool(r.get("passed"))}
            n = qc.seam_numbers(qc.picture_grey(before, self.ffmpeg or "ffmpeg"), qc.picture_grey(first, self.ffmpeg or "ffmpeg"))
            return {**n, "ok": n["corr"] >= float(self.settings["seam_corr"] or qc.SEAM_CORR)}
        except Exception as error:  # noqa: BLE001 - a check that could not run is said, never a failed clip
            return {"ok": None, "error": str(error)[:200]}

