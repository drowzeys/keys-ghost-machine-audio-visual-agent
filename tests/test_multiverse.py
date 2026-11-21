"""keys Ghost Multiverse Studio (gma/multiverse): the tree model and its states, the store, the voice's words, the
Director's options, the one-at-a-time render queue against the governor's room, export, and the HTTP API on the whole
app. The renders are the stub renderer's (Pillow + ffmpeg) or a fake; no model is loaded."""

import asyncio
import subprocess
import json
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from gma.multiverse import tree as T
from gma.multiverse.service import MultiverseService, keyframe_size
from gma.multiverse.voice import MultiverseAgent, parse
from gma.multiverse.writer import Writer, distinct, similar
from voice_agent.video import Progress, Refused, Renderer

HAS_FFMPEG = shutil.which("ffmpeg") is not None or any(
    __import__("pathlib").Path(p).is_file() for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")

PLANS = [{"label": "he climbs to the lamp", "beat": "the keeper climbs", "prompt": "The keeper climbs the spiral stairs to the lamp."},
         {"label": "the storm breaks the window", "beat": "glass shatters", "prompt": "A gust breaks the window, rain pours into the room."},
         {"label": "a ship's light answers", "beat": "a ship signals back", "prompt": "Far out at sea a ship's light blinks back at the tower."}]


def small_tree() -> T.Tree:
    tree = T.Tree.new("a lighthouse keeper in a storm", {"label": "The opening", "prompt": "A lighthouse in a storm at night."})
    tree.set(tree.ghost, status="done", quality="full")
    return tree


# ---------------------------------------------------------------------------------------------------- the tree

def test_new_tree_has_the_ghost_on_the_root():
    tree = small_tree()
    assert tree.ghost == tree.data["root"] and tree.active_path() == [tree.ghost]
    assert tree.states() == {tree.ghost: "ghost"}
    with pytest.raises(T.TreeError):
        T.Tree.new("   ", {})


def test_options_are_lettered_and_ahead_of_the_ghost():
    tree = small_tree()
    a, b, c = tree.add_options(tree.ghost, PLANS)
    assert [o["letter"] for o in (a, b, c)] == ["A", "B", "C"] and {o["round"] for o in (a, b, c)} == {1}
    assert [o["id"] for o in tree.options()] == [a["id"], b["id"], c["id"]]
    assert all(tree.states()[o["id"]] == "option" for o in (a, b, c))
    assert a["depth"] == 1 and a["parent"] == tree.data["root"]


def test_active_path_is_highlighted_and_roads_not_taken_are_dimmed():
    tree = small_tree()
    root = tree.ghost
    a, b, c = tree.add_options(root, PLANS)
    tree.choose(b["id"])
    tree.set(b["id"], status="done", quality="full")
    states = tree.states()
    assert tree.ghost == b["id"] and tree.active_path() == [root, b["id"]]
    assert states[root] == "active" and states[b["id"]] == "ghost"
    assert states[a["id"]] == "dimmed" and states[c["id"]] == "dimmed"  # roads not taken stay, dimmed
    d, e = tree.add_options(b["id"], PLANS[:2])
    assert tree.states()[d["id"]] == "option" and tree.states()[e["id"]] == "option"
    # back to the root: B's branch is no longer on the active timeline, A and C are its options again
    tree.goto(root)
    states = tree.states()
    assert states[root] == "ghost" and tree.active_path() == [root]
    assert states[b["id"]] == "option" and states[d["id"]] == "dimmed" and states[a["id"]] == "option"
    snap = tree.snapshot()
    assert snap["active"] == [root] and snap["states"] == states and snap["ahead"] == [a["id"], b["id"], c["id"]]


def test_goto_needs_a_walked_node_and_choose_needs_an_option():
    tree = small_tree()
    a, b, _ = tree.add_options(tree.ghost, PLANS)
    with pytest.raises(T.TreeError):
        tree.goto(a["id"])  # never taken
    with pytest.raises(T.TreeError):
        tree.choose(tree.data["root"])
    with pytest.raises(T.TreeError):
        tree.get("nope")
    tree.choose(a["id"])
    tree.goto(tree.data["root"])
    tree.goto(a["id"])  # taken once: the Ghost can walk forward to it again
    assert tree.ghost == a["id"]


def test_a_fresh_round_dims_the_older_one_and_a_mix_joins_the_newest():
    tree = small_tree()
    old = tree.add_options(tree.ghost, PLANS)
    new = tree.add_options(tree.ghost, PLANS[:2])
    assert {o["round"] for o in new} == {2} and [o["letter"] for o in new] == ["A", "B"]
    assert [o["id"] for o in tree.options()] == [o["id"] for o in new]
    assert all(tree.states()[o["id"]] == "dimmed" for o in old)
    mixed = tree.add_options(tree.ghost, [PLANS[2]], mixed_from=[new[0]["id"], new[1]["id"]], same_round=True)
    assert mixed[0]["letter"] == "C" and mixed[0]["round"] == 2 and mixed[0]["mixed_from"] == [new[0]["id"], new[1]["id"]]
    assert len(tree.options(every_round=True)) == 6


def test_store_round_trip_and_list(tmp_path):
    store = T.Store(tmp_path)
    tree = small_tree()
    a, *_ = tree.add_options(tree.ghost, PLANS)
    tree.choose(a["id"])
    store.save(tree)
    assert (tmp_path / tree.id / "tree.json").is_file() and not list((tmp_path / tree.id).glob(".tree.*"))
    again = T.Store(tmp_path).load(tree.id)
    assert again.ghost == a["id"] and again.states() == tree.states()
    listed = T.Store(tmp_path).list()
    assert listed[0]["id"] == tree.id and listed[0]["nodes"] == 4 and listed[0]["depth"] == 1
    with pytest.raises(T.TreeError):
        store.load("../etc")
    with pytest.raises(T.TreeError):
        store.node_folder(tree.id, "../../x")


# ---------------------------------------------------------------------------------------------------- the voice

def test_voice_words():
    tree = small_tree()
    a, b, c = tree.add_options(tree.ghost, PLANS)
    assert parse("make a film about a lighthouse keeper in a storm", None) == ("new", {"idea": "a lighthouse keeper in a storm"})
    assert parse("let's test a new concept", tree) == ("new", {"idea": ""})
    assert parse("make the film darker", tree) is None  # a film exists: steering words are not a new film
    assert parse("take the left path", tree) == ("pick", {"node": a["id"]})
    assert parse("I'll go right", tree) == ("pick", {"node": c["id"]})
    assert parse("the middle one", tree) == ("pick", {"node": b["id"]})
    assert parse("option B", tree) == ("pick", {"node": b["id"]})
    assert parse("B.", tree) == ("pick", {"node": b["id"]})
    assert parse("take a", tree) == ("pick", {"node": a["id"]})
    assert parse("the one where the storm breaks the window", tree) == ("pick", {"node": b["id"]})
    assert parse("let's follow the ship", tree) == ("pick", {"node": c["id"]})
    assert parse("show me other options here", tree)[0] == "others"
    assert parse("mix A with C", tree) == ("mix", {"a": "A", "b": "C"})
    assert parse("export this path", tree) == ("export", {})
    assert parse("what are the options?", tree) == ("read", {})
    assert parse("what's the weather tomorrow", tree) is None
    tree.choose(b["id"])
    assert parse("go back", tree) == ("back", {"node": tree.data["root"]})
    assert parse("go back to the start", tree) == ("back", {"node": tree.data["root"]})


def test_agent_takes_words_only_while_the_page_is_open(tmp_path):
    svc = MultiverseService(T.Store(tmp_path), Renderer())
    agent = MultiverseAgent(svc)
    assert not agent.wants("make a film about a fox")
    svc.seen_at = time.monotonic()
    assert agent.wants("make a film about a fox") and not agent.wants("how are you today")


# ---------------------------------------------------------------------------------------------------- the Director

class FakeBrain:
    def __init__(self, replies):
        self.replies, self.asked = list(replies), []

    async def complete(self, messages, **settings):
        self.asked.append(messages[-1]["content"])
        return self.replies.pop(0) if self.replies else ""


def test_options_are_distinct_and_padded_from_the_fallback():
    dupes = {"options": [dict(PLANS[0]), dict(PLANS[0], label="he climbs up to the lamp"), dict(PLANS[1])]}
    brain = FakeBrain([json.dumps(dupes)])
    story = [{"label": "The opening", "beat": "a storm", "prompt": "A lighthouse in a storm."}]
    plans = asyncio.run(Writer(brain).options("a lighthouse keeper in a storm", story, 3))
    assert len(plans) == 3 and plans[0]["label"] == PLANS[0]["label"] and plans[1]["label"] == PLANS[1]["label"]
    assert all(similar(x, y) < 0.5 for i, x in enumerate(plans) for y in plans[i + 1:])
    assert "genuinely different" in brain.asked[0] and "A lighthouse in a storm." in brain.asked[0]


def test_fallback_without_a_brain_and_others_avoid_what_was_offered():
    w = Writer()
    story = [{"label": "o", "beat": "", "prompt": "p" * 30}]
    first = asyncio.run(w.options("a fox in the snow", story, 3))
    again = asyncio.run(w.options("a fox in the snow", story, 3, avoid=first, seed=9))
    assert len(first) == 3 and len(distinct(first)) == 3
    assert not {p["label"] for p in first} & {p["label"] for p in again}
    assert "the fox" in first[0]["prompt"] or "The fox" in first[0]["prompt"]
    mixed = asyncio.run(w.mix("a fox", story, dict(first[0]), dict(first[1])))
    assert first[0]["label"] in mixed["label"] and first[1]["label"] in mixed["label"]


# ---------------------------------------------------------------------------------------------------- the queue

class CountingRenderer(Renderer):
    """Makes no pictures: counts how many renders run at once, and says "no room" ``refuse`` times first."""

    name = "fake"

    def __init__(self, folder, refuse=0):
        self.folder, self.refuse, self.active, self.most, self.rooms, self.jobs = folder, refuse, 0, 0, 0, {}

    async def room(self, clip):
        self.rooms += 1
        if self.refuse:
            self.refuse -= 1
            return Refused("No room for the clip now: 3.1 GB short")
        return None

    async def _job(self, kind, ext):
        handle = f"{kind}{len(self.jobs)}"
        self.active += 1
        self.most = max(self.most, self.active)
        await asyncio.sleep(0.02)
        out = self.folder / f"{handle}.{ext}"
        out.write_bytes(b"\x89PNG fake" if ext == "png" else b"fake mp4")
        self.active -= 1
        self.jobs[handle] = str(out)
        return handle

    async def keyframe(self, prompt, seed, references=None, size=None):
        return await self._job("k", "png")

    async def submit(self, clip):
        return await self._job("v", "mp4")

    async def poll(self, handle):
        return Progress("done", output=self.jobs[handle])

    async def picture(self, ref):
        return open(ref, "rb").read()

    async def fetch(self, output, dest):
        shutil.copyfile(output, dest)


async def until(check, limit=20.0):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        if check():
            return True
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


def test_queue_renders_one_at_a_time_and_waits_for_room(tmp_path):
    async def go():
        r = CountingRenderer(tmp_path, refuse=2)
        svc = MultiverseService(T.Store(tmp_path / "films"), r, Writer(), {"room_retry_s": 0.01, "poll_s": 0.01})
        svc.frames = lambda clip, folder: (folder / "last.jpg").write_bytes(b"x") and {"last": "last.jpg"}  # no ffmpeg needed
        spoken = []
        svc.say = lambda text, mood="happy": spoken.append(text)
        tree = await svc.create("a lighthouse keeper in a storm")
        await until(lambda: len(tree.options()) == 3 and all(o["quality"] == "draft" for o in tree.options()))
        root = tree.get(tree.data["root"])
        assert root["quality"] == "full" and root["files"]["keyframe"] == "keyframe.png" and root["files"]["clip"] == "clip.mp4"
        assert r.most == 1 and r.rooms >= 3  # never two renders at once; two "no room" answers were waited out
        assert spoken and spoken[0].startswith("Here are the paths. A:")
        b = tree.options()[1]
        await svc.choose(tree.id, b["id"])
        await until(lambda: tree.get(b["id"])["quality"] == "full" and len(tree.options()) == 3)
        assert tree.ghost == b["id"] and r.most == 1
        assert tree.get(b["id"])["cost"]["full_spec"] == "864x480 · 5 s" and "draft_s" in tree.get(b["id"])["cost"]
        # "show me other options": a second round, the first stays (dimmed)
        await svc.request_options(tree.id, fresh=True)
        assert len(tree.options(every_round=True)) == 6 and {o["round"] for o in tree.options()} == {2}
        await until(lambda: all(o["quality"] == "draft" for o in tree.options()))
        # back to the root and branch again from there
        await svc.goto(tree.id, tree.data["root"])
        assert tree.states()[b["id"]] == "option"
        mixed = await svc.mix(tree.id, tree.options()[0]["id"], tree.options()[2]["id"])
        assert mixed["letter"] == "D" and mixed["mixed_from"]
        await until(lambda: tree.get(mixed["id"])["quality"] == "draft")
        assert r.most == 1
        svc.worker.cancel()
    asyncio.run(go())


def test_a_picked_branch_skips_its_siblings_queued_drafts(tmp_path):
    async def go():
        r = CountingRenderer(tmp_path)
        svc = MultiverseService(T.Store(tmp_path / "films"), r, Writer(), {"poll_s": 0.01, "auto_options": False})
        svc.frames = lambda clip, folder: (folder / "last.jpg").write_bytes(b"x") and {"last": "last.jpg"}
        tree = await svc.create("a fox in the snow")
        await until(lambda: tree.get(tree.ghost)["quality"] == "full")
        opts = tree.add_options(tree.ghost, PLANS)
        for o in opts:
            svc.enqueue(tree.id, o["id"], "draft")
        await svc.choose(tree.id, opts[2]["id"])
        await until(lambda: tree.get(opts[2]["id"])["quality"] == "full")
        await asyncio.sleep(0.1)
        made = [o for o in opts if tree.get(o["id"])["quality"] == "draft"]
        assert len(made) <= 1  # at most the draft already at work when the pick came; the rest were dropped
        svc.worker.cancel()
    asyncio.run(go())


def test_a_render_failure_is_the_nodes_and_the_queue_goes_on(tmp_path):
    class Failing(CountingRenderer):
        async def poll(self, handle):
            return Progress("failed", error="worker died") if handle.startswith("v") else await super().poll(handle)

    async def go():
        svc = MultiverseService(T.Store(tmp_path / "films"), Failing(tmp_path), Writer(), {"poll_s": 0.01})
        tree = await svc.create("a fox")
        await until(lambda: tree.get(tree.ghost)["status"] == "failed")
        assert "worker died" in tree.get(tree.ghost)["error"]
        assert (await svc.retry(tree.id, tree.ghost))["id"] == tree.ghost
    asyncio.run(go())


@needs_ffmpeg
def test_stub_renderer_end_to_end_with_continuity_and_export(tmp_path):
    from gma.multiverse.stub import StubRenderer

    async def go():
        svc = MultiverseService(T.Store(tmp_path / "films"), StubRenderer(tmp_path / "stub", delay_s=0.05), Writer(),
                                {"poll_s": 0.02, "draft": {"seconds": 1.5, "width": 320, "height": 176}, "full": {"seconds": 2, "width": 480, "height": 272}})
        tree = await svc.create("a lighthouse keeper in a storm")
        await until(lambda: len(tree.options()) == 3 and all(o["quality"] == "draft" for o in tree.options()), 120)
        a = tree.options()[0]
        folder = svc.store.node_folder(tree.id, a["id"])
        assert {"keyframe", "draft", "poster", "first", "last"} <= set(a["files"]) and (folder / "draft.mp4").stat().st_size > 1000
        assert a["qc"]["seam_draft"]["corr"] > 0.5  # the draft opens on the opening's last frame
        await svc.choose(tree.id, a["id"])
        await until(lambda: tree.get(a["id"])["quality"] == "full", 120)
        film = await svc.export(tree.id)
        assert film.is_file() and film.stat().st_size > 1000 and tree.data["film"]["path"] == tree.active_path()
        assert svc.probe(film)[0]  # the film has a sound track (silence here: the stub's clips have none)
        svc.worker.cancel()
    asyncio.run(go())


# ---------------------------------------------------------------------------------------------------- the app

@needs_ffmpeg
def test_the_app_serves_the_multiverse(monkeypatch, tmp_path):
    from voice_agent import server

    monkeypatch.setattr(server, "Recognizer", lambda spec: (lambda audio: ""))
    profile = {
        "name": "test", "platform": "mac", "asr": {"kind": "parakeet-mlx", "model": "x"},
        "brain": {"url": "http://127.0.0.1:9/v1", "model": "stub", "system": "You are the Ghost."},
        "voice": {"url": "http://127.0.0.1:9/v1", "model": "kokoro-82m", "keep_warm": False},
        "engines": {}, "supervisor": {"home": str(tmp_path / "home"), "monitor_s": 60},
        "multiverse": {"dir": str(tmp_path / "mv"), "renderer": "stub", "writer": "plain", "stub_delay_s": 0.05, "poll_s": 0.05,
                       "draft": {"seconds": 1, "width": 320, "height": 176}, "full": {"seconds": 1.5, "width": 320, "height": 176}},
    }
    with TestClient(server.build(profile)) as client:
        page = client.get("/")
        assert page.status_code == 200 and "keys Ghost Multiverse Studio" in page.text
        assert "Ghost Machine" in client.get("/classic").text or client.get("/classic").status_code == 200
        assert client.get("/multiverse/tree.js").status_code == 200
        assert client.get("/multiverse/../server.py").status_code == 404
        assert next(iter(client.app.state.multiverse.store.root.parent.iterdir()))  # the store's folder exists
        made = client.post("/api/multiverse/projects", json={"idea": "a lighthouse keeper in a storm"}).json()
        pid, root = made["id"], made["root"]
        assert made["states"] == {root: "ghost"}
        assert client.post("/api/multiverse/projects", json={"idea": ""}).status_code == 400
        view = {}
        for _ in range(600):
            view = client.get(f"/api/multiverse/projects/{pid}").json()
            if len(view["ahead"]) == 3 and all(view["nodes"][n]["quality"] == "draft" for n in view["ahead"]):
                break
            time.sleep(0.05)
        a, b, c = view["ahead"]
        assert client.get(f"/api/multiverse/projects/{pid}/files/{a}/draft.mp4").headers["content-type"] == "video/mp4"
        assert client.get(f"/api/multiverse/projects/{pid}/files/{a}/secret.txt").status_code == 404
        chose = client.post(f"/api/multiverse/projects/{pid}/choose", json={"node": b}).json()
        assert chose["tree"]["ghost"] == b and chose["tree"]["states"][a] == "dimmed" and chose["tree"]["states"][root] == "active"
        assert client.post(f"/api/multiverse/projects/{pid}/goto", json={"node": c}).status_code == 409  # never taken
        assert client.post(f"/api/multiverse/projects/{pid}/choose", json={}).status_code == 400
        for _ in range(600):
            if client.get(f"/api/multiverse/projects/{pid}").json()["nodes"][b]["quality"] == "full":
                break
            time.sleep(0.05)
        assert client.post(f"/api/multiverse/projects/{pid}/export").json()["ok"]
        assert client.get(f"/api/multiverse/projects/{pid}/film.mp4").status_code == 200
        assert client.get("/api/multiverse/projects").json()["projects"][0]["id"] == pid
        names = [c["name"] for c in client.get("/api/credits").json()]
        assert "keys Ghost Multiverse Studio" in names and "Loom" in names


def test_a_drafts_keyframe_beside_the_last_frame_is_scaled_into_the_edits_pixels():
    # 11-08 on the DGX Spark: every 640x352 draft keyframe failed ("an edit is made at 512x512 to 1536x1536 pixels in all")
    for w, h in [(640, 352), (512, 288), (864, 480), (2048, 2048), (1920, 1080)]:
        kw, kh = keyframe_size(w, h, edit=True)
        assert 512 * 512 <= kw * kh <= 1536 * 1536 and kw % 16 == 0 and kh % 16 == 0
        assert abs(kw / kh - w / h) < 0.03
    assert keyframe_size(640, 352, edit=False) == (640, 352)
    assert keyframe_size(864, 480, edit=True) == (864, 480)


def test_draft_and_full_clips_are_in_h3s_5_to_15_seconds():
    # 11-08 on the DGX Spark: a 3 s draft was refused ("MiniMax-H3 generates 5-15 seconds at 24 fps")
    from gma.multiverse.service import DEFAULTS
    for kind in ("draft", "full"):
        assert 5 <= DEFAULTS[kind]["seconds"] <= 15


def test_renders_cut_off_by_a_restart_go_back_on_the_queue(tmp_path):
    # 11-08 on the DGX Spark: after an app restart B stayed "drafting" and C "planned" with nothing at work on them
    store = T.Store(tmp_path)
    tree = small_tree()
    a, b, c = tree.add_options(tree.ghost, PLANS)
    tree.set(a["id"], status="draft", quality="draft")
    tree.set(b["id"], status="drafting")
    store.save(tree)

    async def go():
        svc = MultiverseService(store, Renderer(), Writer(), {"auto_options": False})
        svc.worker = asyncio.get_running_loop().create_future()  # no worker: only the queue is looked at
        assert svc.resume() == 2 and svc.resume() == 0
        queued = {k[1]: k[2] for k in svc.queued}
        assert queued == {b["id"]: "draft", c["id"]: "draft"}
        assert store.load(tree.id).get(b["id"])["status"] == "planned"
        svc.worker.cancel()

    asyncio.run(go())


def test_going_back_drafts_the_options_dropped_when_a_sibling_was_taken(tmp_path):
    # 11-08 on the DGX Spark: A was taken before B and C were drafted; back at that step, B and C stayed "planned" forever
    store = T.Store(tmp_path)
    tree = small_tree()
    a, b, c = tree.add_options(tree.ghost, PLANS)
    tree.set(a["id"], status="done", quality="full")
    tree.choose(a["id"])
    store.save(tree)

    async def go():
        svc = MultiverseService(store, Renderer(), Writer(), {"auto_options": False})
        svc.resumed = True
        svc.worker = asyncio.get_running_loop().create_future()  # no worker: only the queue is looked at
        await svc.goto(tree.id, tree.data["root"])
        assert {k[1]: k[2] for k in svc.queued} == {b["id"]: "draft", c["id"]: "draft"}
        svc.queued.clear()
        await svc.retry(tree.id, c["id"])  # the page's "Draft it" on one never drafted
        assert {k[1]: k[2] for k in svc.queued} == {c["id"]: "draft"}
        svc.worker.cancel()

    asyncio.run(go())


def test_a_restart_does_not_draft_options_dropped_elsewhere_in_the_tree(tmp_path):
    store = T.Store(tmp_path)
    tree = small_tree()
    a, b, c = tree.add_options(tree.ghost, PLANS)
    tree.set(a["id"], status="done", quality="full")
    tree.choose(a["id"])  # b and c: dropped, never drafted
    x, y, z = tree.add_options(a["id"], PLANS)  # ahead of the Ghost, not drafted yet
    store.save(tree)

    async def go():
        svc = MultiverseService(store, Renderer(), Writer(), {"auto_options": False})
        svc.worker = asyncio.get_running_loop().create_future()
        assert svc.resume() == 3
        assert {k[1] for k in svc.queued} == {x["id"], y["id"], z["id"]}
        svc.worker.cancel()

    asyncio.run(go())


@needs_ffmpeg
def test_export_keeps_each_clips_sound(tmp_path):
    # 11-08 on the DGX Spark: the exported film had no sound although every H3 clip has its own (concat ... a=0)
    from voice_agent.video import find_ffmpeg
    ff = find_ffmpeg()
    store = T.Store(tmp_path)
    tree = small_tree()
    a, *_ = tree.add_options(tree.ghost, PLANS)
    tree.set(a["id"], status="done", quality="full")
    tree.choose(a["id"])
    for nid, sound in ((tree.data["root"], True), (a["id"], False)):
        folder = store.node_folder(tree.id, nid)
        folder.mkdir(parents=True, exist_ok=True)
        src = ["-f", "lavfi", "-i", "testsrc=s=320x176:r=24:d=2"] + (["-f", "lavfi", "-i", "sine=f=440:d=2"] if sound else [])
        subprocess.run([ff, "-v", "error", "-y", *src, "-shortest", "-pix_fmt", "yuv420p", str(folder / "clip.mp4")], check=True)
        tree.set(nid, files={"clip": "clip.mp4"})
    store.save(tree)
    svc = MultiverseService(store, Renderer(), Writer())
    film = asyncio.run(svc.export(tree.id))
    has_audio, seconds = svc.probe(film)
    assert has_audio and 3.8 < seconds < 4.4
    level = subprocess.run([ff, "-v", "info", "-t", "1.5", "-i", str(film), "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    assert "mean_volume: -inf" not in level and "mean_volume" in level  # the first clip's tone is heard
