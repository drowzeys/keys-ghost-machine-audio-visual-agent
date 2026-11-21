"""Invisible mode (gma/watch): the ledger and its search, the sources it watches by itself, the open API for agents,
the describer's turn-taking, the voice's words, and the app end to end with the stub renderer. No model is loaded."""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gma.watch import sources
from gma.watch.describer import Describer, fallback
from gma.watch.ledger import Ledger, id_of, query_of
from gma.watch.service import WatchError, WatchService
from gma.watch.voice import parse

FFMPEG = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).is_file()), None)
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")


def clip(path: Path, seconds: float = 1.0, colour: str = "blue") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if FFMPEG:
        subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={colour}:s=160x96:d={seconds}", "-pix_fmt", "yuv420p",
                        str(path)], check=True, capture_output=True)
    else:
        path.write_bytes(b"not a real clip")
    return path


# ---------------------------------------------------------------------------------------------------- the ledger

def test_a_creation_keeps_its_id_and_description_when_seen_again(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    c, changed = led.put("mv:p:n1", kind="video", agent="Multiverse Director", source="multiverse", at=10, title="He climbs",
                         prompt="a keeper climbs the stairs", file="/x/clip.mp4")
    assert changed and c["id"] == id_of("mv:p:n1") and c["status"] == "new"
    led.describe(c["id"], "A keeper climbs a spiral stair", "Rain on the glass.", ["stairs", "storm"])
    again, changed = led.put("mv:p:n1", kind="video", agent="Multiverse Director", source="multiverse", at=10, title="He climbs",
                             prompt="a keeper climbs the stairs", file="/x/clip.mp4")
    assert not changed and again["status"] == "described" and again["summary"] == "A keeper climbs a spiral stair"
    redone, changed = led.put("mv:p:n1", kind="video", agent="Multiverse Director", source="multiverse", at=10, title="He climbs",
                              prompt="a keeper climbs the stairs", file="/x/clip-full.mp4")
    assert changed and redone["status"] == "new"  # a new file: the Ghost looks again


def test_search_finds_words_in_the_description_and_by_prefix(tmp_path):
    led = Ledger(tmp_path / "l.sqlite")
    a, _ = led.put("a", kind="image", agent="hermes", source="api", at=1, title="still one")
    b, _ = led.put("b", kind="video", agent="autopilot", source="autopilot", at=2, title="clip two", parents=["a"])
    led.describe(a["id"], "A lighthouse in a storm", "Waves break on black rocks.", ["lighthouse"])
    led.describe(b["id"], "A fox on the ice", "Dawn over a frozen river.", ["fox"])
    assert [c["id"] for c in led.search("light")] == [a["id"]]
    assert [c["id"] for c in led.search("frozen river")] == [b["id"]]
    assert led.search("fox", kind="image") == []
    assert [c["id"] for c in led.search("", agent="hermes")] == [a["id"]]
    assert [c["id"] for c in led.children(a["id"])] == [b["id"]]  # a parent named by its ref
    assert query_of('he said "x" OR y*') == '"he"* AND "said"* AND "x"* AND "or"* AND "y"*'  # words only, no FTS syntax
    assert led.stats()["total"] == 2 and led.stats()["agents"] == {"hermes": 1, "autopilot": 1}


# ---------------------------------------------------------------------------------------------------- the sources

def test_multiverse_films_become_an_idea_with_its_branching_clips(tmp_path):
    root = tmp_path / "mv"
    folder = root / "p1"
    for nid in ("n0", "nA", "nB"):
        clip(folder / "nodes" / nid / ("clip.mp4" if nid != "nB" else "draft.mp4"))
    tree = {"id": "p1", "title": "a keeper", "idea": "a lighthouse keeper in a storm", "created": 100.0, "root": "n0",
            "nodes": {"n0": {"id": "n0", "parent": None, "label": "rain on the rocks", "prompt": "rain", "files": {"clip": "clip.mp4"}},
                      "nA": {"id": "nA", "parent": "n0", "label": "he climbs", "prompt": "climb", "letter": "A", "taken": True,
                             "files": {"clip": "clip.mp4"}},
                      "nB": {"id": "nB", "parent": "n0", "label": "he falls", "prompt": "fall", "letter": "B",
                             "files": {"draft": "draft.mp4"}},
                      "nC": {"id": "nC", "parent": "n0", "label": "not drawn yet", "prompt": "x", "files": {}}}}
    (folder / "tree.json").write_text(json.dumps(tree))
    found = {r["ref"]: r for r in sources.multiverse(root)}
    assert set(found) == {"mv:p1", "mv:p1:n0", "mv:p1:nA", "mv:p1:nB"}  # nothing for a node with no file yet
    assert found["mv:p1"]["kind"] == "text" and found["mv:p1:n0"]["parents"] == ["mv:p1"]
    assert found["mv:p1:nA"]["parents"] == found["mv:p1:nB"]["parents"] == ["mv:p1:n0"]  # the fork
    assert found["mv:p1:nA"]["title"] == "He climbs" and found["mv:p1:nA"]["meta"]["open"] == "/#p1"


def test_an_autopilot_retake_is_a_fork_and_the_film_joins_the_kept_takes(tmp_path):
    root = tmp_path / "video"
    run = root / "r1"
    for rel in ("segments/S0001/take1.mp4", "segments/S0002/take1.mp4", "segments/S0002/take2.mp4", "film.mp4", "film-final.mp4"):
        clip(run / rel)
    (run / "narration.wav").write_bytes(b"RIFF")
    seg = lambda i, takes, out: {"index": i, "id": f"S{i:04d}", "beat": f"beat {i}", "prompt": f"prompt {i}", "output": out,
                                 "takes": [{"take": t, "output": f"r1/segments/S{i:04d}/take{t}.mp4", "passed": t == takes[-1]} for t in takes]}
    data = {"id": "r1", "idea": "a keeper lights the lamp", "mode": "autopilot", "created": "2026-11-07T18:26:30", "clips": [],
            "autopilot": {"name": "The Keeper", "segments": [seg(1, [1], "r1/segments/S0001/take1.mp4"),
                                                             seg(2, [1, 2], "r1/segments/S0002/take2.mp4")],
                          "film": {"path": "r1/film.mp4", "segments": 2},
                          "film_final": {"path": "r1/film-final.mp4", "narration": {"file": "narration.wav", "text": "He climbed."}}}}
    (run / "run.json").write_text(json.dumps(data))
    found = {r["ref"]: r for r in sources.runs(root)}
    assert found["run:r1:S0002:t1"]["parents"] == found["run:r1:S0002:t2"]["parents"] == ["run:r1:S0001:t1"]
    assert found["run:r1:S0002:t2"]["meta"]["kept"] and not found["run:r1:S0002:t1"]["meta"]["kept"]
    assert found["run:r1:film"]["parents"] == ["run:r1:S0002:t2"]  # the film follows the kept take
    assert found["run:r1:narration"]["kind"] == "audio"
    assert found["run:r1:final"]["parents"] == ["run:r1:film", "run:r1:narration"]  # a merge


# ---------------------------------------------------------------------------------------------------- the open API

def service(tmp_path, describer=None, busy=None) -> WatchService:
    return WatchService(Ledger(tmp_path / "w" / "ledger.sqlite"), {"allow": [str(tmp_path / "ok")]}, describer, busy, FFMPEG)


def test_agents_add_texts_files_and_uploads_but_not_files_outside_the_allowed_folders(tmp_path):
    svc = service(tmp_path)
    idea = svc.add({"agent": "hermes", "text": "a fox crossing a frozen river", "title": "fox idea"})
    assert idea["kind"] == "text" and idea["media"] is None
    (tmp_path / "ok").mkdir()
    still = tmp_path / "ok" / "still.png"
    still.write_bytes(b"\x89PNG fake")
    made = svc.add({"agent": "hermes", "path": str(still), "parents": [idea["id"]], "title": "the fox"})
    assert made["kind"] == "image" and made["parents"] == [idea["id"]] and made["media"].endswith("/file")
    secret = tmp_path / "elsewhere.png"
    secret.write_bytes(b"x")
    with pytest.raises(WatchError, match="outside"):
        svc.add({"agent": "hermes", "path": str(secret)})
    with pytest.raises(WatchError, match="agent"):
        svc.add({"text": "no agent"})
    with pytest.raises(WatchError, match="needs its file"):
        svc.add({"agent": "x", "kind": "video"})
    up = svc.upload("../../etc/clip.mp4", b"data", {"agent": "claude", "parents": [made["id"]]})
    assert up["kind"] == "video" and svc.media(up["id"]).parent.parent == svc.files  # kept in the Ghost's own folder
    assert svc.detail(made["id"])["child_list"][0]["id"] == up["id"]
    own = svc.add({"agent": "hermes", "text": "x", "summary": "my own words", "description": "described by the agent"})
    assert own["status"] == "described" and own["summary"] == "my own words"


def test_the_ghost_describes_only_when_the_machine_is_free_and_newest_first(tmp_path):
    class Fake:
        def __init__(self):
            self.seen = []

        async def describe(self, c, parents):
            self.seen.append((c["title"], [p["title"] for p in parents]))
            return {"summary": f"about {c['title']}", "description": "d", "tags": ["t"]}, "fake"

    busy = {"now": True}
    fake = Fake()
    svc = service(tmp_path, fake, lambda: busy["now"])
    a = svc.add({"agent": "a", "text": "first", "title": "first", "at": 1})
    svc.add({"agent": "a", "text": "second", "title": "second", "at": 2, "parents": [a["id"]]})
    assert asyncio.run(svc.describe_one()) is False and fake.seen == []  # a render is running: wait
    busy["now"] = False
    assert asyncio.run(svc.describe_one()) is True and fake.seen == [("second", ["first"])]
    asyncio.run(svc.describe_one())
    assert svc.ledger.get(a["id"])["summary"] == "about first"
    assert svc.ledger.get(a["id"])["meta"]["described_by"] == "fake"
    assert asyncio.run(svc.describe_one()) is False  # nothing left


def test_with_no_models_the_creations_own_words_describe_it(tmp_path):
    words = asyncio.run(Describer().describe({"kind": "text", "title": "", "prompt": "A fox crosses the ice. Dawn breaks.",
                                              "agent": "x", "file": ""}, []))
    assert words[0]["summary"] == "A fox crosses the ice." and words[1] == "its own words"
    assert "fox" in fallback({"kind": "video", "title": "the fox", "prompt": ""})["tags"]


@needs_ffmpeg
def test_a_clip_gets_a_thumbnail(tmp_path):
    svc = service(tmp_path)
    made = svc.add({"agent": "x", "path": str(clip(tmp_path / "ok" / "c.mp4", 1.5))})
    thumb = svc.thumb(made["id"])
    assert thumb and thumb.suffix == ".jpg" and thumb.stat().st_size > 100


def test_the_voice_finds_whats_new_and_one_agents_work():
    assert parse("find the lighthouse clips") == ("find", {"q": "lighthouse", "kind": "video"})
    assert parse("show me the pictures of the fox") == ("find", {"q": "fox", "kind": "image"})
    assert parse("what's new") == ("new", {})
    assert parse("what did hermes make") == ("agent", {"agent": "hermes"})
    assert parse("how are you today") is None


# ---------------------------------------------------------------------------------------------------- the app

@needs_ffmpeg
def test_the_app_watches_a_multiverse_film_and_an_agents_creation(monkeypatch, tmp_path):
    from voice_agent import server

    monkeypatch.setattr(server, "Recognizer", lambda spec: (lambda audio: ""))
    profile = {
        "name": "test", "platform": "mac", "asr": {"kind": "parakeet-mlx", "model": "x"},
        "brain": {"url": "http://127.0.0.1:9/v1", "model": "stub", "system": "You are the Ghost."},
        "voice": {"url": "http://127.0.0.1:9/v1", "model": "kokoro-82m", "keep_warm": False},
        "engines": {}, "supervisor": {"home": str(tmp_path / "home"), "monitor_s": 60},
        "multiverse": {"dir": str(tmp_path / "mv"), "renderer": "stub", "writer": "plain", "stub_delay_s": 0.05, "poll_s": 0.05,
                       "draft": {"seconds": 1, "width": 320, "height": 176}, "full": {"seconds": 1.5, "width": 320, "height": 176}},
        "watch": {"dir": str(tmp_path / "watch"), "runs": str(tmp_path / "video"), "scan_s": 0.3, "describe": True},
    }
    with TestClient(server.build(profile)) as client:
        assert "Invisible" in client.get("/invisible").text
        assert 'href="/invisible"' in client.get("/").text  # the mode switch on the Multiverse page
        assert client.get("/invisible/timeline.js").status_code == 200
        assert client.get("/invisible/../server.py").status_code == 404
        made = client.post("/api/multiverse/projects", json={"idea": "a lighthouse keeper in a storm"}).json()
        pid = made["id"]
        idea = client.post("/api/watch/creations", json={"agent": "hermes", "title": "fox idea", "text": "a fox on the ice"}).json()
        assert idea["agent"] == "hermes"
        assert client.post("/api/watch/creations", json={"agent": "x", "path": "/etc/passwd"}).status_code == 400
        up = client.post(f"/api/watch/upload?agent=claude&name=still.png&parents={idea['id']}", content=b"\x89PNG").json()
        assert up["parents"] == [idea["id"]]
        deadline = time.time() + 60
        while time.time() < deadline:
            line = client.get("/api/watch/timeline").json()
            refs = {c["ref"] for c in line["creations"]}
            described = [c for c in line["creations"] if c["source"] == "multiverse" and c["status"] == "described"]
            if f"mv:{pid}" in refs and any(r.startswith(f"mv:{pid}:") for r in refs) and described:
                break
            time.sleep(0.3)
        else:
            raise AssertionError(f"the film never reached the timeline: {refs}")
        found = client.get("/api/watch/creations", params={"q": "lighthouse"}).json()["creations"]
        assert any(c["ref"] == f"mv:{pid}" for c in found)
        opening = next(c for c in line["creations"] if c["ref"].startswith(f"mv:{pid}:"))
        assert client.get(f"/api/watch/creations/{opening['id']}/file").status_code == 200
        assert client.get(f"/api/watch/creations/{opening['id']}/thumb").headers["content-type"] == "image/jpeg"
        assert client.get(f"/api/watch/creations/{opening['id']}").json()["parent_list"][0]["ref"] == f"mv:{pid}"
        assert client.get("/api/watch/status").json()["total"] >= 4
