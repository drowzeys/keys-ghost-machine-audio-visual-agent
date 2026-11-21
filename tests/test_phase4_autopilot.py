"""Phase 4's AutoPilot fixes (gma/ap/story.py and its use in gma/ap/autopilot.py): the arc planned up front with an
end state a piece and the payoff, the place's first keyframe as the reference picture of its later ones (and a check
against that picture), the person's spoken steers, "redo that" and "keep going". No GPU, no network, no model: the
Director and the eyes are scripted, the queue makes tiny clips with ffmpeg."""

import json
import re
import shutil
import threading
import time

import pytest

from gma.ap import autopilot, story
from test_autopilot import BEATS, HOME, SEED, Queue, Watcher, client, shot  # noqa: F401 - HOME: the run folder's root

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

ENDS = ["Vex is airborne over the tram gap", "Vex is between two hover-trams", "Vex lies flat under the fallen sign",
        "Vex stands at the rooftop relay", "the drone carries the core away", "the city lights blaze on"]


class ArcDirector:
    """The Director stood in: the arc when asked for it, a shot a piece (``scene_change`` at the pieces in ``cuts``)."""

    def __init__(self, cuts=()):
        self.asked, self.systems, self.cuts = [], [], set(cuts)

    def __call__(self, run, system, prompt):
        self.asked.append(prompt)
        self.systems.append(system)
        if system == story.ARC_SYSTEM:
            start = int(re.search(r"Plan segments (\d+) to (\d+)", prompt)[1])
            end = int(re.search(r"Plan segments (\d+) to (\d+)", prompt)[2])
            return json.dumps({"payoff": "the city lights blaze on",
                               "beats": [{"segment": i, "beat": BEATS[(i - 1) % 6], "end_state": ENDS[(i - 1) % 6]} for i in range(start, end + 1)]})
        if "Fold this stretch" in prompt:
            return json.dumps({"title": "The run", "summary": "Vex crossed the city.", "carried_forward": []})
        if '"keyframe": "<60 words max>"' in prompt:
            return json.dumps({"keyframe": "Vex again, wide shot"})
        n = len(autopilot.committed(run))
        index = n + 1
        return json.dumps(shot(BEATS[n % 6] + (" again" if "THE TAKE YOU WROTE" in prompt else ""), change=index == 1 or index in self.cuts,
                               scene="Elsewhere" if index in self.cuts else "Flooded avenue"))


def yes_to_all(no=()):
    """The dense checks' answers: "yes" to every question, except "no" to the keys in ``no`` (a set of (call, key))."""

    calls = [0]

    def answer(prompt, frames):
        calls[0] += 1
        keys = re.findall(r'^"(\w+)":', prompt, re.M)
        out = {}
        for k in keys:
            expect_no = any(w in prompt.split(f'"{k}":', 1)[1].split("\n", 1)[0] for w in ("second person", "change into", "malformed", "out of place", "slow motion"))
            out[k] = "no" if expect_no else "yes"
        for key, when in no:
            if key in out and when(prompt):
                out[key] = "no"
        return json.dumps(out)
    return answer


@pytest.fixture
def loop4(monkeypatch):
    queue = Queue()
    monkeypatch.setattr(autopilot, "submit", queue.submit)
    monkeypatch.setattr(autopilot, "job_of", queue.lookup)
    monkeypatch.setattr(autopilot, "cancel", lambda job_id: None)
    monkeypatch.setattr(autopilot, "room", lambda params: None)
    monkeypatch.setattr(autopilot, "gate", lambda clip, prompt, sheet=None, brain=None: {"passed": True, "audio": {"result": "pass", "heard": ""}, "identity": {"result": "pass"}})
    monkeypatch.setattr(autopilot, "ears_ready", lambda: {"runs": True, "transcript": True})
    director, watcher = ArcDirector(), Watcher(yesno=yes_to_all())
    monkeypatch.setattr(autopilot, "ask_text", director)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    return {"queue": queue, "director": director, "watcher": watcher, "monkeypatch": monkeypatch}


def make(segments=3, **settings):
    said = client.post("/api/autopilot/runs", json={"name": "Storm Courier", "seed": SEED,
                                                    "settings": {"segments": segments, "preview_every": 1, "continuity": {"film_pass": False}, **settings}})
    assert said.status_code == 200, said.text
    return autopilot.load(said.json()["id"])


def go(run):
    autopilot.work(run, threading.Event())
    return autopilot.load(run["id"])


def piece_prompts(director):
    return [p for p, sysm in zip(director.asked, director.systems) if sysm == autopilot.DIRECTOR_SYSTEM and "THE TAKE YOU WROTE" not in p]


# ---------- the arc ----------

def test_arc_answers_are_checked_and_merged():
    assert "list of 3" in story.arc_problem({"beats": [{"beat": "a", "end_state": "x"}]}, 3)
    assert "same state" in story.arc_problem({"payoff": "p", "beats": [{"beat": "a", "end_state": "x"}, {"beat": "b", "end_state": "X"}]}, 2)
    assert story.arc_problem({"payoff": "p", "beats": [{"beat": "a", "end_state": "x"}, {"beat": "b", "end_state": "y"}]}, 2) == ""
    arc = story.merge_arc(None, {"payoff": "p", "beats": [{"beat": "a", "end_state": "x"}, {"beat": "b", "end_state": "y"}]}, 1, 2)
    arc = story.merge_arc(arc, {"payoff": "q", "beats": [{"beat": "c", "end_state": "z"}]}, 3, 1)
    assert [b["index"] for b in arc["beats"]] == [1, 2, 3] and arc["payoff"] == "q" and arc["through"] == 3


def test_the_arc_is_planned_before_the_first_piece_and_each_piece_must_end_on_its_state(loop4):
    run = go(make(3, keyframes="off"))
    director = loop4["director"]
    assert run["state"] == "done" and director.systems[0] == story.ARC_SYSTEM
    assert "Plan segments 1 to 3" in director.asked[0] and "payoff" in director.asked[0]
    asked = piece_prompts(director)
    for i, prompt in enumerate(asked[:3], 1):
        assert f"This is segment {i}: it must COMPLETE beat {i}" in prompt and f"END on: {ENDS[i - 1]}" in prompt
    assert "This is the LAST segment: it must reach the payoff on screen: the city lights blaze on" in asked[2]
    assert "LAST segment" not in asked[1]
    # the last frames were asked whether each piece reached its end state
    story_qs = [p for p, _ in loop4["watcher"].yesno if "By the end of these frames" in p]
    assert len(story_qs) == 3 and ENDS[2] in story_qs[2]
    assert all(s["continuity"]["dense"]["arc"]["reached"] for s in run["segments"])
    assert (autopilot.run_dir(run["id"]) / "arc-0001.json").is_file()


def test_a_missed_end_state_is_a_note_and_the_next_piece_catches_up(loop4):
    no_end_of_first = {("s1", lambda p: ENDS[0] in p)}
    loop4["watcher"].answer_yesno = yes_to_all(no_end_of_first)
    run = go(make(3, keyframes="off"))
    asked = piece_prompts(loop4["director"])
    assert run["state"] == "done" and len(run["segments"][0]["takes"]) == 1  # not a retake: a note
    assert "CATCH UP" in asked[1] and f"Segment 1 did not reach its end state ({ENDS[0]})" in asked[1]
    assert "CATCH UP" not in asked[2]  # piece 2 reached its own: caught up
    found = run["segments"][0]["continuity"]["findings"]
    assert any(f["category"] == "story" and f["status"] == "minor" for f in found)


def test_the_last_piece_that_misses_the_payoff_is_made_again(loop4):
    seen = [0]

    def payoff_once(prompt):
        if ENDS[2] in prompt:
            seen[0] += 1
            return seen[0] <= 3  # the first take's question and its votes say no; the retake shows it
        return False
    loop4["watcher"].answer_yesno = yes_to_all({("s1", payoff_once)})
    run = go(make(3, keyframes="off"))
    last = run["segments"][2]
    assert run["state"] == "done" and len(last["takes"]) == 2 and last["passed"]
    assert any("payoff" in r for r in last["takes"][0]["reasons"])


def test_an_infinity_film_plans_its_arc_a_span_at_a_time(loop4):
    run = make(0, keyframes="off", infinity=True, max_film_seconds=41)
    run = go(run)
    arcs = [p for p, s in zip(loop4["director"].asked, loop4["director"].systems) if s == story.ARC_SYSTEM]
    assert "Plan segments 1 to 4" in arcs[0] and "Plan segments 5 to 8" in arcs[1]
    assert len(autopilot.committed(run)) >= 5


# ---------- the place ----------

def test_the_first_keyframe_anchors_the_place_and_later_pieces_continue_its_last_frame(loop4):
    run = go(make(3, keyframes="on"))
    bodies = [b for b in loop4["queue"].bodies if b["kind"] in ("scout", "edit")]
    assert run["state"] == "done" and [b["kind"] for b in bodies] == ["scout"]  # only the opening is drawn
    first = run["segments"][0]["keyframe"]
    assert run["locations"][0]["from"] == 1 and run["locations"][0]["image"] == first
    assert [s["mode"] for s in run["segments"]] == ["keyframe", "previous", "previous"]
    assert "THE PLACE IS FIXED" in piece_prompts(loop4["director"])[1]


def test_a_scene_change_opens_a_new_place_with_its_own_picture(loop4):
    loop4["director"].cuts = {3}
    run = go(make(3, keyframes="on"))
    bodies = [b for b in loop4["queue"].bodies if b["kind"] in ("scout", "edit")]
    assert [b["kind"] for b in bodies] == ["scout", "scout"]  # the opening and the new scene; piece 2 continues
    assert [p["from"] for p in run["locations"]] == [1, 3]
    assert story.anchor_of(run, 4)["from"] == 3


def test_a_continuing_piece_draws_no_keyframe_and_a_checklist_finding_never_redraws(loop4):
    count = [0]

    def once(prompt):
        if "keyframe" in prompt.lower():
            count[0] += 1
            return count[0] <= 3
        return False
    loop4["watcher"].answer_yesno = yes_to_all({("p1", once)})
    run = go(make(2, keyframes="on"))
    first, second = run["segments"]
    assert run["state"] == "done" and len(first["keyframes"]) == 1 and not second.get("keyframes")
    assert second["mode"] == "previous"


def test_the_anchor_rules():
    run = {"settings": {}, "segments": [{"index": 1, "scene_change": True}, {"index": 2, "scene_change": False},
                                        {"index": 3, "scene_change": True}]}
    assert story.set_anchor(run, run["segments"][0], "a.png")["from"] == 1
    assert story.set_anchor(run, run["segments"][1], "b.png") is None  # a continuing piece keeps the place's picture
    assert story.anchor_of(run, 2)["image"] == "a.png" and story.anchor_of(run, 3) is None  # a new scene: none
    assert story.set_anchor(run, run["segments"][0], "a2.png")["image"] == "a2.png"  # the opener drawn again
    story.set_anchor(run, run["segments"][2], "c.png")
    assert story.anchor_of(run, 4)["image"] == "c.png"


# ---------- steers, redo, keep going ----------

def test_a_spoken_steer_reaches_the_next_piece_the_bible_and_its_check(loop4):
    run = make(3, keyframes="off")
    real = loop4["queue"].submit

    def submit(body):
        job = real(body)
        if body["kind"] == "video" and body["params"]["autopilot"]["segment"] == 1:
            autopilot.steer(run["id"], "make it rain")  # said while piece 1 renders
        return job
    loop4["monkeypatch"].setattr(autopilot, "submit", submit)
    autopilot.runs[run["id"]] = {"thread": threading.current_thread(), "stop": threading.Event(), "run": run}
    try:
        run = go(run)
    finally:
        autopilot.runs.pop(run["id"], None)
    asked = piece_prompts(loop4["director"])
    assert "make it rain" not in asked[0] and 'NEW: the person asked: "make it rain"' in asked[1] and "Keep it." in asked[2]
    assert run["steers"][0]["from"] == 2 and run["steers"][0]["applied_at"] == 2
    bible = autopilot.bible_of(run)
    assert any(f["subject"] == "_scene" and f["value"] == "make it rain" and f["from"] == 2 for f in bible["facts"])
    steered = [p for p, _ in loop4["watcher"].yesno if "Is this clearly visible in these frames: make it rain" in p]
    assert len(steered) == 1


def test_a_new_scene_steer_forces_a_scene_change(loop4):
    run = make(2, keyframes="off")
    story.add_steer(run, "the harbour at dawn", "scene", 2)
    autopilot.save(run)
    run = go(run)
    assert run["segments"][1]["scene_change"] is True
    assert "the person asked for a new scene: the harbour at dawn" in piece_prompts(loop4["director"])[1]


def test_redo_takes_the_last_piece_out_and_makes_it_again(loop4):
    run = go(make(2, keyframes="off"))
    first_output = run["segments"][1]["output"]
    autopilot.redo(run["id"])
    deadline = time.time() + 120
    while autopilot.alive(run["id"]) and time.time() < deadline:
        time.sleep(0.1)
    run = autopilot.load(run["id"])
    last = run["segments"][1]
    assert run["state"] == "done" and len(last["takes"]) == 2 and last["output"] != first_output
    assert "asked to redo" in json.dumps(last["fixed"])
    memory = json.loads((autopilot.run_dir(run["id"]) / "memory.json").read_text())
    assert memory["observed_through"] == 2  # the memory was put back before piece 2, then piece 2 committed again


def test_keep_going_makes_a_finished_film_longer(loop4):
    run = go(make(2, keyframes="off"))
    autopilot.extend(run["id"], 1)
    deadline = time.time() + 120
    while autopilot.alive(run["id"]) and time.time() < deadline:
        time.sleep(0.1)
    run = autopilot.load(run["id"])
    assert run["state"] == "done" and len(autopilot.committed(run)) == 3
    arcs = [p for p, s in zip(loop4["director"].asked, loop4["director"].systems) if s == story.ARC_SYSTEM]
    assert "Plan segments 3 to 3" in arcs[-1] and "The earlier arc's payoff was" in arcs[-1]


def test_go_sharper_asks_the_sharp_size_and_schedule(loop4):
    run = make(1, keyframes="off")
    autopilot.sharpen(run["id"])
    run = go(autopilot.load(run["id"]))
    video = [b for b in loop4["queue"].bodies if b["kind"] == "video"][0]["params"]
    assert (video["width"], video["height"], video["schedule"]) == (1376, 768, "sharp")


def test_an_opening_keyframe_from_the_seed_starts_before_the_director(loop4):
    said = client.post("/api/autopilot/runs", json={"name": "Lamp", "seed": {**SEED, "opening_keyframe": "Vex on a rooftop at dusk, wide"},
                                                    "settings": {"segments": 1, "keyframes": "on", "continuity": {"film_pass": False}}})
    run = autopilot.load(said.json()["id"])
    order = []
    real_submit, real_director = loop4["queue"].submit, loop4["director"]
    loop4["monkeypatch"].setattr(autopilot, "submit", lambda body: (order.append(body["kind"]), real_submit(body))[1])
    loop4["monkeypatch"].setattr(autopilot, "ask_text", lambda r, s, p: (order.append("director"), real_director(r, s, p))[1])
    run = go(run)
    assert order[0] == "scout" and order[1] == "director"
    assert run["segments"][0]["keyframes"][0].get("opening") and "Vex on a rooftop at dusk" in run["segments"][0]["keyframe_prompt"]
    assert "already being drawn" in piece_prompts(real_director)[0]


def test_keep_going_on_a_film_stopped_early_carries_it_on_without_adding(loop4):
    run = go(make(1, keyframes="off"))
    run["settings"]["segments"], run["state"] = 2, "failed"
    autopilot.save(run)
    autopilot.extend(run["id"], 1)
    deadline = time.time() + 120
    while autopilot.alive(run["id"]) and time.time() < deadline:
        time.sleep(0.1)
    run = autopilot.load(run["id"])
    assert run["state"] == "done" and run["settings"]["segments"] == 2 and len(autopilot.committed(run)) == 2


def test_stop_now_while_the_director_plans_sends_nothing_more_to_the_renderer(loop4):
    # "stop now" said while the Director wrote piece 2: its keyframe was drawn anyway (the Mac play run, 2026-10-27)
    stop, real_director = threading.Event(), loop4["director"]
    queue = loop4["queue"]

    def director(run, system, prompt):
        answer = real_director(run, system, prompt)
        if system == autopilot.DIRECTOR_SYSTEM and len(autopilot.committed(run)) == 1:
            stop.set()
        return answer
    loop4["monkeypatch"].setattr(autopilot, "ask_text", director)
    run = make(2, keyframes="on")
    autopilot.work(run, stop)
    run = autopilot.load(run["id"])
    assert run["state"] == "paused" and len(autopilot.committed(run)) == 1
    assert not [b for b in queue.bodies if (b["params"].get("autopilot") or {}).get("segment") == 2]


def test_a_keyframe_with_no_room_yet_is_drawn_again_shortly(loop4):
    loop4["monkeypatch"].setattr(autopilot, "KEYFRAME_WAIT_S", 0.01)
    queue, refused = loop4["queue"], [0]
    real = queue.submit

    def submit(body):
        job = real(body)
        if body["kind"] in ("scout", "edit") and not refused[0]:
            refused[0] = 1
            job.update(status="failed", error="this render needs about 31 GB at its peak (give or take 4); this machine has "
                                              "52 GB for it and keeps 20 GB free as its floor")
        return job
    loop4["monkeypatch"].setattr(autopilot, "submit", submit)
    run = go(make(1, keyframes="on"))
    assert run["state"] == "done" and run["segments"][0]["keyframe_waits"] == 1


def test_a_piece_that_failed_before_its_keyframe_draws_it_again_on_resume(loop4):
    run = make(1, keyframes="on")
    run["segments"] = [{"index": 1, "id": "S0001", "status": "failed", "scene": "x", "scene_change": True, "beat": "b",
                        "fields": shot("b")["shot"], "seed": 1, "frames": 24, "seconds": 1.0, "start": 0, "takes": [],
                        "job": None, "output": None, "keyframe_prompt": "Vex, wide", "keyframes": [], "pace": {}}]
    run["arc"] = {"beats": [{"index": 1, "beat": "b", "end_state": ENDS[0]}], "payoff": "p", "through": 1}
    autopilot.save(run)
    autopilot.runs.pop(run["id"], None)
    autopilot.begin(run)
    deadline = time.time() + 120
    while autopilot.alive(run["id"]) and time.time() < deadline:
        time.sleep(0.1)
    run = autopilot.load(run["id"])
    assert run["state"] == "done" and run["segments"][0]["keyframes"] and run["segments"][0]["mode"] == "keyframe"
