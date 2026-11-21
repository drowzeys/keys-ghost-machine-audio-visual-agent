# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, tests/test_autopilot_checks.py at 6cc6e67c5e138499ea2c17183eda261b710d1217. Apache-2.0:
# LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: imports from the port (gma/ap); the vision server's switch is gma/ap/eyes.py's; the fight-sequence tests
# (fight.py) are left out with the feature.
"""AutoPilot's dense checks, pace plan, reasoning Director, memory-first lengths and Infinity mode (app/checks.py,
app/autopilot.py, app/agents.py's thinking switch, spark/memory_guard.py's expected load): the loop of
tests/test_autopilot.py with a scripted Director and a stand-in Cosmos. No GPU, no network beyond loopback, no model."""
import http.server
import json
import re
import sys
import threading
from pathlib import Path

import pytest

from test_autopilot import BEATS, HOME, ROOT, Director, Watcher, autopilot, go, loop, new_run, shot  # noqa: F401

checks = autopilot.checks


# ---------- the answers, read in code ----------

def test_answers_are_read_tolerantly_and_unclear_is_never_a_failure():
    extract = autopilot.extract_json
    # the Viking v2 film's malformed answer: every key nested under the first one
    nested = '{"same": {"copper_red": "Yes", "brown_coat": "Yes", "sea": "No"}, "note": "x"}'
    assert checks.read(nested, ["copper_red", "sea", "missing"], extract) == {"copper_red": "yes", "sea": "no", "missing": None}
    assert checks.read("<think>the coat looks brown</think>```json\n{'q1': True, 'q2': 'unclear', 'q3': 'No, it is not',}\n```", ["q1", "q2", "q3"], extract) == \
        {"q1": "yes", "q2": None, "q3": "no"}
    assert checks.read("I cannot tell.", ["q1"], extract) == {"q1": None}
    assert checks.read('{"q1": {"answer": "yes"}}', ["q1"], extract) == {"q1": "yes"}


def test_a_failed_major_question_fails_only_by_a_majority_of_votes():
    q = {"key": "q1", "category": "identity", "text": "same person?", "expect": "yes", "major": True}
    said = iter(['{"q1": "yes"}', '{"q1": "YES"}'])
    log = []
    assert checks.vote(lambda *a: next(said), [], "h", q, "no", 3, autopilot.extract_json, log, {"window": 1}) == "yes"
    assert log[-1]["tally"] == ["no", "yes", "yes"] and log[-1]["requests"] == 2
    said = iter(['{"q1": "no"}', '{"q1": "unclear"}'])
    assert checks.vote(lambda *a: next(said), [], "h", q, "no", 3, autopilot.extract_json, log, {}) == "no"
    said = iter(['{"q1": "yes"}'])
    assert checks.vote(lambda *a: next(said), [], "h", q, "no", 2, autopilot.extract_json, log, {}) is None  # a tie fails nothing


def test_a_window_too_few_answered_is_asked_again_once():
    asked = [{"key": f"q{i}", "category": "identity", "text": "t", "expect": "yes", "major": True} for i in range(1, 5)]
    said = iter(["not json at all", '{"q1": "yes", "q2": "yes", "q3": "no", "q4": "yes"}'])
    log = []
    found = checks.ask_window(lambda *a: next(said), [], "h", asked, autopilot.extract_json, log, {"window": 1})
    assert found == {"q1": "yes", "q2": "yes", "q3": "no", "q4": "yes"} and log[0]["requests"] == 2
    assert checks.summary(log) == {"requests": 2, "questions": 4, "failed_answers": 1, "votes": 0, "seconds": log[0]["seconds"]}


def test_windows_cover_the_piece():
    spans = checks.windows(10.125, 2.0, 3)
    assert len(spans) == 5 and all(len(w) == 3 for w in spans)
    assert spans[0][0] > 0 and spans[-1][-1] < 10.125 and all(a < b for w in spans for a, b in zip(w, w[1:]))
    assert checks.windows(1.0, 2.0, 2) == [[0.25, 0.75]]


# ---------- the pace ----------

FIGHT = {"kind": "fight", "beats": [{"from": 0, "to": 6, "speed": "fast"}, {"from": 6, "to": 8, "speed": "slow", "what": "the centerline punch"},
                                    {"from": 8, "to": 10, "speed": "fast"}]}
TIMED = "0-6 s rapid real-time exchange of punches; 6-8 s slow motion on the centerline punch; 8-10 s back to full speed"


def test_the_pace_plan_keeps_fights_fast_with_slow_motion_for_the_key_move_only():
    plan = checks.pace_of(FIGHT)
    assert checks.pace_problem(plan, TIMED, "Handheld, whip pans between the fighters", 10) == ""
    assert "pace.beats" in checks.pace_problem(checks.pace_of({"kind": "fight"}), "they fight", "handheld", 10)
    assert "mostly fast" in checks.pace_problem(checks.pace_of({"kind": "fight", "beats": [{"from": 0, "to": 7, "speed": "slow"}, {"from": 7, "to": 10, "speed": "fast"}]}),
                                                "0-7 s slow motion fight; 7-10 s fast", "handheld", 10)
    assert "camera" in checks.pace_problem(plan, TIMED, "The camera arcs slowly left", 10)
    assert checks.pace_problem(plan, "they trade blows and one punch lands in slow motion", "handheld", 10) == ""  # pace_line adds its seconds
    assert "slow motion" in checks.pace_problem(checks.pace_of({"kind": "calm"}), "she walks in slow motion", "static", 10)
    line = checks.pace_line(plan, 10)
    assert "fast, real-time speed" in line and "slow motion only from 6 to 8 s on the centerline punch" in line
    assert checks.pace_line(checks.pace_of({}), 10) == "" and checks.pace_of(None)["kind"] == "calm"
    assert [checks.planned(plan, t, 10) for t in (1, 6.5, 9)] == ["fast", "slow", "fast"]


def test_a_fight_that_comes_out_slow_throughout_fails_and_one_at_speed_passes():
    plan = checks.pace_of(FIGHT)
    slow = [0.8] * 10
    fast = [12.0] * 6 + [1.0, 1.0] + [10.0] * 2
    eyes_slow = [(1, "yes"), (3, "yes"), (5, "yes"), (7, "yes"), (9, "yes")]
    assert checks.pace_verdict(plan, slow, eyes_slow, 10)["status"] == "major"
    assert checks.pace_verdict(plan, fast, [(1, "no"), (3, "no"), (9, "no")], 10)["status"] == "ok"
    late = [12.0, 12.0] + [0.8] * 8
    assert checks.pace_verdict(plan, late, [(1, "no"), (3, "no"), (9, "no")], 10)["status"] == "ok"  # the eyes disagree: not a retake
    # an opening that eases in from a still frame is a retake by measurement alone, whatever the rest does
    opening = checks.pace_verdict(plan, [3.1, 4.9] + [4.0] * 8, [(1, "no"), (3, "no")], 10)
    assert opening["status"] == "major" and "opens slowly" in opening["evidence"] and opening["opening"] == [3.1, 4.9]
    # a slow lead that a fast second follows is cut from the film (ease_in): a note, not a retake
    assert checks.pace_verdict(plan, [3.1, 4.9] + [14.0] * 8, [(1, "no"), (3, "no")], 10)["status"] == "minor"
    assert checks.pace_verdict(plan, [3.1, 9.0] + [14.0] * 8, [], 10)["status"] == "ok"
    assert checks.pace_opening(plan).startswith("From the very first frame") and checks.pace_opening(checks.pace_of({"kind": "calm"})) == ""
    assert checks.pace_verdict(plan, slow, [], 10)["status"] == "major"  # no answer from the eyes: the measurement, stricter
    assert checks.pace_verdict(checks.pace_of({"kind": "calm"}), slow, [], 10)["status"] == "not_planned"


def test_motion_is_measured_a_second_at_a_time(tmp_path):
    import subprocess
    moving, still = tmp_path / "moving.mp4", tmp_path / "still.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24:duration=2", "-pix_fmt", "yuv420p", str(moving)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:size=160x96:rate=24:duration=2", "-pix_fmt", "yuv420p", str(still)], check=True)
    assert len(checks.motion(moving)) == 2 and min(checks.motion(moving)) > max(checks.motion(still)) and max(checks.motion(still)) < checks.SLOW


# ---------- the Director reasons first ----------

class Server(http.server.BaseHTTPRequestHandler):
    bodies, replies = [], []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.answer({"data": [{"id": "nvidia-nemotron-3.5-bolt"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Server.bodies.append(body)
        code, reply = Server.replies.pop(0)
        if code != 200:
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps({"error": reply}).encode())
            return
        self.answer({"choices": [{"message": reply, "finish_reason": "stop"}]})

    def answer(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def bolt_like():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Server.bodies, Server.replies = [], []
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_the_director_asks_bolt_with_thinking_on_and_keeps_only_the_answer():
    """The Director (gma/ap/service.py) on the app's brain: reasoning on with a bounded thinking_budget, the reasoning
    dropped; an empty answer (the budget used up) asked again without thinking, its length bounded."""
    import asyncio

    from gma.ap.service import AutoPilotService

    class Brain:
        def __init__(self, replies):
            self.replies, self.bodies = list(replies), []

        async def complete(self, messages, **settings):
            self.bodies.append(settings)
            return self.replies.pop(0)

    class Video:
        root = HOME
        renderer = None

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    hooks = ("submit", "job_of", "cancel", "director", "eyes", "ears", "on_change", "shell")
    saved = {k: getattr(autopilot, k) for k in hooks}
    try:
        brain = Brain(['{"scene": "x"}'])
        svc = AutoPilotService(Video(), brain=brain, loop=loop)
        assert svc.director("sys", "write it", 4096) == '{"scene": "x"}'
        body = brain.bodies[-1]
        assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["thinking_budget"] == 4096
        assert body["max_tokens"] == 4096 + 8000
        brain.replies = ["<think>the coat is brown</think>\n{\"a\": 1}"]
        assert svc.director("sys", "q", 512) == '{"a": 1}'
        brain.replies = ["", '{"b": 2}']
        assert svc.director("sys", "q", 512) == '{"b": 2}'
        assert brain.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False} and brain.bodies[-1]["max_tokens"] == 9000
        brain.replies = ["plain"]
        assert svc.director("sys", "q", 0) == "plain" and "thinking_budget" not in brain.bodies[-1]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        for k, v in saved.items():
            setattr(autopilot, k, v)


def test_the_run_asks_its_brain_with_the_reasoning_budget(monkeypatch):
    seen = []
    monkeypatch.setattr(autopilot, "director", lambda system, prompt, think: seen.append(think) or "{}")
    run = {"settings": {**autopilot.DEFAULTS}}
    autopilot.ask_text(run, "s", "p")
    run["settings"]["think"] = False
    autopilot.ask_text(run, "s", "p")
    assert seen == [4096, 0]


# ---------- the dense checks in the loop ----------

def answers_for(rules):
    """A stand-in Cosmos for the yes/no checks: every question answered as the bible expects, except those whose text
    matches a rule (regex -> answer, or a list of answers in turn)."""

    def answer(prompt, frames):
        keys = re.findall(r'^"(\w+)": (.*)$', prompt, re.M)
        out = {}
        for key, text in keys:
            out[key] = "yes" if not re.search(r"change into a different|malformed|out of place|slow motion|second person who looks like|float, hover|bend, stretch|merge into each other|more than two arms", text) else "no"
            for pattern, said in rules.items():
                if re.search(pattern, text):
                    out[key] = said.pop(0) if isinstance(said, list) else said
        return json.dumps(out)
    return answer


def test_every_piece_is_checked_in_windows_and_the_checks_are_counted(loop, monkeypatch):
    watcher = Watcher(yesno=answers_for({}))
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, continuity={"film_pass": False}))
    first, second = run["segments"]
    assert first["passed"] and second["passed"]
    stats = first["check_stats"]
    assert stats["requests"] >= 1 and stats["questions"] >= 3 and stats["failed_answers"] == 0
    assert run["check_stats"]["requests"] == first["check_stats"]["requests"] + second["check_stats"]["requests"]
    dense = second["continuity"]["dense"]
    assert "identity" in dense["passed"] and (HOME / dense["file"]).is_file()
    joins = [p for p, _ in watcher.yesno if "The join between segment 1 and segment 2" in p]
    assert len(joins) == 1  # the continuation's join, picture 1 the last frame before
    log = json.loads((HOME / dense["file"]).read_text())
    assert {e["kind"] for e in log} >= {"window", "join"} and all("seconds" in e for e in log)


def test_a_major_seen_by_the_majority_and_the_second_look_retakes_with_the_finding(loop, monkeypatch):
    director = Director()
    # the setting question fails in segment 2's first take (every vote agrees), and passes on the retake
    state = {"take": 0}

    def setting(prompt, frames):
        if "Window 1 of" in prompt and "segment 2" in prompt and '"q1":' in prompt and '"q2":' in prompt:
            state["take"] += 1
        return answers_for({r"Is the place in these frames": "no" if state["take"] == 1 else "yes"})(prompt, frames)
    watcher = Watcher(yesno=setting)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    monkeypatch.setattr(autopilot, "ask_text", director)
    run = go(new_run(2, continuity={"film_pass": False}))
    second = run["segments"][1]
    assert len(second["takes"]) == 2 and second["takes"][0]["passed"] is False and second["passed"] is True
    found = [f for f in second["takes"][0]["continuity"]["findings"] if f.get("source") == "yes/no"]
    assert found and found[0]["category"] == "setting" and found[0]["action"] == "retake" and found[0]["second_look"]["confirmed"] is True
    assert "Is the place in these frames" in found[0]["evidence"]
    assert any(e.get("vote") for e in json.loads((HOME / second["takes"][0]["continuity"]["dense"]["file"]).read_text()))


def test_a_noisy_no_outvoted_is_not_a_retake(loop, monkeypatch):
    said = {"n": 0}

    def noisy(prompt, frames):
        if "Is the place in these frames" in prompt and prompt.count('": ') == 1:  # a vote: asked alone
            said["n"] += 1
            return json.dumps({re.findall(r'^"(\w+)"', prompt, re.M)[0]: "yes"})
        return answers_for({r"Is the place in these frames": "no"})(prompt, frames)
    watcher = Watcher(yesno=noisy)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(1, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert len(seg["takes"]) == 1 and seg["passed"] is True and said["n"] >= 2


def test_a_checklist_major_that_no_window_saw_becomes_a_correction(loop, monkeypatch):
    major = {"setting": {"verdict": "major", "frame": 3, "subject": "", "evidence": "a brick wall at sea"}}
    watcher = Watcher(checks={1: major}, yesno=answers_for({}))
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(1, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert len(seg["takes"]) == 1 and seg["passed"] is True
    entry = seg["continuity"]["checks"]["setting"]
    assert entry["status"] == "minor" and entry["action"] == "correct" and "every window" in entry["demoted"]


class FightDirector(Director):
    def __call__(self, run, system, prompt):
        self.asked.append(prompt)
        n = len(autopilot.committed(run))
        answer = shot(BEATS[n % len(BEATS)], change=n == 0)
        answer["shot"]["action"] = "Vex fights the drone: " + TIMED + "."
        answer["shot"]["camera"] = "Handheld, whip pans with every strike"
        answer["pace"] = FIGHT
        return json.dumps(answer)


def test_a_fight_comes_out_fast_with_the_key_move_timed_and_a_slow_one_is_retaken(loop, monkeypatch):
    director = FightDirector()
    monkeypatch.setattr(autopilot, "ask_text", director)
    watcher = Watcher(yesno=answers_for({r"slow motion": ["yes"] + ["no"] * 20}))
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    measured = iter([[0.5] * 10, [12.0] * 6 + [1.0] * 2 + [12.0] * 2])
    monkeypatch.setattr(autopilot.checks, "motion", lambda clip, env=None: next(measured))
    run = go(new_run(1, fight={"sequence": "off"}, continuity={"film_pass": False}))
    seg = run["segments"][0]
    prompt = loop["queue"].bodies[0]["params"]["prompt"]
    assert "fast, real-time speed" in prompt and "slow motion only from 6 to 8 s on the centerline punch" in prompt
    assert len(seg["takes"]) == 2 and "pace" in seg["takes"][0]["reasons"][0] and seg["passed"] is True
    assert seg["takes"][0]["continuity"]["dense"]["pace"]["status"] == "major" and seg["continuity"]["dense"]["pace"]["status"] == "ok"
    rewrite = [p for p in director.asked if "FAILED ITS CONTINUITY CHECK" in p]
    assert rewrite and "The video model made it slow" in rewrite[0]
    # the eyes were asked about slow motion in each window
    assert any("slow motion" in p for p, _ in watcher.yesno)


def test_a_fight_plan_that_is_slow_throughout_is_turned_back(loop, monkeypatch):
    answers = []

    class SlowFirst(FightDirector):
        def __call__(self, run, system, prompt):
            said = json.loads(super().__call__(run, system, prompt))
            if not answers:
                said["shot"]["camera"] = "The camera arcs slowly around them"
            answers.append(1)
            return json.dumps(said)
    monkeypatch.setattr(autopilot, "ask_text", SlowFirst())
    monkeypatch.setattr(autopilot.checks, "motion", lambda clip, env=None: [6.0] * 10)
    run = go(new_run(1, fight={"sequence": "off"}, continuity={"film_pass": False}))
    assert "camera" in run["segments"][0]["rejected"][0]


# ---------- memory first ----------

def test_a_piece_that_does_not_fit_is_made_shorter_at_the_same_size(loop, monkeypatch):
    monkeypatch.setattr(autopilot, "room", lambda params: {"fits": params["frames"] <= 192, "nearest": {"frames": 192}})
    run = go(new_run(1, preset="sharp", seconds=10, continuity={"film_pass": False}))
    seg = run["segments"][0]
    body = loop["queue"].bodies[0]["params"]
    assert body["frames"] == 192 and body["preset"] == "sharp" and seg["frames"] == 192
    assert seg["memory"][0]["from"] == 243 and seg["memory"][0]["to"] == 192 and seg["memory"][0]["retry"] is False


def test_a_render_stopped_by_the_memory_guard_is_made_again_shorter_and_the_run_goes_on(loop, monkeypatch):
    queue = loop["queue"]
    real = queue.submit
    stops = {"left": 2}

    def submit(body):
        job = real(body)
        if body["kind"] == "video" and stops["left"]:
            stops["left"] -= 1
            job.update(status="failed", error="the render was stopped to protect the machine: 33825 MB available and falling by 28270 MB a second")
        return job
    monkeypatch.setattr(autopilot, "submit", submit)
    run = go(new_run(1, preset="sharp", seconds=10, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert run["state"] == "done" and seg["status"] == "done"
    frames = [b["params"]["frames"] for b in queue.bodies if b["kind"] == "video"]
    assert frames[0] == 243 and frames[1] < frames[0] and frames[2] < frames[1] and len(frames) == 3
    assert {b["params"]["preset"] for b in queue.bodies} == {"sharp"} and [e["retry"] for e in seg["memory"]] == [True, True]
    assert any("stopped for memory" in line["text"] for line in run["log"])


def test_the_queues_refusal_names_the_length_that_fits(loop, monkeypatch):
    queue = loop["queue"]
    real = queue.submit

    def submit(body):
        if body["kind"] == "video" and body["params"]["frames"] > 209:
            raise autopilot.Bad("this clip needs about 71 GB ... What fits: 209 frames (8.7 s) at this size or 864x480 at this length.")
        return real(body)
    monkeypatch.setattr(autopilot, "submit", submit)
    run = go(new_run(1, preset="sharp", seconds=10, continuity={"film_pass": False}))
    assert run["segments"][0]["frames"] == 209 and queue.bodies[0]["params"]["frames"] == 209



# ---------- Infinity ----------

class EndingDirector(Director):
    """Writes beats until the fourth, which ends the story; folds and summaries answered."""

    def __call__(self, run, system, prompt):
        if "Write the story so far" in prompt:
            return json.dumps({"text": "Vex crossed the drowned city with the core."})
        if "Fold this stretch" in prompt:
            return super().__call__(run, system, prompt)
        self.asked.append(prompt)
        n = len(autopilot.committed(run))
        answer = shot(BEATS[n % len(BEATS)], change=n == 0)
        if n == 3:
            answer["end"] = True
        return json.dumps(answer)


def test_infinity_goes_on_until_the_director_ends_it_with_a_checkpoint_after_every_piece(loop, monkeypatch):
    director = EndingDirector()
    monkeypatch.setattr(autopilot, "ask_text", director)
    run = go(new_run(2, infinity=True, story_every=2, film_every=2, continuity={"film_pass": True}))
    done = autopilot.committed(run)
    assert len(done) == 4 and run["state"] == "done" and done[-1]["end"] is True  # the segment count of 2 does not stop it
    assert "the Director ended the story" in run["log"][-1]["text"]
    folder = autopilot.run_dir(run["id"])
    points = sorted(p.name for p in (folder / "checkpoints").iterdir())
    assert points == ["S0001.json", "S0002.json", "S0003.json", "S0004.json"]
    point = json.loads((folder / "checkpoint.json").read_text())
    assert point["segments"] == 4 and [p["seed"] for p in point["pieces"]] == [1, 2, 3, 4] and point["film"]["segments"] == 4
    assert run["story_so_far"]["text"].startswith("Vex crossed") and "The story so far (a running summary)" in director.asked[-1]
    assert "has no fixed length" in director.asked[-1] and run["film_checks"] and run["film_checks"][0]["window"] == [1, 2]


def test_infinity_stops_at_its_length_limit(loop, monkeypatch):
    run = go(new_run(0, seconds=1, infinity=True, max_film_seconds=1.5, continuity={"film_pass": False}))
    assert len(autopilot.committed(run)) == 2 and run["state"] == "done" and "the film reached" in run["log"][-1]["text"]


def test_a_killed_infinity_run_resumes_from_its_checkpoint_by_itself(loop, monkeypatch):
    """The app dies in the middle of segment 3 (run.json says running, the take rendered and not checked): on the next
    start resume_all picks it up where it was, nothing asked of anyone."""

    queue = loop["queue"]

    class Killed(BaseException):
        pass
    calls = {"n": 0}
    real_check = autopilot.check

    def dies(run, seg):
        if seg["index"] == 3 and not calls["n"]:
            calls["n"] += 1
            raise Killed()
        return real_check(run, seg)
    monkeypatch.setattr(autopilot, "check", dies)
    run = new_run(0, seconds=1, infinity=True, max_film_seconds=3.5, continuity={"film_pass": False})
    run["state"] = "running"
    autopilot.save(run)
    with pytest.raises(Killed):
        autopilot.work(run, threading.Event())
    on_disk = autopilot.load(run["id"])
    assert on_disk["state"] == "running" and len(autopilot.committed(on_disk)) == 2
    assert json.loads((autopilot.run_dir(run["id"]) / "checkpoint.json").read_text())["next"]["index"] == 3
    rendered = len(queue.bodies)
    autopilot.runs.pop(run["id"], None)
    autopilot.resume_all()
    autopilot.runs[run["id"]]["thread"].join(120)
    after = autopilot.load(run["id"])
    assert after["state"] == "done" and len(autopilot.committed(after)) == 4
    assert len(queue.bodies) == rendered + 1  # segment 3's rendered take was checked, not rendered again; then segment 4
    assert any("Picked up after a restart" in line["text"] for line in after["log"])


def test_the_eyes_are_asked_with_their_reasoning_off(bolt_like):
    """Cosmos3-Edge reasons by default and used its whole 2000-token budget on a check, answering nothing (53 s):
    the checks ask with thinking off; a server that refuses the switch is asked without it. Gemma's leaked tokens go."""
    from gma.ap import eyes

    one = eyes.Eyes({"engine": "cosmos", "url": bolt_like})
    Server.replies = [(200, {"content": '{"q1": "yes"}'})]
    assert one.chat(bolt_like + "/v1", "sys", [], "q") == '{"q1": "yes"}'
    assert Server.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    Server.replies = [(400, "unknown field chat_template_kwargs"), (200, {"content": "ok"})]
    assert one.chat(bolt_like + "/v1", "sys", [], "q") == "ok" and "chat_template_kwargs" not in Server.bodies[-1]
    Server.replies = [(400, "too many image rows")]
    with pytest.raises(eyes.TooMany):
        one.chat(bolt_like + "/v1", "sys", [], "q")
    gemma = eyes.Eyes({"engine": "gemma", "url": bolt_like})
    Server.replies = [(200, {"content": '<image|>{"q1": "no"}<audio|>'})]
    assert gemma.chat(bolt_like + "/v1", "sys", [], "q") == '{"q1": "no"}' and Server.bodies[-1]["temperature"] == 0.0


def test_identity_questions_that_pass_in_every_window_overrule_the_watchers_drift(loop, monkeypatch):
    watcher = Watcher(drift={2}, yesno=answers_for({}))
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, retakes=1, continuity={"film_pass": False}))
    second = run["segments"][1]
    assert len(second["takes"]) == 1 and second["passed"] is True
    assert second["watch"]["identity_drift"] is False and "every window" in second["watch"]["drift_overruled"]


def test_the_watchers_drift_flag_is_read_leniently():
    data = {"ending": "Eirik over the chest", "identity_drift": "True"}
    assert autopilot.check_watch(data) == "" and data["identity_drift"] is True
    data = {"ending": "Eirik over the chest"}
    assert autopilot.check_watch(data) == "" and data["identity_drift"] is False


def test_each_window_asks_about_a_double_and_the_sheets_clothes():
    continuity = autopilot.continuity
    cast = {"Eirik": "copper-red hair in a braid, short red beard, grey wool tunic"}
    bible = continuity.new({"characters": [{"name": "Eirik", "look": cast["Eirik"]}]}, cast)
    seg = {"fields": {"action": "Eirik swings his shield", "setting": "a shingle beach"}, "mode": "text"}
    asked = checks.questions(bible, 1, seg, cast, list(continuity.WATCHED), continuity)
    texts = [q["text"] for q in asked]
    assert any("second person who looks like Eirik" in t for t in texts)
    assert any(q["category"] == "wardrobe" and "grey wool tunic" in q["text"] for q in asked)


# ---------- the wardrobe, one exact line ----------

def test_each_sheet_gets_one_exact_wardrobe_line_said_in_every_prompt(loop, monkeypatch):
    continuity = autopilot.continuity
    assert continuity.wardrobe_line({"look": "a man of thirty, copper-red hair tied back, brown wool coat over a cream linen shirt, black boots"}) == \
        "brown wool coat over a cream linen shirt, black boots"
    assert continuity.wardrobe_line({"look": "a man of twenty-five with copper braids and a brown frock coat"}) == "a brown frock coat"
    assert continuity.wardrobe_line({"look": "x", "wardrobe": "a brown waistcoat, cream shirt"}) == "a brown waistcoat, cream shirt"
    run = go(new_run(2, continuity={"film_pass": False}))
    bible = json.loads((autopilot.run_dir(run["id"]) / "bible.json").read_text())
    fact = next(f for f in bible["facts"] if f["category"] == "wardrobe" and f["source"] == "seed")
    assert fact["value"] == "chrome visor, orange reflective jacket"  # the sheet's clothes, canonical: what the eyes see never replaces it
    for body in loop["queue"].bodies:
        assert "Vex wears exactly this in every shot: chrome visor, orange reflective jacket." in body["params"]["prompt"]


def test_a_fight_prompt_opens_in_motion_and_its_keyframe_is_asked_for_motion(loop, monkeypatch):
    monkeypatch.setattr(autopilot, "ask_text", FightDirector())
    monkeypatch.setattr(autopilot.checks, "motion", lambda clip, env=None: [12.0] * 10)
    watcher = Watcher(yesno=answers_for({}))
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(1, fight={"sequence": "off"}, keyframes="on", continuity={"film_pass": False}))
    video = next(b for b in loop["queue"].bodies if b["kind"] == "video")["params"]["prompt"]
    assert "From the very first frame, already in motion at full real-time speed: Vex fights" in video
    assert any("caught in the middle of a fast movement" in p for p, _ in watcher.yesno)


def test_the_film_leaves_out_a_fight_pieces_slow_ease_in():
    fight = {"pace": FIGHT, "continuity": {"dense": {"pace": {"motion": [3.1, 2.2, 12.0, 14.0, 9.0]}}}}
    assert autopilot.ease_in(fight) == 2.0
    assert autopilot.ease_in({**fight, "continuity": {"dense": {"pace": {"motion": [3.1, 16.3, 10.0]}}}}) == 1.0
    assert autopilot.ease_in({**fight, "continuity": {"dense": {"pace": {"motion": [2.0, 2.0, 2.0, 2.0]}}}}) == 0.0  # nothing fast after it
    assert autopilot.ease_in({**fight, "continuity": {"dense": {"pace": {"motion": [9.0, 2.0, 12.0]}}}}) == 0.0
    assert autopilot.ease_in({"pace": {"kind": "calm"}, "continuity": {"dense": {"pace": {"motion": [1.0, 12.0]}}}}) == 0.0


def test_a_joined_film_is_shorter_by_the_cut_ease_in(loop, monkeypatch):
    class Varied(FightDirector):
        def __call__(self, run, system, prompt):
            said = json.loads(super().__call__(run, system, prompt))
            said["shot"]["action"] = BEATS[len(autopilot.committed(run)) % len(BEATS)] + ", trading quick strikes with the drone."
            return json.dumps(said)
    monkeypatch.setattr(autopilot, "ask_text", Varied())
    monkeypatch.setattr(autopilot.checks, "motion", lambda clip, env=None: [2.0, 12.0, 12.0])
    run = go(new_run(2, fight={"sequence": "off"}, seconds=3, continuity={"film_pass": False}))
    assert run["state"] == "done", run["log"][-3:]
    assert [autopilot.ease_in(s) for s in run["segments"]] == [1.0, 1.0]
    clip = sum(autopilot.duration(autopilot.HOME / s["output"]) for s in run["segments"])
    assert run["film"]["seconds"] < clip - 0.9  # each 1 s test piece cut to its last half second (the floor), a dissolve between
