"""AutoPilot's clip judge (gma/ap/judge.py): d1-2B against a fake /v1/systemone server. The calibrated decision
(person, cut, clothes retake; doubt retakes; notes never do), the split when a build reads too few rows a pass, a
server that is down, and whole runs where d1 alone keeps or retakes a take, with every probability in the report.
No GPU, no model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gma.ap import autopilot, judge
from test_autopilot import go, loop, new_run  # noqa: F401

CLEAN = {"person": 0.997, "cut": 0.01, "wardrobe": 0.03, "place": 0.99, "morph": 0.02}  # c01 of the calibration set
SWAP = {"person": 0.25, "cut": 0.98, "wardrobe": 0.24, "place": 0.08, "morph": 0.32}  # s09
HIDDEN_SWAP = {"person": 0.93, "cut": 0.31, "wardrobe": 0.19, "place": 0.74, "morph": 0.31}  # s12: only the cut is in doubt
HUE = {"person": 0.96, "cut": 0.11, "wardrobe": 0.82, "place": 0.99, "morph": 0.35}  # w17


class FakeD1(BaseHTTPRequestHandler):
    """/v1/systemone: answers each question from ``FakeD1.p`` (0.9 for a context question), at most ``rows`` questions
    a pass (a build that reads too few prompt rows refuses more), or every request with ``status``."""

    p: dict = {}
    rows = 99
    status = 200
    bodies: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeD1.bodies.append(body)
        if self.path != "/v1/systemone" or FakeD1.status != 200:
            return self.reply(FakeD1.status if FakeD1.status != 200 else 404, {"error": {"message": "down"}})
        if len(body["questions"]) > FakeD1.rows:
            return self.reply(400, {"error": {"message": "a packed decision of 1083 tokens is past the 1024 rows one pass reads"}})
        answers = {k: {"type": "noul", "noul": FakeD1.p.get(k, 0.9)} for k in body["questions"]}
        self.reply(200, {"model": "d1", "answers": answers})

    def reply(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


@pytest.fixture
def d1():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeD1)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeD1.p, FakeD1.rows, FakeD1.status, FakeD1.bodies = dict(CLEAN), 99, 200, []
    yield judge.D1Judge({"url": f"http://127.0.0.1:{server.server_address[1]}/v1"})
    server.shutdown()
    autopilot.judge = None


def clip(tmp_path, seconds=2):
    import subprocess
    out = tmp_path / "take.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x96:rate=24:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


# ---------- the decision ----------

def test_the_calibrated_rule_on_the_calibration_clips():
    j = judge.D1Judge()
    clean, swap, hidden, hue = (j.decide(p) for p in (CLEAN, SWAP, HIDDEN_SWAP, HUE))
    assert not clean["retake"] and clean["faults"] == clean["doubts"] == clean["notes"] == []
    assert swap["retake"] and {"person", "cut"} <= set(swap["faults"]) and "place" in swap["notes"]  # place: a note only
    assert hidden["retake"] and hidden["faults"] == [] and hidden["doubts"] == ["cut"]  # retaken on doubt, no confirmer
    assert hue["retake"] and hue["faults"] == ["wardrobe"]
    q = swap["questions"]["person"]
    assert q["p_yes"] == 0.25 and q["p"] == round(judge.calibrate(0.25, 1.337), 4) and q["p_fault"] == round(1 - q["p"], 4)


def test_the_temperature_softens_and_the_person_bar_retakes_doubt_about_the_person():
    assert judge.calibrate(0.5, 1.337) == pytest.approx(0.5) and 0.5 < judge.calibrate(0.9, 1.337) < 0.9
    j = judge.D1Judge()
    unsure = j.decide({**CLEAN, "person": 0.62})  # 0.66 raw is the bar at 0.6 after the temperature
    assert unsure["retake"] and unsure["doubts"] == ["person"] and unsure["questions"]["person"]["status"] == "doubt"
    assert not j.decide({**CLEAN, "person": 0.7})["retake"]  # a clean animal close-up's 0.69: kept
    calm = judge.D1Judge({"retake_on_doubt": False}).decide(HIDDEN_SWAP)
    assert not calm["retake"] and calm["doubts"] == ["cut"]  # recorded, not retaken


def test_notes_never_retake_and_become_no_findings():
    j = judge.D1Judge()
    noted = j.decide({**CLEAN, "place": 0.2, "morph": 0.9, "beat": 0.05, "dress": 0.5},
                     judge.context_questions("Vex leaps the gap", "", "Vex", "orange jacket"))
    assert not noted["retake"] and set(noted["notes"]) == {"place", "morph", "beat", "dress"}
    assert noted["questions"]["beat"]["text"] == "Does the clip show this: Vex leaps the gap?" and judge.findings(noted) == []
    found = judge.findings(j.decide(SWAP))
    assert {f["category"] for f in found} == {"identity", "setting"} and all(f["action"] == "retake" and f["source"] == "d1" for f in found)


def test_the_profile_picks_d1_by_default_and_gemma_only_when_asked():
    assert isinstance(judge.make({}), judge.D1Judge) and judge.make({}).url == "http://127.0.0.1:8412"
    assert judge.make({"autopilot": {"judge": "eyes"}}) is None
    picked = judge.make({"autopilot": {"judge": "d1", "judge_choices": {"d1": {"url": "http://127.0.0.1:9999/v1", "fault": 0.8}}}})
    assert picked.url == "http://127.0.0.1:9999" and picked.spec["fault"] == 0.8 and picked.spec["person_min"] == 0.6


# ---------- the server ----------

def test_a_take_is_asked_in_one_pass_with_eight_frames(d1, tmp_path):
    FakeD1.p = dict(SWAP)
    verdict = d1.judge(clip(tmp_path), tmp_path / "j", 2.0, judge.context_questions("Vex leaps", "a neon avenue"))
    assert verdict["retake"] and verdict["passes"] == 1 and len(FakeD1.bodies) == 1
    body = FakeD1.bodies[0]
    assert len(body["images"]) == 8 and all(i.startswith("data:image/png;base64,") for i in body["images"])
    assert set(body["questions"]) == {"person", "cut", "wardrobe", "place", "morph", "beat", "setting"}
    assert all(q["type"] == "noul" for q in body["questions"].values()) and "2-second video clip" in body["state"]
    assert verdict["at"] == [0.12, 0.38, 0.62, 0.88, 1.12, 1.38, 1.62, 1.88] and verdict["ms"] > 0
    assert d1.requests == 1 and d1.describe()["frames"] == 8


def test_a_build_that_reads_fewer_rows_is_asked_in_halves(d1, tmp_path):
    FakeD1.rows = 3
    verdict = d1.judge(clip(tmp_path), tmp_path / "j", 2.0, judge.context_questions("Vex leaps", "a neon avenue", "Vex", "orange jacket"))
    assert verdict["passes"] == 4 and set(verdict["questions"]) == {"person", "cut", "wardrobe", "place", "morph", "beat", "setting", "dress"}
    assert not verdict["retake"] and all(len(b["images"]) == 8 for b in FakeD1.bodies)


def test_a_judge_that_does_not_answer_keeps_the_take(d1, tmp_path):
    FakeD1.status = 503
    verdict = d1.judge(clip(tmp_path), tmp_path / "j", 2.0)
    assert verdict["retake"] is False and "503" in verdict["error"]
    assert judge.D1Judge().judge(tmp_path / "missing.mp4", tmp_path / "k", 2.0)["error"] == "no frames could be read from the take"


# ---------- whole runs ----------

def test_d1_alone_keeps_clean_takes_and_the_eyes_are_never_asked(loop, d1, monkeypatch):
    monkeypatch.setattr(autopilot, "judge", d1)
    run = go(new_run(2))
    assert run["state"] == "done" and [len(s["takes"]) for s in run["segments"]] == [1, 1]
    assert loop["watcher"].calls == 0 and loop["watcher"].yesno == [] and loop["watcher"].checklists == []  # no Gemma anywhere
    take = run["segments"][0]["takes"][0]
    assert take["passed"] and take["judge"]["questions"]["person"]["p_yes"] == 0.997 and take["judge"]["by"] == "d1-2B"
    report = json.loads((autopilot.run_dir(run["id"]) / "segments" / "S0001" / "continuity-take1.json").read_text())
    assert set(report["judge"]["questions"]) >= {"person", "cut", "wardrobe", "place", "morph", "beat", "setting"}
    memory = json.loads((autopilot.run_dir(run["id"]) / "memory.json").read_text())
    assert memory["graph"]["events"] and memory["graph"]["nodes"][0]["label"] == "Vex"  # the plan, as the d1 path records it
    assert run["check_stats"]["requests"] == 2


def test_doubt_about_the_person_is_a_note_and_a_sure_fault_retakes_with_the_shot_rewritten(loop, d1, monkeypatch):
    """Only a confident, visible failure retakes: d1-2B's doubt (person 0.6) is a note for the next piece; its fault
    (a swap) renders the piece again with the shot rewritten."""

    monkeypatch.setattr(autopilot, "judge", d1)
    answers = [dict(CLEAN, person=0.6), dict(CLEAN)]
    original = judge.D1Judge.ask

    def ask(self, state, questions, images):
        FakeD1.p = answers.pop(0) if answers else dict(CLEAN)
        return original(self, state, questions, images)
    monkeypatch.setattr(judge.D1Judge, "ask", ask)
    run = go(new_run(1))
    seg = run["segments"][0]
    assert [t["passed"] for t in seg["takes"]] == [True] and seg["takes"][0]["judge"]["doubts"] == ["person"]
    assert any("d1-2B, identity: unsure" in n for n in seg["takes"][0]["notes"])
    answers[:] = [dict(SWAP), dict(CLEAN)]
    run = go(new_run(1))
    seg = run["segments"][0]
    assert [t["passed"] for t in seg["takes"]] == [False, True] and seg["fixed"][0]["findings"][0]["category"] == "identity"
    assert any("FAILED ITS CONTINUITY CHECK" in p for p in loop["director"].asked)


def test_when_retakes_run_out_the_best_take_is_kept(loop, d1, monkeypatch):
    monkeypatch.setattr(autopilot, "judge", d1)
    FakeD1.p = dict(SWAP)
    run = go(new_run(1, retakes=1))
    seg = run["segments"][0]
    assert len(seg["takes"]) == 2 and seg["status"] == "done" and not seg["passed"]
    assert any("no take passed" in r for r in seg["reasons"])
