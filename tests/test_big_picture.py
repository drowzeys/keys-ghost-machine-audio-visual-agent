"""The big-picture Director (gma/ap/big_picture.py): the whole film's findings numbered, the score, the plan's rules
(every finding merged or dropped, 3 fixes at most), the draft moved aside and put back ("do no harm"), and the fix
written into the Director's prompt for the regenerated pieces. No model, no GPU."""

import json

from gma.ap import autopilot as ap
from gma.ap import big_picture as bp


def seg(i, checks, output=None):
    return {"index": i, "id": f"S{i:04d}", "status": "done", "beat": f"beat {i}", "output": output or f"r1/segments/S{i:04d}/take1.mp4",
            "fields": {"action": f"action {i}", "camera": "camera", "setting": "setting"}, "qc_checks": checks}


def c(check, passed, **more):
    return {"check": check, "passed": passed, "p": 1.0 if passed else 0.0, "state": check, "frame": "f", "confident": not passed, **more}


def test_findings_score_and_the_plan_rules():
    segs = [seg(1, [c("direction", False), c("end place", False), c("end state", True)]),
            seg(2, [c("seam", True, corr=0.99), c("end state", True), c("contact", False)]),
            seg(3, [c("seam", True, corr=0.84), c("end state", True)])]
    found = bp.findings_of(segs)
    assert [f["id"] for f in found] == ["F1", "F2", "F3"] and found[0]["piece"] == 1
    s = bp.score_of(segs)
    assert s["pieces"] == [round(1 / 3, 3), 0.5, 1.0] and s["low"] == 0.84
    seen = {"findings": found}
    assert "F3" in bp.plan_problem({"fixes": [{"piece": 1, "merges": ["F1", "F2"], "change": "climbs UP"}], "dropped": []}, seen, 3)
    assert bp.plan_problem({"fixes": [{"piece": 1, "merges": ["F1", "F2"], "change": "climbs UP"}],
                            "dropped": [{"finding": "F3", "why": "cosmetic"}]}, seen, 3) == ""
    four = [{"piece": 1, "merges": [], "change": "x"}] * 4
    assert "merge them" in bp.plan_problem({"fixes": four, "dropped": []}, seen, 3)


def test_the_draft_is_moved_aside_and_put_back(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "HOME", tmp_path)  # never ap.attach: its HOME would outlive the test
    run = {"id": "r1", "seed": {"world": "w"}, "settings": ap.clean_settings({"big_picture": True}), "cast": {},
           "segments": [seg(1, [c("direction", False)]), seg(2, [c("seam", True, corr=0.9)])], "steers": [], "locations": [{"from": 1, "image": "a"}]}
    folder = ap.run_dir("r1")
    for s in run["segments"]:
        (folder / "segments" / s["id"]).mkdir(parents=True)
        (folder / "segments" / s["id"] / "take1.mp4").write_bytes(b"draft")
        ap.write_json(folder / "memory" / f"{s['id']}.json", {"graph": {"nodes": [], "events": [], "chapters": []}, "archive": [], "observed_through": s["index"]})
    (folder / "film.mp4").write_bytes(b"draft film")
    kept = bp.stash(run, 1, folder / "drafts" / "pass1")
    assert run["segments"] == [] and not (folder / "segments" / "S0001").exists() and run["locations"] == []
    assert (folder / "drafts" / "pass1" / "segments" / "S0002" / "take1.mp4").read_bytes() == b"draft"
    run["big_picture"] = {"state": "regenerating", "start": 1, "pass": 1, "passes": [], "draft": kept,
                          "fixes": [{"piece": 1, "change": "Ewan climbs UP the stairs, camera below looking up", "root_cause": "wrong way"}]}
    text = bp.plan_text(run, 1)
    assert "climbs UP" in text and "action 1" in text and bp.plan_text(run, 2).count("THE FIX") == 0
    (folder / "segments" / "S0001").mkdir(parents=True)
    (folder / "segments" / "S0001" / "take1.mp4").write_bytes(b"new")
    run["segments"] = [seg(1, [c("direction", False)])]
    bp.restore(run, kept, 1, folder / "drafts" / "pass1", folder / "drafts" / "pass1-rejected")
    assert (folder / "segments" / "S0001" / "take1.mp4").read_bytes() == b"draft" and len(run["segments"]) == 2
    assert (folder / "drafts" / "pass1-rejected" / "segments" / "S0001" / "take1.mp4").read_bytes() == b"new"
    assert run["locations"] == [{"from": 1, "image": "a"}]


def test_no_per_piece_retakes_while_a_big_picture_film_is_made_and_the_rewrite_rules():
    assert ap.retakes_now({"settings": ap.clean_settings({"big_picture": True, "retakes": 2})}) == 0
    assert ap.retakes_now({"settings": ap.clean_settings({"retakes": 2})}) == 2
    rules = ap.rewrite_rules([{"check": "direction"}, {"check": "direction"}, {"check": "end place"}])
    assert rules.count("REWRITE THE MOVEMENT") == 1 and "UP" in rules and "REWRITE THE ENDING" in rules
    assert ap.clean_settings({"fix_budget": 5})["fix_budget"] == 2
