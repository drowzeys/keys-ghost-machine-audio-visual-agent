"""AutoPilot's self-check (gma/ap/qc.py) on planted failures, with the numbers Gemma 4 12B and d1-2B gave on the original
film's frames: a keyframe with the lamp already lit, a take going the wrong way, a walk out through the door instead of
in, a duplicated hero, a lamp lit before the lighting across pieces, a hard cut and a jump at the seam; and the rule
that only a confident, visible failure retakes (everything else is a note). A fake looker, synthetic frames: no model,
no GPU."""

import json
from pathlib import Path

import numpy as np

from gma.ap import autopilot as ap
from gma.ap import qc

PLAN = qc.clean_qc({"start": ["the oil lamp is unlit, its wick dark"], "end": ["the oil lamp is lit, its flame inside the glass chimney"],
                    "motion": "Arthur climbs the stairs upward, away from the camera",
                    "wrong_motion": "Arthur walks down the stairs toward the camera",
                    "contact": "the match flame touches the wick inside the glass chimney",
                    "end_place": "Arthur is inside the lantern room, the door behind him",
                    "wrong_place": "Arthur is outside on the gallery, beyond the door",
                    "people": [{"who": "an old man with short white hair in a brown wool sweater", "count": 1}]})


class Looker:
    """``yes``: the first entry of ``table`` whose key is in the statement, else ``p``; ``pick``: ``picks`` by the right
    one's words, else ``p``. Remembers what it was asked."""

    def __init__(self, p=0.9, table=None, picks=None, two=0.05):
        self.p, self.table, self.picks, self.two, self.asked = p, table or {}, picks or {}, two, []

    def yes(self, pictures, statement):
        self.asked.append(("yes", [c for c, _ in pictures], statement))
        if statement.startswith("two or more"):
            return self.two
        return next((v for k, v in self.table.items() if k in statement), self.p)

    def pick(self, pictures, question, right, wrong):
        self.asked.append(("pick", [c for c, _ in pictures], right))
        if right.startswith("exactly one"):
            return 1 - self.two
        return next((v for k, v in self.picks.items() if k in right), self.p)

    def describe(self, pictures):
        return "an old man stands in an open doorway, looking out at the rain"


def frames(tmp_path, n=8):
    out = []
    for i in range(n):
        p = tmp_path / f"f{i}.png"
        p.write_bytes(b"P")
        out.append((f"Frame {i + 1}", p))
    return out


def failed(results):
    return {r["check"]: r for r in results if not r["passed"]}


def test_a_keyframe_with_the_lamp_already_lit_is_drawn_again_once(tmp_path):
    # Gemma 4 12B on the original piece 3 keyframe: "the oil lamp is unlit" 0.00
    results = qc.keyframe_checks(Looker(table={"unlit": 0.0}), PLAN, tmp_path / "kf.png")
    found = qc.failures(results, {})
    assert [f["check"] for f in found] == ["keyframe start"] and found[0]["retake"] and "START" in found[0]["evidence"]
    unsure = qc.failures(qc.keyframe_checks(Looker(table={"unlit": 0.4}), PLAN, tmp_path / "kf.png"), {})
    assert unsure[0]["retake"] is False and unsure[0]["status"] == "minor"  # not sure: a note, no redraw


def test_the_wrong_way_on_the_stair_and_out_through_the_door_retake_when_sure(tmp_path):
    f = frames(tmp_path)
    # Gemma, both orders: climbing up 0.165 / 1 - 0.706 -> 0.23; ending inside vs out on the gallery 0.011
    look = Looker(picks={"climbs the stairs upward": 0.23, "inside the lantern room": 0.011}, table={"unlit": 0.9})
    bad = failed(qc.take_checks(look, PLAN, f))
    assert bad["direction"]["confident"] and bad["end place"]["confident"]
    picks = [a for a in look.asked if a[0] == "pick"]
    assert picks[0][1] == ["The picture"] and picks[1][1] == ["First frame", "Middle frame", "Last frame"]
    found = {x["check"]: x for x in qc.failures(qc.take_checks(look, PLAN, f), {})}
    assert found["direction"]["retake"] and found["end place"]["retake"]
    assert "Anchor the camera" in found["direction"]["evidence"] and "wrong side of a door" in found["end place"]["evidence"]
    # a lean the wrong way, not sure: a note
    unsure = {x["check"]: x for x in qc.failures(qc.take_checks(Looker(picks={"climbs": 0.4}), PLAN, f), {})}
    assert unsure["direction"]["retake"] is False


def test_contact_and_end_state_are_notes_never_retakes(tmp_path):
    f = frames(tmp_path)
    # Gemma said yes (0.98) to a flame on the wick that burned outside the glass: contact cannot retake
    found = {x["check"]: x for x in qc.failures(qc.take_checks(Looker(table={"touches the wick": 0.0, "flame inside": 0.0}), PLAN, f), {})}
    assert not found["contact"]["retake"] and not found["end state"]["retake"]
    assert any(n.startswith("contact:") for n in qc.notes_of({"findings": list(found.values())}))


def test_a_second_copy_of_the_hero_retakes(tmp_path):
    f = frames(tmp_path)
    bad = failed(qc.take_checks(Looker(two=0.9), PLAN, f))
    assert bad["people"]["confident"] and bad["people"]["p"] == 0.9
    assert "window" in qc.many("an old man")


def test_a_lamp_the_piece_before_left_unlit_cannot_open_lit(tmp_path):
    record = {"checks": [{"check": "end state", "state": "the oil lamp is unlit, its wick dark", "passed": True, "p": 0.8},
                         {"check": "end place", "state": "Arthur is inside the lantern room", "passed": True, "p": 0.7},
                         {"check": "end state", "state": "the door is shut", "passed": False, "p": 0.2}]}
    carried = qc.verified_end(record)
    assert carried == ["the oil lamp is unlit, its wick dark", "Arthur is inside the lantern room"]
    bad = failed(qc.take_checks(Looker(table={"the oil lamp is unlit, its wick dark": 0.0}), qc.clean_qc({}), frames(tmp_path), carried))
    assert bad["carried"]["confident"] and bad["carried"]["frame"] == "the first frame"


def grey(seed, shift=0):
    return np.roll(np.kron(np.random.default_rng(seed).uniform(0, 255, (9, 16)), np.ones((4, 4))), shift, axis=1).astype(np.float32)


def test_the_seam_is_measured_and_a_jump_retakes(tmp_path, monkeypatch):
    a = grey(1)
    pictures = {"before.png": a, "same.png": grey(1, 1), "jump.png": grey(2)}
    monkeypatch.setattr(qc, "picture_grey", lambda p, ffmpeg="ffmpeg": pictures[Path(p).name])
    ok = qc.seam(Looker(p=0.9), tmp_path / "before.png", tmp_path / "same.png")
    assert ok["passed"] and ok["corr"] > qc.SEAM_CORR
    jump = qc.seam(Looker(p=0.9), tmp_path / "before.png", tmp_path / "jump.png")
    assert not jump["passed"] and jump["confident"] and jump["corr"] < qc.SEAM_CORR  # numbers alone are enough
    seen = qc.seam(Looker(p=0.1), tmp_path / "before.png", tmp_path / "same.png")
    assert not seen["passed"] and seen["confident"]  # the looker sure it is another place
    assert qc.failures([jump], {})[0]["retake"]


def layouts(seed):
    return np.random.default_rng(seed).uniform(0, 255, (9, 16))


def clip_of(*parts, fps=12):
    out = []
    for layout, seconds, flash in parts:
        for i in range(int(seconds * fps)):
            frame = np.roll(np.kron(layout, np.ones((4, 4))), i // 3, axis=1)
            out.append(np.clip(frame * (2.0 if flash and i == 6 else 1.0), 0, 255))
    return np.stack(out).astype(np.float32)


def test_a_hard_cut_is_found_and_a_pan_or_a_lightning_flash_is_not():
    cut = qc.cuts_of(clip_of((layouts(1), 4, False), (layouts(2), 4, False)))
    assert len(cut["cuts"]) == 1 and abs(cut["cuts"][0]["t"] - 4.0) < 0.5
    smooth = qc.cuts_of(clip_of((layouts(1), 8, True)))
    assert smooth["cuts"] == [] and smooth["min"] > qc.CUT_NCC
    found = qc.failures([], cut, judge_cut=0.19)
    assert found[0]["check"] == "cut" and found[0]["retake"] and "0.19" in found[0]["evidence"]


def test_the_takes_record_retakes_only_on_sure_failures_and_keeps_notes(tmp_path, monkeypatch):
    f = frames(tmp_path)
    monkeypatch.setattr(ap, "looker", lambda: Looker(table={"touches the wick": 0.1}, picks={"inside the lantern room": 0.01}))
    monkeypatch.setattr(qc, "hard_cuts", lambda clip, ffmpeg="ffmpeg": {"cuts": []})
    monkeypatch.setattr(ap, "previous_last", lambda run, seg, folder: None)
    record = ap.qc_take({"id": "r1"}, {"index": 1, "id": "S0001", "qc": PLAN}, tmp_path / "take1.mp4", tmp_path, 1, f)
    saved = json.loads((tmp_path / "qc-take1.json").read_text())
    actions = {x["check"]: x["action"] for x in record["findings"]}
    assert actions == {"end place": "retake", "contact": "note"}
    assert saved["described"].startswith("an old man") and saved["notes"][0].startswith("contact:")


def test_a_plan_without_checks_asks_nothing(tmp_path):
    assert qc.clean_qc(None)["people"] == [] and qc.clean_qc({"start": "lit", "end": ["  x  "]})["end"] == ["x"]
    look = Looker()
    assert qc.take_checks(look, qc.clean_qc(None), frames(tmp_path)) == [] and not look.asked


def test_a_continuing_piece_opens_on_the_last_frame_and_a_retake_keeps_the_anchor():
    run = {"settings": {"keyframes": "on"}, "seed": {}}
    assert ap.keyframes_for(run, 1) and ap.keyframes_for(run, 3, scene_change=True)
    assert not ap.keyframes_for(run, 2) and not ap.keyframes_for(run, 2, scene_change=False)


def test_a_continuing_piece_inherits_its_first_frame_and_vague_states_are_notes(tmp_path):
    # baseline film r1107-175548-17aa8: piece 3 opened on piece 2's glowing lamp (seam corr 0.971); "lamp unlit" 0.001
    # there is a note for the Director, not a retake (piece 2 cannot be sent back); "iron railing: untouched" 0.007 too
    results = [{"check": "seam", "state": "s", "p": 0.99, "passed": True, "confident": False},
               qc.result("start state", "lamp: unlit, wick dark", "the first frame", 0.001, True),
               qc.result("carried", "Arthur is inside the lamp room", "the first frame", 0.001, True)]
    assert results[1]["confident"]  # where a piece opens, a lit lamp before the lighting retakes
    qc.inherit(results)
    assert not results[1]["confident"] and results[1]["inherited"] and not results[2]["confident"]
    assert not any(f["retake"] for f in qc.failures(results, {}))
    vague = qc.result("start state", "iron railing: untouched", "the first frame", 0.007, True)
    assert not vague["confident"]


class GhostEyes:
    """Ghost's answers as text: ratings and reasoned A/B lines; records how many pictures each request carried."""

    engine = "ghost"

    def __init__(self, rating="0", letter_for_right=True):
        self.rating, self.right, self.sent = rating, letter_for_right, []

    def look(self, system, frames, sheet, text):
        self.sent.append(len(frames))
        if system == qc.RATE_SYSTEM:
            return self.rating, "ghost"
        if system == qc.PICK_SYSTEM:
            a = text.split("\nA: ")[1].split("\nB: ")[0]
            says_right = "upward" in a  # the right one in this fake is the climb
            return ("frame 1: ... \nANSWER: " + ("A" if says_right == self.right else "B")), "ghost"
        return "a man on a stair", "ghost"


def test_ghost_looks_at_one_picture_a_request_and_tiles_the_frames(tmp_path, monkeypatch):
    tiled = []
    monkeypatch.setattr(qc, "tile", lambda pictures, target, ffmpeg="ffmpeg": tiled.append(len(pictures)) or target.write_bytes(b"P") or target)
    eyes = GhostEyes(rating="0")
    look = qc.GhostLooker(eyes, tmp_path)
    f = frames(tmp_path)
    assert look.yes([f[0]], "the oil lamp is unlit") == 0.0  # Ghost on the lit-lamp keyframe: 0 of 10
    assert look.pick([f[0], f[4], f[-1]], "Which?", "he climbs upward", "he walks down") == 1.0  # both orders right
    assert look.pick([f[0], f[4], f[-1]], "Which?", "he climbs upward", "he walks down") == 1.0
    wrong = qc.GhostLooker(GhostEyes(letter_for_right=False), tmp_path)
    assert wrong.pick([f[0], f[4], f[-1]], "Which?", "he climbs upward", "he walks down") == 0.0
    assert set(eyes.sent) == {1} and tiled  # never more than one picture a request; the three frames tiled once


class StrictEyes:
    """Ghost under the strict looker: a description, then A/B lines with a RATING; ``holds`` decides the true side."""

    engine = "ghost"

    def __init__(self, holds=True, rating=10, flip=False):
        self.holds, self.rating, self.flip, self.sent, self.described = holds, rating, flip, [], 0

    def look(self, system, frames, sheet, text):
        self.sent.append(len(frames))
        if system == qc.SEE_SYSTEM:
            self.described += 1
            return "a man on a stair, the lamp dark", "ghost"
        if system == qc.COUNT_SYSTEM:
            return "1", "ghost"
        assert system == qc.STRICT_SYSTEM and "described before this question" in text  # the description comes first
        a = text.split("\nA: ")[1].split("\nB: ")[0]
        a_is_right = a.startswith("Yes:") or "upward" in a
        pick_a = a_is_right == self.holds
        if self.flip:  # leans to one letter whatever the order
            pick_a = True
        return f"checked.\nRATING: {self.rating}\nANSWER: {'A' if pick_a else 'B'}", "ghost"


def test_the_strict_ghost_passes_only_on_both_orders_and_a_9_or_10(tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "tile", lambda pictures, target, ffmpeg="ffmpeg": target.write_bytes(b"P") or target)
    f = frames(tmp_path)
    sure = StrictEyes(holds=True, rating=10)
    look = qc.StrictGhostLooker(sure, tmp_path)
    assert look.yes([f[0]], "the oil lamp is unlit") == 1.0
    assert look.pick([f[0], f[4], f[-1]], "Which?", "he climbs upward", "he walks down") == 1.0
    assert sure.described == 2 and set(sure.sent) == {1}  # described once a picture; one picture a request
    assert qc.StrictGhostLooker(StrictEyes(holds=True, rating=8), tmp_path).yes([f[0]], "the lamp is lit") == 0.45  # an 8 fails
    assert qc.StrictGhostLooker(StrictEyes(flip=True), tmp_path).yes([f[0]], "the lamp is lit") == 0.3  # orders disagree: fails
    no = qc.StrictGhostLooker(StrictEyes(holds=False, rating=10), tmp_path).yes([f[0]], "the lamp is lit")
    assert no == 0.0 and no <= qc.CONFIDENT["start state"]  # false both ways: a confident failure
    r = qc.result("end state", "lamp: lit", "the last frame", 0.45, 0.45 < qc.YES)
    assert not r["passed"] and not r["confident"]  # a note for the Director, not a retake


def test_rich_shots_add_the_director_rule_only_when_set():
    from gma.ap import autopilot as ap
    assert ap.director_system({"settings": {}}) == ap.DIRECTOR_SYSTEM
    rich = ap.director_system({"settings": ap.clean_settings({"rich_shots": True})})
    assert rich.startswith(ap.DIRECTOR_SYSTEM) and "camera MOVE" in rich and "music" in rich and "pace.beats" in rich


class Counter(Looker):
    """A looker that counts people: ``counts`` by the picture's file name, 1 otherwise."""

    def __init__(self, counts=None, **kw):
        super().__init__(**kw)
        self.counts = counts or {}

    def count(self, picture):
        return self.counts.get(Path(picture[1]).name, 1)


def test_people_are_counted_against_the_cast_and_one_person_plainly_shown_passes(tmp_path):
    f = frames(tmp_path)
    one = [r for r in qc.take_checks(Counter(two=0.9), PLAN, f) if r["check"] == "people"]
    assert len(one) == 1 and one[0]["passed"] and one[0]["counted"] == [1, 1, 1]  # "two or more" is not asked of a counter
    doubled = [r for r in qc.take_checks(Counter(counts={"f4.png": 2, "f7.png": 2}), PLAN, f) if r["check"] == "people"][0]
    assert not doubled["passed"] and not doubled["confident"]  # 2 of 3 frames: a note
    everywhere = [r for r in qc.take_checks(Counter(counts={"f0.png": 2, "f4.png": 2, "f7.png": 2}), PLAN, f) if r["check"] == "people"][0]
    assert everywhere["confident"]  # a second copy in every frame: a retake
    pair = dict(PLAN, people=[{"who": "Ewan", "count": 1}, {"who": "a girl", "count": 1}])
    assert [r for r in qc.take_checks(Counter(counts={"f0.png": 2, "f4.png": 2, "f7.png": 2}), pair, f) if r["check"] == "people"][0]["passed"]


def test_a_take_that_keeps_its_opening_picture_is_not_asked_its_start_again(tmp_path):
    f = frames(tmp_path)
    grey = np.tile(np.linspace(0, 255, 64, dtype=np.float32), (36, 1))
    import gma.ap.qc as q
    original = q.picture_grey
    q.picture_grey = lambda path, ffmpeg="ffmpeg": grey
    try:
        look = Looker(table={"unlit": 0.0})  # the strict looker's false start: it would fail
        opened = qc.take_checks(look, PLAN, f, ["Arthur stands at the stair foot"], opens_on=f[0][1])
        checks = {r["check"] for r in opened}
        assert "opening" in checks and not checks & {"start state", "start place", "carried"}
        assert next(r for r in opened if r["check"] == "opening")["passed"]
        q.picture_grey = lambda path, ffmpeg="ffmpeg": grey if Path(path).name != "f0.png" else grey[::-1, ::-1].copy()
        jumped = qc.take_checks(look, PLAN, f, [], opens_on=tmp_path / "kf.png") if (tmp_path / "kf.png").write_bytes(b"P") else None
        assert {"opening", "start state"} <= {r["check"] for r in jumped}  # left its picture: the start is asked again
    finally:
        q.picture_grey = original
