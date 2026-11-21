# Ported from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman), the AutoPilot
# branch, tests/test_autopilot_continuity.py at 6cc6e67c5e138499ea2c17183eda261b710d1217. Apache-2.0:
# LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: imports from the port (gma/ap) and its stand-in client.
"""AutoPilot's continuity bible and its checks (app/continuity.py, autopilot.check_continuity): the loop of
tests/test_autopilot.py with a scripted Director and a stand-in Cosmos that answers the checklist, the second look
and the film-level pass. No GPU, no network, no model."""
import json

import pytest

from test_autopilot import BEATS, HOME, SEED, Director, Watcher, autopilot, client, go, loop, new_run, shot  # noqa: F401

continuity = autopilot.continuity

# From the 60 s Viking test film (the first long real run): what the QA gate and the watcher said on segments 2 and 10
VIKING = {
    "segment_2_audio": {"result": "fail", "heard": "Eric!", "wanted": "Eirik!", "note": "0 of the line's 1 words heard"},
    "segment_10": [
        {"qa": {"passed": True, "identity": {"result": "pass", "note": "local model: The same man with red hair and brown coat appears in all three frames"}},
         "drift_check": {"same": False, "why": "The description of the copper-haired man changes across frames, indicating different individuals."}},
        {"qa": {"passed": False, "identity": {"result": "fail", "note": "local model: The main subject's wardrobe changes from brown to red across the frames, violating the consistency requirement."}}},
    ],
    "watcher_notes": ["Remains the tall Viking man in chain mail and dented iron helmet throughout the sequence.",
                      "Maintains her Viking woman appearance in a green dress and brooch-pinned shawl.",
                      "Dark clouds, lightning, and rough sea persist as the storm intensifies."],
}


def bible_of(run):
    return json.loads((autopilot.run_dir(run["id"]) / "bible.json").read_text())


def live(bible, index, subject, key):
    return [f["value"] for f in continuity.current(bible, index, subject) if f["key"] == key]


# ---------- the bible ----------

class EraDirector(Director):
    """Segment 3 jumps fifteen years to Boston: a new scene with an era label and its rules."""

    def __call__(self, run, system, prompt):
        self.asked.append(prompt)
        n = len(autopilot.committed(run))
        beat = ["The longship pitches in the storm", "A wave throws the boy overboard", "Years later Eirik walks the docks of Boston",
                "Eirik faces a redcoat patrol at the quay"][n]
        answer = shot(beat, scene="Boston harbour" if n >= 2 else "The longship", change=n in (0, 2))
        if n == 0:
            answer.update(era="Norse seas, 1758", era_rules=["wool, furs, iron and wood"])
        if n == 2:
            answer.update(era="Boston, 1773", era_rules=["tricorn hats, muskets, brick and cobblestones", "out of place: anything electric"])
        return json.dumps(answer)


def test_the_bible_versions_facts_across_an_era_change(loop, monkeypatch):
    director = EraDirector()
    watcher = Watcher(present="Eirik", facts={1: [{"category": "identity", "subject": "Eirik", "key": "scar", "value": "small crescent scar, left cheekbone"},
                                                  {"category": "wardrobe", "subject": "Eirik", "key": "outfit", "value": "rough undyed wool tunic, soaked"}]})
    monkeypatch.setattr(autopilot, "ask_text", director)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    seed = {**SEED, "characters": [{"name": "Eirik the boy", "look": "a ten-year-old with copper braids", "segments": "1-2"},
                                   {"name": "Eirik the man", "look": "a man of twenty-five with copper braids and a brown frock coat", "segments": "3-"}]}
    said = client.post("/api/autopilot/runs", json={"name": "Saga", "seed": seed, "settings": {"segments": 4, "continuity": {"film_pass": False}, "arc": False, "location_anchor": False}})
    assert said.status_code == 200, said.text
    run = go(autopilot.load(said.json()["id"]))
    assert run["state"] == "done"
    bible = bible_of(run)
    assert [(e["id"], e["from"], e["label"]) for e in bible["eras"]] == [(1, 1, "Norse seas, 1758"), (2, 3, "Boston, 1773")]
    assert "eirik" in bible["subjects"] and bible["subjects"]["eirik"]["names"] == ["Eirik the boy", "Eirik the man"]  # one person, two ages
    # each era's sheet only in its own segments; the scar (film-scoped) carries over, the tunic (era-scoped) does not
    assert live(bible, 2, "eirik", "look") == ["a ten-year-old with copper braids"]
    assert live(bible, 3, "eirik", "look") == ["a man of twenty-five with copper braids and a brown frock coat"]
    assert live(bible, 4, "eirik", "marks") == ["small crescent scar, left cheekbone"]
    # the man's sheet brings its own wardrobe line for its era (continuity.wardrobe_line), canonical from segment 3
    assert live(bible, 2, "eirik", "wardrobe") == ["rough undyed wool tunic, soaked"] and live(bible, 3, "eirik", "wardrobe") == ["a brown frock coat"]
    tunic = next(f for f in bible["facts"] if f["key"] == "wardrobe" and "tunic" in f["value"])
    assert (tunic["from"], tunic["to"], tunic["era"]) == (1, 2, 1)  # closed, not overwritten: the history stays
    # the eyes check segment 3 against the new era; the Director of segment 4 reads it
    check3 = watcher.checklists[2]
    assert "Boston, 1773" in check3 and "anything electric" in check3 and "crescent scar" in check3 and "wool tunic" not in check3
    asked4 = director.asked[3]
    assert "Boston, 1773" in asked4 and "crescent scar" in asked4 and "a man of twenty-five" in asked4 and "wool tunic" not in asked4
    # a snapshot of the bible after every segment (a branch starts from one)
    assert all((autopilot.run_dir(run["id"]) / "bible" / f"S{i:04d}.json").is_file() for i in range(1, 5))


def test_each_speaker_keeps_one_voice_and_the_180_degree_rule_is_measured():
    bible = continuity.new({"characters": [{"name": "Vex", "look": "orange jacket"}, {"name": "Ada", "look": "grey coat"}]})
    first = continuity.voices(bible, 1, [{"speaker": "Vex", "voice": "a low rasp", "text": "Go."}])
    later = continuity.voices(bible, 4, [{"speaker": "Vex", "voice": "a bright chirp", "text": "Now."}])
    assert first[0]["voice"] == later[0]["voice"] == "a low rasp"
    continuity.absorb(bible, 2, {"checks": {}, "findings": [], "positions": [{"who": "Vex", "last": "left", "moving": "right"}, {"who": "Ada", "last": "right"}]}, [])
    flips = continuity.screen_flip(bible, 3, [{"who": "Vex", "first": "right", "moving": "left"}, {"who": "Ada", "first": "left"}])
    assert len(flips) == 2 and all(f["category"] == "spatial" and f["status"] == "minor" for f in flips)
    assert "swap sides" in flips[0]["evidence"] and "moved screen right" in flips[1]["evidence"]
    assert continuity.screen_flip(bible, 3, [{"who": "Vex", "first": "left"}, {"who": "Ada", "first": "right"}]) == []
    continuity.planned(bible, 3, {"scene_change": True, "scene": "Roof"}, {})  # a new scene: positions start fresh
    assert continuity.screen_flip(bible, 3, [{"who": "Vex", "first": "right"}, {"who": "Ada", "first": "left"}]) == []


# ---------- what findings do ----------

class FixingDirector(Director):
    """Answers a retake request with the shot corrected."""

    def __call__(self, run, system, prompt):
        if "FAILED ITS CONTINUITY CHECK" in prompt:
            self.asked.append(prompt)
            n = len(autopilot.committed(run))
            answer = shot(BEATS[n])
            answer["shot"]["action"] = "Vex lands mid-air from the jump already under way, the core in her right hand."
            return json.dumps(answer)
        return super().__call__(run, system, prompt)


def test_a_major_finding_retakes_with_the_corrected_prompt_and_a_new_seed(loop, monkeypatch):
    director = FixingDirector()
    major = {"action": {"verdict": "major", "frame": 3, "subject": "Vex", "evidence": "the jump restarts from the platform instead of continuing mid-air"}}
    watcher = Watcher(checks={2: major})
    monkeypatch.setattr(autopilot, "ask_text", director)
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, continuity={"film_pass": False}))
    second = run["segments"][1]
    assert [t["seed"] for t in second["takes"]] == [2, 3] and second["takes"][0]["passed"] is False and second["passed"] is True
    assert "continuity, action: the jump restarts" in second["takes"][0]["reasons"][0]
    found = second["takes"][0]["continuity"]["checks"]["action"]
    assert found["status"] == "major" and found["second_look"]["confirmed"] is True and found["action"] == "retake"
    # the second look saw the frame named beside the last frame of the segment before
    prompt, pictures = watcher.looks[0]
    assert "the jump restarts" in prompt and len(pictures) == 2
    # the Director rewrote the shot from the finding; the new take renders it at the next seed
    assert "FAILED ITS CONTINUITY CHECK" in director.asked[-1] and "(frame 3): the jump restarts" in director.asked[-1]
    bodies = loop["queue"].bodies
    assert [b["params"]["seed"] for b in bodies] == [1, 2, 3]
    assert "already under way" in bodies[2]["params"]["prompt"] and "already under way" not in bodies[1]["params"]["prompt"]
    assert second["fixed"][0]["findings"][0]["category"] == "action"
    assert (autopilot.run_dir(run["id"]) / "segments" / "S0002" / "director-take2.json").is_file()
    assert autopilot.DEFAULTS["retakes"] == 2


def test_when_every_take_fails_the_best_one_is_kept_with_a_warning(loop, monkeypatch):
    monkeypatch.setattr(autopilot, "ask_text", FixingDirector())
    action = {"verdict": "major", "frame": 2, "evidence": "the jump restarts from the platform"}
    minor = lambda what: {"verdict": "minor", "frame": 4, "evidence": f"{what} slightly off"}
    watcher = Watcher(checks={2: {"action": action, "wardrobe": minor("jacket")},
                              3: {"action": action, "wardrobe": minor("jacket"), "props": minor("core")},
                              4: {"action": action, "wardrobe": minor("jacket"), "props": minor("core"), "lighting": minor("light")}})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(2, continuity={"film_pass": False}))
    second = run["segments"][1]
    assert len(second["takes"]) == 3 and second["passed"] is False
    assert second["output"].endswith("take1.mp4") and second["seed"] == 2  # take 1 had the fewest problems
    assert "kept take 1 of 3" in second["reasons"][-1]
    assert any("did not pass" in line["text"] for line in run["log"])


def test_a_minor_finding_becomes_the_directors_correction(loop, monkeypatch):
    watcher = Watcher(checks={1: {"props": {"verdict": "minor", "frame": 4, "subject": "Vex", "evidence": "the data core is in her left hand, not the right"}}})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = new_run(2, continuity={"film_pass": False})
    run["settings"]["segments"] = 1
    run = go(run)
    first = run["segments"][0]
    assert len(first["takes"]) == 1 and first["passed"] is True  # a minor finding is never a retake
    assert [c["category"] for c in bible_of(run)["corrections"]] == ["props"]
    run["settings"]["segments"] = 2
    run = go(run)
    asked = loop["director"].asked[1]
    assert "Continuity corrections from the last checks" in asked and "data core is in her left hand" in asked
    assert bible_of(run)["corrections"] == []  # props checked ok in segment 2: put right


def test_a_false_positive_is_filtered_by_the_second_look(loop, monkeypatch):
    watcher = Watcher(checks={1: {"identity": {"verdict": "major", "frame": 5, "subject": "Vex", "evidence": "the courier's face changes as she mounts the bike",
                                               "box": [0.3, 0.1, 0.6, 0.6]},
                                  "wardrobe": {"verdict": "major", "frame": 0, "evidence": "the jacket"},
                                  "props": {"verdict": "minor", "frame": 2, "evidence": "the core remains the same glowing blue"}}},
                      second=[False])
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(1, continuity={"film_pass": False}))
    seg = run["segments"][0]
    checks = seg["takes"][0]["continuity"]["checks"]
    assert checks["identity"]["status"] == "filtered" and checks["identity"]["second_look"] == {"confirmed": False, "why": "looked closer",
                                                                                                "pictures": [f"{run['id']}/segments/S0001/look-take1-identity.png"]}
    assert checks["wardrobe"]["status"] == "no_evidence" and checks["props"]["status"] == "benign"
    assert len(watcher.looks) == 1  # only a major finding with evidence is looked at again
    assert (HOME / f"{run['id']}/segments/S0001/look-take1-identity.png").is_file()  # the crop of where it is
    assert len(seg["takes"]) == 1 and seg["passed"] is True and bible_of(run)["corrections"] == []


@pytest.mark.parametrize("strictness", ["lenient", "standard", "strict"])
def test_strictness_decides_which_majors_retake(strictness):
    """Each strictness against one take with a confirmed major in props, story and identity in turn."""

    for category in ("props", "story", "identity"):
        entry = {"category": category, "status": "major", "frame": 1, "evidence": "x"}
        assert continuity.act(entry, {"strictness": strictness}) == ("retake" if category in continuity.STRICTNESS[strictness]["retake"] else "correct")
    unsure = continuity.settle({"category": "identity", "status": "major", "frame": 1, "evidence": "x"}, {"error": "no answer"}, {"strictness": strictness})
    assert unsure["status"] == {"lenient": "filtered", "standard": "minor", "strict": "major"}[strictness]


def test_settings_choose_strictness_categories_and_the_budget():
    said = client.post("/api/autopilot/runs", json={"seed": SEED, "settings": {"continuity": {"strictness": "wild"}}})
    assert said.status_code == 400 and "strictness" in said.text
    run = new_run(1, retakes=4, continuity={"strictness": "strict", "categories": ["identity", "era", "audio"], "film_pass": False})
    assert run["settings"]["retakes"] == 4 and run["settings"]["continuity"] == {"strictness": "strict", "categories": ["identity", "era", "audio"], "film_pass": False}
    assert client.get(f"/api/autopilot/runs/{run['id']}").json()["bible"]["subjects"]["vex"]["kind"] == "character"


def test_categories_left_out_are_not_asked(loop, monkeypatch):
    watcher = Watcher()
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    go(new_run(1, continuity={"categories": ["identity", "era"], "film_pass": False}))
    asked = watcher.checklists[0]
    assert "- identity:" in asked and "- era:" in asked and "- props:" not in asked and "- lighting:" not in asked
    off = Watcher()
    monkeypatch.setattr(autopilot, "ask_vision", off)
    run = go(new_run(1, continuity={"categories": [], "film_pass": False}))
    assert off.checklists == [] and "no category" in run["segments"][0]["continuity"]["skipped"]


# ---------- recursive memory ----------

def test_folding_the_scene_graph_keeps_every_canonical_fact(loop, monkeypatch):
    monkeypatch.setattr(autopilot, "COMPACT_AT", 2)
    monkeypatch.setattr(autopilot, "FOLD", 2)
    watcher = Watcher(facts={1: [{"category": "identity", "subject": "Vex", "key": "scar", "value": "thin scar through the left eyebrow"},
                                 {"category": "story", "subject": "core", "value": "the data core was cracked in the fall"}]})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(5, continuity={"film_pass": False}))
    folder = autopilot.run_dir(run["id"])
    memory = json.loads((folder / "memory.json").read_text())
    assert memory["graph"]["chapters"] and memory["archive"]  # event detail folded
    assert "swerves past obstacle 1" not in json.dumps(memory["graph"]["events"])
    bible = bible_of(run)
    assert live(bible, 6, "vex", "marks") == ["thin scar through the left eyebrow"]
    story = [s["text"] for s in bible["story"]]
    assert "the data core was cracked in the fall" in story and all(any(f"obstacle {i}" in t for t in story) for i in range(1, 6))
    asked = next(p for p in loop["director"].asked if "You are writing segment 5" in p)  # after two folds
    assert "thin scar through the left eyebrow" in asked and "swerves past obstacle 1" in asked and "cracked in the fall" in asked


# ---------- the film-level pass ----------

def test_the_film_level_pass_reports_drift_and_retakes_nothing(loop, monkeypatch):
    watcher = Watcher(film={"same": False, "different": [2, 9], "why": "segment 2 shows another face"},
                      world={"issues": [{"category": "props", "segments": [1, 3], "evidence": "the data core turns from blue to red"},
                                        {"category": "other", "segments": [2], "evidence": "everything is consistent"}]})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(3))
    report = run["film_check"]
    assert report["identity"] == [{"subject": "Vex", "era": 1, "segments": [1, 2, 3], "same": False, "different": [2], "why": "segment 2 shows another face"}]
    assert report["world"] == [{"category": "props", "evidence": "the data core turns from blue to red", "segments": [1, 3]}]
    assert len(loop["queue"].bodies) == 3  # a report only
    assert json.loads((autopilot.run_dir(run["id"]) / "film-check.json").read_text())["identity"][0]["different"] == [2]
    assert "SAME CHARACTER ACROSS THE FILM" in watcher.films[0] and "segment 1, segment 2, segment 3" in watcher.films[0]
    # again by hand, in the background
    said = client.post(f"/api/autopilot/runs/{run['id']}/film-check", json={})
    assert said.status_code == 200, said.text
    import threading
    for thread in threading.enumerate():
        if thread.name == "autopilot-film-" + run["id"]:
            thread.join(30)
    assert autopilot.load(run["id"])["film_check"]["state"] == "done"


# ---------- regressions from the 60 s Viking test film ----------

def test_a_name_heard_as_a_word_that_sounds_alike_is_minor(loop, monkeypatch):
    """Segment 2: the gate heard "Eric!" for "Eirik!" and failed both takes; the same words by ear are a minor note."""

    audio = VIKING["segment_2_audio"]
    assert continuity.sound_key("Eirik") == continuity.sound_key("Eric") == "E620"
    assert continuity.heard_alike(audio["wanted"], audio["heard"]) and not continuity.heard_alike("Eirik!", "Olaf!")
    assert not continuity.heard_alike("Hold the line, Eirik", "we'll be right back")
    monkeypatch.setattr(autopilot, "gate", lambda clip, prompt, sheet=None, brain=None: {"passed": False, "audio": dict(audio), "identity": {"result": "pass"}})
    run = go(new_run(1, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert len(seg["takes"]) == 1 and seg["passed"] is True and seg["reasons"] == []
    note = next(f for f in seg["takes"][0]["continuity"]["findings"] if f["category"] == "audio")
    assert note["status"] == "minor" and note["action"] == "note" and '"Eirik!" was heard as "Eric!"' in note["evidence"]


def test_a_coat_that_changes_colour_is_retaken_with_the_finding_and_the_best_take_kept(loop, monkeypatch):
    """Segment 10: drift on take 1, then the identity gate saw the coat turn from brown to red, and the old loop kept
    the last take after one plain retake. Now the Director gets each finding, the budget is two, the best take stays."""

    takes = VIKING["segment_10"]
    answers = iter([takes[0]["qa"], takes[1]["qa"], takes[1]["qa"]])
    monkeypatch.setattr(autopilot, "gate", lambda clip, prompt, sheet=None, brain=None: {**next(answers), "audio": {"result": "pass"}})
    watcher = Watcher(drift={1})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    director = FixingDirector()
    monkeypatch.setattr(autopilot, "ask_text", director)
    run = go(new_run(1, continuity={"film_pass": False}))
    seg = run["segments"][0]
    assert len(seg["takes"]) == 3 and all(not t["passed"] for t in seg["takes"])
    retakes = [p for p in director.asked if "FAILED ITS CONTINUITY CHECK" in p]
    assert len(retakes) == 2 and "identity" in retakes[0] and "wardrobe: local model: The main subject's wardrobe changes from brown to red" in retakes[1]
    assert seg["output"].endswith("take3.mp4") and "kept take 3 of 3" in seg["reasons"][-1]  # equals: the latest, with the most corrections
    assert continuity.gate_category(takes[1]["qa"]["identity"]["note"]) == "wardrobe"


def test_notes_that_say_nothing_changed_are_not_problems():
    for note in VIKING["watcher_notes"][:2]:
        assert continuity.BENIGN.search(note), note
    assert not continuity.BENIGN.search("the scar is on the right cheek now")


def test_each_era_gets_only_its_own_style_and_a_setting_from_another_is_a_retake(loop, monkeypatch):
    """The Viking test film: one style named the sea, the island and 1770s Boston, and Boston's red-brick streets came
    into the storm at sea. Styles now hold for their segments only, and the checklist asks whether the place is the one
    asked for; a street in the sea shot is a major finding that renders the segment again."""

    styles = [{"style": "storm-dark North Sea, longships, lightning", "segments": "1-2"},
              {"style": "1770s Boston, red-brick streets, gas-less lanterns", "segments": "3-"}]
    watcher = Watcher(checks={1: {"setting": {"verdict": "major", "frame": 4, "evidence": "red-brick Boston houses stand on the horizon of the open sea"}}})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    run = go(new_run(3, style="35 mm film grain", styles=styles, continuity={"film_pass": False}))
    assert run["settings"]["styles"] == [{"style": styles[0]["style"], "from": 1, "to": 2}, {"style": styles[1]["style"], "from": 3}]
    prompts = [b["params"]["prompt"] for b in loop["queue"].bodies]
    assert len(prompts) == 4  # segment 1 twice: its first take showed the wrong setting
    for sea in prompts[:3]:
        assert "North Sea" in sea and "film grain" in sea and "Boston" not in sea
    assert "Boston" in prompts[3] and "North Sea" not in prompts[3]
    assert "Boston" not in loop["director"].asked[0] and "North Sea" in loop["director"].asked[0]
    first = run["segments"][0]
    assert first["takes"][0]["continuity"]["checks"]["setting"]["action"] == "retake" and first["passed"] is True
    assert "- setting: Does the setting match" in watcher.checklists[0] and "The setting it asked for:" in watcher.checklists[0]
    assert "Its visual style: 35 mm film grain storm-dark North Sea" in watcher.checklists[0]
    assert autopilot.style_at(run["settings"], 3) == "35 mm film grain 1770s Boston, red-brick streets, gas-less lanterns"


def test_an_outline_run_opens_on_a_checked_keyframe_and_continues_from_its_last_frame(loop, monkeypatch):
    """Fully autonomous by default: with an outline the film opens on a keyframe the Director wrote and the eyes
    checked (the checklist's finding is a note, never a redraw: only a confident self-check failure draws it again);
    a piece that continues opens on the last frame of the piece before, kept, with no keyframe of its own."""

    watcher = Watcher(keyframes={1: {"setting": {"verdict": "major", "frame": 1, "evidence": "a red-brick street fills the sea shot"}}})
    monkeypatch.setattr(autopilot, "ask_vision", watcher)
    said = client.post("/api/autopilot/runs", json={"name": "Saga", "seed": {**SEED, "outline": ["Vex launches off the platform", "Vex threads the trams"]},
                                                    "settings": {"segments": 2, "style": "teal night", "continuity": {"film_pass": False}, "arc": False, "location_anchor": False}})
    assert said.status_code == 200, said.text
    run = autopilot.load(said.json()["id"])
    assert (run["settings"]["preset"], run["settings"]["seconds"], run["settings"]["keyframes"]) == ("small", 10, "on")
    run = go(run)
    kinds = [b["kind"] for b in loop["queue"].bodies]
    assert kinds == ["scout", "video", "video"]
    first, second = run["segments"]
    assert [k["take"] for k in first["keyframes"]] == [1] and first["keyframe"] == first["keyframes"][0]["path"]
    assert first["keyframes"][0]["continuity"]["findings"][0]["category"] == "setting"  # recorded, not redrawn
    videos = [b["params"] for b in loop["queue"].bodies if b["kind"] == "video"]
    assert videos[0]["image"] == first["keyframe"] and videos[1]["first_frame_from"] == first["output"]
    assert [s["mode"] for s in run["segments"]] == ["keyframe", "previous"]
    assert loop["queue"].bodies[0]["params"]["prompt"].startswith("teal night") and loop["queue"].bodies[0]["params"]["preset"] == "small"
    assert "fill \"keyframe\"" in loop["director"].asked[0]


def test_the_director_keeps_prompts_short():
    run = {"settings": dict(autopilot.DEFAULTS), "seed": {}, "segments": []}
    long = shot("Vex climbs")
    long["shot"]["action"] = " ".join(["word"] * 70)
    assert "keep it to 60" in autopilot.director_problem(run, long, 1, None)
