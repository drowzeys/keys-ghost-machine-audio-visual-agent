"""The live stage: the director's reading of a note, and the stage service against a stand-in picture engine."""

import asyncio
import json
import struct
import sys
import textwrap

import pytest

from gma.live import director
from gma.live.stage import Refused, Stage

SCENE = "Medium shot, static camera. Mara, a woman with short black hair in a red raincoat, waits at a tram stop at night in the rain."


def test_the_cast_is_read_from_its_lines_and_folded_into_the_opening():
    text = "Wide shot of a pier at dusk.\nTom: a tall man with a grey beard\nMara: a woman in a red raincoat"
    assert director.cast_of(text) == [{"name": "Tom", "look": "a tall man with a grey beard"}, {"name": "Mara", "look": "a woman in a red raincoat"}]
    opening = director.opening(text)
    assert opening.startswith("Wide shot of a pier at dusk.") and "Tom is a tall man with a grey beard." in opening and "Tom:" not in opening


def test_a_rewrite_is_read_and_a_line_needs_both_who_and_words():
    said = director.read('here: {"scene": "Close-up. Mara, a woman with short black hair in a red raincoat, smiles in the rain at night.", '
                         '"who": "Mara", "line": "I waited an hour.", "ack": "Moving in."}', SCENE, "closer, and Mara says how long she waited")
    assert said.by == "brain" and said.scene.startswith("Close-up.") and (said.who, said.line, said.ack) == ("Mara", "I waited an hour.", "Moving in.")
    half = director.read('{"scene": "Close-up. Mara, a woman with short black hair in a red raincoat, smiles in the rain.", "who": "Mara", "line": null}', SCENE, "closer")
    assert half.who is None and half.line is None


@pytest.mark.parametrize("reply", ["", "Sure, moving in!", '{"scene": "too short"}', '{"scene": 3}', "{broken"])
def test_a_reply_that_cannot_be_read_tacks_the_note_on(reply):
    said = director.read(reply, SCENE, "she walks to the camera.")
    assert said.by == "plain" and said.scene == SCENE + " Now: she walks to the camera."


def test_a_brain_that_fails_never_fails_the_note():
    class Away:
        async def complete(self, messages, **settings):
            raise RuntimeError("held for a render")
    said = asyncio.run(director.direct(Away(), SCENE, "make it dawn"))
    assert said.by == "plain" and said.scene.endswith("Now: make it dawn.")


ENGINE = textwrap.dedent('''
    import json, socket, struct, sys, time
    spec = json.loads(sys.argv[1]); server = socket.socket(socket.AF_UNIX); server.bind(spec["socket"]); server.listen(1)
    conn, _ = server.accept(); conn.settimeout(0.02)
    def say(header, payload=b""):
        head = json.dumps(header).encode(); conn.sendall(struct.pack(">I", len(head)) + head + payload)
    say({"type": "ready", "size": [64, 48], "fps": 16, "window": 12})
    f, live, buffer = 0, False, b""
    while True:
        try:
            piece = conn.recv(65536)
            if not piece: break
            buffer += piece
        except socket.timeout:
            pass
        while len(buffer) >= 4 and len(buffer) >= 4 + struct.unpack(">I", buffer[:4])[0]:
            size = struct.unpack(">I", buffer[:4])[0]; order = json.loads(buffer[4:4 + size]); buffer = buffer[4 + size:]
            if order["op"] == "start": f, live = 0, True; say({"type": "started", "seed": order["seed"], "prompt": 0})
            elif order["op"] == "prompt": say({"type": "prompt", "id": order["id"], "at": f, "ms": 1.0})
            elif order["op"] == "stop": live = False; say({"type": "stopped", "frames": f})
            elif order["op"] == "take": say({"type": "take", "name": order["name"], "state": "done", "file": order["name"] + ".mp4", "seconds": order["frames"] / 4, "took_s": 0.1, "script": order["script"]})
            elif order["op"] == "quit": sys.exit(0)
        if live:
            say({"type": "frame", "f": f, "ms": 200.0, "late_ms": 0, "prompt": 0, "sizes": [3, 3]}, b"abcdef"); f += 1
''')


class Brain:
    async def complete(self, messages, **settings):
        asked = messages[-1]["content"]
        if "The director says: " in asked:  # at the table: the note becomes a beat; a question to the cast is answered
            note = asked.rsplit("The director says: ", 1)[1]
            beats = json.loads(asked.split("Beats so far: ", 1)[1].split("\n", 1)[0])
            talk = note.startswith("Mara,")
            return json.dumps({"scene": asked.split("Opening picture:\n", 1)[1].split("\n", 1)[0],
                               "beats": beats if talk else [*beats, {"s": 0.5 * (len(beats) + 1), "note": note}],
                               "who": "Mara" if talk else None, "line": "Tired, and a little hopeful." if talk else None, "ack": "Noted."})
        note = asked.rsplit("Director's note: ", 1)[1]
        return json.dumps({"scene": f"Close-up. Mara, a woman with short black hair in a red raincoat, at night in the rain; {note}.",
                           "who": "Mara" if "say" in note else None, "line": "I waited an hour." if "say" in note else None, "ack": "Done."})


def make(tmp_path, **spec) -> Stage:
    script = tmp_path / "engine.py"
    script.write_text(ENGINE)

    def launcher(given):
        return [sys.executable, str(script), json.dumps({"socket": str(tmp_path / "run" / "stage.sock")})]
    return Stage({"run": str(tmp_path / "run"), "need_gb": 0, "floor_gb": 0, "settle_s": 0, "name": "ghost-live-test", **spec}, brain=Brain(), launcher=launcher)


def test_a_scene_plays_takes_notes_and_is_filmed_again_with_the_same_script(tmp_path):
    async def go():
        stage = make(tmp_path)
        stage._kill = lambda: asyncio.sleep(0)  # no container in the test
        queue = stage.watch()
        with pytest.raises(Refused):
            await stage.note("closer")  # nothing plays yet
        await stage.start(SCENE + "\nMara: a woman with short black hair in a red raincoat", seed=7)
        frames = 0
        while frames < 3:
            kind, what = await asyncio.wait_for(queue.get(), 5)
            if kind == "frame":
                size = struct.unpack(">I", what[:4])[0]
                assert json.loads(what[4:4 + size])["sizes"] == [3, 3] and what[4 + size:] == b"abcdef"
                frames += 1
        assert stage.phase == "live"
        said = await stage.note("move closer")
        assert said["by"] == "brain" and said["line"] is None and stage.scene.endswith("move closer.")
        said = await stage.note("Mara, say how long you waited", heard=True)
        assert said["who"] == "Mara" and said["line"] == "I waited an hour."
        line = [e for e in stage.log if e["kind"] == "line"][0]
        assert line["who"] == "Mara" and line["voice"] == stage.spec["voices"][0]
        await asyncio.sleep(0.2)
        state = await stage.stop()
        assert state["phase"] == "ready" and state["frames"] >= 3
        # the script the take is filmed from: the opening, then each note as tacked on and as rewritten, in order
        assert stage.script[0] == [0, stage.script[0][1]] and len(stage.script) == 5
        assert [f for f, _ in stage.script] == sorted(f for f, _ in stage.script)
        await stage.take()
        for _ in range(100):
            if stage.phase == "ready":
                break
            await asyncio.sleep(0.05)
        take = stage.takes[-1]
        assert take["state"] == "done" and take["script"] == stage.script and take["directions"] == 4
        saved = json.loads((tmp_path / "run" / "takes" / f"{take['name']}.json").read_text())
        assert saved["seed"] == 7 and saved["script"] == stage.script
        await stage.shutdown()
        assert stage.phase == "off"
    asyncio.run(go())


def test_the_stage_is_not_started_without_the_memory_it_needs(tmp_path):
    async def go():
        stage = make(tmp_path, need_gb=100000)
        with pytest.raises(Refused, match="not enough free memory"):
            await stage.start(SCENE)
        assert stage.phase == "off" and stage.process is None
    asyncio.run(go())


def test_an_engine_on_another_machine_is_started_there_by_ssh_with_that_machines_paths():
    from gma.live.stage import DEFAULTS, launch

    spec = {**DEFAULTS, "host": "10.0.0.2", "model": "ff-l", "lean": True, "claim": "~/CLAIM.txt",
            "there": {"home": "/home/other", "uid": 1234, "gid": 1234}}
    command = launch(spec)
    assert command[0] == "ssh" and command[-2] == "10.0.0.2"
    remote = command[-1]
    assert "--network none" in remote and "--user 1234:1234" in remote
    assert "/home/other/ff-try/repo:/work:ro" in remote and "/home/other/ghost-studio/live-engine:/engine:ro" in remote
    assert '"model": "ff-l"' in remote and '"send_takes": true' in remote and "~" not in remote.replace("~/CLAIM", "")
    assert remote.count("/home/other/CLAIM.txt") == 2  # a line when it starts, a line when it has gone
    assert launch({**DEFAULTS})[0] == "docker"  # here: no ssh, and takes are read from the shared folder


@pytest.mark.parametrize("said, order", [("Action!", "action"), ("And... action", "action"), ("Cut.", "cut"), ("cut, cut!", "cut"),
                                         ("That's a cut", "cut"), ("cut to a close-up of her face", None), ("she springs into action", None)])
def test_action_and_cut_are_orders_only_when_said_alone(said, order):
    assert director.command(said) == order


def test_a_heard_shout_is_taken_loosely_but_only_where_it_makes_sense():
    assert director.command("Got there.", heard=True, filming=True) == "cut"
    assert director.command("Got there.", heard=True, filming=False) is None  # at the table it is a note
    assert director.command("Got there.", heard=False, filming=True) is None  # typed words are taken as typed
    assert director.command("Auction!", heard=True, filming=False) == "action"
    assert director.command("got her to smile at him", heard=True, filming=True) is None


def test_a_rehearsal_plans_without_filming_then_action_films_the_plan_and_cut_stops_it(tmp_path):
    async def go():
        stage = make(tmp_path)
        watching = stage.watch()
        try:
            await table(stage)
        finally:
            stage.unwatch(watching)
            await stage.shutdown()

    async def table(stage):
        await stage.rehearse(SCENE + "\n\nMara: a woman with short black hair in a red raincoat")
        answer = await stage.note("Mara, how do you feel tonight?", heard=True)
        assert answer["line"] == "Tired, and a little hopeful." and answer["beats"] == []
        await stage.note("she walks toward the camera")
        await stage.note("push in to a close-up")
        assert [b["note"] for b in stage.table["beats"]] == ["she walks toward the camera", "push in to a close-up"]
        assert stage.frames == 0 and stage.phase != "live"  # nothing filmed at the table
        assert (await stage.note("And... action!"))["command"] == "action"
        for _ in range(400):
            if len(stage.script) >= 5:  # the opening, then each beat's interim and rewritten scene
                break
            await asyncio.sleep(0.02)
        assert (await stage.note("Cut!"))["command"] == "cut"
        assert stage.phase == "ready" and stage.frames > 0 and stage.table is not None
        assert "she walks toward the camera" in stage.script[2][1] and "push in to a close-up" in stage.script[4][1]
        first = stage.seed
        await stage.note("action")  # the same plan again: another try, the next seed
        assert stage.phase == "live" and stage.seed == first + 1
        await stage.note("cut")
        with pytest.raises(Refused):
            await stage.note("cut")
    asyncio.run(go())


class Renderer:
    """A stand-in for the app's renderer: stills and clips are small files, made after a few polls."""

    width, height = 864, 480

    def __init__(self, folder):
        self.folder, self.jobs, self.asked, self.cancelled = folder, {}, [], []

    def job(self, kind, **more):
        handle = f"{kind[0]}{len(self.jobs)}"
        self.jobs[handle] = {"kind": kind, "state": "rendering", "polls": 0, **more}
        self.asked.append((kind, more))
        return handle

    async def keyframe(self, prompt, seed, references=None, size=None):
        return self.job("keyframe", prompt=prompt, references=references)

    async def submit(self, clip):
        return self.job("video", clip=clip)

    async def room(self, clip):
        return None

    async def poll(self, handle):
        from voice_agent.video import Progress
        job = self.jobs[handle]
        job["polls"] += 1
        if job["state"] == "rendering" and job["polls"] > 1:
            out = self.folder / f"{handle}.{'png' if job['kind'] == 'keyframe' else 'mp4'}"
            out.write_bytes(b"made")
            job.update(state="done", output=str(out))
        return Progress(job["state"], 1, 8, "denoising", output=job.get("output"))

    async def fetch(self, output, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(open(output, "rb").read())

    async def cancel(self, handle):
        self.cancelled.append(handle)
        self.jobs[handle]["state"] = "cancelled"


def test_with_the_studios_engines_the_rehearsal_has_a_picture_and_action_films_the_plan_in_parts(tmp_path, monkeypatch):
    from gma.live import film

    async def join(self, parts, out, ffmpeg="ffmpeg"):
        out.write_bytes(b"".join(p.read_bytes() for p in parts))
    monkeypatch.setattr(film.Set, "join", join)
    monkeypatch.setattr(film.asyncio, "sleep", lambda s, real=asyncio.sleep: real(0.001))

    async def go():
        renderer = Renderer(tmp_path)
        stage = Stage({"run": str(tmp_path / "run"), "render": "h3", "anchors": False}, brain=Brain(), renderer=renderer)
        await stage.rehearse(SCENE + "\n\nMara: a woman with short black hair in a red raincoat")
        await stage.set.task  # the reference shot, then the loop where she listens and the one where she talks
        assert stage.set.still == "ref-1.png" and stage.set.loops == {"idle": "idle-1.mp4", "talk": "talk-1.mp4"}
        assert [kind for kind, _ in renderer.asked] == ["keyframe", "video", "video"] and stage.process is None  # no stream engine
        assert "<d>" in renderer.asked[2][1]["clip"]["prompt"] and renderer.asked[1][1]["clip"]["image_file"].endswith("ref-1.png")
        await stage.note("she walks toward the camera")       # 0.5 s: the first part
        for _ in range(11):
            await stage.note("filler beat")                    # ... up to 6 s: the last beats are in the second part
        stage.table["beats"][-1]["note"] = "Mara, say why you waited"
        assert (await stage.note("action"))["command"] == "action" and stage.phase == "filming"
        assert (await stage.note("look left"))["by"] == "queued"   # a note while it films goes into the next part to be filmed
        await stage.filming
        take = stage.takes[-1]
        assert take["state"] == "done" and take["parts"] == [take["name"] + "-p1.mp4", take["name"] + "-p2.mp4"]
        first, second = renderer.asked[3][1]["clip"], renderer.asked[4][1]["clip"]
        assert first["image_file"].endswith("ref-1.png") and second["opens_on"] == renderer.jobs["v3"]["output"]
        walked = next(e for e in stage.log if e["kind"] == "note" and e.get("planned") and e["text"] == "she walks toward the camera")
        assert walked["part"] == 1 and "look left" in first["prompt"] and "<d>[English] I waited an hour.</d>" in second["prompt"]
        assert stage.phase == "off" and (tmp_path / "run" / "takes" / take["file"]).is_file()
        await stage.note("action")                                # again: "cut" ends the take on the part being filmed
        await stage.note("cut")
        assert stage.phase == "filming" and stage.state()["wrapping"]
        for _ in range(400):
            if stage.phase != "filming":
                break
            await asyncio.sleep(0.02)
        assert stage.takes[-1]["state"] == "done" and stage.takes[-1]["cut"] and len(stage.takes[-1]["parts"]) == 1
        await stage.note("action")                                # and a second "cut" drops the part instead
        await stage.note("cut")
        await stage.note("cut")
        assert stage.takes[-1]["state"] == "failed" and stage.phase == "off"
        await stage.shutdown()
    asyncio.run(go())


# ----- the blocking preview and the open microphone -----

def test_a_floor_plan_from_the_brain_is_kept_on_the_floor_and_to_the_cast():
    from gma.live import blocking
    cast = [{"name": "Mara", "look": "a woman in a red raincoat"}, {"name": "Jon", "look": "a tall man"}]
    said = {"marks": [{"name": "lamp", "at": [-2, 40]}, {"at": [0, 0]}],
            "cast": [{"name": "mara", "start": [-1, -1], "facing": 450, "cues": [{"t": 9, "do": "turn", "facing": 90}, {"t": 4, "do": "walk", "to": [99, 2]},
                                                                                 {"t": 5, "do": "teleport"}, {"t": 6, "do": "fly", "secs": 99}, {"t": 7, "do": "run"}, "junk"]},
                     {"name": "Stranger", "start": [0, 0]}]}
    plan = blocking.checked(said, cast, [])
    assert [p["name"] for p in plan["cast"]] == ["Mara", "Jon"] and plan["marks"] == [{"name": "lamp", "at": [-2.0, 5.0]}]
    mara, jon = plan["cast"]
    assert mara["facing"] == 90 and [c["t"] for c in mara["cues"]] == [4.0, 6.0, 9.0] and mara["cues"][0]["to"] == [5.0, 2.0]
    assert mara["cues"][1] == {"t": 6.0, "do": "fly", "secs": 6.0}  # an action or a gesture is kept; a run to nowhere and an unknown word are not
    assert jon["cues"] == [] and jon["start"] == [0.75, 0.0]  # the brain forgot him: he stands on his mark
    whole = blocking.whole(plan, [{"s": 4, "note": "she walks to the camera"}, {"s": 21, "note": "push in"}])
    assert whole["seconds"] == 26 and whole["beats"][1] == {"s": 21.0, "note": "push in"}
    assert blocking.checked("nonsense", [], [])["cast"][0]["name"] == "Lead"


def test_the_rehearsal_draws_its_blocking_again_when_the_plan_moves(tmp_path):
    drawn = []

    def drawer(spec, plan, work, film):
        drawn.append(plan)
        film.parent.mkdir(parents=True, exist_ok=True)
        film.write_bytes(b"film")
        return {}

    class Planner(Brain):
        async def complete(self, messages, **settings):
            if "floor plan" in messages[0]["content"]:
                return json.dumps({"cast": [{"name": "Mara", "start": [0, -2], "facing": 0, "cues": [{"t": 4, "do": "walk", "to": [0, 2]}]}]})
            return json.dumps({"scene": SCENE, "beats": [{"s": 4, "note": "she walks to the camera"}], "who": None, "line": None, "ack": "Got it."})

    async def play():
        stage = make(tmp_path, blocking={"wait_s": 0, "floor_gb": 0})
        stage.brain = stage.blocking.brain = Planner()
        stage.blocking.drawer = drawer
        try:
            await stage.rehearse(SCENE + "\nMara: a woman in a red raincoat")
            await stage.blocking.task
            first = stage.state()["blocking"]
            assert first["file"] == "block-1.mp4" and first["by"] == "brain" and drawn[0]["beats"] == []
            await stage.note("she walks to the camera")
            await stage.blocking.task
            state = stage.state()["blocking"]
            assert state["file"] == "block-2.mp4" and not (tmp_path / "run" / "set" / "block-1.mp4").exists()
            assert drawn[1]["cast"][0]["cues"][0]["to"] == [0.0, 2.0] and drawn[1]["seconds"] == 10
            assert stage.blocking.file("block-2.mp4") is not None and stage.blocking.file("../x") is None
        finally:
            await stage.shutdown()
    asyncio.run(asyncio.wait_for(play(), 20))


def test_the_open_microphone_hands_over_one_sentence_at_a_time():
    import numpy as np
    from gma.live.ear import WINDOW, Ear

    class Vad:
        def reset(self):
            pass

        def __call__(self, window):
            return 0.9 if abs(float(window[0])) > 0.1 else 0.0

    async def play():
        heard, marks = [], []

        async def hear(audio):
            heard.append(len(audio))

        async def speech(on):
            marks.append(on)

        ear = Ear(Vad(), hear, speech=speech)
        loud, quiet = (np.full(WINDOW, 0.5) * 32767).astype(np.int16).tobytes(), np.zeros(WINDOW, np.int16).tobytes()
        await ear.audio(quiet * 20 + loud * 2 + quiet * 30)   # a click: not a sentence
        await ear.audio(loud * 30 + quiet * 30)               # a sentence and its silence
        await ear.audio(loud * 6)
        ear.drop()                                            # the page spoke over it
        await ear.audio(quiet * 30)
        await asyncio.sleep(0.05)
        ear.close()
        assert marks == [True, False, True] and len(heard) == 1  # the dropped start is the page's to clear and heard[0] >= 30 * WINDOW
    asyncio.run(play())


def test_her_lips_are_prepared_from_the_loop_and_a_line_comes_back_as_a_film(tmp_path):
    from gma.live.lips import Lips
    asked, up = [], {"on": False}

    async def runner(command, data=None, timeout=0):
        asked.append((command, len(data or b"")))
        if "/health" in command:
            return (0, b'{"avatars": []}', b"") if up["on"] else (7, b"", b"")
        if "run.sh start" in command:
            up["on"] = True
            return 0, b"", b""
        if "/prepare" in command:
            return 0, b'{"id": "a1", "frames": 248, "fps": 24}', b""
        if "/speak" in command:
            return (0, b"F" * 4000, b'{"out": "x", "frames": 48, "next": 60, "done": true}') if up["on"] else (7, b"", b"")
        return 0, b"", b""

    async def play():
        loop = tmp_path / "idle-1.mp4"
        loop.write_bytes(b"loop")
        lips = Lips({"home": "~/lips", "idle_s": 600}, runner=runner)
        assert await lips.speak(b"wav") is None  # nothing prepared: the plain talking loop is used
        lips.follow(loop)
        await lips.task
        assert lips.state()["phase"] == "ready" and any("/prepare" in c and n == 4 for c, n in asked)
        film, said = await lips.speak(b"RIFFwav", start=12)
        assert film == b"F" * 4000 and said["next"] == 60
        sent = next(c for c, n in asked if "/speak" in c and n == 7)
        assert '"start": 12' in sent and f'"id": "a{lips.tag}1"' in sent and sent.rstrip().endswith(".mp4")
        lips.home = str(tmp_path)                 # a streamed line with no service behind it: none, and the service is free again
        (tmp_path / "run" / "io").mkdir(parents=True)
        assert await lips.stream(b"RIFFwav") is None and not lips.lock.locked()
        up["on"] = False                          # its guard stopped it: the next line is spoken without the film
        assert await lips.speak(b"RIFFwav") is None and lips.state()["phase"] == "off"
        await lips.stop()
    asyncio.run(asyncio.wait_for(play(), 20))


def test_action_before_the_reference_shot_is_ready_waits_for_it_and_the_character_comes_back_after(tmp_path, monkeypatch):
    from gma.live import film

    async def join(self, parts, out, ffmpeg="ffmpeg"):
        out.write_bytes(b"".join(p.read_bytes() for p in parts))
    monkeypatch.setattr(film.Set, "join", join)
    monkeypatch.setattr(film, "there_and_back", lambda clip, loop, *a, **k: loop.write_bytes(clip.read_bytes()))
    monkeypatch.setattr(film.asyncio, "sleep", lambda s, real=asyncio.sleep: real(0.001))

    async def go():
        renderer = Renderer(tmp_path)
        stage = Stage({"run": str(tmp_path / "run"), "render": "h3", "anchors": False}, brain=Brain(), renderer=renderer)
        await stage.rehearse(SCENE + "\n\nMara: a woman with short black hair in a red raincoat")
        assert stage.set.still is None
        await stage.note("action")                      # said at once, the still not yet made (the director's first try, 2026-11-19)
        await stage.filming
        take = stage.takes[-1]
        assert take["state"] == "done" and stage.set.still == "ref-1.png", take
        assert [kind for kind, _ in renderer.asked][:2] == ["keyframe", "video"] and not renderer.cancelled
        await stage.set.task                            # back at the table: her loops are made now
        assert stage.set.loops == {"idle": "idle-1.mp4", "talk": "talk-1.mp4"}
        await stage.shutdown()
    asyncio.run(asyncio.wait_for(go(), 20))


def test_talk_in_the_room_is_not_a_direction(tmp_path):
    assert director.chatter("Um well") and director.chatter("Thanks.") and not director.chatter("she walks to the camera")

    class Listener(Brain):
        async def complete(self, messages, **settings):
            if "open microphone" in messages[-1]["content"] and "patient" in messages[-1]["content"]:
                return '{"ignore": true}'
            return await super().complete(messages, **settings)

    async def play():
        stage = make(tmp_path)
        stage.brain = Listener()
        try:
            await stage.rehearse(SCENE)
            assert (await stage.note("Um well", heard=True, overheard=True)) == {"ignored": True}
            assert (await stage.note("He's gonna go get the patient now.", heard=True, overheard=True)) == {"ignored": True}
            kept = await stage.note("she walks to the camera", heard=True, overheard=True)
            assert [b["note"] for b in kept["beats"]] == ["she walks to the camera"]
            assert [e["text"] for e in stage.log if e["kind"] == "note"] == ["she walks to the camera"]
        finally:
            await stage.shutdown()
    asyncio.run(asyncio.wait_for(play(), 20))


def test_a_held_shot_names_no_shot_size_and_no_moves():
    from gma.live import film

    scene = film.held("Medium close-up, static camera. A woman stands under a street lamp at night.")
    assert scene == film.HOLD + " A woman stands under a street lamp at night."
    assert film.held("A woman stands under a street lamp. She looks at the camera.").endswith("lamp. She looks at the camera.")
    for word in ("zoom", "push", "pan", "drift", "close-up"):  # the clip model takes a move that is named, also after "no"
        assert word not in scene.lower()


def test_the_take_is_told_the_blocking_it_was_shown():
    from gma.live.blocking import staging

    plan = {"cast": [{"name": "Mara", "start": [-1.0, -1.0], "facing": 0.0, "cues": [{"t": 6.0, "do": "walk", "to": [-1.0, 1.4], "facing": 0.0},
                                                                                       {"t": 10.0, "do": "turn", "facing": 180.0}]},
                     {"name": "Jon", "start": [5.0, -1.0], "facing": 270.0, "cues": [{"t": 2.0, "do": "walk", "to": [0.5, -1.0], "facing": 270.0}]}]}
    first, second, third = staging(plan, 0, 5.17), staging(plan, 5.17, 10.33), staging(plan, 10.33, 1e9)
    assert first == "Jon comes in from screen right and walks to screen left, stops at the middle of the frame and faces screen left."
    assert second.startswith("Mara walks a few steps toward the camera") and "Jon" not in second
    assert third == "Mara turns and has their back to the camera." or "turns" in second
    assert staging(None, 0, 5) == "" and staging({"cast": []}, 0, 5) == ""


def test_the_clock_counts_steps_and_learns_what_they_take(tmp_path):
    from gma.live.clock import Clock

    clock = Clock(tmp_path / "usual.json")
    clock.plan([("still", "still", "Reference shot"), ("lips", "lips", "Lip-sync")])
    assert [s["state"] for s in clock.state()] == ["waiting", "waiting"]
    clock.begin("still"); clock.steps["still"]["began"] -= 30; clock.end("still")
    said = clock.state()[0]
    assert said["state"] == "done" and 29 <= said["elapsed_s"] <= 31 and 39 <= Clock(tmp_path / "usual.json").usual["still"] <= 41
    clock.seen("lips", "running"); clock.seen("lips", "done")
    assert clock.state()[1]["state"] == "done" and clock.usual["lips"] == Clock().usual["lips"]  # seen, not measured


def test_a_take_is_filmed_between_fixed_pictures_so_faces_hold(tmp_path, monkeypatch):
    from gma.live import film

    async def join(self, parts, out, ffmpeg="ffmpeg"):
        out.write_bytes(b"".join(p.read_bytes() for p in parts))
    monkeypatch.setattr(film.Set, "join", join)
    monkeypatch.setattr(film.asyncio, "sleep", lambda s, real=asyncio.sleep: real(0.001))

    async def go():
        renderer = Renderer(tmp_path)
        stage = Stage({"run": str(tmp_path / "run"), "render": "h3"}, brain=Brain(), renderer=renderer)
        await stage.rehearse("A woman waits at a tram stop.\n\nMara: a woman in a red raincoat", seed=3)
        await stage.note("she walks toward the camera")
        await stage.note("action")
        for _ in range(600):
            if stage.phase != "filming":
                break
            await asyncio.sleep(0.02)
        take = stage.takes[-1]
        assert take["state"] == "done", take
        kinds = [kind for kind, _ in renderer.asked]
        clips = [said["clip"] for kind, said in renderer.asked if kind == "video" and "last_file" in said["clip"]]
        assert len(clips) == len(take["parts"]) >= 1 and kinds.count("keyframe") == 1 + len(clips)  # the reference shot and one end picture a part
        assert clips[0]["image_file"].endswith("ref-1.png") and clips[0]["last_file"].endswith("-e1.png")
        assert all(c["image_file"] == before["last_file"] for before, c in zip(clips, clips[1:]))  # each part opens on the picture the one before ends on
        assert "Picture 2" in clips[0]["prompt"]
        await stage.shutdown()
    asyncio.run(go())


def test_action_plays_a_run_through_first_and_cut_asks_retake_or_film(tmp_path, monkeypatch):
    """With a blocking preview, "action" plays the plan in real time and films nothing; "cut" ends it at its second;
    "again" plays it once more and "render it" films it up to the cut, with the lines the run-through spoke."""

    from gma.live import film

    async def join(self, parts, out, ffmpeg="ffmpeg"):
        out.write_bytes(b"".join(p.read_bytes() for p in parts))
    monkeypatch.setattr(film.Set, "join", join)
    monkeypatch.setattr(film.asyncio, "sleep", lambda s, real=asyncio.sleep: real(0.001))

    def drawer(spec, plan, work, film):
        film.parent.mkdir(parents=True, exist_ok=True)
        film.write_bytes(b"film")
        return {}

    class Planner(Brain):
        directed = 0

        async def complete(self, messages, **settings):
            said = messages[0]["content"]
            if "floor plan" in said:
                return json.dumps({"cast": [{"name": "Mara", "start": [0, -2], "facing": 0, "cues": [{"t": 2, "do": "walk", "to": [0, 2]}]}]})
            if '"beats"' in said:
                return json.dumps({"scene": "A woman waits at a tram stop.", "who": None, "line": None, "ack": "Got it.",
                                   "beats": [{"s": 2, "note": "she walks to the camera"}, {"s": 12, "note": "Mara says she is tired of waiting"}]})
            if "run through in real time" in said:
                return json.dumps({"line": "Soon, I hope."})  # spoken to while the run-through plays
            if "You keep the description" not in said:
                return "ready"  # the brain being warmed
            Planner.directed += 1
            return json.dumps({"scene": "A woman waits at a tram stop at night, nearer to the camera now.", "who": "Mara", "line": "I am tired of waiting.", "ack": "Okay."})

    async def go():
        renderer = Renderer(tmp_path)
        stage = Stage({"run": str(tmp_path / "run"), "render": "h3", "anchors": False, "run_count_s": 0.05,
                       "blocking": {"wait_s": 0, "floor_gb": 0}}, brain=Planner(), renderer=renderer)
        stage.blocking.drawer, stage.blocking.free_gb = drawer, None
        await stage.rehearse("A woman waits at a tram stop.\n\nMara: a woman in a red raincoat", seed=3)
        await stage.note("she walks to the camera, then says she is tired of waiting")
        await stage.preparing  # what the beats do is written when the note is given, behind the rehearsal
        assert Planner.directed == 2
        assert (await stage.note("action"))["command"] == "action"
        assert stage.phase == "running" and not stage.takes
        for _ in range(300):
            if stage.through["state"] == "playing":
                break
            await asyncio.sleep(0.01)
        run = stage.state()["run"]
        assert run["film"].startswith("block-") and run["seconds"] == 17 and [c["s"] for c in run["cues"]] == [2.0, 12.0]
        assert run["cues"][1]["line"] == "I am tired of waiting." and run["cues"][1]["voice"] == "af_heart"
        assert Planner.directed == 2 and run["count_s"] == 0.05  # "action" asked the brain nothing
        assert (await stage.note("I am tired of waiting.", heard=True, overheard=True)) == {"ignored": True}  # her own voice in the microphone
        stage.through_began -= 5  # five seconds in, the director speaks to her and she answers on the spot
        said = await stage.note("Mara, is the tram coming?")
        assert said["line"] == "Soon, I hope." and said["who"] == "Mara" and stage.phase == "running"
        assert [b["note"] for b in stage.table["beats"]] == ["she walks to the camera", 'Mara says: "Soon, I hope."', "Mara says she is tired of waiting"]
        assert 5 <= stage.table["beats"][1]["s"] <= 5.6 and [c.get("aside", False) for c in stage.through["cues"]] == [False, True, False]
        assert stage.log[-1]["kind"] == "line" and stage.log[-1]["voice"] == "af_heart" and Planner.directed == 2
        stage.through_began -= 2  # seven seconds in
        assert (await stage.note("cut", heard=True))["command"] == "cut"
        run = stage.state()["run"]
        assert stage.phase != "running" and run["state"] == "cut" and 6.9 <= run["cut_s"] <= 7.6 and not stage.takes
        asked = Planner.directed
        assert (await stage.note("again"))["command"] == "action" and stage.phase == "running"  # the retake
        for _ in range(300):
            if stage.through["state"] == "playing":
                break
            await asyncio.sleep(0.01)
        assert [c["line"] for c in stage.through["cues"]] == ["I am tired of waiting.", "Soon, I hope.", "I am tired of waiting."]  # her answer is in the plan now
        stage.through_began -= 7
        await stage.stop("cut")
        assert Planner.directed == asked
        assert (await stage.note("render it"))["command"] == "film"
        filming = stage.filming
        assert stage.phase != "filming" and stage.rendering() and stage.state()["rendering"] and stage.takes[-1]["behind"]  # filmed behind the rehearsal
        assert "command" not in await stage.note("she looks up at the lamp")  # which goes on: a note is a note
        with pytest.raises(Refused):
            await stage.rehearse("Another scene altogether, in a field.")  # but not a new scene while the take is filmed
        await filming
        take = stage.takes[-1]
        assert take["state"] == "done" and len(take["parts"]) == 1 and take["directions"] == 2, take  # 7 s: one part, the beats before the cut
        assert any("Soon, I hope." in job[1]["clip"]["prompt"] for job in renderer.asked if job[0] == "video")  # what she answered in the run-through is in the take
        assert Planner.directed == asked and not stage.rendering()  # the take films what the run-through played: nothing is written again
        assert stage.through is None
        # "auto_take": a cut films what was played at once, behind the rehearsal; a cut in the first seconds does not
        stage.spec["auto_take"] = True
        for seconds, filmed in ((1, False), (7, True)):
            before = len(stage.takes)
            assert (await stage.note("action"))["command"] == "action"
            for _ in range(300):
                if stage.through["state"] == "playing":
                    break
                await asyncio.sleep(0.01)
            stage.through_began -= seconds
            await stage.stop("cut")
            await asyncio.sleep(0.05)
            assert (len(stage.takes) > before) == filmed and stage.rendering() == filmed and stage.phase != "filming", (seconds, stage.takes[-1])
        await stage.filming
        assert stage.takes[-1]["state"] == "done" and stage.takes[-1]["behind"]
        await stage.shutdown()
    asyncio.run(go())


def test_a_take_is_as_long_as_the_run_through_up_to_its_cut():
    from gma.live.stage import run_parts
    assert [run_parts(t) for t in (0.5, 3.7, 5.9, 7.5, 8.0, 10.5, 13.5, 17)] == [1, 1, 1, 1, 2, 2, 3, 3]


def test_words_after_a_run_through():
    assert director.after_run("retake") == "action" and director.after_run("Again.") == "action" and director.after_run("Let's do it again") == "action"
    assert director.after_run("render it") == "film" and director.after_run("Proceed with rendering.") == "film" and director.after_run("print it") == "film"
    assert director.after_run("go ahead") == "film"
    assert director.after_run("she walks to the door again and keeps her eyes on the tram") is None
    assert director.after_run("Mara, how do you feel?") is None


def test_a_scene_that_names_nobody_gets_its_cast_from_the_words_about_it(tmp_path):
    """A set-up with no ``Name: look`` line: the brain reads who is on screen, and they can be spoken to."""

    class Reader(Brain):
        async def complete(self, messages, **settings):
            if "named nobody" in messages[0]["content"]:
                return json.dumps({"cast": [{"name": "warrior", "look": "a young heroic warrior in silver armor and a dark blue cape, him"},
                                            {"name": "", "look": "nobody"}, {"name": "Warrior", "look": "the same again"}],
                                   "scene": "Wide shot from behind. Warrior, a young man in silver armor, steps into a dark stone chamber lit by candles.",
                                   "beats": [{"s": 5, "note": "the walls tremble"}, {"s": 10, "note": "a ghost comes out of the dark"}]})
            return await super().complete(messages, **settings) if len(messages) > 1 else "ready"

    async def go():
        stage = Stage({"run": str(tmp_path / "run")}, brain=Reader())
        stage._load = lambda: asyncio.sleep(0)
        await stage.rehearse("A young heroic warrior wearing silver armor and a dark blue cape slowly enters an abandoned chamber.")
        assert stage.state()["casting"]
        await stage.finding
        assert stage.cast == [{"name": "Warrior", "look": "a young heroic warrior in silver armor and a dark blue cape, him"}]
        assert stage.to() == "Warrior" and stage.voice_of("Warrior").startswith("am") and not stage.state()["casting"]
        assert any(e["kind"] == "cast" for e in stage.log)
        assert stage.table["scene"].startswith("Wide shot from behind.") and [b["s"] for b in stage.table["beats"]] == [5.0, 10.0]  # the story as a plan
        await stage.preparing
        await stage.rehearse(SCENE + "\n\nMara: a woman in a red raincoat")  # named on a line: nobody is asked
        assert stage.finding.done() and [c["name"] for c in stage.cast] == ["Mara"]
        await stage.shutdown()
    asyncio.run(go())

    class Away:
        async def complete(self, messages, **settings):
            raise RuntimeError("away")
    assert asyncio.run(director.casting(Away(), "A man waits.")) == []


def test_the_casts_own_voices_in_the_microphone_are_not_words_to_them():
    assert director.echo("I am tired of waiting", ["I am tired of waiting."])
    assert director.echo("tired of waiting", ["I am tired of waiting."])
    assert not director.echo("Mara, are you cold tonight?", ["I am tired of waiting."])
    assert not director.echo("", ["I am tired of waiting."]) and not director.echo("cut", [])
