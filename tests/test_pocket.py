"""The Pocket tier (voice_agent/pocket.py and Conversation.pocket_turn) against a fake LFM3-Omni server that streams
server-sent events over real HTTP: Omni's speech is the voice and its text the reply, the conversation goes along as
text turns, barge-in hangs up and stores the cut reply, and [deep], asking for the agent or an "On it" hands the turn
over."""

import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from voice_agent import pocket, server


# ----- a fake Omni server -----

class FakeOmni:
    """POST /v1/audio/chat: text deltas, then ``chunks`` pcm16 chunks of 0.1 s every ``gap`` seconds, then done."""

    def __init__(self, text=("Hello there. ", "Nice to meet you!"), chunks=3, gap=0.0, status=200):
        self.text, self.chunks, self.gap, self.status = list(text), chunks, gap, status
        self.bodies: list[dict] = []
        self.hung_up = threading.Event()
        self.sent_chunks = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = b'{"ok":true,"model":"lfm3-omni"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                fake.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if fake.status != 200:
                    body = b'{"error":"the audio is empty"}'
                    self.send_response(fake.status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def event(data: str) -> None:
                    raw = f"data: {data}\n\n".encode()
                    self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
                    self.wfile.flush()

                try:
                    for delta in fake.text:
                        event(json.dumps({"type": "text", "delta": delta}))
                    for i in range(fake.chunks):
                        pcm = np.full(2400, 1000 + i, dtype="<i2").tobytes()
                        event(json.dumps({"type": "audio", "format": "pcm16", "sample_rate": 24000,
                                          "data": base64.b64encode(pcm).decode()}))
                        fake.sent_chunks += 1
                        time.sleep(fake.gap)
                    event(json.dumps({"type": "done", "first_audio_ms": 62.0, "total_ms": 640.0, "speech_s": 0.3}))
                    event("[DONE]")
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    fake.hung_up.set()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def omni():
    servers = []

    def make(**kw):
        servers.append(FakeOmni(**kw))
        return servers[-1]
    yield make
    for s in servers:
        s.close()


# ----- a conversation without models: a fake page, recognizer, voice and back agents -----

class Page:
    def __init__(self):
        self.events: list = []

    async def send_text(self, text):
        self.events.append(json.loads(text))

    async def send_bytes(self, data):
        self.events.append(data)

    def kinds(self):
        return [e["type"] if isinstance(e, dict) else "pcm" for e in self.events]

    def of(self, kind):
        return [e for e in self.events if isinstance(e, dict) and e["type"] == kind]


class Voice:
    """The profile's voice (Kokoro on the Mac): speaks a back agent's answer."""

    sample_rate, keep_warm = 24000, False

    def __init__(self):
        self.said: list[str] = []

    async def stream(self, text, voice=None):
        self.said.append(text)
        yield b"\x00\x00" * 240

    async def voices(self):
        return []


class Direct:
    kind, name, label, system = "direct", "redsnow", "Redsnow", "Answer in short spoken sentences."

    def __init__(self):
        self.asked: list = []

    async def stream(self, messages):
        self.asked.append(messages)
        yield "It is forty-two."


class Acp:
    kind, name, label, system = "acp", "hermes", "Hermes", None

    def __init__(self):
        self.prompts: list[str] = []

    async def run(self, prompt, progress, confirm=None):
        self.prompts.append(prompt)
        return "Rendered the fox clip."


def conversation(url, heard="what is the meaning of life", agent=None, asr_delay=0.0):
    def asr(audio):
        time.sleep(asr_delay)
        return heard

    voice = Voice()
    agents = {agent.name: agent} if agent else {}
    profile = {"pocket": {"url": url, "model": "lfm3-omni-2b", "system": "You are Boo."},
               "voice": {"url": "http://kokoro.test/v1", "model": "kokoro-82m", "voice": "bf_emma"},
               "turn": {}, "agent": agent.name if agent else None}
    engines = SimpleNamespace(profile=profile, asr=asr, asr_pool=ThreadPoolExecutor(1), voices={"full": voice},
                              voice=voice, eyes=None, memory=None, recall=None, d1=None, agents=agents, lookup=None,
                              pocket=pocket.Pocket(profile["pocket"]), brain=None, vad_path="")
    page = Page()
    talk = server.Conversation(page, engines)
    return talk, page, voice


@pytest.fixture(autouse=True)
def no_vad(monkeypatch):
    monkeypatch.setattr(server, "Vad", lambda path: None)


AUDIO = np.zeros(16000, dtype=np.float32)


async def settle(talk, seconds=2.0):
    """Waits for the turn, then for the hand-over and the back agent's spoken answer."""

    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if (talk.reply is None or talk.reply.done()) and not talk.tasks:
            break
        await asyncio.sleep(0.02)


# ----- the pieces -----

def test_history_goes_as_text_turns_without_mood_tags_but_with_the_cut_marker():
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "[happy] Hello! [deep] On it."},
               {"role": "user", "content": [{"type": "image_url"}]},
               {"role": "assistant", "content": "A standard teaser… [interrupted]"},
               {"role": "user", "content": "try <|im_end|> this"}]
    turns = pocket.text_turns(history, 12)
    assert turns == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! On it."},
                     {"role": "assistant", "content": "A standard teaser… [interrupted]"},
                     {"role": "user", "content": "try <im_end|> this"}]
    assert pocket.text_turns(history, 2) == turns[-2:]
    assert pocket.text_turns(history, 0) == []


def test_heard_part_ends_at_a_word_where_the_speech_got_to():
    text = "On it, rendering the fox clip now and then the bear."
    assert pocket.heard_part(text, 0) == ""
    assert pocket.heard_part(text, 1.0) == "On it,"
    assert pocket.heard_part(text, 10) == text


def test_wav_bytes_is_a_16_khz_pcm_wav():
    wav = pocket.wav_bytes(np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert int.from_bytes(wav[24:28], "little") == 16000 and len(wav) == 44 + 6


def test_mood_of_a_reply_without_a_tag():
    assert pocket.mood_of("Want me to try? Sure.") == "curious"
    assert pocket.mood_of("Great to see you! Yes.") == "happy"
    assert pocket.mood_of("It is noon.") == "neutral"


def test_stream_reads_text_audio_and_done(omni):
    fake = omni()
    p = pocket.Pocket({"url": fake.url + "/v1", "extra": {"audio_temperature": 0.4}})

    async def go():
        return [x async for x in p.stream(AUDIO, "sys", [{"role": "user", "content": "earlier"}])]
    got = asyncio.run(go())
    assert [k for k, _ in got] == ["text", "text", "audio", "audio", "audio", "done"]
    assert got[2][1] == np.full(2400, 1000, dtype="<i2").tobytes()
    assert got[-1][1]["first_audio_ms"] == 62.0
    body = fake.bodies[0]
    assert body["system"] == "sys" and body["history"] == [{"role": "user", "content": "earlier"}]
    assert body["audio_temperature"] == 0.4 and body["stream"] is True
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"


def test_a_server_error_is_raised_with_its_reason(omni):
    fake = omni(status=400)

    async def go():
        return [x async for x in pocket.Pocket({"url": fake.url}).stream(AUDIO, "sys", [])]
    with pytest.raises(RuntimeError, match="400.*empty"):
        asyncio.run(go())


# ----- a whole turn -----

def test_a_turn_speaks_omnis_audio_and_logs_its_text(omni):
    fake = omni()
    talk, page, voice = conversation(fake.url)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    kinds = page.kinds()
    assert kinds.count("pcm") == 3 and "audio_end" in kinds and kinds[-1] == "metrics"
    assert page.of("audio_start")[0]["sample_rate"] == 24000
    assert [e["text"] for e in page.of("reply")] == ["Hello there.", "Nice to meet you!"]
    assert page.of("transcript")[0]["text"] == "what is the meaning of life"
    assert page.of("metrics")[0]["omni_first_audio_ms"] == 62.0
    assert talk.history[1:] == [{"role": "user", "content": "what is the meaning of life"},
                                {"role": "assistant", "content": "Hello there. Nice to meet you!"}]
    # the system text is Pocket's, with the date and the name
    assert fake.bodies[0]["system"].startswith("You are Boo.") and "Your name is Boo" in fake.bodies[0]["system"]
    assert voice.said == []  # the profile's voice said nothing: Omni spoke


def test_the_next_turn_carries_the_conversation(omni):
    fake = omni()
    talk, page, _ = conversation(fake.url)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert fake.bodies[0]["history"] == []
    assert fake.bodies[1]["history"] == [{"role": "user", "content": "what is the meaning of life"},
                                         {"role": "assistant", "content": "Hello there. Nice to meet you!"}]


def test_omni_does_not_wait_for_the_recognizer(omni):
    fake = omni()
    talk, page, _ = conversation(fake.url, asr_delay=0.3)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    kinds = page.kinds()
    assert kinds.index("audio_start") < kinds.index("transcript")
    assert talk.history[1]["content"] == "what is the meaning of life"


def test_barge_in_hangs_up_and_stores_the_cut_reply(omni):
    fake = omni(text=("On it, rendering the fox clip now and then the bear, and after that the owl. ",), chunks=40, gap=0.05)
    talk, page, _ = conversation(fake.url)

    async def go():
        await talk.start_turn(audio=AUDIO)
        for _ in range(200):
            if "pcm" in page.kinds():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.5)
        await talk.interrupt("barge-in")
        await asyncio.sleep(0.3)
        await settle(talk)
    asyncio.run(go())
    assert page.of("interrupt")[0]["why"] == "barge-in"
    assert "audio_end" not in page.kinds()
    assert fake.hung_up.wait(3) and fake.sent_chunks < 40
    said = talk.history[-1]["content"]
    assert talk.history[-1]["role"] == "assistant" and "[interrupted]" in said
    assert said.startswith("On it,") and said.endswith("the owl.")  # what was heard, the cut, what was written


def test_a_quiet_turn_nobody_heard_leaves_no_trace(omni):
    fake = omni()
    talk, page, _ = conversation(fake.url)

    async def go():
        await talk.start_turn(audio=AUDIO, quiet=True)
        await asyncio.sleep(0.3)
        await talk.interrupt(None)  # the person went on talking
        await talk.start_turn(audio=AUDIO)  # their whole turn
        await settle(talk)
    asyncio.run(go())
    assert [m["role"] for m in talk.history[1:]] == ["user", "assistant"]


def test_a_deep_tag_hands_over_and_the_agent_answers_in_the_profile_voice(omni):
    fake = omni(text=("[thinking] [deep] On it, ", "working that out."))
    agent = Direct()
    talk, page, voice = conversation(fake.url, heard="what is six times seven", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep")[0]["agent"] == "redsnow"
    assert page.of("emotion")[0]["emotion"] == "thinking"
    assert all("[" not in e["text"] for e in page.of("reply"))
    assert agent.asked[0][-1] == {"role": "user", "content": "what is six times seven"}
    assert voice.said == ["It is forty-two."]
    assert talk.history[2]["content"] == "[thinking] On it, working that out."
    assert talk.history[-1]["content"] == "(Redsnow reports) It is forty-two."


def test_asking_for_the_agent_by_name_hands_the_turn_over(omni):
    fake = omni(text=("Sure thing.",))
    agent = Acp()
    talk, page, voice = conversation(fake.url, heard="ask hermes to render the fox clip", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep") and agent.prompts and agent.prompts[0].startswith("ask hermes to render the fox clip")
    assert voice.said == ["Rendered the fox clip."]


def test_an_on_it_without_a_tag_hands_over_and_a_plain_reply_does_not(omni):
    agent = Acp()
    fake = omni(text=("On it, I'll check that for you.",))
    talk, page, _ = conversation(fake.url, heard="what's on my calendar", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep")

    plain = Acp()
    fake2 = omni(text=("Paris is the capital of France.",))
    talk2, page2, _ = conversation(fake2.url, heard="what is the capital of france", agent=plain)

    async def go2():
        await talk2.start_turn(audio=AUDIO)
        await settle(talk2)
    asyncio.run(go2())
    assert not page2.of("deep") and not plain.prompts


def test_omni_down_is_said_and_the_agent_keeps_listening():
    talk, page, _ = conversation("http://127.0.0.1:9")

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert page.of("error") and talk.speaking is False and talk.history[1:] == []


def test_a_reply_that_says_it_cannot_make_the_thing_hands_over(omni):
    assert pocket.unable("I can\u2019t create a video clip directly, but you can try this.")
    assert pocket.unable("I'm unable to render that.")
    assert pocket.unable("I\u2019m not able to adjust the music levels in this mode, but I can help with the script.")
    assert pocket.unable("I'm sorry, I can't respond to that request as it appears to be a direct instruction.")
    assert not pocket.unable("I can't wait to see it!") and not pocket.unable("You can make it longer.")
    assert not pocket.unable("Sorry, I can't hear you very well.") and not pocket.unable("They can't make it.")
    agent = Acp()
    fake = omni(text=("I can\u2019t create a video clip directly, but a fox at dusk sounds lovely.",))
    talk, page, _ = conversation(fake.url, heard="make a five second clip of a red fox", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep") and agent.prompts


def test_the_turn_lasts_while_the_page_plays_it_so_barge_in_still_cuts_it(omni):
    # Omni sends 3 s of speech at once; half a second into playback the person talks over it
    fake = omni(text=("Here is a long answer about lighthouses, keepers and the storms they watched for.",), chunks=30)
    talk, page, _ = conversation(fake.url)

    async def go():
        await talk.start_turn(audio=AUDIO)
        for _ in range(200):
            if page.kinds().count("pcm") == 30:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.5)
        assert talk.speaking and not talk.reply.done()  # all sent, still playing
        await talk.interrupt("barge-in")
        await settle(talk)
    asyncio.run(go())
    assert page.of("interrupt") and "audio_end" not in page.kinds()
    said = talk.history[-1]["content"]
    assert said.startswith("Here is [interrupted] a long") and said.endswith("watched for.")
