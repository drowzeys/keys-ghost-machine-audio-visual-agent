"""Speech recognition on a TensorFold speech server (voice_agent/engines.py) against a fake chat server: the request it
sends, its deadline, and the local recognizer taking over when the server is slow, refuses or is down."""

import base64
import io
import json

import httpx
import numpy as np
import soundfile as sf

from voice_agent.engines import SAMPLE_RATE, TRANSCRIBE_PROMPT, Recognizer, SpeechServer, wav_bytes


def fake(text="Make a five second clip of a red fox.", status=200, fail=None, seen=None, body=None):
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        if fail:
            raise fail
        if body is not None:
            return httpx.Response(status, json=body)
        return httpx.Response(status, json={"choices": [{"message": {"role": "assistant", "content": text}}]})
    return httpx.MockTransport(handle)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Local(Recognizer):
    """The Recognizer's choice between server and local, with a local recognizer that loads no model."""

    def __init__(self, server, text="local words"):
        self.server, self.where, self.text, self.calls = server, None, text, 0
        self.url, self.model = "", object()  # no shared Parakeet; the local one is "loaded"

    def local(self, audio):
        self.calls += 1
        return self.text


SPEC = {"url": "http://transcribe.test/v1", "model": "transcribe", "timeout_ms": 50}
AUDIO = (0.1 * np.sin(np.arange(SAMPLE_RATE) / 10)).astype(np.float32)


def test_wav_is_16_bit_mono_16k_and_keeps_the_samples():
    data, rate = sf.read(io.BytesIO(wav_bytes(AUDIO)), dtype="float32")
    assert rate == SAMPLE_RATE and data.ndim == 1 and len(data) == len(AUDIO)
    assert np.max(np.abs(data - AUDIO)) < 2 / 32767  # rounded to 16 bits (soundfile reads them back over 32768)
    loud, _ = sf.read(io.BytesIO(wav_bytes(np.array([2.0, -2.0], dtype=np.float32))), dtype="float32")
    assert loud[0] > 0.999 and loud[1] < -0.999  # clipped, not wrapped


def test_the_request_is_one_input_audio_part_greedy_without_thinking():
    seen = []
    s = SpeechServer(SPEC, fake("  Hello there.  ", seen=seen))
    assert s(AUDIO) == "Hello there."
    body = seen[0]
    assert body["model"] == "transcribe" and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    text, part = body["messages"][0]["content"]
    assert text == {"type": "text", "text": TRANSCRIBE_PROMPT}
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    data, rate = sf.read(io.BytesIO(base64.b64decode(part["input_audio"]["data"])), dtype="float32")
    assert rate == SAMPLE_RATE and len(data) == len(AUDIO)


def test_a_profile_prompt_replaces_the_default():
    seen = []
    SpeechServer({**SPEC, "prompt": "Transcribe with punctuation. Context terms: Tanaka, ProRes."}, fake(seen=seen))(AUDIO)
    assert seen[0]["messages"][0]["content"][0]["text"] == "Transcribe with punctuation. Context terms: Tanaka, ProRes."


def test_server_text_is_used_and_the_local_recognizer_is_not_called():
    r = Local(SpeechServer(SPEC, fake("Turn the music down.")))
    assert r(AUDIO) == "Turn the music down." and r.where == "server" and r.calls == 0


def test_without_a_server_the_local_recognizer_answers():
    r = Local(None)
    assert r(AUDIO) == "local words" and r.where == "local"


def test_a_refused_connection_falls_back_and_skips_the_server_for_retry_s():
    seen = []
    clock = Clock()
    s = SpeechServer({**SPEC, "retry_s": 30}, fake(fail=httpx.ConnectError("refused"), seen=seen), clock=clock)
    r = Local(s)
    assert r(AUDIO) == "local words" and r.where == "local"
    assert r(AUDIO) == "local words" and len(seen) == 1  # skipped: no second request while it is down
    clock.t += 31
    s.client = httpx.Client(transport=fake("Back again."))
    assert r(AUDIO) == "Back again." and r.where == "server"


def test_a_slow_server_falls_back_and_two_misses_in_a_row_skip_it():
    clock = Clock()
    s = SpeechServer(SPEC, fake(fail=httpx.ReadTimeout("slow")), clock=clock)
    r = Local(s)
    assert r(AUDIO) == "local words" and s.down_until == 0  # one miss: tried again next turn
    assert r(AUDIO) == "local words" and s.down_until > clock.t
    assert r.calls == 2


def test_a_miss_then_an_answer_resets_the_count():
    s = SpeechServer(SPEC, fake(fail=httpx.ReadTimeout("slow")), clock=Clock())
    assert s(AUDIO) is None and s.misses == 1
    s.client = httpx.Client(transport=fake("On time."))
    assert s(AUDIO) == "On time." and s.misses == 0


def test_the_deadline_is_timeout_ms():
    s = SpeechServer({**SPEC, "timeout_ms": 800}, fake())
    assert s.client.timeout.read == 0.8 and s.client.timeout.connect == 0.8
    assert SpeechServer({**SPEC, "timeout_ms": 3000}, fake()).client.timeout.connect == 1.0


def test_errors_and_bad_replies_fall_back():
    for transport, down in [(fake(status=503), True), (fake(status=400), False), (fake(body={"error": "x"}), False)]:
        clock = Clock()
        s = SpeechServer(SPEC, transport, clock=clock)
        r = Local(s)
        assert r(AUDIO) == "local words" and r.where == "local"
        assert (s.down_until > clock.t) == down


def test_silence_from_the_server_is_an_empty_transcript_not_a_fallback():
    r = Local(SpeechServer(SPEC, fake("")))
    assert r(AUDIO) == "" and r.where == "server" and r.calls == 0
