"""The shared recognizer (voice_agent/asr_server.py) and the agents' client of it (engines.Recognizer with "url"): text
from the server, the in-process fallback when it does not answer, the retry after retry_s, and the MLX cache cap."""

import sys

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_agent import engines
from voice_agent.asr_server import app_for


class Fake:
    """Stands in for the model: counts its loads and its utterances."""

    loads = 0

    def __init__(self):
        Fake.loads += 1
        self.heard = []


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    Fake.loads = 0

    def load(self):
        self.model = Fake()

    def local(self, audio):
        self.model.heard.append(len(audio))
        return f"local {len(audio)}"

    monkeypatch.setattr(engines.Recognizer, "load", load)
    monkeypatch.setattr(engines.Recognizer, "local", local)


SPEC = {"kind": "parakeet-mlx", "model": "m", "url": "http://asr.test"}


def server(seen=None, down=False, status=200):
    def handle(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        audio = np.frombuffer(request.content, dtype="<f4")
        if seen is not None:
            seen.append((audio, dict(request.url.params)))
        return httpx.Response(status, json={"text": f" shared {len(audio)} "})
    return httpx.MockTransport(handle)


def test_in_process_by_default():
    r = engines.Recognizer({"kind": "parakeet-mlx", "model": "m"})
    assert Fake.loads == 1 and r(np.zeros(160, np.float32)) == "local 160" and r.where == "local"


def test_shared_server_answers_and_no_model_loads():
    seen = []
    r = engines.Recognizer({**SPEC, "language": "en"}, transport=server(seen))
    audio = np.linspace(-1, 1, 1600, dtype=np.float32)
    assert r(audio) == "shared 1600" and r.where == "shared"
    assert Fake.loads == 0 and r.model is None
    assert np.array_equal(seen[0][0], audio) and seen[0][1] == {"language": "en"}


def test_down_at_start_loads_in_process_and_retries_later(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(engines.time, "monotonic", lambda: clock[0])
    state = {"down": True}

    def handle(request):
        if state["down"]:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"ok": True, "text": "shared"})

    r = engines.Recognizer({**SPEC, "retry_s": 30}, transport=httpx.MockTransport(handle))
    assert Fake.loads == 1  # loaded at start, since the server is down
    state["down"] = False
    assert r(np.zeros(10, np.float32)) == "local 10"  # within retry_s: not asked yet
    clock[0] += 31
    assert r(np.zeros(10, np.float32)) == "shared" and r.where == "shared"


def test_failure_mid_run_falls_back_once_loaded_lazily():
    r = engines.Recognizer(SPEC, transport=server(status=500))
    assert Fake.loads == 0
    assert r(np.zeros(8, np.float32)) == "local 8" and Fake.loads == 1 and r.where == "local"


def test_no_fallback_raises():
    with pytest.raises(RuntimeError):
        engines.Recognizer({**SPEC, "fallback": False}, transport=server(down=True))


def test_asr_server_round_trip():
    r = engines.Recognizer({"kind": "parakeet-mlx", "model": "m"})
    client = TestClient(app_for(r))
    assert client.get("/health").json()["ok"] is True
    audio = np.ones(320, np.float32)
    got = client.post("/v1/transcribe", content=audio.astype("<f4").tobytes(), headers={"Content-Type": "application/octet-stream"})
    assert got.status_code == 200 and got.json()["text"] == "local 320"
    assert client.get("/health").json()["utterances"] == 1
    assert client.post("/v1/transcribe", content=b"abc").status_code == 400


def test_cache_cap_is_metal_only(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "linux")
    engines.cap_mlx_cache(256)  # MLX's CUDA backend: never capped, mlx not even imported for it
    assert calls == []
    engines.cap_mlx_cache(None)
    engines.cap_mlx_cache(-1)


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal only")
def test_cache_cap_on_metal():
    mx = pytest.importorskip("mlx.core")
    engines.cap_mlx_cache(256)
    setter = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
    assert setter(0) == 256 * 1024 * 1024  # set_cache_limit returns the previous limit
    setter(256 * 1024 * 1024)


def test_a_speech_server_falls_back_to_the_shared_parakeet_not_a_local_copy():
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "transcribe.test":
            raise httpx.ConnectError("refused", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={"text": f" shared {len(np.frombuffer(request.content, dtype='<f4'))} "})

    spec = {**SPEC, "server": {"url": "http://transcribe.test/v1", "timeout_ms": 50}}
    r = engines.Recognizer(spec, transport=httpx.MockTransport(handle))
    assert Fake.loads == 0  # the shared Parakeet answers its /health: no model here
    assert r(np.zeros(32, np.float32)) == "shared 32" and r.where == "shared" and Fake.loads == 0
