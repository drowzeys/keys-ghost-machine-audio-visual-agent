"""The whole app (voice_agent/server.py build) with the recognizer faked: the supervisor's routes, adoption at start-up
and the governor's refusal over HTTP. No model, no GPU; the engines are stand-ins on free loopback ports."""

import threading

import pytest
from fastapi.testclient import TestClient

from gm_core import engine as E
from voice_agent import server

from fake_engine import serve


@pytest.fixture
def brain_port():
    one = serve(E.free_port())
    threading.Thread(target=one.serve_forever, daemon=True).start()
    yield one.server_address[1]
    one.shutdown()
    one.server_close()


def test_app_adopts_running_engines_and_shows_them(monkeypatch, tmp_path, brain_port):
    monkeypatch.setattr(server, "Recognizer", lambda spec: (lambda audio: ""))
    voice_port = E.free_port()
    profile = {
        "name": "test", "platform": "spark", "asr": {"kind": "nemo", "model": "nvidia/parakeet-tdt-0.6b-v3"},
        "brain": {"url": f"http://127.0.0.1:{brain_port}/v1", "model": "gemma-4-12b-it", "system": "You are Boo."},
        "voice": {"url": f"http://127.0.0.1:{voice_port}/v1", "model": "kokoro-82m", "keep_warm": False},
        "engines": {"brain": {"role": "brain", "mode": "adopt", "port": brain_port},
                    "kokoro": {"role": "voice", "mode": "adopt", "port": voice_port},
                    "asr": {"role": "asr", "mode": "inprocess", "resident_gb": 3}},
        "governor": {"schedules": {"live": {"render": {"need_gb": 500}, "keep": ["brain", "asr"]}}},
        "supervisor": {"home": str(tmp_path), "monitor_s": 60},
    }
    with TestClient(server.build(profile)) as client:
        engines = {e["name"]: e for e in client.get("/api/engines").json()["engines"]}
        assert engines["brain"]["state"] == "adopted" and engines["brain"]["adopted"]
        assert engines["kokoro"]["state"] == "unloaded"  # nothing answers there, and adopt mode starts nothing
        assert engines["asr"]["state"] == "loaded" and engines["asr"]["mode"] == "inprocess"
        health = client.get("/health").json()
        assert health["brains"]["main"] == "gemma-4-12b-it" and health["governor"] == "talking"
        refused = client.post("/api/governor/phase", json={"phase": "rendering", "schedule": "live"})
        assert refused.status_code == 409 and refused.json()["said"] == "There isn't enough memory to render that right now."
        assert client.get("/api/governor").json()["phase"] == "talking"
        assert client.get("/api/governor/plan", params={"schedule": "nope"}).status_code == 400
        assert any(c["name"] == "Imagination Studio" for c in client.get("/api/credits").json())
        assert client.get("/api/lanes").json()["voice_turns"] == 0
        load = client.post("/api/engines/kokoro/load")
        assert load.status_code == 409 and load.json()["said"] == "My voice isn't running right now."
