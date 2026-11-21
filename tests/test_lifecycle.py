"""The engine lifecycle (gm_core/engine.py) and the leash (gm_core/leash.py) with a stand-in server
(tests/fake_engine.py): no model, no GPU. Every process here is one these tests started, stopped by its own PID."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from gm_core import engine as E
from gm_core.engine import Engine, Held, Refusal
from gm_core.memstat import MemReader

from fake_engine import serve

HERE = Path(__file__).resolve().parent
FAKE = str(HERE / "fake_engine.py")


def roomy() -> MemReader:
    """A Spark with plenty available (faked)."""
    return MemReader("spark", meminfo=lambda: "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n")


def managed(tmp_path, port=0, **spec) -> Engine:
    return Engine("fake", {"role": "brain", "mode": "managed", "port": port, "argv": ["{python}", FAKE, "--port", "{port}"],
                           "load_timeout_s": 20, "idle_s": 600, **spec}, roomy(), log_dir=tmp_path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie is gone for our purposes
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


@pytest.fixture
def running():
    """A server already answering on a port (as Bolt, Drama 3 or Cosmos do on the Mac)."""
    server = serve(E.free_port())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_managed_engine_loads_on_demand_under_the_leash_and_unloads(tmp_path):
    one = managed(tmp_path)
    assert one.state() == "unloaded"
    with one.request() as base:
        assert base.startswith("http://127.0.0.1:") and one.in_flight == 1 and one.state() == "loaded"
        leash = one.process.pid
        assert alive(leash) and not one.adopted
    assert one.in_flight == 0 and one.load_s_measured is not None
    assert (tmp_path / "logs" / "fake.log").exists()
    one.stop("test")
    assert one.state() == "unloaded" and one.port == 0
    deadline = time.time() + 5
    while alive(leash) and time.time() < deadline:
        time.sleep(0.05)
    assert not alive(leash)


def test_a_server_already_on_the_port_is_adopted_not_started_twice(tmp_path, running):
    port = running.server_address[1]
    one = managed(tmp_path, port=port)
    with one.request() as base:
        assert base == f"http://127.0.0.1:{port}"
    assert one.adopted and one.state() == "adopted" and one.process is None
    assert one.release("a render") is False  # held, never ended: it is not this app's
    assert one.hold == "a render"
    with pytest.raises(Held):
        one.acquire(wait=False)
    one.unhold()
    one.stop("let go")
    assert one.state() == "unloaded" and not one.adopted
    # the adopted server still answers: stopping only let it go
    with one.request():
        assert one.adopted


def test_adopt_mode_never_starts_anything_and_says_so_shortly(tmp_path):
    one = Engine("voice", {"role": "voice", "mode": "adopt", "port": E.free_port()}, roomy(), log_dir=tmp_path)
    with pytest.raises(Refusal) as refused:
        one.acquire()
    assert one.process is None and refused.value.said == "My voice isn't running right now."
    assert len(refused.value.said) <= 90


def test_adopted_server_that_goes_away_is_let_go_on_the_next_tick(tmp_path):
    server = serve(E.free_port())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    one = Engine("vision", {"role": "vision", "mode": "adopt", "port": server.server_address[1]}, roomy(), log_dir=tmp_path)
    with one.mutex:
        assert one.adopt()
    one.tick()
    assert one.state() == "adopted"
    server.shutdown()
    server.server_close()
    one.tick()
    assert one.state() == "unloaded"


def test_idle_unload_waits_for_nothing_in_flight(tmp_path):
    one = managed(tmp_path, idle_s=1)
    base = one.acquire()
    assert base
    one.tick(now=time.time() + 10)
    assert one.loaded()  # a request is in flight: never unloaded under it
    one.release_request()
    one.tick(now=time.time() + 10)
    assert one.state() == "unloaded"


def test_release_waits_for_the_answer_in_flight_then_holds(tmp_path):
    one = managed(tmp_path)
    one.acquire()
    done = []
    worker = threading.Thread(target=lambda: done.append(one.release("a sharp render")))
    worker.start()
    time.sleep(0.4)
    assert one.loaded() and one.hold == "a sharp render" and not done  # waits for the answer
    one.release_request()
    worker.join(30)
    assert done == [True] and one.state() == "held"
    with pytest.raises(Held) as held:
        one.acquire(wait=False)
    assert held.value.said == "My brain is paused while I render. Give me a moment."
    one.unhold()
    assert one.state() == "unloaded"


def test_admit_can_refuse_a_load(tmp_path):
    def admit(engine):
        raise Refusal("needs 43 GB, 20 available", "I don't have the memory to start my brain right now.")

    one = managed(tmp_path)
    one.admit = admit
    with pytest.raises(Refusal, match="43 GB"):
        one.acquire()
    assert one.process is None and one.state() == "unloaded"


def test_missing_command_is_refused_before_anything_starts(tmp_path):
    one = Engine("bolt", {"role": "brain", "mode": "managed", "port": E.free_port(), "argv": ["/nonexistent/tensorfold-native", "serve"]},
                 roomy(), log_dir=tmp_path)
    with pytest.raises(Refusal, match="No bolt engine"):
        one.acquire()


def test_kill_guard_kills_the_group_under_the_floor(tmp_path):
    low = MemReader("spark", meminfo=lambda: "MemTotal: 127000000 kB\nMemAvailable: 1000000 kB\n")
    one = Engine("fake", {"role": "brain", "mode": "managed", "argv": ["{python}", FAKE, "--port", "{port}", "--delay", "30"],
                          "guard_gb": 8}, low, log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="memory guard killed"):
        one.acquire()
    assert one.state() == "unloaded"


def test_leash_ends_its_child_when_the_pipe_closes():
    leash = subprocess.Popen([sys.executable, "-I", str(Path(E.__file__).with_name("leash.py")), "--", sys.executable, "-c",
                              "import time, os; print(os.getpid(), flush=True); time.sleep(600)"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
    child = int(leash.stdout.readline())
    assert alive(child)
    leash.stdin.close()
    assert leash.wait(30) == 128 + signal.SIGTERM or leash.returncode in (-signal.SIGTERM, 143)
    deadline = time.time() + 5
    while alive(child) and time.time() < deadline:
        time.sleep(0.05)
    assert not alive(child)


def test_expand_fills_the_command_line():
    out = E.expand(["~/engines/bin/serve", "--port", "{port}", "{python}", "{repo}/x"], 8091)
    assert out[0] == os.path.expanduser("~/engines/bin/serve") and out[2] == "8091" and out[3] == sys.executable
    assert out[4].endswith("/x") and E.idle_seconds("never") == 0 and E.idle_seconds(None) == 600


def test_child_engines_live_on_loopback_only(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        Engine("x", {"host": "192.0.2.10", "port": 1}, roomy(), log_dir=tmp_path)
