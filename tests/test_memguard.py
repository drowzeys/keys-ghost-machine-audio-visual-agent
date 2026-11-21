"""Ghost Studio's memory guard (gma/memguard.py and the governor's admission): idle models released biggest first to
admit a load, a load that cannot fit refused with the biggest users named (never started), the live guard's order
(release idle models, then stop the step), foreign processes named and only killed when opted in. Faked memory: no
model, no GPU."""

import pytest

from gm_core.engine import Refusal
from gma import memguard
from tests.test_governor import FakeEngine, Machine, spark


def test_a_load_that_does_not_fit_releases_idle_models_biggest_first(monkeypatch):
    monkeypatch.setattr(memguard, "biggest", lambda top=5, gpu=None, me=None: [{"pid": 7, "mb": 21000, "where": "gpu", "ours": False, "what": "/x/transcribe serve"}])
    machine, registry, governor = spark(available_mb=70_000, loaded=("bolt", "kokoro", "cosmos", "drama3"))
    big = FakeEngine("ghost", {"role": "looker", "mode": "managed", "port": 1, "need_gb": 30, "resident_gb": 22}, registry.reader, machine)
    registry.engines["ghost"] = big
    before = machine.available
    governor.admit(big)  # fits only once Bolt (30.4 GB, the biggest idle one) is released
    assert not registry["bolt"].loaded() and registry["kokoro"].loaded() and registry["drama3"].loaded()
    assert governor.released[-1]["engine"] == "bolt" and machine.available > before
    assert not registry["bolt"].hold  # released, not held: it loads again when admitted


def test_a_load_that_cannot_fit_is_refused_with_the_biggest_users_named(monkeypatch):
    monkeypatch.setattr(memguard, "biggest", lambda top=5, gpu=None, me=None: [{"pid": 7, "mb": 21000, "where": "gpu", "ours": False, "what": "/x/boo-transcribe/bin/tensorfold-native serve"}])
    machine, registry, governor = spark(available_mb=40_000, loaded=())
    huge = FakeEngine("fp4", {"role": "clips", "mode": "managed", "port": 1, "need_gb": 70, "resident_gb": 52}, registry.reader, machine)
    with pytest.raises(Refusal) as refused:
        governor.admit(huge)
    assert "tensorfold-native" in str(refused.value) and "not ours" in str(refused.value) and not huge.loaded()


class Reader:
    def __init__(self, values):
        self.values = list(values)

    def available_mb(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def test_the_live_guard_releases_idle_models_then_stops_the_step(tmp_path, monkeypatch):
    monkeypatch.setattr(memguard, "biggest", lambda top=5, gpu=None, me=None: [{"pid": 9, "mb": 52000, "where": "gpu", "ours": True, "what": "fp4_worker.py"}])
    machine, registry, governor = spark(available_mb=118_700, loaded=("bolt", "kokoro"))
    stopped = []
    guard = memguard.LiveGuard(registry, Reader([30_000]), {"memguard": {"cooldown_s": 0}}, tmp_path,
                               stop_step=lambda why: stopped.append(why) or "ended fp4", psi=lambda: 0.0)
    assert guard.tick() is None  # 30 GB, no pressure: nothing to do
    guard.reader = Reader([15_000])
    assert guard.tick()["engine"] == "bolt"  # under the 20 GB floor: the biggest idle model first
    assert guard.tick()["engine"] == "kokoro"
    act = guard.tick()
    assert act["what"] == "stopped the current step" and stopped and "fp4_worker" in stopped[0]
    assert guard.tick()["what"] == "nothing left to release"  # the step is stopped once
    assert (tmp_path / "memguard.jsonl").read_text().count("\n") == 4


def test_pressure_alone_triggers_and_an_answering_model_is_never_released(tmp_path, monkeypatch):
    monkeypatch.setattr(memguard, "biggest", lambda top=5, gpu=None, me=None: [])
    machine, registry, governor = spark(loaded=("bolt", "kokoro"))
    registry["bolt"].in_flight = 1
    guard = memguard.LiveGuard(registry, Reader([30_000]), {"memguard": {"cooldown_s": 0}}, tmp_path, psi=lambda: 35.0)
    assert guard.tick()["engine"] == "kokoro" and registry["bolt"].loaded()


def test_foreign_processes_are_only_killed_when_opted_in(monkeypatch, tmp_path):
    killed = []
    monkeypatch.setattr(memguard.os, "kill", lambda pid, sig: killed.append(pid))
    users = [{"pid": 7, "mb": 21000, "where": "gpu", "ours": False, "what": "/home/k/boo-transcribe/bin/tensorfold-native serve"}]
    machine, registry, governor = spark(loaded=())
    assert memguard.LiveGuard(registry, Reader([1]), {}, tmp_path).kill_foreign(users) == [] and not killed
    opted = memguard.LiveGuard(registry, Reader([1]), {"memguard": {"kill_foreign": ["boo-transcribe"]}}, tmp_path)
    assert opted.kill_foreign(users) and killed == [7]


def test_pressure_reads_some_avg10(tmp_path):
    p = tmp_path / "memory"
    p.write_text("some avg10=23.50 avg60=4.00 avg300=1.00 total=1\nfull avg10=2.00 avg60=0.00 avg300=0.00 total=1\n")
    assert memguard.pressure(str(p)) == 23.5 and memguard.pressure(str(tmp_path / "none")) is None


# ---------- the thermal gate and the thermal stop ----------

class Sensors:
    def __init__(self, readings):
        self.readings = list(readings)

    def __call__(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def gate(readings, **spec):
    clock, logged, said = Clock(), [], []
    record = memguard.thermal_gate({"thermal": spec}, "clip 3", read=Sensors(readings), sleep=clock.sleep, clock=clock,
                                   log=lambda what, **k: logged.append(what), say=said.append)
    return record, logged, said


def test_a_cool_machine_passes_the_gate_at_once():
    record, logged, said = gate([{"gpu_c": 60, "zone_c": 80}])
    assert record["waited_s"] == 0 and record["outcome"] == "ok" and not logged and not said


def test_the_gate_waits_for_the_gpu_and_the_zone_to_cool():
    record, logged, said = gate([{"gpu_c": 82, "zone_c": 90}, {"gpu_c": 76, "zone_c": 88}, {"gpu_c": 70, "zone_c": 86}, {"gpu_c": 68, "zone_c": 84}])
    assert record["outcome"] == "ok" and record["waited_s"] == 6.0 and record["end"]["gpu_c"] == 68
    assert logged == ["thermal gate: waited"] and "82" in said[0] and "70" in said[0]


def test_after_the_timeout_a_warm_machine_starts_with_a_warning_and_a_hot_one_fails():
    record, logged, _ = gate([{"gpu_c": 78, "zone_c": 86}], timeout_s=10)
    assert record["outcome"] == "timeout: started warm" and record["waited_s"] >= 10
    with pytest.raises(memguard.TooHot):
        gate([{"gpu_c": 89, "zone_c": 95}], timeout_s=10)


def test_the_governor_gate_refuses_cleanly_when_too_hot(monkeypatch):
    machine, registry, governor = spark()
    governor.spec = {"thermal": {"timeout_s": 0}}
    monkeypatch.setattr(memguard, "sensors", lambda: {"gpu_c": 91, "zone_c": 97, "reasons": 0x40})
    with pytest.raises(Refusal) as refused:
        governor.thermal_gate("clip 1")
    assert "too hot" in str(refused.value) and governor.cooling is None


def thermal_guard(tmp_path, readings):
    machine, registry, governor = spark(loaded=())
    stopped = []
    guard = memguard.LiveGuard(registry, Reader([90_000]), {"thermal": {}}, tmp_path, stop_step=lambda why: stopped.append(why) or "ended fp4", psi=lambda: 0.0)
    guard.read_sensors = Sensors(readings)
    return guard, stopped


def test_the_thermal_stop_on_temperature_acts_once(tmp_path):
    memguard.LAST_STOP.clear()
    guard, stopped = thermal_guard(tmp_path, [{"gpu_c": 85, "zone_c": 95, "reasons": 0x20}, {"gpu_c": 90, "zone_c": 96, "reasons": 0x20}])
    assert guard.tick(0) is None
    act = guard.tick(1)
    assert act["what"].startswith("thermal stop") and len(stopped) == 1 and memguard.LAST_STOP["kind"] == "thermal"
    assert guard.tick(2) is None and len(stopped) == 1  # still hot: one action a step


def test_the_thermal_stop_on_sustained_hardware_slowdown(tmp_path):
    guard, stopped = thermal_guard(tmp_path, [{"gpu_c": 88, "zone_c": 95, "reasons": 0x48}])
    assert all(guard.tick(t) is None for t in range(0, 11))  # 10 s of hardware slowdown: not yet
    assert guard.tick(11)["what"].startswith("thermal stop") and "slowdown" in stopped[0]
    software_only, none = thermal_guard(tmp_path, [{"gpu_c": 88, "zone_c": 95, "reasons": 0x20}])
    assert all(software_only.tick(t) is None for t in range(0, 30)) and not none  # software thermal slowdown alone: no stop


def test_pressure_with_plenty_of_memory_is_a_load_not_a_danger(tmp_path, monkeypatch):
    monkeypatch.setattr(memguard, "biggest", lambda top=5, gpu=None, me=None: [])
    machine, registry, governor = spark(loaded=("bolt", "kokoro"))
    guard = memguard.LiveGuard(registry, Reader([45_000]), {"memguard": {"cooldown_s": 0}}, tmp_path, psi=lambda: 49.0)
    assert guard.tick() is None and registry["kokoro"].loaded()  # 49 pressure, 45 GB free: a model loading from disk
