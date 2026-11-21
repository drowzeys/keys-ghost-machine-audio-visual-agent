"""The memory governor (gma/governor.py) and its readers (gm_core/memstat.py), on faked /proc/meminfo and vm_stat, with
the FP4 clip worker's measured peaks stood in for (gma/clips.py NEED_MB): no model, no GPU, no Studio."""

import asyncio

import pytest

from gm_core.engine import Engine, Held, Refusal
from gm_core.memstat import MemReader, parse_meminfo, parse_vm_stat
from gma.governor import SAID, Governor
from gma.registry import Registry

SPARK_KB = 124546 * 1024  # the Spark the Studio's estimates were measured on


class Machine:
    """A faked machine: MemAvailable (MB) that falls when an engine loads and climbs when it is unloaded."""

    def __init__(self, available_mb: int, total_mb: int = 124546) -> None:
        self.available, self.total = available_mb, total_mb

    def meminfo(self) -> str:
        return f"MemTotal: {self.total * 1024} kB\nMemFree: 1000 kB\nMemAvailable: {self.available * 1024} kB\n"

    def reader(self) -> MemReader:
        return MemReader("spark", meminfo=self.meminfo)


class FakeEngine(Engine):
    """An engine that is "loaded" without a process (its memory taken from the faked machine)."""

    def __init__(self, name, spec, reader, machine):
        super().__init__(name, spec, reader)
        self.machine, self.up = machine, False

    def bring_up(self):
        self.up, self.phase = True, "loaded"
        self.machine.available -= int(self.resident_gb * 1024)

    def loaded(self):
        return self.up or self.phase == "adopted"

    def stop(self, why):
        with self.mutex:
            if self.up:
                self.machine.available += int(self.resident_gb * 1024)
            self.up, self.phase = False, "unloaded"
            self.changed.notify_all()


# measured on one Spark (2026-10-27): Bolt 30.4 GB, Cosmos3 3.4, Kokoro 3.5, Parakeet 5.4; Drama 3 from GM's profile
SPEC = {"bolt": ("brain", 43, 30.4), "drama3": ("voice", 12, 8), "kokoro": ("voice", 5, 3.5), "cosmos": ("vision", 6, 3.4)}
APP_MB = 2048  # the app process itself


class FakeRenderer:
    """The Studio's estimate as the headless renderer gives it, with a cache that can be freed (its ComfyUI's /free).
    "fh8-480" is the render these schedules were measured on (61 +- 6 GB, 2026-10-27)."""

    def __init__(self, platform="spark"):
        from gma.governor import StatedBudget
        self.stub, self.freed, self.label = StatedBudget(platform), [], "fake"

    def memory_request(self, render):
        if render.get("preset") == "fh8-480":
            return {"need_mb": 61 * 1024, "error_mb": 6 * 1024, "route": "fh8-480", "rows": None, "source": "measured"}
        return self.stub.memory_request(render)

    def floor_mb(self):
        return 12460  # the Studio's floor on that Spark

    def free_cache(self):
        self.freed.append(True)
        return {"freed": True, "why": "test"}


def spark(available_mb=118_700, loaded=("bolt", "kokoro", "cosmos", "drama3"), adopted=(), schedules=None):
    machine = Machine(available_mb)
    reader = machine.reader()
    registry = Registry({"engines": {"asr": {"role": "asr", "mode": "inprocess", "resident_gb": 3}}}, reader)
    for name, (role, need, resident) in SPEC.items():
        one = FakeEngine(name, {"role": role, "mode": "managed", "port": 1, "need_gb": need, "resident_gb": resident}, reader, machine)
        registry.engines[name] = one
        if name in adopted:
            one.phase, one.adopted = "adopted", True
            machine.available -= int(resident * 1024)
        elif name in loaded:
            one.bring_up()
    registry.engines["asr"].resident_gb = 5.4
    registry.engines["asr"].set(object())
    machine.available -= int(5.4 * 1024) + APP_MB
    governor = Governor({"schedules": schedules} if schedules else None, registry, reader, platform="spark",
                        renderer=FakeRenderer())
    registry.set_admit(governor.admit)
    reader.settle = lambda *a, **k: None
    return machine, registry, governor


# ---------- readers ----------

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              1000000.
Pages active:                            5000000.
Pages inactive:                          3000000.
Pages speculative:                         10000.
Pages throttled:                               0.
Pages wired down:                        2000000.
Pages purgeable:                           20000.
File-backed pages:                       1500000.
Anonymous pages:                         6000000.
"""


def test_mac_vm_stat_counts_free_file_backed_and_purgeable_only():
    assert parse_vm_stat(VM_STAT, 1 << 40) == (1_000_000 + 1_500_000 + 20_000) * 16384 // 2 ** 20  # 39375 MB
    mac = MemReader("mac", vm_stat=lambda: VM_STAT, pressure_critical=lambda: False, total_mb=262144)
    assert mac.available_mb() == 39375 and mac.total_mb() == 262144
    critical = MemReader("mac", vm_stat=lambda: VM_STAT, pressure_critical=lambda: True, total_mb=262144)
    assert critical.available_mb() == 0  # macOS says its pressure is critical: nothing to give
    assert MemReader("mac", vm_stat=lambda: None, total_mb=1).available_mb() == -1
    older = VM_STAT.replace("File-backed pages", "Old line")  # an older macOS: free, inactive, speculative, purgeable
    assert parse_vm_stat(older, 1 << 40) == (1_000_000 + 3_000_000 + 10_000 + 20_000) * 16384 // 2 ** 20


def test_spark_meminfo_reader():
    spark_ = MemReader("spark", meminfo=lambda: "MemTotal: 127535104 kB\nMemAvailable: 121548800 kB\n")
    assert spark_.available_mb() == 118700 and spark_.total_mb() == 124546
    assert parse_meminfo("MemAvailable: 2048 kB\nbad line\n") == {"MemAvailable": 2}
    assert MemReader("spark", meminfo=lambda: None).available_mb() == -1


# ---------- loads and renders ----------


def test_a_load_that_would_break_the_floor_is_refused_with_a_short_line():
    machine, registry, governor = spark(available_mb=50_000, loaded=())
    governor.renderer.free_cache = lambda: {"freed": False, "why": "no renderer running"}
    with pytest.raises(Refusal) as refused:
        governor.admit(registry["bolt"])  # 43 GB of ~43 GB with a 12 GB floor
    assert refused.value.said == "I don't have the memory to start my brain right now."
    assert "43 GB" in str(refused.value) and governor.refusals
    governor.admit(registry["kokoro"])  # a small one fits


# ---------- the governor: rendering ----------

def test_spark_live_fit_is_explicit_and_falls_back_under_the_floor():
    machine, registry, governor = spark()
    moves = []
    governor.hooks["asr_to_cpu"] = moves.append
    found = governor.plan("live")
    budget = found["budget"]
    # the measured render (fh8 864x480: budgeted 61 +- 6 GB) beside Bolt, Kokoro, Parakeet and Cosmos: under the floor
    assert found["request"]["need_mb"] == 61 * 1024 and found["floor_mb"] == 12460  # the Studio's floor
    assert budget["left_before_fallbacks_mb"] < budget["floor_mb"]
    assert found["evict"] == ["drama3"] and set(found["keep"]) == {"bolt", "kokoro", "asr", "cosmos", "fp4"}
    # Parakeet to the CPU is not enough here; the brain released for the DiT pass is
    assert [s["step"] for s in found["fallbacks"]] == ["asr_to_cpu", "brain_during_dit"]
    assert found["dit_evict"] == ["bolt"] and found["fits"] and budget["left_mb"] >= budget["floor_mb"]

    async def go():
        await governor.enter("live")
        assert moves == [True] and registry["bolt"].loaded()  # the brain answers until the DiT pass starts
        await governor.dit(True)
        assert not registry["bolt"].loaded()
        with pytest.raises(Held) as held:
            registry["bolt"].acquire(wait=False)
        assert held.value.said == "My brain is paused while I render. Give me a moment."
        with pytest.raises(Held):  # Drama 3 was evicted for the render: it may not load beside it
            registry["drama3"].acquire(wait=False)
        with pytest.raises(Refusal) as busy:
            await governor.enter("sharp")
        assert busy.value.said == SAID["busy"]
        await governor.leave()
        # the renderer's cache is freed before the brain may load again, and Parakeet is back on the GPU
        assert governor.renderer.freed == [True] and moves == [True, False]
        assert governor.phase == "talking" and not registry["bolt"].hold

    asyncio.run(go())


def test_spark_live_fits_without_fallbacks_when_cosmos_and_parakeet_are_not_loaded():
    machine, registry, governor = spark(loaded=("bolt", "kokoro"))
    machine.available += int(5.4 * 1024)  # no recognizer here
    registry["asr"].phase = "unloaded"
    found = governor.plan("live")
    assert found["fits"] and found["fallbacks"] == []


def test_spark_live_with_parakeet_on_cpu_alone_when_that_closes_the_gap():
    machine, registry, governor = spark(loaded=("bolt", "kokoro", "cosmos"))
    machine.available += 5 * 1024  # a little more room than the measured run: Parakeet on the CPU is enough
    governor.hooks["asr_to_cpu"] = lambda to_cpu: None
    found = governor.plan("live")
    assert found["fits"] and [s["step"] for s in found["fallbacks"]] == ["asr_to_cpu"] and found["dit_evict"] == []


def test_spark_live_with_an_adopted_brain_cannot_fall_back_and_says_so():
    machine, registry, governor = spark(loaded=("kokoro", "cosmos"), adopted=("bolt",))
    found = governor.plan("live")
    assert not found["fits"] and "bolt" in found["adopted_blockers"]


def test_a_brain_load_after_a_render_frees_the_renderers_cache_first():
    machine, registry, governor = spark(available_mb=60_000, loaded=())
    renderer = governor.renderer

    def free_cache():
        renderer.freed.append(True)
        machine.available += 50_000  # ComfyUI's ~50 GB cache given back
        return {"freed": True, "why": "test"}

    renderer.free_cache = free_cache
    governor.admit(registry["bolt"])  # refused without the free (status -6 on the Spark), fits after it
    assert renderer.freed == [True]


def test_spark_sharp_evicts_the_brain():
    machine, registry, governor = spark()
    found = governor.plan("sharp")
    assert "bolt" in found["evict"] and found["fits"]
    assert found["request"]["need_mb"] == 67584  # the FP4 clip worker's measured peak at 1376x768, 243 frames
    # with Bolt kept it would not fit: the same render under "live"'s keep list
    _, _, keep_bolt = spark(schedules={"sharp-with-bolt": {"render": {"width": 1376, "height": 768, "frames": 243},
                                                             "keep": ["bolt", "kokoro", "asr"]}})
    assert not keep_bolt.plan("sharp-with-bolt")["fits"]

    async def go():
        grant = await governor.enter("sharp")
        assert "bolt" in grant["evicted"] and not registry["bolt"].loaded()
        with pytest.raises(Held) as held:  # a voice turn while sharp renders: one short line
            registry["bolt"].acquire(wait=False)
        assert held.value.said == "My brain is paused while I render. Give me a moment."
        await governor.leave()
        assert registry["bolt"].hold == ""

    asyncio.run(go())


def test_sharp_with_an_adopted_brain_is_refused_and_says_why_shortly():
    machine, registry, governor = spark(loaded=("kokoro",), adopted=("bolt",))
    found = governor.plan("sharp")
    assert not found["fits"] and found["adopted_blockers"] == ["bolt"] and "outside this app" in found["reason"]

    async def go():
        await governor.enter("sharp")

    with pytest.raises(Refusal) as refused:
        asyncio.run(go())
    assert refused.value.said == SAID["adopted"] and governor.phase == "talking"


def test_no_room_is_said_shortly_with_the_guards_reason_kept():
    machine, registry, governor = spark(available_mb=60_000, loaded=())

    async def go():
        await governor.enter("live")

    with pytest.raises(Refusal) as refused:
        asyncio.run(go())
    assert refused.value.said == SAID["no_room"] and "needs about 61 GB" in str(refused.value)


def test_every_spoken_refusal_is_short():
    for line in SAID.values():
        assert len(line.format(role="vision")) <= 90 and "GB" not in line


def test_mac_plan_uses_the_measured_mac_peaks_and_the_profile_floor():
    mac = MemReader("mac", vm_stat=lambda: VM_STAT, pressure_critical=lambda: False, total_mb=262144)
    registry = Registry({"engines": {"asr": {"role": "asr", "mode": "inprocess", "resident_gb": 2.5}}}, mac)
    governor = Governor({"floor_gb": 40, "schedules": {"live": {"render": {"kind": "video"}, "keep": ["asr"]}}}, registry, mac)
    found = governor.plan("live")
    assert found["request"]["need_mb"] == 33 * 1024 and found["floor_mb"] == 40 * 1024
    assert not found["fits"]  # 39 GB available (faked) cannot hold a 33 GB render and a 40 GB floor
    assert governor.status()["available_mb"] == 39375


def test_the_stated_budget_says_the_fp4_workers_measured_peaks():
    from gma import clips
    from gma.governor import StatedBudget

    stub = StatedBudget("spark")
    assert stub.free_cache()["freed"] is False  # no renderer: nothing is ever freed
    assert stub.memory_request({"frames": 243})["need_mb"] == clips.NEED_MB[(864, 480, 243)]
    assert stub.memory_request({"width": 1376, "height": 768, "frames": 243})["error_mb"] == clips.ERROR_MB
    assert stub.memory_request({"need_gb": 31, "error_gb": 4}) ["need_mb"] == 31 * 1024


def test_a_recognizer_that_cannot_move_releases_the_brain_for_the_dit_instead():
    machine, registry, governor = spark()

    def broken(to_cpu):  # DGX Spark, 2026-11-07: Parakeet's move raised a CUDA illegal-address error mid-film
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    governor.hooks["asr_to_cpu"] = broken

    async def go():
        grant = await governor.enter("live")
        assert grant["asr_move_failed"].startswith("CUDA error") and not governor.asr_on_cpu
        assert governor.dit_evict == ["bolt"]
        await governor.leave()

    asyncio.run(go())


def test_a_step_that_fails_while_entering_never_leaves_the_rendering_phase_behind():
    machine, registry, governor = spark()

    def stuck(*_a, **_k):
        raise RuntimeError("release failed")
    registry["drama3"].release = stuck

    async def go():
        with pytest.raises(RuntimeError):
            await governor.enter("live")
        assert governor.phase == "talking" and not any(e.hold for e in registry.engines.values() if hasattr(e, "hold"))
        registry["drama3"].release = lambda *a, **k: True
        await governor.enter("live")  # the next render is not "already rendering"
        await governor.leave()

    asyncio.run(go())
