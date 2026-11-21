"""The engine registry (gma/registry.py) and the pluggable brain (gma/brain.py): no model, no GPU, no network beyond
a stand-in server on loopback."""

import asyncio
import json
import threading
from pathlib import Path

import httpx
import pytest

from gm_core import engine as E
from gm_core.engine import Held
from gm_core.memstat import MemReader
from gma import brain as B
from gma.registry import Registry, derived_engines, engine_for_url

from fake_engine import serve

ROOT = Path(__file__).resolve().parent.parent
ROOMY = MemReader("spark", meminfo=lambda: "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n")


def example(name: str) -> dict:
    return json.loads((ROOT / "profiles" / f"{name}.example.json").read_text())


@pytest.fixture
def running():
    server = serve(E.free_port())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_profile_without_engines_adopts_what_its_urls_name():
    profile = {"brain": {"url": "http://127.0.0.1:8091/v1", "model": "b"}, "voice": {"url": "http://127.0.0.1:8098/v1", "model": "d"},
               "voice_lite": {"url": "http://127.0.0.1:8197/v1", "model": "k"}, "eyes": {"url": "http://127.0.0.1:8099/v1", "model": "c"},
               "asr": {"kind": "parakeet-mlx", "model": "p"}, "agents": {"deep": {"url": "http://192.0.2.1:8188/v1"}}}
    found = derived_engines(profile)
    assert {n: (s["role"], s["mode"], s.get("port")) for n, s in found.items()} == {
        "brain@8091": ("brain", "adopt", 8091), "voice@8098": ("voice", "adopt", 8098),
        "voice@8197": ("voice", "adopt", 8197), "vision@8099": ("vision", "adopt", 8099), "asr": ("asr", "inprocess", None)}
    registry = Registry(profile, ROOMY)
    assert engine_for_url(registry, "http://127.0.0.1:8099/v1") == "vision@8099"
    assert engine_for_url(registry, "http://192.0.2.1:8099/v1") is None  # not this machine's


@pytest.mark.parametrize("name", ["spark", "mac"])
def test_example_profiles_build_a_registry_and_brains(name):
    profile = example(name)
    registry = Registry(profile, ROOMY)
    roles = {e.role for e in registry.engines.values()}
    assert {"brain", "voice", "vision", "asr"} <= roles
    for one in registry.engines.values():
        assert getattr(one, "mode") in ("managed", "adopt", "inprocess")
    brains = B.make_brains(profile, registry)
    assert brains.main.model == profile["brain"]["model"] and brains.eyes is not None
    assert engine_for_url(registry, profile["brain"]["url"]) is not None  # every brain request goes through its engine


def test_registry_adopts_running_engines_and_counts_requests(running):
    profile = {"engines": {"bolt": {"role": "brain", "mode": "managed", "port": running, "argv": ["/nonexistent"]},
                           "drama": {"role": "voice", "mode": "adopt", "port": E.free_port()}}}
    registry = Registry(profile, ROOMY)
    assert registry.adopt_running() == {"bolt": True, "drama": False}

    async def go():
        async with registry.use("bolt") as base:
            assert base == f"http://127.0.0.1:{running}" and registry["bolt"].in_flight == 1
        assert registry["bolt"].in_flight == 0
        async with registry.use(None) as nothing:  # no engine: nothing counted
            assert nothing is None

    asyncio.run(go())
    states = {e["name"]: e["state"] for e in registry.status()["engines"]}
    assert states == {"bolt": "adopted", "drama": "unloaded"}


def test_use_refuses_at_once_on_the_voice_path_while_held(running):
    registry = Registry({"engines": {"bolt": {"role": "brain", "mode": "adopt", "port": running}}}, ROOMY)
    registry.adopt_running()
    registry["bolt"].hold = "a render (sharp)"

    async def go():
        async with registry.use("bolt", wait=False):
            pass

    with pytest.raises(Held) as held:
        asyncio.run(go())
    assert held.value.said.startswith("My brain is paused")


def test_monitor_ticks_and_shutdown_leaves_adopted_servers_running(running):
    registry = Registry({"engines": {"bolt": {"role": "brain", "mode": "adopt", "port": running}}}, ROOMY, monitor_s=0.05)
    registry.adopt_running()
    registry.start_monitor()
    registry.shutdown()
    assert registry["bolt"].state() == "adopted"


# ---------- the pluggable brain ----------

def sse(*pieces: str) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in pieces]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def test_brain_requests_go_through_the_engine_lifecycle(running):
    profile = {"engines": {"bolt": {"role": "brain", "mode": "adopt", "port": running}},
               "brain": {"url": f"http://127.0.0.1:{running}/v1", "model": "nvidia-nemotron-3.5-bolt"}}
    registry = Registry(profile, ROOMY)
    registry.adopt_running()
    seen = []

    def chat(request: httpx.Request) -> httpx.Response:
        seen.append(registry["bolt"].in_flight)
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=sse("Hi", " there."))
        return httpx.Response(200, json={"choices": [{"message": {"content": "a plan"}}]})

    brains = B.make_brains(profile, registry)
    brains.main.client = httpx.AsyncClient(transport=httpx.MockTransport(chat))

    async def go():
        said = [p async for p in brains.main.stream([{"role": "user", "content": "hello"}])]
        whole = await brains.main.complete([{"role": "user", "content": "plan"}], max_tokens=50)
        await brains.main.warm([{"role": "user", "content": "…"}])
        return said, whole

    said, whole = asyncio.run(go())
    assert said == ["Hi", " there."] and whole == "a plan"
    assert seen == [1, 1, 1] and registry["bolt"].in_flight == 0


def test_brain_is_swapped_by_the_profile_alone_and_split_is_expressible():
    # the brain evaluation's candidates are each just a URL and a model name behind the same client
    for model in ("nvidia-nemotron-3.5-bolt", "gemma-4-12b-it", "qwen3.8-27b"):
        brains = B.make_brains({"brain": {"url": "http://127.0.0.1:9000/v1", "model": model}})
        assert brains.main.model == model and not brains.split
    split = B.make_brains({"brain": {"url": "http://127.0.0.1:9000/v1", "model": "qwen3.8-27b", "extra": {"max_tokens": 200}},
                           "eyes": {"url": "http://127.0.0.1:8099/v1", "model": "cosmos3"},
                           "ears": {"url": "http://127.0.0.1:8300/v1", "model": "small-audio-lm"}})
    assert split.split and split.describe() == {"main": "qwen3.8-27b", "eyes": "cosmos3", "ears": "small-audio-lm", "split": True}
    assert split.eyes.extra == {"max_tokens": 200}  # the eyes inherit the brain's settings, as Ghost Machine did
    with pytest.raises(ValueError, match="unknown brain kind"):
        B.make_client({"kind": "telepathy", "url": "http://127.0.0.1:1/v1", "model": "x"})


@pytest.mark.parametrize("name", ["spark", "mac"])
def test_example_profiles_direct_with_qwen_and_keep_bolt_selectable(name):
    profile = example(name)
    registry = Registry(profile, ROOMY)
    ap = profile["autopilot"]
    assert ap["director"] == "qwen" and {"qwen", "bolt"} <= set(ap["director_choices"])
    for choice in ap["director_choices"].values():
        assert choice["engine"] in registry.engines
        assert engine_for_url(registry, choice["url"]) == choice["engine"]
    # Qwen3.8-27B is the brain too, so it directs as the main brain: one engine, one set of lanes
    assert B.director_spec(profile) is None and B.make_brains(profile, registry).director is None
    # Bolt chosen as the Director: a client of its own, through its own engine
    bolt = B.make_brains({**profile, "autopilot": {**ap, "director": "bolt"}}, registry)
    assert bolt.director.model == "nvidia-nemotron-3.5-bolt" and bolt.main.model == "qwen38-a"
    assert bolt.describe()["director"] == "nvidia-nemotron-3.5-bolt"


def test_director_choice_beside_another_voice_brain():
    profile = {"brain": {"url": "http://127.0.0.1:8091/v1", "model": "nvidia-nemotron-3.5-bolt", "system": "voice",
                         "extra": {"max_tokens": 200}},
               "autopilot": {"director": "qwen", "director_choices": {
                   "qwen": {"url": "http://127.0.0.1:8093/v1", "model": "qwen38-a"}}}}
    brains = B.make_brains(profile)
    assert brains.main.model == "nvidia-nemotron-3.5-bolt" and brains.director.model == "qwen38-a"
    assert brains.director.extra == {"max_tokens": 200}  # the brain's settings; each Director question overrides them
    for none in ("brain", "missing", None):
        assert B.make_brains({**profile, "autopilot": {**profile["autopilot"], "director": none}}).director is None
    inline = {**profile, "autopilot": {"director": {"url": "http://127.0.0.1:8093/v1", "model": "qwen38-a"}}}
    assert B.make_brains(inline).director.model == "qwen38-a"


def test_a_new_brain_kind_registers_itself():
    class Echo:
        kind = "echo"

        def __init__(self, spec, use=None):
            self.model, self.url, self.use = spec["model"], spec.get("url", ""), use

        async def stream(self, messages, image_png=None):
            yield messages[-1]["content"]

    B.KINDS["echo"] = Echo
    try:
        brains = B.make_brains({"brain": {"kind": "echo", "model": "echo-1"}})

        async def go():
            return [p async for p in brains.main.stream([{"role": "user", "content": "ping"}])]

        assert asyncio.run(go()) == ["ping"]
    finally:
        B.KINDS.pop("echo")
