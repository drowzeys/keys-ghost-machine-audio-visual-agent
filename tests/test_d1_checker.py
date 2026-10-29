"""The d1-2B clip checker (video.D1Checker) against a stub /v1/systemone and a stub confirming vision model."""

import asyncio
import json
import shutil
import subprocess

import httpx
import pytest

from voice_agent import d1, video

FFMPEG = video.find_ffmpeg()
pytestmark = pytest.mark.skipif(not FFMPEG, reason="needs ffmpeg")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    out = tmp_path_factory.mktemp("clip") / "c.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24:duration=3",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def d1_stub(p_yes: dict, seen=None, fail=None):
    async def handle(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        if fail:
            raise fail
        return httpx.Response(200, json={"answers": {k: {"type": "noul", "noul": p_yes.get(k, 0.5)} for k in body["questions"]}})
    return d1.D1({"url": "http://d1.test"}, transport=httpx.MockTransport(handle))


def confirm_stub(says: str, asked=None, fail=None):
    async def handle(request):
        if asked is not None:
            asked.append(json.loads(request.content))
        if fail:
            raise fail
        return httpx.Response(200, json={"choices": [{"message": {"content": says}}]})
    return httpx.MockTransport(handle)


CLEAN = {"person": 0.97, "cut": 0.03, "wardrobe": 0.04, "place": 0.96}


def run(coro):
    return asyncio.run(coro)


def test_a_clean_clip_is_kept_with_four_frames_asked_at_once(clip):
    seen = []
    checker = video.D1Checker(d1_stub(CLEAN, seen))
    clip_state = {}
    verdict = run(checker.check(clip, clip_state, {}))
    assert verdict.ok and not verdict.retake and not verdict.notes
    assert len(seen) == 1 and len(seen[0]["images"]) == 4 and set(seen[0]["questions"]) == {"person", "cut", "wardrobe", "place"}
    assert all(i.startswith("data:image/png;base64,") for i in seen[0]["images"])
    assert clip_state["checks"]["by"] == "d1-2B" and clip_state["checks"]["faults"] == []


def test_a_sure_fault_is_retaken_and_a_doubtful_one_only_noted(clip):
    verdict = run(video.D1Checker(d1_stub({**CLEAN, "cut": 0.9, "wardrobe": 0.6})).check(clip, {}, {}))
    assert verdict.retake and not verdict.ok
    assert verdict.notes == ["the clip cut to a different shot", "someone's clothes changed"]
    person = run(video.D1Checker(d1_stub({**CLEAN, "person": 0.1})).check(clip, {}, {}))
    assert person.retake and person.notes == ["the person changed partway through"]


def test_the_stronger_checker_can_clear_a_flagged_clip(clip):
    asked = []
    checker = video.D1Checker(d1_stub({**CLEAN, "person": 0.1}), {"confirm": {"url": "http://gemma.test/v1", "model": "gemma-4-12b"}},
                              transport=confirm_stub("Yes.", asked))
    state = {}
    verdict = run(checker.check(clip, state, {}))
    # Gemma says the same person is there throughout: no retake, the note stays for the planner
    assert not verdict.retake and verdict.notes == ["the person changed partway through"]
    assert len(asked) == 1 and asked[0]["model"] == "gemma-4-12b" and len(asked[0]["messages"][0]["content"]) == 5
    assert state["checks"]["by"] == "d1-2B + gemma-4-12b"


def test_the_stronger_checker_agrees_or_is_down_and_the_fault_stands(clip):
    agree = video.D1Checker(d1_stub({**CLEAN, "person": 0.1}), {"confirm": {"url": "http://gemma.test/v1"}},
                            transport=confirm_stub("No"))
    assert run(agree.check(clip, {}, {})).retake
    down = video.D1Checker(d1_stub({**CLEAN, "cut": 0.95}), {"confirm": {"url": "http://gemma.test/v1"}},
                           transport=confirm_stub("", fail=httpx.ConnectError("down")))
    assert run(down.check(clip, {}, {})).retake


def test_no_d1_answer_keeps_the_clip(clip):
    verdict = run(video.D1Checker(d1_stub(CLEAN, fail=httpx.ConnectError("down"))).check(clip, {}, {}))
    assert verdict.ok and not verdict.retake and not verdict.notes


def test_only_the_named_checks_are_asked(clip):
    seen = []
    run(video.D1Checker(d1_stub(CLEAN, seen), {"checks": ["person", "cut"], "frames": 3}).check(clip, {}, {}))
    assert set(seen[0]["questions"]) == {"person", "cut"} and len(seen[0]["images"]) == 3
