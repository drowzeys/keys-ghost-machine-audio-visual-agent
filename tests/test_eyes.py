"""The camera model's side answers said as the person's own sides (voice_agent/engines.py), and a profile's "thinking":
false reaching every eyes request."""

import asyncio
import json

import httpx
import pytest

from voice_agent import engines
from voice_agent.engines import Brain, picture_sides, sides


@pytest.mark.parametrize("said, spoken", [
    # the long form the question asks for, unchanged
    ("The apple is on the left of the picture.", "The apple is on your left side."),
    ("The mug is on the right side of the image.", "The mug is on your right side."),
    ("The cup is on the left of the picture and the phone is on the right of the picture.",
     "The cup is on your left side and the phone is on your right side."),
    # the side alone
    ("Left.", "On your left side."),
    ("Right", "On your right side."),
    ("right.", "On your right side."),
    ("The right.", "On your right side."),
    ("On the left side.", "On your left side."),
    ("It's on the right.", "On your right side."),
    ("To the left!", "On your left side."),
    # "on the left" in a sentence, without "of the picture"
    ("The banana is on the right.", "The banana is on your right side."),
    ("The ball is to the left, near the edge.", "The ball is on your left side, near the edge."),
])
def test_a_picture_side_is_said_as_the_persons_own(said, spoken):
    assert picture_sides(said) == spoken


@pytest.mark.parametrize("said", [
    "Middle.",
    "The mug is in the middle of the image.",
    "The apple is in the right hand.",  # a hand: the habit the mirrored question avoids, not a picture side
    "It is in the person's left hand.",
    "The book is on the left side of the desk.",
    "It's on the left of you.",
    "Leftover pizza.",
    "I can't tell.",
    "",
])
def test_no_picture_side_gives_none(said):
    assert picture_sides(said) is None


def mock(reply_text, seen):
    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if request.headers.get("accept") == "text/event-stream" or seen[-1].get("stream"):
            chunk = {"choices": [{"delta": {"content": reply_text}}]}
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"choices": [{"message": {"content": reply_text}}]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def test_sides_takes_a_bare_side_from_the_camera_model():
    seen = []
    eyes = Brain({"url": "http://eyes.test/v1", "model": "lfm3-vl-2b"})
    eyes.client = mock("Left.", seen)
    assert asyncio.run(sides(eyes, "iVBOR", "Is the apple in my left hand or my right hand?")) == "On your left side."
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_false_reaches_every_eyes_request():
    brain = {"url": "http://brain.test/v1", "model": "bolt", "extra": {"temperature": 0.7, "max_tokens": 200}}
    eyes = Brain({**brain, "url": "http://eyes.test/v1", "model": "cosmos3", "thinking": False})
    assert eyes.extra == {"temperature": 0.7, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
    assert "chat_template_kwargs" not in brain["extra"]  # the brain's own settings are left as they were
    seen = []
    eyes.client = mock("You're wearing a red hoodie.", seen)

    async def answer():
        return "".join([p async for p in eyes.stream([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], "iVBOR")])

    assert asyncio.run(answer()) == "You're wearing a red hoodie."
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False} and seen[0]["temperature"] == 0.7
    assert seen[0]["messages"][-1]["content"][0]["type"] == "image_url"


def test_other_kwargs_are_kept_and_no_flag_changes_nothing():
    keep = Brain({"url": "http://x.test/v1", "model": "m", "thinking": False,
                  "extra": {"chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": True}}})
    assert keep.extra["chat_template_kwargs"] == {"reasoning_effort": "low", "enable_thinking": False}
    plain = Brain({"url": "http://x.test/v1", "model": "m", "extra": {"temperature": 0}})
    assert plain.extra == {"temperature": 0}
    assert Brain({"url": "http://x.test/v1", "model": "m", "thinking": True}).extra == {}


def test_the_example_profiles_turn_cosmos_thinking_off():
    from pathlib import Path
    root = Path(engines.__file__).resolve().parent.parent / "profiles"
    for name in ("mac.example.json", "spark.example.json", "eyes-lfm3-vl.example.json"):
        profile = json.loads((root / name).read_text())
        assert profile["eyes"]["thinking"] is False, name
