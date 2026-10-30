"""Pocket's hybrid mode (pocket.mode "hybrid"): Omni answers small talk, the brain everything else. Against the fake
Omni server of test_pocket.py and a fake brain: the words decide the route, Omni's held reply is let out or hung up,
the brain's reply is spoken by the profile's voice, both share one conversation, and an Omni reply that repeats an
earlier one goes to the brain instead."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from voice_agent import pocket, server
from test_pocket import AUDIO, Acp, Direct, Eyes, FakeOmni, Page, Voice, sent_picture, settle, with_camera


@pytest.fixture
def omni():
    servers = []

    def make(**kw):
        servers.append(FakeOmni(**kw))
        return servers[-1]
    yield make
    for s in servers:
        s.close()


@pytest.fixture(autouse=True)
def no_vad(monkeypatch):
    monkeypatch.setattr(server, "Vad", lambda path: None)


class Brain:
    """The front brain (Qwen on the Mac): streams its reply a few words at a time."""

    def __init__(self, reply="[happy] A teaser runs about a minute. Long enough to hook, short enough to leave them wanting.",
                 gap=0.0):
        self.reply, self.gap = reply, gap
        self.asked: list = []

    async def stream(self, messages, image_png=None):
        self.asked.append(messages)
        for word in self.reply.split(" "):
            if self.gap:
                await asyncio.sleep(self.gap)
            yield word + " "

    async def warm(self, messages):
        self.warmed = getattr(self, "warmed", []) + [messages]


def hybrid(url, heard="thanks boo", brain=None, agent=None, asr_delay=0.05, eyes=None, **spec):
    def asr(audio):
        time.sleep(asr_delay)
        return heard

    voice = Voice()
    brain = brain or Brain()
    agents = {agent.name: agent} if agent else {}
    profile = {"pocket": {"url": url, "model": "lfm3-omni-2b", "system": "You are Boo.", "mode": "hybrid", **spec},
               "brain": {"system": "You are Boo, the brain."},
               "voice": {"url": "http://kokoro.test/v1", "model": "kokoro-82m", "voice": "bf_emma"},
               "turn": {}, "agent": agent.name if agent else None}
    engines = SimpleNamespace(profile=profile, asr=asr, asr_pool=ThreadPoolExecutor(1), voices={"full": voice},
                              voice=voice, eyes=eyes, memory=None, recall=None, d1=None, agents=agents, lookup=None,
                              pocket=pocket.Pocket(profile["pocket"]), brain=brain, vad_path="")
    page = Page()
    return server.Conversation(page, engines), page, voice, brain


# ----- the pieces -----

@pytest.mark.parametrize("text", ["Hey Boo, thanks!", "Okay, cool.", "Good morning, how are you?", "Thank you so much.",
                                  "That's awesome!", "I'm good, how are you?", "Bye Boo, see you later.", "Hi there",
                                  "Nice to meet you, Boo.", "Haha, that's hilarious.", "Yes."])
def test_small_talk_goes_to_omni(text):
    assert pocket.small_talk(text)


@pytest.mark.parametrize("text", ["How long should a teaser trailer be?", "Okay, render it.", "Who are you?",
                                  "Hello, what can you do?", "Stop. Actually, never mind, go back to the previous shot.",
                                  "Thanks, and what was the second act break again?", "Hey, give me a logline.", "",
                                  "Okay okay okay okay okay okay okay okay okay okay okay okay okay"])
def test_everything_else_goes_to_the_brain(text):
    assert not pocket.small_talk(text)


def test_a_bare_yes_to_a_question_goes_to_the_brain():
    assert not pocket.small_talk("Yes, please.", "[curious] Want me to render it in nine by sixteen?")
    assert not pocket.small_talk("No.", "Should I queue all three? [interrupted]?")
    assert pocket.small_talk("Yes.", "[happy] Done, the fox clip is queued.")
    assert pocket.small_talk("Thanks!", "[curious] Want me to render it?")  # thanks answers nothing


def test_repeats_catches_an_old_answer_said_again():
    earlier = ["On it, rendering the fox clip now.", "A teaser runs about a minute."]
    assert pocket.repeats("On it, rendering the fox clip now, with snow.", earlier)
    assert pocket.repeats("A teaser runs about one minute.", earlier)
    assert not pocket.repeats("You're welcome, anytime!", earlier)
    assert not pocket.repeats("On it.", earlier)  # too short to tell


def test_hybrid_sends_omni_fewer_turns():
    assert pocket.Pocket({"url": "http://x", "mode": "hybrid"}).turns == 4
    assert pocket.Pocket({"url": "http://x"}).turns == 12
    assert pocket.Pocket({"url": "http://x", "mode": "hybrid", "history_turns": 6}).turns == 6
    with pytest.raises(ValueError):
        pocket.Pocket({"url": "http://x", "mode": "both"})


# ----- whole turns -----

def test_small_talk_is_answered_by_omni_and_the_brain_is_not_asked(omni):
    fake = omni(text=("You're welcome! ", "Anytime."))
    talk, page, voice, brain = hybrid(fake.url, heard="thanks boo")

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert brain.asked == [] and voice.said == []
    assert page.kinds().count("pcm") == 3 and [e["text"] for e in page.of("transcript")] == ["thanks boo"]
    assert [e["text"] for e in page.of("reply")] == ["You're welcome!", "Anytime."]
    metrics = page.of("metrics")[0]
    assert metrics["route"] == "omni" and metrics["first_audio_ms"] >= metrics["asr_ms"]  # held until the words chose
    assert talk.history[1:] == [{"role": "user", "content": "thanks boo"},
                                {"role": "assistant", "content": "You're welcome! Anytime."}]


def test_a_question_hangs_omni_up_and_the_brain_answers_in_the_profile_voice(omni):
    fake = omni(text=("Hmm, about thirty seconds. ",), chunks=40, gap=0.03)
    talk, page, voice, brain = hybrid(fake.url, heard="how long should a teaser trailer be", asr_delay=0.1)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert len(fake.bodies) == 1  # Omni was started before the words were in
    assert fake.hung_up.wait(3) and fake.sent_chunks < 40
    assert "pcm" in page.kinds() and page.of("audio_start")[0]["sample_rate"] == 24000
    assert all("thirty" not in e["text"] for e in page.of("reply"))  # nothing of Omni's reply went out
    assert voice.said and "teaser runs about a minute" in " ".join(voice.said)
    assert page.of("metrics")[-1]["route"] == "brain"
    assert brain.asked[0][-1] == {"role": "user", "content": "how long should a teaser trailer be"}
    assert [m["role"] for m in talk.history[1:]] == ["user", "assistant"]
    assert talk.history[2]["content"].startswith("[happy] A teaser runs")


def test_without_speculation_omni_is_never_asked_a_question(omni):
    fake = omni()
    talk, page, voice, brain = hybrid(fake.url, heard="what is a crash zoom", speculate=False)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
        talk.e.asr = lambda audio: "cool, thanks"
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert len(fake.bodies) == 1 and len(brain.asked) == 1  # one turn each
    assert [e["route"] for e in page.of("metrics")] == ["brain", "omni"]


def test_both_models_share_one_conversation(omni):
    fake = omni(text=("Glad you like it!",))
    talk, page, voice, brain = hybrid(fake.url, heard="how long should a teaser trailer be")

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
        talk.e.asr = lambda audio: "that's great, thanks"
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
        talk.e.asr = lambda audio: "and a trailer?"
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    # Omni's second request carried the brain's answer, the brain's second the Omni line
    assert fake.bodies[-1]["history"][-2:] == [
        {"role": "user", "content": "how long should a teaser trailer be"},
        {"role": "assistant", "content": "A teaser runs about a minute. Long enough to hook, short enough to leave them wanting."}]
    assert [m["content"] for m in brain.asked[-1][1:]] == [
        "how long should a teaser trailer be",
        "[happy] A teaser runs about a minute. Long enough to hook, short enough to leave them wanting.",
        "that's great, thanks", "Glad you like it!", "and a trailer?"]


def test_an_omni_reply_that_repeats_an_earlier_one_goes_to_the_brain(omni):
    fake = omni(text=("A teaser runs about a minute. ", "Long enough to hook."))
    talk, page, voice, brain = hybrid(fake.url, heard="cool thanks", brain=Brain("[happy] Anytime, glad it helps."))
    talk.history += [{"role": "user", "content": "how long is a teaser"},
                     {"role": "assistant", "content": "[happy] A teaser runs about a minute."}]

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert voice.said == ["Anytime, glad it helps."]
    assert all("teaser" not in e["text"] for e in page.of("reply"))
    assert page.of("metrics")[-1]["fallback"] == 1 and page.of("metrics")[-1]["route"] == "brain"
    assert len(page.of("transcript")) == 1
    assert talk.history[-1]["content"] == "[happy] Anytime, glad it helps."


def test_omni_down_in_hybrid_the_brain_answers_small_talk_too():
    talk, page, voice, brain = hybrid("http://127.0.0.1:9", heard="hello")

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert voice.said and not page.of("error") and talk.history[-1]["role"] == "assistant"


def test_a_brain_hand_over_goes_to_the_back_agent_as_on_the_live_boo(omni):
    fake = omni()
    agent = Direct()
    brain = Brain("[thinking] [deep] On it, working that out.")
    talk, page, voice, _ = hybrid(fake.url, heard="what is six times seven", brain=brain, agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep")[0]["agent"] == "redsnow"
    assert agent.asked and agent.asked[0][-1]["content"] == "what is six times seven"
    assert " ".join(voice.said).endswith("It is forty-two.")


def test_asking_hermes_by_name_says_on_it_and_starts_the_task(omni):
    fake = omni()
    agent = Acp()
    talk, page, voice, brain = hybrid(fake.url, heard="ask hermes to render the fox clip", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk, 4)
    asyncio.run(go())
    assert page.of("deep") and agent.prompts[0].startswith("ask hermes to render the fox clip")
    assert voice.said[-1] == "Rendered the fox clip." and brain.asked == []  # the fixed "On it" line, no brain guess


def test_barge_in_cuts_a_brain_reply(omni):
    fake = omni()
    long = "[neutral] " + " ".join(["The lighthouse keeper climbs the stairs every night and counts them."] * 6)
    talk, page, voice, brain = hybrid(fake.url, heard="tell me about the keeper", brain=Brain(long, gap=0.01))
    talk.e.voices["full"] = talk.e.voice = SlowVoice()

    async def go():
        await talk.start_turn(audio=AUDIO)
        for _ in range(300):
            if "pcm" in page.kinds():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.5)  # half a second of it heard
        await talk.interrupt("barge-in")
        await settle(talk)
    asyncio.run(go())
    assert page.of("interrupt")[0]["why"] == "barge-in" and "audio_end" not in page.kinds()
    assert "[interrupted]" in talk.history[-1]["content"]


def test_barge_in_cuts_an_omni_reply(omni):
    fake = omni(text=("Good morning to you too, and what a lovely day it is for making films. ",), chunks=40, gap=0.03)
    talk, page, voice, brain = hybrid(fake.url, heard="good morning boo")

    async def go():
        await talk.start_turn(audio=AUDIO)
        for _ in range(300):
            if "pcm" in page.kinds():
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
        await talk.interrupt("barge-in")
        await settle(talk)
    asyncio.run(go())
    assert page.of("interrupt") and fake.hung_up.wait(3) and brain.asked == []
    assert "[interrupted]" in talk.history[-1]["content"]


def test_a_new_turn_while_the_words_are_awaited_stops_the_held_omni(omni):
    fake = omni(chunks=40, gap=0.03)
    talk, page, voice, brain = hybrid(fake.url, heard="hello", asr_delay=0.3)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await asyncio.sleep(0.1)
        await talk.interrupt("barge-in")
        await asyncio.sleep(0.5)
    asyncio.run(go())
    assert fake.hung_up.wait(3) and "pcm" not in page.kinds() and talk.history[1:] == []


# ----- the camera -----

@pytest.mark.parametrize("text", ["Where am I right now?", "What does my sign say?", "Can you read this?",
                                  "What colour is my hat?", "Do I look tired?"])
def test_camera_words_the_eyes_list_misses(text):
    assert pocket.looks(text) and not pocket.looks("Where is the second act break?")


def test_hybrid_with_vision_omni_answers_a_camera_question_with_the_frame(omni):
    fake = omni(text=("You're wearing a red hoodie.",))
    eyes = Eyes()
    talk, page, voice, brain = hybrid(fake.url, heard="what am I wearing", eyes=eyes, vision=True)
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert "images" not in fake.bodies[0]  # the speculative start went without it, and was hung up
    assert sent_picture(fake.bodies[-1]).format == "PNG" and pocket.VISION_NOTE in fake.bodies[-1]["system"]
    assert brain.asked == [] and eyes.asked == [] and voice.said == []
    assert page.of("metrics")[-1]["route"] == "omni-eyes"
    assert talk.history[-1]["content"] == "You're wearing a red hoodie."


def test_hybrid_with_vision_sides_go_mirrored_and_without_the_conversation(omni):
    fake = omni(text=("It's in your left hand.",))
    talk, page, voice, brain = hybrid(fake.url, heard="is the apple in my left hand or my right hand",
                                      eyes=Eyes(), vision=True, vision_mirror="sides", vision_sides="omni")
    talk.history += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    body = fake.bodies[-1]
    assert body["history"] == [] and sent_picture(body).convert("RGB").getpixel((5, 240))[2] > 150  # mirrored
    assert brain.asked == []


def test_hybrid_with_vision_finger_counts_stay_with_the_hand_tracker(omni):
    fake = omni()
    talk, page, voice, brain = hybrid(fake.url, heard="how many fingers am I holding up", eyes=Eyes(), vision=True,
                                      brain=Brain("[neutral] Two fingers on your right hand."))
    talk.hands, talk.hands_at = [{"side": "right", "up": ["index", "middle"], "count": 2}], time.monotonic()
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert "Hand tracking" in brain.asked[0][-1]["content"] and page.of("metrics")[-1]["route"] == "brain"


@pytest.mark.parametrize("heard", ["where am I right now", "what does my sign say"])
def test_hybrid_without_vision_a_camera_question_goes_to_the_eyes_with_the_picture(omni, heard):
    fake = omni()
    eyes = Eyes()
    talk, page, voice, brain = hybrid(fake.url, heard=heard, eyes=eyes)
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert eyes.asked and eyes.asked[0][1] and brain.asked == []  # the camera model, with the picture; never Qwen alone
    assert all("images" not in b for b in fake.bodies)


def test_small_talk_with_the_camera_on_still_goes_to_omni_without_the_picture(omni):
    fake = omni(text=("Hey, good to see you!",))
    talk, page, voice, brain = hybrid(fake.url, heard="hey boo", eyes=Eyes(), vision=True)
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert len(fake.bodies) == 1 and "images" not in fake.bodies[0] and page.of("metrics")[-1]["route"] == "omni"


def test_hybrid_with_vision_sides_go_to_the_eyes_by_default(omni, monkeypatch):
    fake = omni()
    asked = []

    async def sides(eyes, mirrored_png, question):
        asked.append(question)
        return "On your left side."
    monkeypatch.setattr(server, "sides", sides)
    talk, page, voice, brain = hybrid(fake.url, heard="is the apple in my left hand or my right hand", eyes=Eyes(),
                                      vision=True, vision_mirror="sides")
    with_camera(talk)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert asked and voice.said == ["On your left side."] and brain.asked == []
    assert all("images" not in b for b in fake.bodies)


def test_hybrid_warms_the_brain_while_the_person_speaks_and_plain_pocket_does_not(omni):
    fake = omni()
    talk, _, _, brain = hybrid(fake.url)
    from test_pocket import conversation
    plain, _, _ = conversation(fake.url)

    async def go():
        for t in (talk, plain):
            t.audio_at = time.monotonic()
            t.start_warming()
        await asyncio.sleep(0.1)
        talk.stop_warming()
        plain.stop_warming()
    asyncio.run(go())
    assert brain.warmed and "Your name is Boo" in brain.warmed[0][0]["content"]  # the system text as replies have it
    assert brain.warmed[0][-1] == {"role": "user", "content": "\u2026"} and plain.warming is None


class SlowVoice(Voice):
    """Kokoro as it is: a second of speech per sentence, sent at once."""

    async def stream(self, text, voice=None):
        self.said.append(text)
        yield b"\x00\x00" * 24000


def test_a_brain_reply_lasts_while_the_page_plays_it_so_barge_in_cuts_it_where_it_was_heard(omni):
    fake = omni()
    talk, page, _, _ = hybrid(fake.url, heard="tell me about the keeper", brain=Brain(
        "[neutral] The keeper climbs the stairs every night. He counts every step on the way. Then he lights the lamp."))
    talk.e.voices["full"] = talk.e.voice = SlowVoice()

    async def go():
        await talk.start_turn(audio=AUDIO)
        for _ in range(300):
            if page.kinds().count("pcm") == 2:  # two pieces: "The keeper ... night." and the other two sentences
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(1.3)  # all sent at once; the page is in the second piece
        assert not talk.reply.done()
        await talk.interrupt("barge-in")
        await settle(talk)
    asyncio.run(go())
    assert page.of("interrupt") and "audio_end" not in page.kinds()
    said = talk.history[-1]["content"]
    assert said.startswith("[neutral] The keeper climbs the stairs every night. He") and "[interrupted]" in said
    assert said.index("[interrupted]") < said.index("lights the lamp")  # the rest is stored after the cut


def test_a_clarifying_question_is_no_promise():
    from voice_agent import backstage
    assert not backstage.promised("Let me ask: do you want the fox clip in nine by sixteen?")
    assert not backstage.promised("I'm checking: did you mean the Last Tram script? ")
    assert backstage.promised("Let me check the render queue.") and backstage.promised("On it, rendering now.")


def test_a_brain_reply_that_asks_to_confirm_is_not_handed_over(omni):
    fake = omni()
    agent = Acp()
    talk, page, voice, brain = hybrid(fake.url, heard="render the fax clip in nine by six", agent=agent,
                                      brain=Brain("[curious] Let me ask: did you mean the fox clip in nine by sixteen?"))

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert not page.of("deep") and not agent.prompts and voice.said


def test_a_promise_in_omnis_small_talk_hands_nothing_over(omni):
    fake = omni(text=("Sounds good, I'll keep working on it. See you later!",))
    agent = Acp()
    talk, page, voice, brain = hybrid(fake.url, heard="bye boo, see you later", agent=agent)

    async def go():
        await talk.start_turn(audio=AUDIO)
        await settle(talk)
    asyncio.run(go())
    assert not page.of("deep") and not agent.prompts and page.of("metrics")[-1]["route"] == "omni"
