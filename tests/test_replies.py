"""Cut replies and degenerate replies (voice_agent/replies.py): how a reply stopped by barge-in is stored, which
finished replies count as no answer, and the guard around the front brain's stream (hold, one retry, a fallback line
instead of silence) against stub streams."""

import asyncio

from voice_agent import replies


def run(coro):
    return asyncio.run(coro)


async def stream(*pieces):
    for piece in pieces:
        yield piece


def collect(first, earlier=(), again=None, handed=False):
    calls = {"again": 0, "retried": [], "first": 0}

    def make_again():
        calls["again"] += 1
        return stream(*(again or ()))

    def mark():
        calls["first"] += 1

    async def go():
        return [p async for p in replies.guarded(stream(*first), list(earlier), make_again, lambda: handed, mark,
                                                  calls["retried"].append)]
    return "".join(run(go())), calls


# ----- storing a cut reply -----

def test_cut_mid_sentence_gets_ellipsis_and_marker():
    assert replies.cut_off("A standard teaser") == "A standard teaser… [interrupted]"


def test_cut_drops_dangling_punctuation():
    assert replies.cut_off("On it, rendering the fox,") == "On it, rendering the fox… [interrupted]"
    assert replies.cut_off("Well — ") == "Well… [interrupted]"


def test_cut_between_sentences_keeps_full_stop():
    assert replies.cut_off("Sure thing. That works.") == "Sure thing. That works. [interrupted]"


def test_cut_of_nothing_is_nothing_and_marker_is_not_doubled():
    assert replies.cut_off("  ") == ""
    once = replies.cut_off("I am on")
    assert replies.cut_off(once) == once


def test_words_ignore_tags_and_marker():
    assert replies.words("[happy] [deep] On it… [interrupted]") == ["on", "it"]


# ----- what counts as no answer -----

def test_empty_and_tags_only_are_degenerate():
    assert replies.degenerate("", [])
    assert replies.degenerate("[neutral]", [])
    assert replies.degenerate("[thinking] [deep]  ", ["[happy] Hello there, how can I help?"])


def test_short_prefix_of_an_earlier_reply_is_degenerate():
    earlier = ["[neutral] On it, rendering the fox clip now.", "[happy] I am on it… [interrupted]"]
    assert replies.degenerate("[neutral] On it,", earlier)
    assert replies.degenerate("I", earlier)
    assert replies.degenerate("I am", earlier)


def test_short_real_answers_are_not_degenerate():
    earlier = ["[neutral] On it, rendering the fox clip now."]
    assert not replies.degenerate("[happy] Yes!", earlier)
    assert not replies.degenerate("Two hundred forty.", earlier)
    assert not replies.degenerate("On it, rendering the fox clip now.", earlier)  # a whole line again, not a prefix
    assert not replies.degenerate("I am on it", earlier)  # more than two words


def test_copied_marker_is_degenerate():
    assert replies.degenerate("[neutral] On it, applying that\u2026 [interrupted]", [])


def test_marker_copied_mid_reply_is_harmless():
    # a whole reply with the marker inside, as the brain sometimes copies a stored cut reply: not degenerate
    assert not replies.degenerate("[neutral] On it, casting Tanaka [interrupted] and adjusting her voice.", [])


def test_cut_reply_keeps_what_was_written_past_the_cut():
    assert replies.cut_reply("On it, casting Tanaka", "[neutral] On it, casting Tanaka and adjusting her voice.") == \
        "On it, casting Tanaka [interrupted] and adjusting her voice."
    # sentence chunks joined by the voice with other spacing still line up
    assert replies.cut_reply("Sure. That works", "[happy]Sure.  That works for the\nfox clip.") == \
        "Sure. That works [interrupted] for the\nfox clip."


def test_cut_reply_without_rest_or_with_other_words_marks_the_spoken_part():
    assert replies.cut_reply("A standard teaser", "[neutral] A standard teaser") == "A standard teaser\u2026 [interrupted]"
    assert replies.cut_reply("Hello there", "") == "Hello there\u2026 [interrupted]"
    assert replies.cut_reply("Hello there", "Something else entirely.") == "Hello there\u2026 [interrupted]"


def test_cut_line_is_recognised():
    assert replies.is_cut(replies.cut_off("On it, casting"))
    assert not replies.is_cut("[neutral] On it, casting Tanaka now.")


def test_enough_when_wary_waits_for_the_first_sentence_and_no_marker():
    assert not replies.enough("[neutral] On it, applying that", wary=True)
    assert not replies.enough("[neutral] On it, applying that\u2026 [interrupted]", wary=True)
    assert replies.enough("[neutral] On it, casting Tanaka [interrupted] and adjusting her voice.", wary=True)
    assert replies.enough("[neutral] On it, applying that warm grade.", wary=True)
    assert replies.enough("A dolly zoom moves the camera while the lens zooms the other way", wary=True)


def test_enough_waits_for_three_whole_words():
    assert not replies.enough("[neutral] On it,")
    assert not replies.enough("[neutral] On it, re")
    assert replies.enough("[neutral] On it, rendering ")
    assert replies.enough("On it, rendering the")


# ----- the guard around the stream -----

def test_normal_reply_passes_through_unchanged_without_retry():
    text, calls = collect(["[happy] ", "A teaser", " usually runs", " thirty to sixty seconds."])
    assert text == "[happy] A teaser usually runs thirty to sixty seconds."
    assert calls["again"] == 0 and calls["retried"] == [] and calls["first"] == 1


def test_short_real_answer_is_spoken_without_retry():
    text, calls = collect(["[happy] ", "Yes!"], earlier=["[neutral] On it, rendering the fox clip now."])
    assert text == "[happy] Yes!" and calls["again"] == 0


def test_empty_reply_is_retried_and_keeps_first_tags():
    text, calls = collect(["[neutral]"], again=["[neutral] ", "There are 240 frames."])
    assert calls["again"] == 1 and calls["retried"] == ["[neutral]"]
    assert text == "[neutral][neutral] There are 240 frames."


def test_truncated_copy_is_retried():
    text, calls = collect(["[deep] ", "On it,"], earlier=["[neutral] On it, rendering the fox clip now."],
                          again=["On it, queuing those three seeds."])
    assert calls["again"] == 1
    assert text == "[deep]On it, queuing those three seeds."


def test_empty_twice_falls_back_to_a_line_never_silence():
    text, calls = collect([""], again=["[neutral]"])
    assert calls["again"] == 1
    assert text.endswith(replies.FALLBACK)


def test_empty_twice_after_hand_over_says_on_it():
    text, _ = collect(["[deep]"], again=[""], handed=True)
    assert text == "[deep]" + replies.FALLBACK_HANDED


def test_retry_that_is_short_but_not_empty_is_spoken_as_is():
    text, calls = collect(["I"], earlier=["I am on it now."], again=["Okay."])
    assert calls["again"] == 1 and text == "Okay."


def test_copied_marker_after_a_cut_is_retried_and_never_reaches_the_voice():
    earlier = ["[neutral] On it, lowering the music\u2026 [interrupted]"]
    text, calls = collect(["[neutral] ", "On it,", " applying", " that", "\u2026 [interrupted]"], earlier=earlier,
                          again=["[neutral] On it, applying that warm grade."])
    assert calls["again"] == 1
    assert "[interrupted]" not in text
    assert text == "[neutral][neutral] On it, applying that warm grade."


def test_after_a_cut_a_whole_first_sentence_passes_without_retry():
    earlier = ["[neutral] On it, lowering the music\u2026 [interrupted]"]
    text, calls = collect(["[neutral] ", "On it,", " applying", " that", " warm grade.", " Done soon."], earlier=earlier)
    assert calls["again"] == 0 and text == "[neutral] On it, applying that warm grade. Done soon."


def test_written_holds_the_whole_reply_before_the_voice_asks_for_it():
    written = []

    async def go():
        gen = replies.guarded(stream("[happy] ", "One two three. ", "Four five six. ", "Seven eight."), [],
                              lambda: stream(), written=written)
        first = await gen.__anext__()
        await asyncio.sleep(0.01)  # the voice is busy with the first sentence; the reader is not
        got = "".join(written)
        await gen.aclose()
        return first, got
    first, got = run(go())
    assert first == "[happy] One two three. "
    assert got == "[happy] One two three. Four five six. Seven eight."


def test_retry_replaces_what_was_written():
    written = []

    async def go():
        return [p async for p in replies.guarded(stream("[neutral]"), [], lambda: stream("[neutral] There are 240."),
                                                 written=written)]
    run(go())
    assert "".join(written) == "[neutral] There are 240."


def test_closing_stops_the_reader_and_errors_reach_the_caller():
    closed = []

    async def endless():
        try:
            while True:
                yield "word "
                await asyncio.sleep(0)
        finally:
            closed.append(True)

    async def broken():
        yield "one two three four "
        raise RuntimeError("engine down")

    async def go():
        gen = replies.ahead(endless(), [])
        await gen.__anext__()
        await gen.aclose()
        try:
            [p async for p in replies.ahead(broken(), [])]
        except RuntimeError as error:
            return str(error)
    assert run(go()) == "engine down"
    assert closed == [True]
