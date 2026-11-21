"""Keeping the brain's history well formed when a reply is cut, and catching a reply that came back empty or as a
cut-short copy of an earlier one.

A reply stopped by barge-in (or by the next half of a sentence the recognizer split in two) used to be stored as only
the words that were spoken. Stored bare, those fragments read to the brain like finished replies, and a model that
copies the shape of its earlier turns starts ending its own replies the same way: "I am on" -> "I" -> nothing. So a
cut reply is stored whole, as far as the brain had written it, with a marker where the person cut in:
"On it, casting Tanaka [interrupted] and adjusting her voice." The brain sees what was heard and where it stopped, and
the line it may copy is a whole one. When nothing past the cut was written, the spoken part ends with the marker
("A standard teaser\u2026 [interrupted]"). The marker has the form of a tag, and tags are never spoken, so a brain
that copies it cannot make the voice say it.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing

CUT_MARK = "[interrupted]"
TAGS = re.compile(r"\s*\[\w+(?: \w+)*\]")
WORDS = re.compile(r"[\w']+")
SENTENCE_END = re.compile(r"[.!?\u2026](?:\s|$)")
RETRY_NOTE = ("After your mood tag, always say your reply in words: at least one whole sentence, never only a tag, "
              "never stopping midway.")
FALLBACK = "Sorry, I lost my words there. Could you say that again?"
FALLBACK_HANDED = "On it."


def cut_off(spoken: str) -> str:
    """The spoken part of a reply that was stopped midway, marked as cut: "A standard teaser" -> "A standard
    teaser\u2026 [interrupted]"; a reply cut between sentences keeps its full stop: "Sure. [interrupted]"."""

    text = re.sub(r"[\s,;:\u2013\u2014-]+$", "", spoken.strip())
    if not text:
        return ""
    if text.endswith(CUT_MARK):
        return text
    return f"{text} {CUT_MARK}" if text[-1] in ".!?\u2026" else f"{text}\u2026 {CUT_MARK}"


def cut_reply(spoken: str, written: str) -> str:
    """A cut reply as stored: the spoken part, the marker, then the rest of what the brain had written
    (``written``, its raw reply so far, tags and all); just the spoken part and the marker when there is no rest or
    the spoken words cannot be found at the start of what was written."""

    rest = remainder(spoken, TAGS.sub("", written))
    return f"{spoken.strip()} {CUT_MARK} {rest}" if rest else cut_off(spoken)


def remainder(spoken: str, written: str) -> str | None:
    """What follows ``spoken`` in ``written``, comparing everything but whitespace; None when ``written`` does not
    start with it."""

    j, said = 0, re.sub(r"\s+", "", spoken)
    if not said:
        return None
    for i, ch in enumerate(written):
        if ch.isspace():
            continue
        if ch != said[j]:
            return None
        j += 1
        if j == len(said):
            rest = written[i + 1:].strip()
            return rest or None
    return None


def is_cut(line: str) -> bool:
    """Whether a stored line is a cut reply (or a brain's copy of the marker)."""

    return CUT_MARK in line


def ends_cut(line: str) -> bool:
    """Whether a line stops at the marker (a brain that copied a cut reply's ending)."""

    return line.rstrip().endswith(CUT_MARK)


def words(text: str) -> list[str]:
    """The spoken words of a line, lower case, without tags (the mood, the cut marker)."""

    return WORDS.findall(TAGS.sub(" ", text).lower())


def degenerate(reply: str, earlier: list[str]) -> bool:
    """Whether a reply is no answer: nothing but tags, one that stops at a copied cut marker (where an earlier cut
    reply did), or at most two words that are only the start of a longer earlier reply ("On it" after "On it,
    rendering the fox clip now.")."""

    said = words(reply)
    if not said or ends_cut(reply):
        return True
    if len(said) > 2:
        return False
    for line in earlier:
        before = words(line)
        if len(before) > len(said) and before[: len(said)] == said:
            return True
    return False


def enough(text: str, wary: bool = False) -> bool:
    """Whether a reply under way can be spoken as it comes: it has more than two whole words; and, when the
    conversation holds a cut line (``wary``: a brain may copy its marker and stop there), its first sentence has
    ended, or it has more than eight words, and it does not stop at a marker."""

    said = words(text)
    whole = len(said) - (0 if re.search(r"[\w']\W+$", TAGS.sub(" ", text)) else 1)
    if whole <= 2 or (wary and ends_cut(text)):
        return False
    if not wary:
        return True
    return whole > 8 or bool(SENTENCE_END.search(TAGS.sub(" ", text)))


async def ahead(stream: AsyncIterator[str], written: list[str]) -> AsyncIterator[str]:
    """``stream``'s pieces as they are asked for, while it is read to its end as fast as it comes, into ``written``:
    the voice takes a reply a sentence at a time, and a cut reply is stored with what the brain had written past the
    cut. Closing this stops the stream."""

    queue: asyncio.Queue = asyncio.Queue()
    end = object()

    async def read() -> None:
        try:
            async with aclosing(stream) as source:
                async for piece in source:
                    written.append(piece)
                    queue.put_nowait(piece)
        except Exception as error:  # an engine down: raised where the pieces are asked for
            queue.put_nowait(error)
        finally:
            queue.put_nowait(end)

    reader = asyncio.create_task(read())
    try:
        while (item := await queue.get()) is not end:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not reader.done():
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass


async def guarded(stream: AsyncIterator[str], earlier: list[str], again: Callable[[], AsyncIterator[str]],
                  handed: Callable[[], bool] = lambda: False, first: Callable[[], None] | None = None,
                  retried: Callable[[str], None] | None = None, written: list[str] | None = None) -> AsyncIterator[str]:
    """The front brain's reply, piece by piece, with its first words held until there are more than two (the voice
    waits for its first few words anyway, so this costs no time); once the conversation holds a cut line, until its
    first sentence has ended (a few tokens more). A reply that turns out degenerate is asked for once more
    (``again()``: the same conversation with RETRY_NOTE); the first try's tags still go out (its mood, its
    hand-over). If the second try says nothing either, a short line is said instead of silence: "On it." when the
    turn was handed to a back agent (``handed()``), else a request to say it again. With ``written``, the stream is
    read ahead (``ahead``) into it, the retry's in place of the first try's."""

    wary = any(is_cut(line) for line in earlier)
    held: str | None = ""
    if written is not None:
        stream = ahead(stream, written)
    async with aclosing(stream) as source:
        async for piece in source:
            if first:
                first()
                first = None
            if held is None:
                yield piece
                continue
            held += piece
            if enough(held, wary):
                yield held
                held = None
    if held is None:
        return
    if not degenerate(held, earlier):
        if held:
            yield held
        return
    if retried:
        retried(held)
    tags = "".join(t for t in re.findall(r"\[\w+(?: \w+)*\]", held) if t != CUT_MARK)
    if tags:
        yield tags
    said = ""
    retry = again()
    if written is not None:
        written.clear()
        retry = ahead(retry, written)
    async with aclosing(retry) as source:
        async for piece in source:
            said += piece
            yield piece
    if not words(said):
        yield FALLBACK_HANDED if handed() else FALLBACK
