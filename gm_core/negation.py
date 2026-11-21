# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman),
# app/negation.py at 912374e1b6111fdab5d55947cc695c9c303f83b7. Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Unchanged except this header.
"""Things a video prompt names in order to forbid them: "no photographs", "without a cut", "never look at the camera".

MiniMax H3 and FastH3 are run here without a negative prompt (they are distilled to one forward a pass, and the
empty side of the guidance is not used), so every word of the prompt is something to show. In a production run a clip
told "No photographs or portraits appear anywhere" hung the wall with photographs, twice, and came out clean once
every mention of photographs was gone; clips told that nobody "NEVER look at the camera" had people turn to it; "No
cut, no close-up" did not stop cuts to close-ups.

`find` lists such phrases so the page can show them before the job is queued. The prompt is the user's: nothing here
changes a sentence they wrote. The app's own prompts no longer forbid anything by name either: a shot without spoken
lines is told "Mouths stay closed." (where it has people) and "Only ambient sound is heard.", a locked camera is "a
locked, still camera" (server.py QUIET_*, slate.py MOVES). The older sentences ("Nobody speaks.", "No voices, no
speech.") are still passed over in a prompt that has them, as are the control-layer lines of a motion reference.

`ensure_ambient` is the other half, for a text or first-frame prompt that never went through the app.
scripts/video.sh (and spark/video.sh, which serves that script on Linux) calls it before the model sees the text.
A prompt with no spoken `<d>` line and no `overall_soundscape:` heading gains the Studio's own ambient line under
that heading. A MiniMax first-frame lead ("For the target video…") stays ahead of the heading, where `compose()`
puts it. A prompt that already has the heading, a spoken line, a reference (`<Picture>`, `<Video>` or `<Audio>`,
or reference files beside the job), `REF_PLAIN=1`, or a reference video's own soundtrack is returned as it was.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REASON = ("The video model has no negative prompt here: a thing the prompt names tends to appear, also when the prompt "
          "says there is none of it. Take the words out and say what is there instead (\"plain dark walls\" for \"no photographs\").")
# the same sentence server.py writes as QUIET_SOUND. A test holds the two equal.
QUIET_SOUND = "Only ambient sound is heard."
SPOKEN = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
SOUNDSCAPE = re.compile(r"(?im)^[ \t]*overall_soundscape[ \t]*:")
DESCRIPTION = re.compile(r"(?i)integrated_multimodal_description[ \t]*:")
MUSIC = re.compile(r"(?im)^[ \t]*non_diegetic_music[ \t]*:")
# a Ref2VA name in the prompt. The first-frame lead also says <Picture 1>; split_lead takes that paragraph off first.
REFERENCE_TAG = re.compile(r"<(?:Picture|Video|Audio)\b", re.IGNORECASE)
LEAD_SPLIT = re.compile(r"^(For the target video\b.*?)\n[ \t]*\n(.*)$", re.DOTALL)
# what the Studio writes itself, about the sound track and the control layers of a motion reference
OWN = ("Nobody speaks.", "No voices, no speech.", "No speech after the spoken lines.", "No speech.", "non_diegetic_music: N/A",
       "its look is not rendered", "is not rendered", "are not rendered", "These colours are labels only, not the look of the target video.",
       "no skeleton lines, no flat-coloured, depth-map or segmentation-map look", "and no strike lands", "No music")
IDIOMS = ("not only", "no doubt", "no matter", "not just", "no longer than", "no more than", "no less than", "no sooner")
WORD = r"[A-Za-z][A-Za-z'’-]*"
FOUND = re.compile(r"(?<![A-Za-z'’-])(no one|nobody|no|not|never|without|none of|neither|nor|cannot|" + WORD + r"n['’]t)"
                   r"(?![A-Za-z'’-])((?:[ \t]+" + WORD + r"){0,4})", re.IGNORECASE)
FILLER = {"a", "an", "the", "any", "of", "to", "be", "is", "are", "at", "in", "on", "it", "its", "their", "his", "her", "and", "or",
          "ever", "one", "other", "more", "new", "appear", "appears", "anywhere", "there", "that", "this", "with", "for", "into", "from"}
SECTIONS = re.compile(r"\n\s*(overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)


def picture_part(prompt: str) -> str:
    """The part of a composed prompt that describes the picture: up to the soundscape heading."""

    cut = SECTIONS.search(prompt)
    return prompt[:cut.start()] if cut else prompt


def find(text: str, limit: int = 8) -> list[dict]:
    """The negated things in ``text``: ``phrase`` as written (shortened to the negation and what it forbids) and
    ``thing`` (the words after the negation that name it). In reading order, each phrase once."""

    for own in OWN:
        text = re.sub(re.escape(own), " ", text, flags=re.IGNORECASE)
    out, seen = [], set()
    for match in FOUND.finditer(text):
        negation, rest = match.group(1), match.group(2).split()
        if any((negation + " " + " ".join(rest)).lower().startswith(idiom) for idiom in IDIOMS):
            continue
        # the thing: the words that follow, up to three that carry meaning
        kept, thing = [], []
        for word in rest:
            kept.append(word)
            if word.lower() not in FILLER:
                thing.append(word)
            if len(thing) == 3:
                break
        if not thing and negation.lower() not in ("nobody", "no one"):
            continue
        while kept and kept[-1].lower() in FILLER:
            kept.pop()
        phrase = " ".join([negation, *kept])
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        out.append({"phrase": phrase, "thing": " ".join(thing)})
        if len(out) == limit:
            break
    return out


def check(text: str) -> dict:
    """What the page shows: ``negations`` (see ``find``), and ``reason`` when there are any."""

    found = find(picture_part(text or ""))
    return {"negations": found, "reason": REASON if found else ""}


def check_fields(fields: dict) -> dict:
    """The same over the wizard's own boxes, saying which box each phrase is in."""

    found = []
    for key, label in (("scene", "Scene"), ("camera", "Camera"), ("action", "Action")):
        found += [{**one, "field": label} for one in find(str(fields.get(key) or ""))]
    return {"negations": found[:12], "reason": REASON if found else ""}


def split_lead(prompt: str) -> tuple[str, str]:
    """MiniMax's first-frame preamble, and the prompt after it.

    The preamble is the paragraph that starts "For the target video" and is set off by a blank line, as
    ``scripts/studio.sh`` writes it. It names ``<Picture 1>`` for the opening frame, which is not a Ref2VA
    reference. A prompt with no such paragraph comes back with an empty lead and its own text.
    """

    match = LEAD_SPLIT.match(prompt.strip())
    if not match:
        return "", prompt
    return match.group(1).strip(), match.group(2)


def ensure_ambient(prompt: str, video_sound: bool = False, references: bool = False, plain: bool = False) -> str:
    """A raw text or first-frame prompt with no spoken line, plus the Studio's own ambient line.

    The sentences already in ``prompt`` stay. Returned unchanged when ``prompt`` already has an
    ``overall_soundscape:`` heading, a ``<d>`` spoken line, a reference tag, ``references`` (reference
    files were passed), ``plain`` (``REF_PLAIN=1``: the prompt goes as written), or ``video_sound`` (a
    reference video's own soundtrack, which may itself hold speech). A prompt ``compose`` wrote already
    has the heading, so it passes through as it was queued. A first-frame lead stays ahead of the heading.
    """

    if video_sound or references or plain or SPOKEN.search(prompt) or SOUNDSCAPE.search(prompt):
        return prompt
    lead, rest = split_lead(prompt)
    if REFERENCE_TAG.search(rest):
        return prompt
    body = rest.strip()
    if not DESCRIPTION.search(body):
        body = f"integrated_multimodal_description: [Shot 1] {body}".rstrip()
    if MUSIC.search(body):
        prepared = f"{body}\n\noverall_soundscape: {QUIET_SOUND}\n"
    else:
        prepared = f"{body}\n\noverall_soundscape: {QUIET_SOUND}\n\nnon_diegetic_music: N/A\n"
    return f"{lead}\n\n{prepared}" if lead else prepared


def main(argv: list[str] | None = None) -> int:
    """Write the prepared prompt to stdout. A file argument is read; otherwise stdin is."""

    import argparse

    parser = argparse.ArgumentParser(description="Add the Studio's ambient soundscape to a raw video prompt.")
    parser.add_argument("--video-sound", action="store_true", help="a reference video's own soundtrack is in use")
    parser.add_argument("--references", action="store_true", help="REF_IMAGES, REF_VIDEOS or REF_AUDIOS was set")
    parser.add_argument("--plain", action="store_true", help="REF_PLAIN=1: the prompt goes to the model as written")
    parser.add_argument("prompt_file", nargs="?")
    args = parser.parse_args(argv)
    text = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else sys.stdin.read()
    sys.stdout.write(ensure_ambient(text, video_sound=args.video_sound, references=args.references, plain=args.plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
