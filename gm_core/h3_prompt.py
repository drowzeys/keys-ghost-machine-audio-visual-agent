# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman),
# app/server.py at 912374e1b6111fdab5d55947cc695c9c303f83b7 (LEAD, lead, QUIET_PICTURE, QUIET_SOUND, QUIET_AFTER, PEOPLE, quiet, said_quiet, STOPS,
# joined, compose). Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Changed: only these pieces, taken out of the Studio's server into a module of their own (FPS and lengths imported
# from the vendored gm_core/lengths.py); the code itself is unchanged. from_idea at the end is new here.
"""MiniMax H3's prompt form, as the Studio writes it: the three fields (integrated_multimodal_description,
overall_soundscape, non_diegetic_music) and the first-frame lead that names the keyframe."""

from __future__ import annotations

import re

from . import lengths

FPS = lengths.FPS

LEAD = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."


def lead(first: bool, last: bool, seconds: float = 5, frames: int | None = None) -> str:
    """The line that opens a keyframe prompt: which picture stands at which moment of the clip. MiniMax's own form
    for a first frame is LEAD; an end frame is written the same way at the clip's last moment (Picture 2 beside a
    first frame, Picture 1 alone), which is the clip's frame count over 24. The pictures themselves are held as
    keyframes by the model whatever this says."""

    end = f"{(frames or lengths.nearest(float(seconds))) / FPS:.2f}"
    if first and last:
        return LEAD[:-1] + f"; at {end} seconds into the target video, <Picture 2> (from [Shot 1]) is fully referenced."
    if last:
        return f"For the target video, at {end} seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    return LEAD if first else ""


QUIET_PICTURE = "Mouths stay closed."
QUIET_SOUND = "Only ambient sound is heard."
QUIET_AFTER = "After the spoken lines only ambient sound is heard."
PEOPLE = re.compile(r"\b(man|men|woman|women|person|people|boy|boys|girl|girls|child|children|kid|kids|baby|he|she|they|him|her|his|them|their|"
                    r"someone|somebody|everyone|crowd|couple|friends?|students?|dancers?|singers?|actors?|actress|workers?|soldiers?|"
                    r"mother|father|son|daughter|brother|sister|husband|wife|face|faces|\w+ (?:is|are) \d+ years old)\b", re.IGNORECASE)


def quiet(fields: dict, *texts: str) -> str:
    """The sentence a shot without spoken lines ends on: QUIET_PICTURE where the shot has people (``people`` in the
    fields says so, else the words do), nothing for a shot without them."""

    people = fields.get("people")
    if people is None:
        people = bool(PEOPLE.search(" ".join(text or "" for text in texts)))
    return " " + QUIET_PICTURE if people else ""


def said_quiet(sound: str) -> bool:
    """Whether a soundscape already says that there is no speech, in the old words or the new."""

    low = sound.lower()
    return "no speech" in low or "only ambient sound" in low


STOPS = ".!?…:;,\"”’')"


def joined(*parts) -> str:
    """The wizard's parts as one shot: each but the last ends on a full stop where it has no stop of its own, so the
    scene and the action read as two sentences ("…from a window. The lamp turns…", not "…from a window The lamp…")."""

    parts = [part.strip() for part in parts if part and part.strip()]
    return " ".join([part if part[-1] in STOPS else part + "." for part in parts[:-1]] + parts[-1:])


def compose(fields: dict, first_frame: bool, last_frame: bool = False, seconds: float = 5, frames: int | None = None) -> str:
    """MiniMax H3's three-field prompt from the wizard's parts."""

    shot = joined(fields.get("scene", ""), fields.get("camera", ""), fields.get("action", ""))
    spoken = []
    for index, line in enumerate(fields.get("dialogue") or [], start=1):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        voice = (line.get("voice") or "a natural voice").strip()
        verb = "sings" if line.get("sung") else "says"
        language = (line.get("language") or "English").strip()
        spoken.append(f"{(line.get('speaker') or '').strip() or 'The speaker'}, in {voice} (S{index}), {verb}: <d>[{language}] {text}</d>")
    if spoken:
        shot += (" " + " ".join(spoken) + " Mouth movements follow the words. After the last word the speaker stops"
                 " talking, closes the mouth and stays silent until the end of the clip.")
    else:
        shot += quiet(fields, fields.get("scene", ""), fields.get("action", ""))
    sound = (fields.get("soundscape") or "Natural ambient sound for the scene.").strip()
    if not said_quiet(sound):
        sound += " " + QUIET_AFTER if spoken else " " + QUIET_SOUND
    music = (fields.get("music") or "N/A").strip()
    body = (f"integrated_multimodal_description: [Shot 1] {shot.strip()}\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: {music}")
    opening = lead(first_frame, last_frame, seconds, frames)
    return f"{opening}\n\n{body}" if opening else body


def from_idea(prompt: str, first_frame: bool, frames: int | None = None, seconds: float = 5) -> str:
    """A clip's prompt from one plain description (the planner's two to four sentences): the description is the
    shot, the soundscape the Studio's ambient default; with a keyframe the first-frame lead opens it."""

    return compose({"scene": re.sub(r"\s+", " ", (prompt or "").strip())}, first_frame, False, seconds, frames)
