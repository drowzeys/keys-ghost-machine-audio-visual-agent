"""The finished film's sound, made by the headless Studio (gma/studio.py) once AutoPilot's last piece is kept.

- **narration** (``settings.narration``): the Director writes a short voice-over for the whole film from its arc and
  the story so far, timed to the film's length (about 2.3 words a second, with room to breathe); Fish Audio Drama 3
  speaks it as a take of the Studio's (``POST /api/voice/takes``); it is laid over the film here (ffmpeg), the film's own
  sound ducked under it.
- **score** (``settings.score``): MiniMax Music 3 writes an instrumental in the film's style as long as the film
  (``POST /api/music/takes``), and the Studio lays it under the film, ducked under the dialogue
  (``POST /api/music/soundtrack``).

The result is ``film-final.mp4`` beside ``film.mp4``, which stays as it was. Each part that cannot be made (no room, no
engine, a refusal) is said in the run's log and left out; the film is never lost for it. Both run inside one governor
session (schedule ``audio``: the brain and the judge out, the voice loop kept), so Drama 3 and Music 3 get the memory.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import time
from pathlib import Path

NARRATION_SYSTEM = ("You write the voice-over of a short film. Answer with the words the narrator speaks and nothing else: "
                    "no title, no stage directions, no quotation marks, no emoji.")
NARRATION_PROMPT = """The film: {name}. {idea}
What happens, piece by piece:
{beats}

Write its narration: {words} words at most, in {sentences} short sentences, plain and warm, past tense, as a storyteller \
would tell it over the pictures. Do not describe the camera. Say no line any character says."""
WORDS_A_SECOND = 2.3


def narration_text(run: dict, director, seconds: float) -> str:
    beats = "\n".join(f"{s['index']}. {s.get('beat') or (s.get('fields') or {}).get('action') or ''}"
                      for s in run.get("segments") or [] if s.get("status") == "done")
    words = max(8, int(seconds * WORDS_A_SECOND * 0.7))
    said = director(NARRATION_SYSTEM, NARRATION_PROMPT.format(name=run.get("name") or "", idea=run.get("idea") or "",
                                                              beats=beats or "(one piece)", words=words,
                                                              sentences=max(2, min(6, words // 12))), 0)
    said = re.sub(r"<think>.*?</think>", "", said or "", flags=re.S)
    said = " ".join(said.replace('"', "").split())
    cut = said.split()
    return " ".join(cut[:int(words * 1.2)]).strip()


def score_caption(run: dict) -> str:
    style = str(run["settings"].get("style") or "cinematic").strip().rstrip(".")
    pace = "driving, rhythmic" if "brisk" in str(run["settings"].get("guidance") or "").lower() else "gentle, slow-building"
    return f"Instrumental film score, {pace}, orchestral strings and piano, no vocals, matching: {style}"[:300]


def has_audio(path: Path) -> bool:
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
                           str(path)], capture_output=True, text=True, timeout=60)
    return bool(done.stdout.strip())


def lay_narration(film: Path, voice: Path, target: Path, lead_s: float = 0.6) -> None:
    """The narration over the film: the film's sound ducked under the voice (sidechain), the voice in after ``lead_s``."""

    delay = int(lead_s * 1000)
    if has_audio(film):
        graph = (f"[1:a]aresample=48000,aformat=channel_layouts=stereo,adelay={delay}|{delay},apad,asplit=2[v1][v2];"
                 f"[0:a]aresample=48000,aformat=channel_layouts=stereo[f];"
                 f"[f][v1]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=400[duck];"
                 f"[duck][v2]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.89[a]")
    else:
        graph = f"[1:a]aresample=48000,aformat=channel_layouts=stereo,adelay={delay}|{delay},apad[a]"
    done = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(film), "-i", str(voice), "-filter_complex", graph,
                           "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", str(target)], capture_output=True, text=True, timeout=600)
    if done.returncode or not target.is_file():
        raise RuntimeError("ffmpeg could not lay the narration: " + done.stderr.strip()[-300:])


def finish_film(run: dict, client, director, say, home: Path, timings: dict | None = None) -> dict | None:
    """Narration and score for ``run``'s film (each when its setting is on); the final file's record or None."""

    s = run["settings"]
    if not (s.get("narration") or s.get("score")) or not run.get("film"):
        return None
    film = home / run["film"]["path"]
    seconds = float(run["film"].get("seconds") or 0)
    if not film.is_file() or seconds <= 0:
        return None
    folder = film.parent
    current, made = film, {}
    timings = timings if timings is not None else {}
    if s.get("narration"):
        began = time.time()
        try:
            text = narration_text(run, director, seconds)
            if not text:
                raise RuntimeError("the Director wrote no narration")
            say(run, f"Narration: {text[:160]}", stage="sound")
            take = client.voice_take(text, title=f"Ghost Studio · {run.get('name', '')[:40]} · narration",
                                     **({"seed": int(s["narration_seed"])} if s.get("narration_seed") is not None else {}))
            job = client.wait(take["id"], every=1.0, limit_s=900)
            if job.get("status") != "done" or not job.get("outputs"):
                raise RuntimeError(str(job.get("error") or "the voice take did not finish")[:200])
            voice = folder / "narration.wav"
            client.fetch(job["outputs"][0], voice)
            target = folder / "film-narrated.mp4"
            lay_narration(current, voice, target)
            current = target
            made["narration"] = {"text": text, "take": take["id"], "file": voice.name}
            timings["narration_s"] = round(time.time() - began, 1)
        except Exception as error:  # noqa: BLE001 - the film stands without it
            say(run, "The narration was left out: " + str(error)[:200], stage="sound")
    if s.get("score"):
        began = time.time()
        try:
            song = client.music_take(score_caption(run), max(10, math.ceil(seconds + 1)),
                                     title=f"Ghost Studio · {run.get('name', '')[:40]} · score")
            say(run, "Score: MiniMax Music 3 is writing the music.", stage="sound")
            job = client.wait(song["id"], every=2.0, limit_s=1800)
            if job.get("status") != "done":
                raise RuntimeError(str(job.get("error") or "the song did not finish")[:200])
            under = client.inside(current)
            if under is None:
                raise RuntimeError("the film is outside the Studio's folder")
            laid = client.soundtrack(song["id"], under, float(s.get("score_db", -18)))
            job = client.wait(laid["id"], every=1.0, limit_s=600)
            if job.get("status") != "done" or not job.get("outputs"):
                raise RuntimeError(str(job.get("error") or "the soundtrack did not finish")[:200])
            target = folder / "film-scored.mp4"
            client.fetch(job["outputs"][0], target)
            current = target
            made["score"] = {"take": song["id"], "caption": score_caption(run)}
            timings["score_s"] = round(time.time() - began, 1)
        except Exception as error:  # noqa: BLE001
            say(run, "The score was left out: " + str(error)[:200], stage="sound")
    if current == film:
        return None
    final = folder / "film-final.mp4"
    shutil.copyfile(current, final)
    record = {"path": str(final.relative_to(home)), "seconds": seconds, **made}
    run["film_final"] = record
    return record
