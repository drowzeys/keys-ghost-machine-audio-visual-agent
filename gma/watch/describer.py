"""The Ghost's look at each creation in invisible mode: what it sees (the eyes, on a picture or three frames of a clip),
what it hears (the ears, on a sound), and then one line and a short description in its own words (the brain, on the
Director's lane, which yields to every voice turn). With no eyes, ears or brain, the creation's own words stand in, so
every creation still gets a summary and can be found.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

from gma.multiverse.writer import parse_json

EYES = "You look at generated pictures and frames of generated film clips, and say plainly what is seen."
LOOK = ("Describe what is seen: the subject, the place, the light and colour, any action and the camera. Two or three "
        "plain sentences. No guessing at names.")
SYSTEM = ("You are the Ghost, who watches over everything the studio's agents make: pictures, film clips, sounds and "
          "texts. You keep a catalogue: for each creation, one plain line and a short description, so it can be found later.")
ASK = ("A new creation.\nKind: {kind}\nMade by: {agent}\nTitle: {title}\nIts prompt or text: {prompt}\nMade from: {parents}\n"
       "What the eyes saw: {seen}\nWhat the ears heard: {heard}\n\nReply with JSON only: {{\"summary\": \"what it is, under "
       "twelve plain words\", \"description\": \"two to four plain sentences: what it shows or says, its look or sound, and "
       "how it differs from what it was made from\", \"tags\": [\"three to six single lowercase words\"]}}")
STOP = set("a an the and or of in on at to with from by for is are was were it its this that his her their as into over "
           "under up down out then than".split())


def fallback(c: dict, seen: str = "", heard: str = "") -> dict:
    """A summary from the creation's own words (no brain)."""
    text = re.sub(r"\s+", " ", c.get("prompt") or "").strip()
    first = re.split(r"(?<=[.!?])\s", text)[0] if text else ""
    summary = (c.get("title") or first or c["kind"]).strip()
    description = " ".join(x for x in (seen, ("Heard: " + heard) if heard else "", text if text != summary else "") if x)
    words = [w for w in re.findall(r"[a-z]{3,}", f"{c.get('title', '')} {text} {seen}".lower()) if w not in STOP]
    tags = list(dict.fromkeys(words))[:5]
    return {"summary": summary[:120], "description": (description or summary)[:1200], "tags": tags}


class Describer:
    def __init__(self, brain=None, lanes=None, eyes=None, ears=None, ffmpeg: str | None = None) -> None:
        self.brain, self.lanes, self.eyes, self.ears, self.ffmpeg = brain, lanes, eyes, ears, ffmpeg
        self.looked = 0

    def abilities(self) -> dict:
        """What the Ghost has to describe with (the page and the status show it)."""
        return {"sees": getattr(self.eyes, "label", None) if self.eyes is not None else None,
                "hears": self.ears is not None, "writes": self.brain is not None}

    def frames(self, c: dict, scratch: Path) -> list[tuple[str, Path]]:
        path = Path(c.get("file") or "")
        if not path.is_file():
            return []
        if c["kind"] == "image":
            return [("The picture", path)]
        if c["kind"] != "video" or not self.ffmpeg:
            return []
        length = self.length(path)
        out = []
        for i, at in enumerate((0.15, 0.5, 0.85)):
            frame = scratch / f"f{i}.jpg"
            subprocess.run([self.ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, length * at):.2f}", "-i", str(path),
                            "-frames:v", "1", "-vf", "scale=640:-2", str(frame)], capture_output=True, timeout=60)
            if frame.is_file():
                out.append((f"Frame {i + 1} of 3 ({['start', 'middle', 'end'][i]})", frame))
        return out

    def length(self, path: Path) -> float:
        probe = str(Path(self.ffmpeg).with_name("ffprobe")) if self.ffmpeg else "ffprobe"
        try:
            done = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                                  capture_output=True, text=True, timeout=30)
            return float(done.stdout.strip() or 0)
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0

    async def describe(self, c: dict, parents: list[dict]) -> tuple[dict, str]:
        """({summary, description, tags}, who described it)."""
        seen = heard = ""
        by = []
        with tempfile.TemporaryDirectory() as scratch:
            frames = await asyncio.to_thread(self.frames, c, Path(scratch))
            if frames and self.eyes is not None:
                try:
                    seen, who = await asyncio.to_thread(self.eyes.look, EYES, frames, None, LOOK)
                    seen = re.sub(r"\s+", " ", seen or "").strip()[:900]
                    self.looked += 1
                    by.append(who)
                except Exception:  # noqa: BLE001 - the eyes are busy or gone: the words stand in
                    seen = ""
        if c["kind"] == "audio" and self.ears is not None and Path(c.get("file") or "").is_file():
            try:
                heard = (await asyncio.to_thread(self.ears.transcribe, Path(c["file"])) or "").strip()[:900]
                by.append("ears")
            except Exception:  # noqa: BLE001
                heard = ""
        if c["kind"] == "text":
            heard = ""
        data = None
        if self.brain is not None:
            prompt = ASK.format(kind=c["kind"], agent=c["agent"], title=c.get("title") or "-", prompt=(c.get("prompt") or "-")[:1500],
                                parents="; ".join(p.get("summary") or p.get("title") or "" for p in parents)[:600] or "nothing (a first)",
                                seen=seen or "-", heard=heard or "-")
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            settings = {"max_tokens": 400, "temperature": 0.4, "timeout": 300}
            try:
                if self.lanes is not None:
                    said = await self.lanes.director(lambda: self.brain.complete(messages, **settings), retries=30)
                else:
                    said = await self.brain.complete(messages, **settings)
                data = parse_json(said)
            except Exception:  # noqa: BLE001 - held or yielded too often: the words stand in
                data = None
        plain = fallback(c, seen, heard)
        if isinstance(data, dict) and str(data.get("summary") or "").strip():
            tags = data.get("tags") if isinstance(data.get("tags"), list) else plain["tags"]
            out = {"summary": str(data["summary"]).strip()[:160], "description": str(data.get("description") or plain["description"]).strip()[:1500],
                   "tags": [re.sub(r"[^\w-]", "", str(t).lower())[:24] for t in tags if str(t).strip()][:6]}
            by.append("brain")
        else:
            out = plain
            by.append("its own words")
        if heard:
            out["description"] = (out["description"] + f" Heard: “{heard[:300]}”").strip()
        return out, " + ".join(by)
