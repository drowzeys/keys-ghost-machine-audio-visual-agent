"""What the camera has seen, through Eyes of Kai (tonyd2wild/eyes-of-kai, MIT): a timestamped log of captions, so a turn
can ask about the past ("what was I doing five minutes ago?") as well as the present. The labelled context follows
Kai's own prompt layout.

Captions are written here, not by Kai, by the vision model that answers camera questions (Cosmos3-Edge on TensorFold, on
the Mac and on the Spark), then pushed into Kai's log: one every ``interval_s`` while the camera is on (frames only
arrive then), 3 s unless the page sets another (1 s to 15 minutes)."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone

import httpx

# Kai's pattern for questions about the past, narrowed where it caught ordinary questions, plus spoken forms
PAST = re.compile(r"\b(ago|earlier|before (?:i|you|that)|previous|last (?:night|hour|time|week)|"
                  r"this (?:morning|afternoon|evening)|yesterday|while i was|when i was|how long (?:have|was|did) i|"
                  r"since i|been doing|did (?:i|you) (?:see|notice)|"
                  r"what happened|catch me up|miss(?:ed)? anything|what was i|have you seen|where did i (?:put|leave))\b", re.I)
AGO = re.compile(r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|\d+)\s+"
                 r"(second|minute|hour)s?\s+ago\b", re.I)
WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}
UNIT_S = {"second": 1, "minute": 60, "hour": 3600}


def window_for(text: str) -> tuple[float, float]:
    """The stretch of time a question about the past points at, as (seconds back from, seconds back to)."""

    m = AGO.search(text)
    if not m:
        return 30 * 60, 0  # Kai's default: the last half hour
    n = WORDS.get(m.group(1).lower()) or int(m.group(1))
    back = n * UNIT_S[m.group(2).lower()]
    return back * 1.5 + 30, max(0, back * 0.5 - 30)  # around the moment named, with room either side


# one sentence per frame: shorter captions merge better and make a smaller prompt to look back through
CAPTION_PROMPT = ("Describe exactly what is actually visible in this image in one short sentence. Be literal. "
                  "Do not guess or invent detail.")  # Kai's own prompt, held to one sentence
STOP = {"a", "an", "the", "is", "are", "on", "in", "of", "with", "and", "to", "at", "its", "it", "there", "visible",
        "image", "shows", "displays", "sitting", "placed", "sits", "surface", "background"}


def words(caption: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", caption.lower()) if w not in STOP}


def compact(segments: list[dict], keep: int = 12) -> list[dict]:
    """Neighbouring segments that name the same things joined (a small captioner rewords a still scene every frame),
    each with its first wording; past ``keep``, the briefest go first (one-frame glitches), never the long scenes."""

    out: list[dict] = []
    for seg in segments:
        last = out[-1] if out else None
        b = words(seg["caption"])
        # overlap with everything the scene has been called so far, relative to the shorter: paraphrases still match
        if last and b and len(last["words"] & b) / min(len(last["words"]), len(b)) >= 0.5:
            last["until"], last["words"] = seg["until"], last["words"] | b
        else:
            out.append({"from": seg["from"], "until": seg["until"], "caption": seg["caption"], "words": b or {"?"}})
    while len(out) > keep:
        out.pop(min(range(len(out)), key=lambda k: out[k]["until"] - out[k]["from"]))
    for seg in out:
        del seg["words"]
        seg["caption"] = re.split(r"(?<=[.!?])\s", seg["caption"].strip())[0][:120]
    return out


def clock(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone().strftime("%H:%M:%S")



class Memory:
    def __init__(self, spec: dict) -> None:
        self.url = spec["url"].rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20, connect=2))
        self.captioner = spec.get("captioner")  # {"url", "model"}: an OpenAI-compatible vision model
        self.interval_s = float(spec.get("interval_s", 3))
        self.busy = False
        self.quiet_until = 0.0  # no captions while a camera question is being answered
        self.captioned_at = 0.0
        self.latest: tuple[float, str] | None = None  # (monotonic time it was last true, caption) of the newest caption
        self.count = 0
        self.recall = None  # recall.Recall: each caption is also indexed by meaning

    def every(self, seconds: float) -> None:
        """How often to caption while the camera is on: the page's setting, 1 s to 15 minutes."""

        self.interval_s = min(900.0, max(1.0, float(seconds)))

    async def keep(self, days: int) -> None:
        """How many days the log keeps (1 to 7); Kai prunes anything older at once."""

        try:
            await self.client.post(self.url + "/api/retention", json={"days": min(7, max(1, int(days)))})
        except (httpx.HTTPError, ValueError):
            pass

    def hold(self, seconds: float = 4) -> None:
        self.quiet_until = time.monotonic() + seconds

    def look(self, jpeg_b64: str) -> None:
        """Caption a frame and log it, in the background, once ``interval_s`` has passed since the last caption; a
        frame that arrives while one is in flight, or during a camera question, is dropped."""

        now = time.monotonic()
        if self.busy or now < self.quiet_until or now - self.captioned_at < self.interval_s:
            return
        self.busy = True

        async def run():
            try:
                self.captioned_at = now
                text = await self.caption(jpeg_b64)
                if text:
                    self.latest = (now, text)
                    self.count += 1
                    if self.recall:
                        self.recall.add("seen", text)
                    await self.client.post(self.url + "/api/vision-push", json={"description": text, "source": "camera"})
            except (httpx.HTTPError, ValueError, OSError):
                pass
            finally:
                self.busy = False

        asyncio.create_task(run())

    async def caption(self, jpeg_b64: str) -> str | None:
        if not self.captioner:  # Kai's own captioner
            r = await self.client.post(self.url + "/api/vision-caption", json={"image": jpeg_b64, "prompt": CAPTION_PROMPT})
            return r.json().get("caption") if r.status_code == 200 else None
        from .engines import jpeg_to_png

        png = await asyncio.to_thread(jpeg_to_png, jpeg_b64, 512)
        r = await self.client.post(self.captioner["url"].rstrip("/") + "/chat/completions", json={
            "model": self.captioner["model"], "max_tokens": 60, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
                {"type": "text", "text": CAPTION_PROMPT}]}]})
        if r.status_code != 200:
            return None
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return re.split(r"(?<=[.!?])\s", text)[0][:200] if text else None

    def now(self) -> str | None:
        """The newest caption while it is still current (within one scan interval and a little)."""

        if self.latest and time.monotonic() - self.latest[0] < max(6.0, self.interval_s + 2):
            return self.latest[1]
        return None

    async def past(self, text: str) -> str | None:
        """The log's segments for the time a question asks about, or None if it is not about the past."""

        if not PAST.search(text):
            return None
        start, end = window_for(text)
        now = time.time()
        iso = lambda t: datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            # raw frames, not Kai's summary: its summary keeps the first 40 segments, and a small captioner's reworded
            # frames make so many that a window's later minutes would be dropped
            r = await self.client.get(self.url + "/api/vision/window", params={
                "from": iso(now - start), "to": iso(now - end), "summarize": "false", "limit": 2400})
            s = r.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not s.get("ok"):
            return None
        frames = s.get("observations") or []
        if not frames:
            return "Nothing was recorded then: the camera was off or the app was not running."
        lines = [f"{clock(seg['from'])} to {clock(seg['until'])}: {seg['caption']}" for seg in
                 compact([{"from": o["ts"], "until": o["ts"], "caption": o["caption"]} for o in frames])]
        gaps = [(a["ts"], b["ts"]) for a, b in zip(frames, frames[1:]) if b["ts"] - a["ts"] > 30_000]
        if gaps:
            lines.append("No observations (camera off or away): " + ", ".join(f"{clock(a)} to {clock(b)}" for a, b in gaps))
        return f"(It is {datetime.now().strftime('%H:%M:%S')} now.)\n" + "\n".join(lines)


def with_context(said: str, seen: str | None, earlier: str | None, user: str = "The person",
                 recalled: str | None = None) -> str:
    """Kai's layout: what the camera sees, what the log remembers and what was said, kept apart; ``recalled`` is the
    search by meaning's dated lines (recall.py), seen and said, from any day the index keeps."""

    parts = []
    if seen:
        parts.append(f"[What the user's camera sees right now: the person you are talking to and their surroundings, "
                     f"not you]\n{seen}")
    if earlier:
        parts.append(f"[Earlier observations of the user from their camera, from the continuity log]\n{earlier}")
    if recalled:
        parts.append(f"[Earlier, found by meaning: what the camera saw and what was said, in time order]\n{recalled}")
    if not parts:
        return said
    parts.append(f"[{user} said]\n{said}")
    return "\n\n".join(parts)
