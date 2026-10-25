"""Studio tools: a back agent that turns spoken requests into Imagination Studio actions, through the Studio's own HTTP
API (the same routes its page uses), and answers in a sentence or two.

    "what's in the render queue?"            GET  /api/jobs
    "how long until the queue is done?"      GET  /api/jobs (each job's measured estimate and elapsed time)
    "cancel the last job"                    POST /api/jobs/{id}/cancel                      (asked first)
    "list my Slate shots"                    GET  /api/slate/projects, /api/slate/projects/{id}
    "start a render of the opening shot at 480p"
                                             POST /api/slate/compile, POST /api/jobs, the take logged in the film
                                             (as the page's own Render does; a long render is asked first)
    "make shot 3 darker and re-render it"    the shot's prompt revised by the brain from the note (the old one kept
                                             in the shot's History, as the page's Regenerate does), then rendered
    "play take 2"                            the take's clip, shown by the Studio's Voice panel

A profile turns it on as one of its "agents":

    "studio": {"kind": "studio", "label": "Studio", "url": "http://127.0.0.1:7870",
               "key_file": "~/ImaginationStudio/access-key", "key_env": "STUDIO_ACCESS_KEY_VALUE",
               "confirm_over_s": 120, "always": true, "brain": {...optional, the profile's brain by default}}

The access key is read from the environment variable or the file each time it is needed (a key made anew in the
Studio's Settings is picked up), never written anywhere else. A Studio on the same computer needs none.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable

import httpx

from .engines import Brain

# words that make a turn the Studio's, whichever back agent the page has picked (with "always")
WANTS = re.compile(r"\b(render(s|ed|ing)?|re-?render|queue|shots?|takes?|slate|film|jobs?|clips?|scene|studio)\b")

NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
           "eighteen": 18, "nineteen": 19, "twenty": 20, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
           "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "to": 2, "too": 2, "for": 4, "won": 1}
NUM = r"(\d{1,3}|" + "|".join(sorted(NUMBERS, key=len, reverse=True)) + r")(?:st|nd|rd|th)?"
SHOT_N = re.compile(rf"\bshot (?:number )?{NUM}\b")
N_SHOT = re.compile(rf"\b(?:the )?{NUM} shot\b")
TAKE_N = re.compile(rf"\btake (?:number )?{NUM}\b")
N_TAKE = re.compile(rf"\b(?:the )?{NUM} take\b")
FIRST_SHOT = re.compile(r"\b(opening|first) shot\b")
LAST_SHOT = re.compile(r"\b(last|final|closing) shot\b")
THIS = re.compile(r"\b(it|this one|that one|this shot|that shot|the same shot)\b")
RES = re.compile(r"\b(480|720)\s*p?\b|\b(sharp)\b")
RERENDER = re.compile(r"\b(re-?render|render (it|that|this|again)|and render|then render|render the (new|result))\b")

CANCEL = re.compile(r"\b(cancel|abort|kill|stop)\b")
DELETE = re.compile(r"\b(delete|remove|erase|trash)\b")
ETA = re.compile(r"\b(how long|when will|when'?s|eta|time left|until|finish(ed)?|be done|done yet|is it done|ready yet)\b")
QUEUE = re.compile(r"\b(queue|queued|rendering( now)?|running|in progress|what'?s (cooking|going on|happening))\b")
FILMS = re.compile(r"\b(films|projects|movies)\b")
SHOTS = re.compile(r"\bshots\b")
TAKES = re.compile(r"\btakes\b")
LISTING = re.compile(r"\b(list|what|which|read|tell|show me|name|how many)\b")
PLAY = re.compile(r"\b(play|watch|show|open|preview)\b")
RENDER = re.compile(r"\b(render|re-?render|shoot|make a take|queue up)\b")
REVISE = re.compile(r"\b(make|change|turn|give|add|set)\b")
FILM_NAMED = re.compile(r"\b(?:film|project|movie) (?:called |named )?[\"“]?([\w' -]{2,40}?)[\"”]?(?=$|[,.?!]| at | and )")

FEEDBACK = ("SHOT \"{name}\" WAS RENDERED FROM THIS PROMPT:\n{prompt}\n\nTHE DIRECTOR ASKED FOR THIS CHANGE:\n{note}\n\n"
            "Revise the prompt so the next render makes exactly that change. Make it concretely, in the words that set "
            "it (the light, colour, time of day, weather, movement), rewriting those words rather than adding the note's "
            "word at the end. Change only what the note asks for; keep everything else word for word where you can, "
            "and keep its sections and labels. Never write what must not "
            "be seen (\"no X\", \"never X\"): the video model has no negative prompt, and a thing the prompt names "
            "tends to appear; say what is there instead. Return ONLY the full revised prompt inside a fenced code "
            "block (```prompt ... ```).")
SYSTEM = ("You are the prompt-craft brain of a film studio. You write vivid, concrete, physical, photographable "
          "descriptions for a video model, never filler adjectives.")
INTENT = ("You turn a film director's spoken request to their video studio into one action. Reply with JSON only: "
          "{\"action\": one of queue, eta, cancel, films, shots, takes, play, render, revise, none; "
          "\"shot\": the shot as said (\"3\", \"opening\", \"last\", \"it\") or null; \"take\": a number or \"last\" or null; "
          "\"which\": for cancel one of last, current, all; \"res\": \"480p\", \"720p\" or null; "
          "\"change\": for revise, the change in the director's words; \"render\": true when they also want it rendered}. "
          "revise is a change to how a shot looks or what happens in it.")


class StudioError(Exception):
    """Said to the person as it is."""


def number(word: str) -> int | None:
    return int(word) if word.isdigit() else NUMBERS.get(word)


def plain(text: str) -> str:
    low = text.lower().replace("’", "'")
    low = re.sub(r"[^\w' .,?!-]", " ", low)
    return re.sub(r"\s+", " ", low).strip()


def minutes(seconds: float) -> str:
    """A time as said aloud: "about 40 seconds", "about 3 minutes", "about an hour and 10 minutes"."""

    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"about {max(5, round(seconds / 5) * 5)} seconds"
    whole = round(seconds / 60)
    if whole < 60:
        return "about a minute" if whole == 1 else f"about {whole} minutes"
    hours, rest = divmod(whole, 60)
    head = "an hour" if hours == 1 else f"{hours} hours"
    return f"about {head}" + (f" and {rest} minutes" if rest else "")


def counted(n: int, one: str, many: str | None = None) -> str:
    words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return f"{words.get(n, n)} {one if n == 1 else (many or one + 's')}"


def title_of(job: dict) -> str:
    """A job's title as said: "Night Run, Opening" for Slate's "Night Run · Opening"."""

    return re.sub(r"\s*·\s*", ", ", str(job.get("title") or job.get("kind") or "a job")).strip()


def left_of(job: dict) -> float | None:
    """Seconds a job still needs: its measured estimate less what it has run, or None where nothing was measured."""

    estimate = job.get("estimate")
    if not estimate:
        return None
    if job.get("status") == "running":
        return max(5.0, float(estimate) - float(job.get("elapsed") or 0))
    return float(estimate)


def fence(text: str) -> str | None:
    found = re.search(r"```(?:prompt)?\s*\n?(.*?)```", text or "", re.S)
    body = (found.group(1) if found else "").strip()
    return body or None


class Api:
    """The Studio's HTTP API with its access key (``X-Studio-Key``), read from the environment or a file per call."""

    def __init__(self, url: str, key_file: str | None = None, key_env: str | None = None, timeout: float = 15,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.url = url.rstrip("/")
        self.key_file, self.key_env = key_file, key_env
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=3), transport=transport)

    def key(self) -> str:
        if self.key_env and os.environ.get(self.key_env, "").strip():
            return os.environ[self.key_env].strip()
        if self.key_file:
            try:
                return Path(self.key_file).expanduser().read_text(encoding="utf-8").strip()
            except OSError:
                return ""
        return ""

    async def call(self, method: str, path: str, body: dict | None = None):
        key = self.key()
        headers = {"X-Studio-Key": key} if key else {}
        try:
            reply = await self.client.request(method, self.url + path, json=body, headers=headers)
        except httpx.HTTPError as error:
            raise StudioError(f"I can't reach the Studio right now ({type(error).__name__}).") from None
        if reply.status_code == 401:
            raise StudioError("The Studio wants its access key, and the one I have doesn't work. Check the key in my settings.")
        if reply.status_code >= 400:
            try:
                said = reply.json()
                said = said.get("error") or said.get("detail") or reply.reason_phrase
            except ValueError:
                said = reply.reason_phrase
            raise StudioError(f"The Studio said: {str(said)[:160]}")
        return reply.json()

    async def get(self, path: str):
        return await self.call("GET", path)

    async def post(self, path: str, body: dict):
        return await self.call("POST", path, body)

    async def put(self, path: str, body: dict):
        return await self.call("PUT", path, body)


class Studio:
    """The back agent. ``run`` takes one spoken request and returns the spoken answer; ``confirm(title, "studio")``
    (the page's aloud-confirm flow) is asked before a cancel, a delete or a long render."""

    kind = "studio"

    def __init__(self, name: str, spec: dict, brain: dict | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name, self.label = name, spec.get("label", "Studio")
        self.api = Api(spec.get("url", "http://127.0.0.1:7870"), spec.get("key_file"), spec.get("key_env"),
                       spec.get("timeout_s", 15), transport)
        self.confirm_over = float(spec.get("confirm_over_s", 120))  # a render expected to take longer is asked first
        self.always = bool(spec.get("always", True))  # Studio words come here even when another agent is picked
        given = spec.get("brain") or brain
        self.brain = Brain(given) if given else None  # reads requests the patterns miss, revises a shot's prompt
        self.memory = {"film": None, "shot": None, "job": None}  # what "it" and "that" mean next time

    # ----- routing -----

    def wants(self, text: str) -> bool:
        return bool(WANTS.search(plain(text)))

    # ----- reading a request -----

    def parse(self, text: str) -> dict | None:
        """The request as {"action", ...}, from patterns; None when they do not say (the brain is asked then)."""

        low = plain(text)
        shot = self.shot_said(low)
        take = self.take_said(low)
        res = RES.search(low)
        res = (res.group(1) + "p") if res and res.group(1) else ("sharp" if res else None)
        film = FILM_NAMED.search(low)
        base = {"shot": shot, "take": take, "res": res, "film": film.group(1).strip() if film else None}
        if DELETE.search(low) and (shot or take or SHOTS.search(low) or TAKES.search(low) or FILMS.search(low)):
            return {**base, "action": "delete"}
        if CANCEL.search(low) and (QUEUE.search(low) or re.search(r"\b(job|render|clip|that|it|everything|all|last|current)\b", low)):
            which = "all" if re.search(r"\b(all|everything|whole queue|every)\b", low) else \
                "current" if re.search(r"\b(current|running|this render|one that'?s running)\b", low) else "last"
            return {**base, "action": "cancel", "which": which}
        if PLAY.search(low) and (take or shot or re.search(r"\b(latest|last|newest) (render|clip|take)\b", low)):
            return {**base, "action": "play"}
        change = self.change_said(low)
        if change and (shot or THIS.search(low)):
            return {**base, "action": "revise", "change": change, "render": bool(RERENDER.search(low) or "render" in low)}
        if RENDER.search(low) and (shot or THIS.search(low)) and not LISTING.search(low.split("render")[0][-12:]):
            return {**base, "action": "render"}
        if ETA.search(low) and (QUEUE.search(low) or re.search(r"\b(render|job|clip|it|everything)\b", low)):
            return {**base, "action": "eta"}
        if FILMS.search(low) and LISTING.search(low):
            return {**base, "action": "films"}
        if TAKES.search(low) and (LISTING.search(low) or shot):
            return {**base, "action": "takes"}
        if SHOTS.search(low) and (LISTING.search(low) or "slate" in low):
            return {**base, "action": "shots"}
        if QUEUE.search(low):
            return {**base, "action": "queue"}
        return None

    @staticmethod
    def shot_said(low: str) -> str | None:
        if FIRST_SHOT.search(low):
            return "first"
        if LAST_SHOT.search(low):
            return "last"
        for pattern in (SHOT_N, N_SHOT):
            found = pattern.search(low)
            if found and number(found.group(1)):
                return str(number(found.group(1)))
        named = re.search(r"\bshot (?:called |named )[\"“]?([\w' -]{2,40}?)[\"”]?(?=$|[,.?!]| at | and )", low)
        if named:
            return named.group(1).strip()
        if THIS.search(low):
            return "it"
        return None

    @staticmethod
    def take_said(low: str) -> str | None:
        if re.search(r"\b(last|latest|newest|most recent) take\b", low):
            return "last"
        for pattern in (TAKE_N, N_TAKE):
            found = pattern.search(low)
            if found and number(found.group(1)):
                return str(number(found.group(1)))
        return None

    @staticmethod
    def change_said(low: str) -> str | None:
        """The look asked for in "make shot 3 darker and re-render it": "darker"."""

        found = re.search(rf"\b(?:make|turn|change|set|give|add)\s+(?:the\s+)?(?:(?:opening|first|last|final|closing)\s+shot|"
                          rf"shot\s+(?:number\s+)?{NUM}|{NUM}\s+shot|it|this(?:\s+shot)?|that(?:\s+shot)?)\s+(?:to\s+|into\s+|look\s+|feel\s+|more\s+)?(.+)", low)
        if not found:
            found = re.search(rf"\b(?:shot\s+{NUM}|it|this shot|that shot)\s+(?:should|needs to|has to)\s+(?:be|look|feel)\s+(.+)", low)
        if not found:
            return None
        change = found.groups()[-1]
        change = re.split(r"\s*(?:,\s*)?\b(?:and|then)\s+(?:re-?render|render|show|play)\b", change)[0]
        change = re.sub(r"\b(please|for me|thanks|thank you)\b", "", change).strip(" ,.?!")
        if not change or change in ("render", "re-render"):
            return None
        return change

    async def understand(self, text: str) -> dict | None:
        """The brain's reading of a request the patterns did not catch."""

        if not self.brain:
            return None
        said = await self.complete([{"role": "system", "content": INTENT}, {"role": "user", "content": text}], 120)
        found = re.search(r"\{.*\}", said or "", re.S)
        try:
            out = json.loads(found.group(0)) if found else None
        except ValueError:
            return None
        if not isinstance(out, dict) or out.get("action") in (None, "none"):
            return None
        for key in ("shot", "take"):
            if out.get(key) is not None:
                out[key] = str(out[key]).lower().strip() or None
        return out

    async def complete(self, messages: list[dict], max_tokens: int) -> str:
        body = {"model": self.brain.model, "messages": messages, **self.brain.extra, "max_tokens": max_tokens,
                "temperature": 0.2, "stream": False}
        try:
            reply = await self.brain.client.post(self.brain.url, json=body)
            return reply.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return ""

    # ----- doing it -----

    async def run(self, prompt: str, progress: Callable[[str, str], None], confirm: Callable | None = None) -> str:
        text = prompt.split("\n\n[Do it now", 1)[0].strip()
        try:
            asked = self.parse(text) or await self.understand(text)
            if not asked:
                return ("I can tell you what's in the queue and how long it has left, list your films, shots and takes, "
                        "render or re-render a shot, change how a shot looks, play a take, or cancel a job.")
            progress("tool", f"studio_{asked['action']}")
            act = getattr(self, "do_" + asked["action"], None)
            if not act:
                return "I can't do that in the Studio from here yet."
            return await act(asked, progress, confirm)
        except StudioError as error:
            return str(error)

    async def ask(self, confirm: Callable | None, words: str) -> bool:
        if not confirm:
            return False  # nobody to ask: never act on a destructive or long request unasked
        return (await confirm(words, "studio")) in ("allow_once", "allow_session")

    async def jobs(self) -> list[dict]:
        return await self.api.get("/api/jobs")

    async def do_queue(self, asked: dict, progress, confirm) -> str:
        jobs = await self.jobs()
        running = [j for j in jobs if j.get("status") == "running"]
        waiting = [j for j in jobs if j.get("status") == "queued"]
        if not running and not waiting:
            last = next((j for j in jobs if j.get("status") in ("done", "failed", "cancelled")), None)
            tail = f" The last one, {title_of(last)}, {'finished' if last['status'] == 'done' else last['status']}." if last else ""
            return "The queue is empty." + tail
        parts = []
        for job in running[:1]:
            progress_ = job.get("progress") or {}
            stage = progress_.get("stage") if isinstance(progress_, dict) else None
            left = left_of(job)
            parts.append(f"Rendering {title_of(job)}" + (f", {stage}" if stage and len(stage) < 50 else "") +
                         (f", {minutes(left)} left" if left else "") + ".")
            self.memory["job"] = job["id"]
        if waiting:
            names = ", ".join(title_of(j) for j in reversed(waiting[-3:]))
            parts.append(f"{counted(len(waiting), 'job')} waiting" + (f": {names}" if len(waiting) <= 3 else f", next {names}") + ".")
        return " ".join(part[:1].upper() + part[1:] for part in parts)

    async def do_eta(self, asked: dict, progress, confirm) -> str:
        jobs = [j for j in await self.jobs() if j.get("status") in ("running", "queued")]
        if not jobs:
            return "Nothing is rendering; the queue is empty."
        known = [left_of(j) for j in jobs]
        total = sum(t for t in known if t)
        unknown = sum(1 for t in known if not t)
        count = counted(len(jobs), "job")
        if not total:
            return f"There {'is' if len(jobs) == 1 else 'are'} {count} in the queue, and none has a measured time, so I can't say."
        said = f"The queue should be done in {minutes(total)}, {count} in all"
        if unknown:
            said += f"; {counted(unknown, 'of them has', 'of them have')} no measured time, so it may take longer"
        return said + "."

    async def do_cancel(self, asked: dict, progress, confirm) -> str:
        jobs = [j for j in await self.jobs() if j.get("status") in ("running", "queued")]
        if not jobs:
            return "There's nothing to cancel; the queue is empty."
        which = asked.get("which") or "last"
        if which == "all":
            chosen = jobs
            words = f"cancel all {counted(len(jobs), 'job')} in the queue"
        else:
            if which == "current":
                chosen = [j for j in jobs if j["status"] == "running"][:1]
                if not chosen:
                    return "Nothing is rendering right now; the jobs are waiting."
            else:
                remembered = next((j for j in jobs if j["id"] == self.memory.get("job")), None)
                chosen = [remembered or jobs[0]]  # the newest job still waiting or running (the list is newest first)
            words = f"cancel {title_of(chosen[0])}" + (", which is rendering now" if chosen[0]["status"] == "running" else "")
        if not await self.ask(confirm, words):
            return "Okay, I left the queue as it is."
        for job in chosen:
            await self.api.post(f"/api/jobs/{job['id']}/cancel", {})
        return f"Cancelled {title_of(chosen[0])}." if len(chosen) == 1 else f"Cancelled {counted(len(chosen), 'job')}."

    async def do_delete(self, asked: dict, progress, confirm) -> str:
        # deleting from the voice is left out on purpose in phase 1: the Studio's own page has Trash and Undo for it
        return "I don't delete anything from the Studio by voice. Open the take or the film in the Studio to remove it; it goes to the trash there."

    # ----- films, shots, takes -----

    async def film(self, named: str | None = None) -> dict:
        films = await self.api.get("/api/slate/projects")
        if not films:
            raise StudioError("There are no Slate films in the Studio yet.")
        chosen = None
        if named:
            want = named.lower().strip()
            chosen = next((f for f in films if (f.get("name") or "").lower() == want), None) or \
                next((f for f in films if want in (f.get("name") or "").lower()), None)
            if not chosen:
                raise StudioError(f"I can't find a film called {named}.")
        if not chosen and self.memory.get("film"):
            chosen = next((f for f in films if f["id"] == self.memory["film"]), None)
        chosen = chosen or films[0]  # the film saved last
        self.memory["film"] = chosen["id"]
        return await self.api.get(f"/api/slate/projects/{chosen['id']}")

    @staticmethod
    def shots_of(project: dict) -> list[tuple[dict, dict]]:
        return [(scene, shot) for scene in project.get("scenes") or [] for shot in scene.get("shots") or []]

    def find_shot(self, project: dict, said: str | None) -> tuple[dict, dict, int]:
        shots = self.shots_of(project)
        if not shots:
            raise StudioError(f"The film {project.get('name') or ''} has no shots yet.")
        said = (said or "it").lower()
        index = None
        if said == "first":
            index = 0
        elif said == "last":
            index = len(shots) - 1
        elif said == "it":
            index = next((i for i, (_, s) in enumerate(shots) if s.get("id") == self.memory.get("shot")), None)
            if index is None:
                raise StudioError("Which shot? Say its number, like shot 3.")
        elif said.isdigit():
            n = int(said)
            # a shot named for the number ("Shot 3", "3A") wins over the third in the film's order
            index = next((i for i, (_, s) in enumerate(shots) if re.fullmatch(rf"(shot\s*)?0*{n}", (s.get("name") or "").strip().lower())), None)
            if index is None:
                if not 1 <= n <= len(shots):
                    raise StudioError(f"The film has {counted(len(shots), 'shot')}; there's no shot {n}.")
                index = n - 1
        else:
            index = next((i for i, (_, s) in enumerate(shots) if said in (s.get("name") or "").lower()), None)
            if index is None:
                raise StudioError(f"I can't find a shot called {said}.")
        scene, shot = shots[index]
        self.memory["shot"] = shot.get("id")
        return scene, shot, index

    @staticmethod
    def shot_name(shot: dict, index: int) -> str:
        name = (shot.get("name") or "").strip()
        return name if name and not re.fullmatch(r"(shot\s*)?\d+", name.lower()) else f"shot {index + 1}"

    async def do_films(self, asked: dict, progress, confirm) -> str:
        films = await self.api.get("/api/slate/projects")
        if not films:
            return "There are no Slate films in the Studio yet."
        names = ", ".join(f.get("name") or "an untitled film" for f in films[:5])
        more = f", and {len(films) - 5} more" if len(films) > 5 else ""
        return f"You have {counted(len(films), 'film')}: {names}{more}."

    async def do_shots(self, asked: dict, progress, confirm) -> str:
        project = await self.film(asked.get("film"))
        shots = self.shots_of(project)
        if not shots:
            return f"{project.get('name') or 'The film'} has no shots yet."
        said = []
        for i, (_, shot) in enumerate(shots[:8]):
            takes = [t for t in shot.get("takes") or [] if t.get("mediaPath")]
            said.append(f"{i + 1}, {self.shot_name(shot, i)}" + (f" ({counted(len(takes), 'take')})" if takes else ""))
        more = f", and {len(shots) - 8} more" if len(shots) > 8 else ""
        return f"{project.get('name') or 'Your film'} has {counted(len(shots), 'shot')}: " + "; ".join(said) + more + "."

    async def do_takes(self, asked: dict, progress, confirm) -> str:
        project = await self.film(asked.get("film"))
        _, shot, index = self.find_shot(project, asked.get("shot"))
        takes = [t for t in shot.get("takes") or [] if t.get("mediaPath")]
        if not takes:
            return f"{self.shot_name(shot, index).capitalize()} has no takes yet."
        circled = next((n for n, t in enumerate(reversed(takes), 1) if t.get("rating") == "circled"), None)
        return (f"{self.shot_name(shot, index).capitalize()} has {counted(len(takes), 'take')}"
                + (f"; take {circled} is circled" if circled else "") + ".")

    async def do_play(self, asked: dict, progress, confirm) -> str:
        if not asked.get("shot") and not asked.get("take") or (asked.get("take") == "last" and not asked.get("shot")):
            if not asked.get("shot") and not self.memory.get("shot"):
                done = next((j for j in await self.jobs() if j.get("status") == "done" and j.get("outputs")
                             and str(j["outputs"][0]).endswith((".mp4", ".mov", ".webm", ".wav", ".png", ".jpg"))), None)
                if not done:
                    return "There's no finished render to play yet."
                progress("studio", json.dumps({"action": "play", "path": done["outputs"][0], "title": title_of(done)}))
                return f"Playing {title_of(done)}."
        project = await self.film(asked.get("film"))
        _, shot, index = self.find_shot(project, asked.get("shot"))
        takes = [t for t in shot.get("takes") or [] if t.get("mediaPath")]  # newest first, as Slate keeps them
        if not takes:
            return f"{self.shot_name(shot, index).capitalize()} has no takes yet."
        said = asked.get("take")
        if said and said != "last":
            n = int(said)
            if not 1 <= n <= len(takes):
                return f"{self.shot_name(shot, index).capitalize()} has {counted(len(takes), 'take')}; there's no take {n}."
            take, number_ = list(reversed(takes))[n - 1], n  # take 1 is the first one made
        elif said == "last":
            take, number_ = takes[0], len(takes)
        else:
            take = next((t for t in takes if t.get("rating") == "circled"), None) or \
                next((t for t in takes if t.get("rating") != "no-good"), takes[0])
            number_ = len(takes) - takes.index(take)
        found = await self.api.post("/api/slate/exists", {"paths": [take["mediaPath"]]})
        if not found.get(take["mediaPath"]):
            return f"Take {number_} of {self.shot_name(shot, index)} isn't rendered yet."
        progress("studio", json.dumps({"action": "play", "path": take["mediaPath"],
                                       "title": f"{project.get('name') or 'Film'} · {self.shot_name(shot, index)} · take {number_}"}))
        return f"Playing take {number_} of {self.shot_name(shot, index)}."

    # ----- rendering -----

    async def do_render(self, asked: dict, progress, confirm) -> str:
        project = await self.film(asked.get("film"))
        scene, shot, index = self.find_shot(project, asked.get("shot"))
        return await self.render(project, scene, shot, index, asked.get("res"), progress, confirm)

    async def render(self, project: dict, scene: dict, shot: dict, index: int, res: str | None, progress, confirm,
                     note: str = "") -> str:
        """The shot into the queue as the page's Render does (slate.js slRender, nightshift.render_shot): compiled by
        the Studio, queued, and its take logged in the film."""

        name = self.shot_name(shot, index)
        film = dict(project)
        if res:
            film["tf"] = {**(project.get("tf") or {}), "res": res}  # this render's size only; the film keeps its own
        compiled = await self.api.post("/api/slate/compile", {"project": film, "shotId": shot["id"]})
        if not (compiled.get("memory") or {}).get("fits", True):
            return f"{name.capitalize()} won't fit in the machine's memory: {compiled['memory'].get('reason', '')[:120]}"
        plan = dict(compiled["plan"])
        tf, look = shot.get("tf") or {}, project.get("tf") or {}
        prompt = tf["h3"].strip() if tf.get("h3Auto") is False and (tf.get("h3") or "").strip() else compiled["prompt"]
        if not prompt.strip():
            return f"{name.capitalize()} has no prompt yet, so there's nothing to render."
        estimate = await self.api.post("/api/estimate", {**plan, "text_only": compiled.get("mode") == "text",
                                                         "referenced": bool(plan.get("references"))})
        seconds = estimate.get("seconds")
        size = res or look.get("res") or "480p"
        jobs = [j for j in await self.jobs() if j.get("status") in ("running", "queued")]
        ahead = f", after {counted(len(jobs), 'job')} already in the queue" if jobs else ""
        if seconds is None or seconds > self.confirm_over:
            words = f"render {name} at {size}, " + (f"{minutes(seconds)}" if seconds else "a time it has no measure for") + ahead
            if not await self.ask(confirm, words):
                return f"Okay, I didn't render {name}." + (" Its new prompt is saved." if note else "")
        output = f"slate/{project['id']}/takes/{shot['id']}-{int(time.time() * 1000):x}.mp4"
        params = {**plan, "prompt": prompt, "output": output}
        if not plan.get("references") and (plan.get("loras") or look.get("loras")):
            params["loras"] = plan.get("loras") or look.get("loras")
        defaults = project.get("defaults") or {}
        params["brain"] = {"backend": defaults.get("brain"), "endpoint": defaults.get("localEndpoint"), "model": defaults.get("localModel")}
        params["slate"] = {"project": project["id"], "scene": scene.get("id"), "shot": shot["id"]}
        job = await self.api.post("/api/jobs", {"kind": "video", "title": f"{scene.get('name') or 'Scene'} · {shot.get('name') or name}", "params": params})
        self.memory["job"] = job["id"]
        await self.log_take(project["id"], shot["id"], job, params, prompt, note)
        when = f"; it should take {minutes(seconds)}" if seconds else ""
        return f"{name.capitalize()} is in the queue at {size}{ahead}{when}."

    async def log_take(self, film: str, shot_id: str, job: dict, params: dict, prompt: str, note: str) -> None:
        """The take written into the film as it is now (its own updatedAt, so a page holding an older copy reloads
        rather than overwriting it)."""

        project = await self.api.get(f"/api/slate/projects/{film}")
        for scene in project.get("scenes") or []:
            for shot in scene.get("shots") or []:
                if shot.get("id") == shot_id:
                    asked = job.get("params") or params
                    shot.setdefault("takes", []).insert(0, {
                        "id": f"take-{int(time.time() * 1000):x}-{os.urandom(3).hex()[:5]}", "loggedAt": stamp(),
                        "model": "minimax-h3", "prompt": prompt, "rating": "", "notes": "",
                        "origin": "Voice" + (f": {note}" if note else ""), "mediaPath": params["output"], "jobId": job["id"],
                        "seed": params.get("seed"),
                        "settings": {k: asked[k] for k in ("preset", "quality", "frames", "seconds", "width", "height", "loras") if asked.get(k) is not None}})
                    await self.api.put(f"/api/slate/projects/{film}", project)
                    return

    async def do_revise(self, asked: dict, progress, confirm) -> str:
        change = (asked.get("change") or "").strip()
        if not change:
            return "What should change in the shot?"
        if not self.brain:
            return "I have no brain set up to rewrite a shot's prompt."
        project = await self.film(asked.get("film"))
        scene, shot, index = self.find_shot(project, asked.get("shot"))
        name = self.shot_name(shot, index)
        if not (shot.get("prompt") or "").strip():
            return f"{name.capitalize()} has no prompt for me to change yet."
        progress("tool", "studio_brain")
        said = await self.complete([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": FEEDBACK.format(name=shot.get("name") or name, prompt=shot["prompt"], note=change)}], 1200)
        revised = fence(said)
        if not revised or revised.strip() == shot["prompt"].strip():
            return f"The brain didn't give me a new prompt for {name}, so nothing changed."
        # the film as it is now, the new prompt in place and the old one first in the shot's History (slate.js slKeep)
        project = await self.api.get(f"/api/slate/projects/{project['id']}")
        shot, index = self.same_shot(project, shot["id"])
        shot.setdefault("history", []).insert(0, {"id": f"v-{int(time.time() * 1000):x}", "savedAt": stamp(),
                                                  "label": f"before voice: {change[:40]}", "prompt": shot["prompt"]})
        shot["history"] = shot["history"][:50]
        shot["prompt"], shot["updatedAt"] = revised, stamp()
        shot.setdefault("tf", {})["h3Auto"] = True
        await self.api.put(f"/api/slate/projects/{project['id']}", project)
        if not asked.get("render"):
            return f"I changed {name}'s prompt to make it {change}. The old one is in its History. Say render it when you want a take."
        scene = next(sc for sc, s in self.shots_of(project) if s.get("id") == shot["id"])
        said = await self.render(project, scene, shot, index, asked.get("res"), progress, confirm, note=change)
        return f"I made {name} {change}. " + said

    @staticmethod
    def same_shot(project: dict, shot_id: str) -> tuple[dict, int]:
        for i, (_, shot) in enumerate(Studio.shots_of(project)):
            if shot.get("id") == shot_id:
                return shot, i
        raise StudioError("That shot was removed from the film while I was working on it.")


def stamp() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + f".{int(now * 1000) % 1000:03d}Z"
