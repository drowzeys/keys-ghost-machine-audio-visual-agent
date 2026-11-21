"""AutoPilot's clip judge: d1-2B (LiquidAI's decision model on TensorFold) looks at each take and decides keep or retake.

    "autopilot": {"judge": "d1", "judge_choices": {"d1": {"engine": "d1", "url": "http://127.0.0.1:8412"},
                                                  "eyes": {"engine": "eyes"}}}

``judge`` names the default: ``d1`` (this file) or ``eyes`` (the older path: the vision model of gma/ap/eyes.py, Gemma 4
12B or Cosmos3-Edge, with its watcher, checklist, dense windows and second looks). d1-2B is the only judge by default;
Gemma stays selectable, never resident unless chosen.

How a take is judged. Eight frames evenly through the take (at (i + 0.5) / 8 of its length, 448 px wide) go to the
server's ``/v1/systemone`` with every question at once: d1's set attention reads the frames once and each question as
a branch after them (one pass; a CUDA build that reads 1024 prompt rows a pass is asked again in halves, see ``ask``).
Each answer is a probability of "yes", softened by the calibrated temperature (1.337) before it is read:

  person    "the same person throughout, nobody replaced"   retake when p(same) < ``person_min`` (0.6)
  cut       "a hard cut to another place or shot"            retake when p(cut) >= ``fault`` (0.7)
  wardrobe  "someone's clothing changes colour"              retake when p >= ``fault``
  place     "the same place throughout"                      a note
  morph     "a person changes into another, melts"           a note
  beat      "does it show <the piece's beat>"                a note (the Director's plan against the pixels)
  setting   "is the place this one: <the shot's setting>"    a note
  dress     "is <hero> dressed as <the bible's wardrobe>"    a note

Retake on doubt: an answer to a retake question within ``doubt`` (0.15) of an even chance (0.35 to 0.65 after the
temperature) retakes the piece too, since the person, the cut and the clothes are what a viewer notices first; there
is no second model to ask. Notes never retake: they are recorded with their probabilities in the run report.

Why these numbers (calibrated 2026-10-29 on the DGX Spark, d1-2b-cuda-wip 38724e55): on the 20 labelled synthetic clips (8
clean, 6 person swaps, 2 cuts to an empty scene, 4 clothing colour shifts) the rule retakes 12 of 12 faulty clips
(swaps 6 of 6; one of them only by doubt on the cut) and 0 of 8 clean; Gemma 4 12B's yes/no path retook 11 of 12 and
0 of 8. On the 53 takes of the brain-quality films, looked at by hand (7 with a cut or a person who vanishes, 46
clean), it retakes 4 of the 7 and 1 of the 46 (a fox close-up: no person to be sure of); Gemma failed 18 of those 46
clean takes, mostly on its story, era and wardrobe readings. The place and morph questions are notes because they
flagged clean takes (a fox walking across a field, a zoom); the beat question is a note because its answers on good
takes spread from 0.05 to 0.95.
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULTS = {"engine": "d1", "label": "d1-2B", "url": "http://127.0.0.1:8412", "frames": 8, "width": 448,
            "temperature": 1.337, "person_min": 0.6, "fault": 0.7, "doubt": 0.15, "retake_on_doubt": True,
            "timeout_s": 60}
STATE = "These are frames in time order, evenly spaced across one {seconds:.0f}-second video clip."
# key: (question, the answer that means a fault, retakes)
CHECKS = {
    "person": ("Is the same person (or the same people) shown throughout the clip, with no one replaced by a different person?", "no", True),
    "cut": ("Does the scene change abruptly partway through (a hard cut to a different place or shot)?", "yes", True),
    "wardrobe": ("Does any person's clothing change colour partway through the clip?", "yes", True),
    "place": ("Is it the same place throughout the clip?", "no", False),
    "morph": ("Does any person change into a different person, melt, or morph between these frames?", "yes", False),
}
NOTES = {  # what the fault means, for the run report and the Director
    "person": ("identity", "the person changed partway through (another person, or no one to be sure of)"),
    "cut": ("setting", "the take cuts to a different place or shot partway through"),
    "wardrobe": ("wardrobe", "someone's clothing changes colour partway through"),
    "place": ("location", "the place may change partway through"),
    "morph": ("identity", "a person may change, melt or morph"),
    "beat": ("story", "the take may not show its beat"),
    "setting": ("setting", "the place may not be the one the shot asked for"),
    "dress": ("wardrobe", "the hero may not be dressed as the bible says"),
}


class Refused(RuntimeError):
    pass


def words(text, limit: int) -> str:
    return " ".join(str(text or "").split()[:limit])


def calibrate(p: float, temperature: float) -> float:
    """p(yes) with the logit divided by the temperature (> 1 softens d1's over-sure answers)."""

    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-math.log(p / (1 - p)) / max(temperature, 1e-3)))


def context_questions(beat: str = "", setting: str = "", hero: str = "", dress: str = "") -> dict:
    """The take's own questions from its plan: the beat, the setting, the hero's clothes (each a note; yes is right)."""

    out = {}
    if str(beat or "").strip():
        out["beat"] = f"Does the clip show this: {words(beat, 30)}?"
    if str(setting or "").strip():
        out["setting"] = f"Is the place in these frames this one: {words(setting, 18)}?"
    if hero and str(dress or "").strip():
        out["dress"] = f"Where {hero} is visible, is {hero} dressed as described here: {words(dress, 30)}?"
    return out


class D1Judge:
    def __init__(self, spec: dict | None = None, registry=None) -> None:
        self.spec = {**DEFAULTS, **(spec or {})}
        self.url = str(self.spec["url"]).rstrip("/")
        if self.url.endswith("/v1"):
            self.url = self.url[:-3]
        self.label = self.spec["label"]
        self.registry = registry
        self.name = self.spec.get("registry")  # the registry engine serving it, when this app runs it (a Spark)
        self.requests = 0
        self.seconds = 0.0
        self.loads: list[dict] = []

    def describe(self) -> dict:
        keys = ("engine", "frames", "temperature", "person_min", "fault", "doubt", "retake_on_doubt")
        return {"label": self.label, "url": self.url, "registry": self.name, **{k: self.spec[k] for k in keys},
                "requests": self.requests, "seconds": round(self.seconds, 2), "loads": self.loads[-6:]}

    @contextlib.contextmanager
    def request(self):
        """The server's address for one request, the registry engine loaded first when this app runs it (the same
        as the eyes: counted in flight, so no release cuts the answer)."""

        one = self.registry.get(self.name) if self.registry is not None and self.name else None
        if one is None or not hasattr(one, "acquire"):
            yield self.url
            return
        began, was = time.time(), one.loaded()
        base = one.acquire()
        if not was:
            self.loads.append({"at": began, "seconds": round(time.time() - began, 1)})
        try:
            yield (base or self.url).rstrip("/").removesuffix("/v1")
        finally:
            one.release_request()

    def warm(self) -> None:
        with self.request():
            pass

    # ----- the server -----

    def post(self, base: str, state: str, questions: dict, images: list) -> tuple[dict, int]:
        """{key: p(yes)} and the passes it took. A build that reads fewer prompt rows a pass than the frames and every
        question need (TensorFold CUDA d1-2b-cuda-wip: 1024; 8 frames are ~920) refuses with "past the N rows": the
        questions are asked again in halves over the same frames (set attention: a question's answer does not depend
        on the others, so the halves answer as one pass would)."""

        body = {"state": state, "questions": {k: {"type": "noul", "instructions": v} for k, v in questions.items()}, "images": images}
        call = urllib.request.Request(base + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(call, timeout=float(self.spec["timeout_s"])) as reply:
                answers = json.loads(reply.read())["answers"]
        except urllib.error.HTTPError as error:
            said = error.read().decode(errors="replace")[:300]
            if error.code == 400 and "past the" in said and len(questions) > 1:
                keys = list(questions)
                first, n1 = self.post(base, state, {k: questions[k] for k in keys[:len(keys) // 2]}, images)
                second, n2 = self.post(base, state, {k: questions[k] for k in keys[len(keys) // 2:]}, images)
                return {**first, **second}, n1 + n2
            raise Refused(f"{error.code}: {said}") from None
        return {k: float(answers[k]["noul"]) for k in questions}, 1

    def ask(self, state: str, questions: dict, images: list) -> tuple[dict, int, float]:
        began = time.time()
        try:
            with self.request() as base:
                probs, passes = self.post(base, state, questions, images)
        finally:
            self.requests += 1
            self.seconds += time.time() - began
        return probs, passes, round(1000 * (time.time() - began), 1)

    # ----- a take -----

    def frames(self, clip: Path, folder: Path, seconds: float, env=None) -> list[tuple[float, Path]]:
        n, width = int(self.spec["frames"]), int(self.spec["width"])
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for i in range(n):
            at = round(seconds * (i + 0.5) / n, 2)
            target = folder / f"d1-{i + 1:02d}.png"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(clip), "-frames:v", "1", "-vf",
                                f"scale={width}:-2", str(target)], check=True, env=env, timeout=60, capture_output=True)
            except (subprocess.SubprocessError, OSError):
                continue
            if target.is_file():
                out.append((at, target))
        return out

    def judge(self, clip: Path, folder: Path, seconds: float, context: dict | None = None, env=None) -> dict:
        """The take's verdict: {"retake", "faults", "doubts", "notes", "questions": {key: {text, p_yes, p, p_fault,
        status}}, "ms", "passes", "frames"}; {"error"} (and no retake) when it cannot be judged: a take that cannot be
        looked at is kept, as with no judge."""

        pictures = self.frames(clip, folder, seconds, env)
        if not pictures:
            return {"by": self.label, "error": "no frames could be read from the take", "retake": False}
        images = ["data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() for _, p in pictures]
        questions = {k: v[0] for k, v in CHECKS.items()} | dict(context or {})
        try:
            probs, passes, ms = self.ask(STATE.format(seconds=max(1.0, seconds)), questions, images)
        except Exception as error:  # noqa: BLE001 - a judge that does not answer keeps the take
            return {"by": self.label, "error": str(error)[:300], "retake": False, "frames": [str(p) for _, p in pictures]}
        verdict = self.decide(probs, questions)
        verdict.update(ms=ms, passes=passes, frames=[str(p) for _, p in pictures], at=[t for t, _ in pictures])
        return verdict

    def decide(self, probs: dict, questions: dict | None = None) -> dict:
        s = self.spec
        T, fault, doubt = float(s["temperature"]), float(s["fault"]), float(s["doubt"])
        out = {"by": self.label, "questions": {}, "faults": [], "doubts": [], "notes": []}
        for key, p_yes in probs.items():
            text, bad, retakes = CHECKS.get(key, ((questions or {}).get(key, key), "no", False))
            p = calibrate(p_yes, T)
            p_fault = p if bad == "yes" else 1 - p
            if p_fault >= fault:
                status = "fault"
            elif key == "person" and p < float(s["person_min"]):
                status = "fault" if p < 0.5 - doubt else "doubt"  # under the person bar: a swap, or no one to be sure of
            elif abs(p - 0.5) < doubt:
                status = "doubt"
            else:
                status = "ok"
            if not retakes and status != "ok":
                status = "note"
            out["questions"][key] = {"text": text, "p_yes": round(p_yes, 4), "p": round(p, 4), "p_fault": round(p_fault, 4), "status": status}
            if status == "fault":
                out["faults"].append(key)
            elif status == "doubt":
                out["doubts"].append(key)
            elif status == "note":
                out["notes"].append(key)
        out["retake"] = bool(out["faults"] or (s["retake_on_doubt"] and out["doubts"]))
        return out


def findings(verdict: dict) -> list[dict]:
    """The verdict as continuity findings (what the Director's rewrite and the best take read): a fault or a doubt
    that retakes is a major finding, a note is not a finding (it stays in the report)."""

    out = []
    for key in verdict.get("faults", []) + verdict.get("doubts", []):
        q = verdict["questions"][key]
        category, what = NOTES[key]
        sure = "" if q["status"] == "fault" else "unsure: "
        out.append({"key": key, "category": category, "status": "major", "verdict": "major", "frame": 0, "subject": "", "source": "d1",
                    "evidence": f"{sure}{what} (d1-2B: p(fault) {q['p_fault']:.2f})", "action": "retake"})
    return out


def make(profile: dict, registry=None):
    """The judge a profile names (``autopilot.judge``, d1 unless it says ``eyes``), or None for the eyes' path."""

    ap = profile.get("autopilot") or {}
    spec = ap.get("judge", "d1")
    if isinstance(spec, str):
        name = spec
        spec = (ap.get("judge_choices") or {}).get(name) or ({"engine": "d1"} if name == "d1" else {"engine": name})
    if not spec or spec.get("engine", "d1") != "d1":
        return None
    return D1Judge(spec, registry)
