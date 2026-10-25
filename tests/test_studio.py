"""The Studio tools agent against a stub of the Studio's HTTP API (the routes and shapes of Imagination Studio's
app/server.py and app/slate.py): reading requests, the queue, Slate's films, shots and takes, renders, revisions,
the access key, and asking before a cancel or a long render.

    python -m pytest tests/test_studio.py -q
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_agent import backstage  # noqa: E402
from voice_agent.studio import Studio  # noqa: E402

KEY = "test-key-not-a-secret"


def film() -> dict:
    shot = lambda i, name, prompt, takes=(): {"id": f"s{i}", "name": name, "prompt": prompt, "history": [], "takes": list(takes),
                                              "tf": {"h3Auto": True, "seed": 1}, "spec": {"durationSec": 5}}
    return {"id": "p-night", "name": "Night Run", "updatedAt": "2026-10-25T10:00:00.000Z", "defaults": {"brain": "auto"},
            "tf": {"res": "720p", "quality": "fh8"},
            "scenes": [{"id": "sc1", "name": "Alley", "shots": [
                shot(1, "Opening", "SUBJECT: a runner in a neon alley at night."),
                shot(2, "Shot 2", "SUBJECT: she looks back over her shoulder.", [
                    {"id": "t-b", "mediaPath": "slate/p-night/takes/s2-b.mp4", "rating": "good"},   # newest first
                    {"id": "t-a", "mediaPath": "slate/p-night/takes/s2-a.mp4", "rating": "circled"}]),
                shot(3, "Rooftop", "SUBJECT: she reaches the rooftop at dawn.")]}]}


class Stub:
    """The Studio's routes the agent uses, in memory."""

    def __init__(self) -> None:
        self.projects = {"p-night": film()}
        self.jobs: list[dict] = []
        self.keys: list[str | None] = []
        self.compiled: list[dict] = []
        self.estimate = 90
        self.app = FastAPI()
        app = self.app

        @app.middleware("http")
        async def key(request: Request, call_next):
            self.keys.append(request.headers.get("x-studio-key"))
            if request.headers.get("x-studio-key") != KEY:
                from fastapi.responses import JSONResponse
                return JSONResponse({"error": "This device needs the studio's access key", "access": "key"}, status_code=401)
            return await call_next(request)

        @app.get("/api/jobs")
        def jobs():
            return list(reversed(self.jobs))  # newest first, as the Studio lists them

        @app.post("/api/jobs")
        def submit(body: dict):
            job = {"id": f"j{len(self.jobs) + 1}", "kind": body["kind"], "title": body.get("title"), "status": "queued",
                   "params": body["params"], "estimate": self.estimate, "elapsed": None, "outputs": [], "progress": None}
            self.jobs.append(job)
            return job

        @app.post("/api/jobs/{job_id}/cancel")
        def cancel(job_id: str):
            job = next(j for j in self.jobs if j["id"] == job_id)
            job["status"] = "cancelled"
            return job

        @app.get("/api/slate/projects")
        def projects():
            return [{"id": p["id"], "name": p["name"]} for p in self.projects.values()]

        @app.get("/api/slate/projects/{pid}")
        def project(pid: str):
            return json.loads(json.dumps(self.projects[pid]))

        @app.put("/api/slate/projects/{pid}")
        def save(pid: str, body: dict):
            if body.get("updatedAt") and body["updatedAt"] < self.projects[pid]["updatedAt"]:
                raise HTTPException(409, "the film changed on the server since it was loaded")
            body["updatedAt"] = "2026-10-25T11:00:00.000Z"
            self.projects[pid] = body
            return {"updatedAt": body["updatedAt"]}

        @app.post("/api/slate/compile")
        def compile_(body: dict):
            self.compiled.append(body)
            shot = next(s for sc in body["project"]["scenes"] for s in sc["shots"] if s["id"] == body["shotId"])
            res = body["project"]["tf"]["res"]
            return {"prompt": "H3: " + shot["prompt"], "mode": "text", "warnings": [], "memory": {"fits": True},
                    "plan": {"seconds": 5, "frames": 124, "seed": 1, "quality": "fh8", "preset": "small" if res == "480p" else "p720"}}

        @app.post("/api/estimate")
        def estimate(body: dict):
            return {"seconds": self.estimate}

        @app.post("/api/slate/exists")
        def exists(body: dict):
            return {p: True for p in body["paths"]}


def agent_for(stub: Stub, **spec) -> Studio:
    os.environ["TEST_STUDIO_KEY"] = KEY
    made = Studio("studio", {"kind": "studio", "url": "http://studio.test", "key_env": "TEST_STUDIO_KEY", **spec},
                  transport=httpx.ASGITransport(app=stub.app))
    return made


def run(agent: Studio, text: str, answer: str | None = "allow_once"):
    """The agent's spoken reply, the page events it sent, and what it asked."""

    events, asked = [], []

    def progress(kind, value):
        events.append((kind, value))

    async def confirm(title, kind):
        asked.append(title)
        return answer

    said = asyncio.run(agent.run(text, progress, confirm if answer else None))
    return said, events, asked


# ----- reading requests -----

@pytest.mark.parametrize("text,want", [
    ("What's in the render queue?", {"action": "queue"}),
    ("how long until the queue is done", {"action": "eta"}),
    ("When will the render be finished?", {"action": "eta"}),
    ("cancel the last job", {"action": "cancel", "which": "last"}),
    ("Cancel everything in the queue", {"action": "cancel", "which": "all"}),
    ("stop the current render", {"action": "cancel", "which": "current"}),
    ("list my slate shots", {"action": "shots"}),
    ("What films do I have?", {"action": "films"}),
    ("play take 2", {"action": "play", "take": "2"}),
    ("Play take two of shot three.", {"action": "play", "take": "2", "shot": "3"}),
    ("play the second take of the opening shot", {"action": "play", "take": "2", "shot": "first"}),
    ("start a render of the opening shot at 480p", {"action": "render", "shot": "first", "res": "480p"}),
    ("render shot 3", {"action": "render", "shot": "3"}),
    ("re-render it at 720p", {"action": "render", "shot": "it", "res": "720p"}),
    ("make shot 3 darker and re-render it", {"action": "revise", "shot": "3", "change": "darker", "render": True}),
    ("Make the opening shot feel colder", {"action": "revise", "shot": "first", "change": "colder", "render": False}),
    ("how many takes does shot 2 have", {"action": "takes", "shot": "2"}),
])
def test_requests_are_read(text, want):
    asked = Studio("studio", {"url": "http://x"}).parse(text)
    assert asked is not None, text
    assert {k: asked.get(k) for k in want} == want


def test_other_talk_is_not_the_studios():
    studio = Studio("studio", {"url": "http://x"})
    assert not studio.wants("what's the weather like tomorrow?")
    assert studio.wants("what's in the render queue?")
    assert studio.parse("tell me a joke") is None


# ----- the queue -----

def test_the_queue_is_said_briefly():
    stub = Stub()
    stub.jobs = [{"id": "j1", "title": "Alley · Opening", "status": "running", "estimate": 200, "elapsed": 80,
                  "progress": {"stage": "pass 3 of 8", "at": 3, "of": 8}},
                 {"id": "j2", "title": "Alley · Rooftop", "status": "queued", "estimate": 200}]
    said, _, _ = run(agent_for(stub), "what's in the render queue?")
    assert said == "Rendering Alley, Opening, pass 3 of 8, about 2 minutes left. One job waiting: Alley, Rooftop."
    said, _, _ = run(agent_for(stub), "how long until the queue is done?")
    assert said == "The queue should be done in about 5 minutes, two jobs in all."


def test_an_empty_queue():
    said, _, _ = run(agent_for(Stub()), "what's rendering?")
    assert said.startswith("The queue is empty")


def test_cancel_is_asked_first_and_a_no_leaves_it():
    stub = Stub()
    stub.jobs = [{"id": "j1", "title": "Alley · Opening", "status": "running", "estimate": 200, "elapsed": 10},
                 {"id": "j2", "title": "Alley · Rooftop", "status": "queued", "estimate": 200}]
    said, _, asked = run(agent_for(stub), "cancel the last job", answer="deny")
    assert asked == ["cancel Alley, Rooftop"] and "left the queue" in said
    assert [j["status"] for j in stub.jobs] == ["running", "queued"]
    said, _, asked = run(agent_for(stub), "cancel the last job", answer="allow_once")
    assert said == "Cancelled Alley, Rooftop." and stub.jobs[1]["status"] == "cancelled" and stub.jobs[0]["status"] == "running"


def test_nothing_is_cancelled_without_someone_to_ask():
    stub = Stub()
    stub.jobs = [{"id": "j1", "title": "A", "status": "queued", "estimate": 10}]
    run(agent_for(stub), "cancel everything in the queue", answer=None)
    assert stub.jobs[0]["status"] == "queued"


def test_delete_is_never_done_by_voice():
    stub = Stub()
    said, _, asked = run(agent_for(stub), "delete shot 2")
    assert "don't delete" in said and asked == [] and len(stub.projects["p-night"]["scenes"][0]["shots"]) == 3


# ----- films, shots and takes -----

def test_shots_are_listed():
    said, _, _ = run(agent_for(Stub()), "list my slate shots")
    assert said == "Night Run has three shots: 1, Opening; 2, shot 2 (two takes); 3, Rooftop."


def test_take_two_is_the_second_one_made():
    stub = Stub()
    said, events, _ = run(agent_for(stub), "play take 2 of shot 2")
    assert said == "Playing take 2 of shot 2."
    played = json.loads(next(v for k, v in events if k == "studio"))
    assert played["action"] == "play" and played["path"] == "slate/p-night/takes/s2-b.mp4"
    # "play shot 2" plays its circled take
    said, events, _ = run(agent_for(stub), "play shot 2")
    assert json.loads(next(v for k, v in events if k == "studio"))["path"] == "slate/p-night/takes/s2-a.mp4"
    assert said == "Playing take 1 of shot 2."


# ----- rendering -----

def test_a_render_is_queued_and_logged_like_the_pages():
    stub = Stub()
    stub.estimate = 60  # under the threshold: queued without asking
    said, _, asked = run(agent_for(stub), "start a render of the opening shot at 480p")
    assert asked == []
    assert said == "Opening is in the queue at 480p; it should take about a minute."
    job = stub.jobs[-1]
    assert job["kind"] == "video" and job["title"] == "Alley · Opening"
    assert job["params"]["preset"] == "small" and job["params"]["prompt"] == "H3: SUBJECT: a runner in a neon alley at night."
    assert job["params"]["slate"] == {"project": "p-night", "scene": "sc1", "shot": "s1"}
    assert job["params"]["output"].startswith("slate/p-night/takes/s1-")
    take = stub.projects["p-night"]["scenes"][0]["shots"][0]["takes"][0]
    assert take["jobId"] == job["id"] and take["mediaPath"] == job["params"]["output"] and take["origin"] == "Voice"
    assert stub.projects["p-night"]["tf"]["res"] == "720p"  # the film keeps its own size


def test_a_long_render_is_asked_first():
    stub = Stub()
    stub.estimate = 400
    said, _, asked = run(agent_for(stub), "render shot 3", answer="deny")
    assert asked == ["render Rooftop at 720p, about 7 minutes"] and stub.jobs == [] and "didn't render" in said
    said, _, asked = run(agent_for(stub), "render shot 3", answer="allow_once")
    assert len(stub.jobs) == 1 and "in the queue at 720p" in said


def test_make_shot_3_darker_and_re_render_it():
    stub = Stub()
    stub.estimate = 60
    agent = agent_for(stub)
    agent.brain = object()  # a brain is set; its answer is the stub's below

    async def complete(messages, max_tokens):
        assert "darker" in messages[-1]["content"] and "rooftop at dawn" in messages[-1]["content"]
        return "Here it is:\n```prompt\nSUBJECT: she reaches the rooftop in deep blue pre-dawn darkness.\n```"

    agent.complete = complete
    said, _, _ = run(agent, "make shot 3 darker and re-render it")
    shot = stub.projects["p-night"]["scenes"][0]["shots"][2]
    assert shot["prompt"] == "SUBJECT: she reaches the rooftop in deep blue pre-dawn darkness."
    assert shot["history"][0]["prompt"] == "SUBJECT: she reaches the rooftop at dawn." and "darker" in shot["history"][0]["label"]
    assert stub.jobs[-1]["params"]["prompt"] == "H3: SUBJECT: she reaches the rooftop in deep blue pre-dawn darkness."
    assert shot["takes"][0]["origin"] == "Voice: darker"
    assert said == "I made Rooftop darker. Rooftop is in the queue at 720p; it should take about a minute."
    # "it" is the shot just changed
    said, _, _ = run(agent, "re-render it at 480p")
    assert stub.jobs[-1]["params"]["slate"]["shot"] == "s3" and "480p" in said


# ----- the access key -----

def test_the_key_comes_from_the_environment_or_a_file(tmp_path):
    stub = Stub()
    agent = agent_for(stub)
    run(agent, "what's in the queue")
    assert stub.keys[-1] == KEY
    (tmp_path / "access-key").write_text(KEY + "\n")
    os.environ.pop("TEST_STUDIO_KEY")
    agent = Studio("studio", {"url": "http://studio.test", "key_env": "TEST_STUDIO_KEY", "key_file": str(tmp_path / "access-key")},
                   transport=httpx.ASGITransport(app=stub.app))
    run(agent, "what's in the queue")
    assert stub.keys[-1] == KEY
    (tmp_path / "access-key").write_text("wrong\n")
    said, _, _ = run(agent, "what's in the queue")
    assert "access key" in said


# ----- routing and spoken answers to "should I go ahead?" -----

def test_studio_requests_go_to_the_studio_whichever_agent_is_picked():
    studio = Studio("studio", {"url": "http://x"})
    hermes = backstage.Direct("deep", {"url": "http://x/v1", "model": "m"})
    agents = {"deep": hermes, "studio": studio}
    assert backstage.claims("what's in the render queue?", agents, hermes) is studio
    assert backstage.claims("what's the capital of France?", agents, hermes) is None
    assert backstage.asked_for("cancel the last job", studio)
    off = Studio("studio", {"url": "http://x", "always": False})
    assert backstage.claims("what's in the render queue?", {"deep": hermes, "studio": off}, hermes) is None


@pytest.mark.parametrize("text,want", [("yes", "allow_once"), ("Yes, cancel it.", "allow_once"), ("yeah go ahead", "allow_once"),
                                       ("no", "deny"), ("no, don't cancel it", "deny"), ("wait", "deny"), ("okay, no", "deny"),
                                       ("stop", "deny"), ("what?", None)])
def test_spoken_answers(text, want):
    assert backstage.answer(text) == want


def test_the_confirm_question_is_the_studios_own_words():
    assert backstage.plain_tool("cancel Alley, Rooftop", "studio") == "cancel Alley, Rooftop"
