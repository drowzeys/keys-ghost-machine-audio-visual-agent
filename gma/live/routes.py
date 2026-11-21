"""The live stage's HTTP side, registered on the app by one hook in voice_agent/server.py ``build``:

    GET  /live                     the stage's page; its files: /live/{file}
    GET  /live-action              the same page as Live Action: rehearse, talk to the cast, "action", "cut"
    WS   /ws/live/mic              the open microphone: 16 kHz mono 16-bit audio in, what was heard and done out
    GET  /api/live                 the state: phase, scene, cast, the log of notes, speed, takes
    WS   /ws/live                  the pictures as they are made (binary: 4-byte length, a JSON header, its JPEGs) and
                                   the state each time it changes (text, JSON)
    POST /api/live/start           {"scene": "...", "seed"?}: a scene begins to play (the engine is loaded first)
    POST /api/live/plan            {"scene": "...", "seed"?}: the rehearsal: nothing is filmed, notes shape the plan
    POST /api/live/film            the run-through just played is filmed, up to where it was cut
    POST /api/live/action          the plan is filmed ({"scene"} films a scene with no rehearsal); "action" as a note does the same
    POST /api/live/note            {"text": "..."}: a typed direction
    POST /api/live/hear            the microphone's audio (float32 mono, 16 kHz, raw): heard, then taken as a direction
    POST /api/live/stop            the scene ends
    POST /api/live/take            the rehearsal filmed again as one clean file; GET /api/live/takes/{name}
    POST /api/live/say             {"text", "voice"}: a character's line as a WAV (the lite voice)
    POST /api/live/off             the picture engine is unloaded now

The profile's "live" section turns it on (see gma/live/stage.py for its keys).
"""

from __future__ import annotations

import asyncio
import io
import json
import re
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse

from . import director
from .ear import Ear
from .stage import Refused, Stage

STATIC = Path(__file__).resolve().parents[2] / "voice_agent" / "static" / "live"
TYPES = {".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml"}
MAX_AUDIO = 16_000 * 4 * 30  # thirty seconds of float32 at 16 kHz


def register(app: FastAPI, engines, profile: dict) -> Stage:
    brains = getattr(engines, "brains", None)
    brain = None if brains is None else (brains.director or brains.main)
    stage = Stage(profile.get("live") if isinstance(profile.get("live"), dict) else None, brain=brain,
                  renderer=getattr(getattr(engines, "supervisor", None), "renderer", None))
    engines.live = stage

    def refused(wrong: Exception) -> JSONResponse:
        return JSONResponse({"error": str(wrong)}, status_code=409)

    async def body(request: Request) -> dict:
        try:
            said = await request.json()
        except ValueError:
            said = None
        return said if isinstance(said, dict) else {}

    @app.get("/live")
    def page():
        """The older stream page: Live Action is the one way in now."""

        return RedirectResponse("/live-action")

    @app.get("/live-action")
    def director_page():
        """Live Action: the rehearsal, the talking character, the blocking and the take (the page's own script
        shows that side of it at this address)."""

        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/live/{name}")
    def files(name: str):
        path = STATIC / name
        if not re.fullmatch(r"[\w.-]+", name) or path.suffix not in TYPES or not path.is_file():
            return JSONResponse({"error": "no such file"}, status_code=404)
        return FileResponse(path, media_type=TYPES[path.suffix], headers={"Cache-Control": "no-store"})

    @app.get("/api/live")
    async def state():
        if stage.host and stage.phase in ("off", "failed"):  # the engine's machine is asked how much memory it has free
            await stage.free()
        return JSONResponse(stage.state())

    @app.post("/api/live/start")
    async def start(request: Request):
        said = await body(request)
        try:
            return JSONResponse(await stage.start(str(said.get("scene", "")), int(said.get("seed") or 0)))
        except Refused as wrong:
            return refused(wrong)

    @app.post("/api/live/plan")
    async def plan(request: Request):
        said = await body(request)
        voice = getattr(engines, "voices", {}).get("lite")
        if voice is not None:  # the cast's voice loads now, not at their first line (15 s measured, cold beside a render)
            async def warm():
                try:
                    async for _ in voice.stream("Ready.", None):
                        pass
                except Exception:  # noqa: BLE001 - the first line is slow, then
                    pass
            stage.voice_warming = asyncio.create_task(warm())
        try:
            return JSONResponse(await stage.rehearse(str(said.get("scene", "")), int(said.get("seed") or 0)))
        except Refused as wrong:
            return refused(wrong)

    @app.post("/api/live/action")
    async def action(request: Request):
        said = await body(request)
        try:
            return JSONResponse(await stage.action(str(said.get("scene", "")), int(said.get("seed") or 0)))
        except Refused as wrong:
            return refused(wrong)

    @app.post("/api/live/film")
    async def film():
        try:
            return JSONResponse(await stage.film())
        except Refused as wrong:
            return refused(wrong)

    @app.post("/api/live/note")
    async def note(request: Request):
        said = await body(request)
        try:
            return JSONResponse(await stage.note(str(said.get("text", ""))))
        except Refused as wrong:
            return refused(wrong)

    @app.post("/api/live/hear")
    async def hear(request: Request):
        raw = await request.body()
        if not raw or len(raw) > MAX_AUDIO or len(raw) % 4:
            return JSONResponse({"error": "send up to thirty seconds of float32 mono audio at 16 kHz"}, status_code=400)
        audio = np.frombuffer(raw, dtype=np.float32).copy()
        if len(audio) < 16_000 // 4:
            return JSONResponse({"error": "too short to hear"}, status_code=400)
        text = await asyncio.get_running_loop().run_in_executor(engines.asr_pool, engines.asr, audio)
        text = (text or "").strip()
        if not text:
            return JSONResponse({"heard": "", "error": "I did not catch that."}, status_code=409)
        try:
            return JSONResponse({"heard": text, **(await stage.note(text, heard=True))})
        except Refused as wrong:
            return JSONResponse({"heard": text, "error": str(wrong)}, status_code=409)

    @app.post("/api/live/stop")
    async def stop():
        return JSONResponse(await stage.stop())

    @app.post("/api/live/off")
    async def off():
        await stage.shutdown("")
        return JSONResponse(stage.state())

    @app.post("/api/live/take")
    async def take():
        try:
            return JSONResponse(await stage.take())
        except Refused as wrong:
            return refused(wrong)

    @app.get("/api/live/takes/{name}")
    def take_file(name: str):
        path = stage.take_file(name) if re.fullmatch(r"take-[\w-]+\.(mp4|jpg)", name) else None
        if path is None:
            return JSONResponse({"error": "no such take"}, status_code=404)
        return FileResponse(path, media_type="video/mp4" if name.endswith(".mp4") else "image/jpeg")

    @app.get("/api/live/set/{name}")
    def set_file(name: str):
        """The rehearsal's reference shot and the character's loops (gma/live/film.py)."""

        path = None
        if stage.set is not None and re.fullmatch(r"(ref|idle|talk|face)-\d+(-\d+)?\.(png|jpg|jpeg|webp|mp4)", name):
            path = stage.set.file(name)
        elif stage.blocking is not None and re.fullmatch(r"block-\d+\.mp4", name):
            path = stage.blocking.file(name)
        if path is None:
            return JSONResponse({"error": "no such file"}, status_code=404)
        return FileResponse(path, headers={"Cache-Control": "max-age=3600"})

    lines: dict = {}  # streamed lines waiting for the page: name -> (what the service said, its pieces)

    sounds: dict = {}  # the same lines' sound alone (WAV): the page plays it through its own audio output

    @app.get("/api/live/sound/{name}")
    async def sound(name: str):
        """A streamed line's sound (WAV). A browser that lets a page's films play only silently still plays this."""

        wav = sounds.get(name)
        if wav is None:
            return JSONResponse({"error": "no such line"}, status_code=404)
        return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/api/live/line/{name}")
    async def line(name: str):
        """A streamed line's film (fragmented MP4) as the lip-sync service makes it; once."""

        opened = lines.pop(name, None)
        if opened is None:
            return JSONResponse({"error": "no such line"}, status_code=404)
        opened[0]["claimed"] = True
        return StreamingResponse(opened[1](), media_type="video/mp4", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.post("/api/live/say")
    async def say(request: Request):
        """A character's line in a stock voice of the lite voice engine (Kokoro), as a WAV the page plays. With
        {"lips": true, "start": the listening loop's frame now} and the lip-sync service ready, the line comes back as
        a film instead (video/mp4, its sound in it): the character saying it; X-Lips-Next is the loop's frame to go on from."""

        said = await body(request)
        voice = getattr(engines, "voices", {}).get("lite")
        text = " ".join(str(said.get("text", "")).split())[:400]
        if voice is None or not text:
            return JSONResponse({"error": "no lite voice in this profile" if voice is None else "nothing to say"}, status_code=409)
        try:
            pcm = b"".join([piece async for piece in voice.stream(text, str(said.get("voice") or "") or None)])
        except Exception as wrong:  # noqa: BLE001 - the voice engine is away: the line stays on the page, unspoken
            return JSONResponse({"error": f"the voice could not speak: {str(wrong)[:120]}"}, status_code=503)
        out = io.BytesIO()
        with wave.open(out, "wb") as file:
            file.setnchannels(1); file.setsampwidth(2); file.setframerate(voice.sample_rate); file.writeframes(pcm)
        lips = stage.lips if said.get("lips") else None
        if lips is not None and lips.phase == "ready" and said.get("stream"):
            # as it is made: the page is told where the film is and plays it while it grows (about a second in)
            try:
                opened = await lips.stream(out.getvalue(), int(said.get("start") or 0), str(said.get("who") or "") or None)
            except Exception:  # noqa: BLE001 - the whole line is made instead
                opened = None
            if opened is not None:
                name = f"line-{len(lines) + 1}-{int(asyncio.get_running_loop().time() * 1000) % 100000}"
                lines[name] = opened
                sounds[name] = out.getvalue()
                while len(sounds) > 6:
                    sounds.pop(next(iter(sounds)))
                while len(lines) > 4:
                    lines.pop(next(iter(lines)))
                return JSONResponse({"film": f"/api/live/line/{name}", "sound": f"/api/live/sound/{name}", "next": int(opened[0].get("next") or 0), "seconds": opened[0].get("seconds"),
                                     "first_s": opened[0].get("first_s")})
        if lips is not None and lips.phase == "ready":
            try:
                made = await lips.speak(out.getvalue(), int(said.get("start") or 0), str(said.get("who") or "") or None)
            except Exception:  # noqa: BLE001 - the line is spoken without the film
                made = None
            if made is not None:
                return Response(made[0], media_type="video/mp4", headers={"X-Lips-Next": str(int(made[1].get("next") or 0)), "Cache-Control": "no-store"})
        return Response(out.getvalue(), media_type="audio/wav")

    @app.post("/api/live/quality")
    async def quality(request: Request):
        """{"quality": "draft" | "sharp"}: how the next take is filmed (gma/live/film.py: the draft is fast, the sharp
        one larger and slower, with the brain away while it renders)."""

        try:
            asked = str(json.loads(await request.body()).get("quality") or "")
        except ValueError:
            asked = ""
        if stage.set is None or asked not in ("draft", "sharp") or (asked == "sharp" and not stage.set.sharp):
            return JSONResponse({"error": "no such quality here"}, status_code=409)
        if stage.phase == "filming" or stage.rendering():
            return JSONResponse({"error": "A take is being filmed; choose before the next one."}, status_code=409)
        stage.set.quality = asked
        stage.tell()
        return JSONResponse({"quality": asked})

    @app.post("/api/live/face")
    async def face(request: Request):
        """{"name"}: the character the director is facing now; what is said to nobody by name is said to them."""

        try:
            name = str(json.loads(await request.body()).get("name") or "")
        except ValueError:
            name = ""
        stage.facing = next((c["name"] for c in stage.cast if c["name"].lower() == name.lower()), None)
        stage.tell()
        return JSONResponse({"facing": stage.facing})

    @app.post("/api/live/client")
    async def client(request: Request):
        """What a browser did with a line (played, refused, failed), into the Studio's log with the browser's name."""

        try:
            what = str(json.loads(await request.body()).get("what") or "")[:300]
        except ValueError:
            what = ""
        if what:
            print(time.strftime("%H:%M:%S"), json.dumps({"type": "live page", "what": what, "browser": (request.headers.get("user-agent") or "")[:160]}), flush=True)
        return Response(status_code=204)

    @app.websocket("/ws/live/mic")
    async def microphone(socket: WebSocket):
        """The open microphone (gma/live/ear.py). The page's text messages: {"sound": true} while it plays sound the
        microphone may pick up (then only "cut" and "action" are taken), {"drop": true} when it spoke over the person."""

        await socket.accept()
        from voice_agent.engines import Vad
        sound = {"on": False}

        async def said(what: dict) -> None:
            try:
                await socket.send_text(json.dumps(what))
            except (RuntimeError, ConnectionError, WebSocketDisconnect):
                pass

        async def hear(audio) -> None:
            text = await asyncio.get_running_loop().run_in_executor(engines.asr_pool, engines.asr, audio)
            text = (text or "").strip()
            if not text:
                return
            if sound["on"] or stage.table is None and stage.phase not in ("live", "filming"):
                # the film's own sound may be what was heard, or there is no scene yet: an order or nothing
                if director.command(text, True, stage.phase in ("live", "filming")) is None:
                    return await said({"type": "ignored", "heard": text})
            try:
                done = await stage.note(text, heard=True, overheard=True)
                await said({"type": "ignored" if done.get("ignored") else "heard", "heard": text, **done})
            except Refused as wrong:
                await said({"type": "heard", "heard": text, "error": str(wrong)})

        ear = Ear(await asyncio.to_thread(Vad, engines.vad_path), hear, speech=lambda on: said({"type": "speech", "on": on}))
        try:
            while True:
                got = await socket.receive()
                if got.get("type") == "websocket.disconnect":
                    break
                if got.get("bytes"):
                    await ear.audio(got["bytes"])
                elif got.get("text"):
                    try:
                        flag = json.loads(got["text"])
                    except ValueError:
                        continue
                    if "sound" in flag:
                        sound["on"] = bool(flag["sound"])
                    if flag.get("drop"):
                        ear.drop()
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            ear.close()

    @app.websocket("/ws/live")
    async def pictures(socket: WebSocket):
        await socket.accept()
        queue = stage.watch()
        try:
            if stage.host and stage.phase in ("off", "failed"):
                await stage.free()
            await socket.send_text(json.dumps({"type": "state", **stage.state()}))
            while True:
                kind, what = await queue.get()
                if kind == "frame":
                    await socket.send_bytes(what)
                else:
                    await socket.send_text(json.dumps(what))
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            stage.unwatch(queue)

    return stage
