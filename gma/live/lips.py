"""The character's lips, in step with what she says at the rehearsal: the listening loop and a spoken line go to the
lip-sync service (gma/live/lipsync/server.py: MuseTalk 1.5 in a container with no network, on the machine the profile
names), and the same loop comes back with the mouth following the words and the line's sound in it.

The service is started when a rehearsal has its listening loop, given the loop once (about 20 s), and then makes each
line a little faster than it plays. It is stopped with the stage, and after ``idle_s`` with no line.

The profile's ``live.lipsync``: {"host": "another-machine" (none: this machine), "home": "~/lipsync-try" (MuseTalk, its
models and packages; the service's own code is sent there), "need_gb": 40}.

    lips = Lips(spec, tell)
    lips.follow(loop_file)                       # in the background: the service up, the loop prepared
    film, said = await lips.speak(wav, start)    # the line as MP4 bytes, {"next": the loop's frame to go on from}
    said, pieces = await lips.stream(wav, start) # the same as it is made: she starts after about a second
"""

from __future__ import annotations

import asyncio
import io
import json
import secrets
import shlex
import time
import wave
from pathlib import Path

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15"]
CODE = Path(__file__).resolve().parent / "lipsync"


def quiet(seconds: float, rate: int = 24_000) -> bytes:
    """A WAV of silence."""

    out = io.BytesIO()
    with wave.open(out, "wb") as file:
        file.setnchannels(1); file.setsampwidth(2); file.setframerate(rate); file.writeframes(b"\0\0" * int(seconds * rate))
    return out.getvalue()


class Lips:
    def __init__(self, spec: dict, tell=None, runner=None) -> None:
        self.spec, self.tell = spec, tell or (lambda: None)
        self.host = spec.get("host") or ""
        self.home = str(spec.get("home", "~/lipsync-try"))
        self.runner = runner or self._run
        self.phase, self.error = "off", ""  # off, starting, preparing, ready, failed
        self.avatar: str | None = None  # the listening loop's face (the one-character set)
        self.avatars: dict[str, str] = {}  # the cast's close-ups: who -> the face the service prepared
        self.faces: dict[str, str] = {}  # who -> preparing, ready, failed
        self.casting: list[asyncio.Task] = []
        self.starting = asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.minder: asyncio.Task | None = None
        self.used, self.count, self.times = time.monotonic(), 0, []
        self.tag = secrets.token_hex(2)  # two stages on one service (a test beside the app) keep their own files
        self.lock = asyncio.Lock()

    def state(self) -> dict:
        return {"phase": self.phase, "error": self.error, "last_s": self.times[-1] if self.times else None, "faces": dict(self.faces)}

    def of(self, who: str | None) -> str | None:
        """The prepared face that speaks ``who``'s line: their close-up, else the listening loop's."""

        return self.avatars.get(who or "") or self.avatar

    def say(self, phase: str, error: str = "") -> None:
        self.phase, self.error = phase, error
        self.tell()

    # ----- the service's machine -----

    async def _run(self, command: str, data: bytes | None = None, timeout: float = 120) -> tuple[int, bytes, bytes]:
        """``command`` on the service's machine (this one, or over ssh), ``data`` on its standard input."""

        argv = [*SSH, self.host, command] if self.host else ["bash", "-c", command]
        process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if data is not None else asyncio.subprocess.DEVNULL,
                                                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(data), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            raise
        return process.returncode or 0, out, err

    def ask(self, path: str, body: dict | None = None) -> str:
        """The curl that asks the service (its unix socket); its answer is JSON on standard output."""

        sock = f"{self.home}/run/lipsync.sock"
        if body is None:
            return f"curl -s -m 5 --unix-socket {sock} http://lips{path}"
        return f"curl -s -m 300 --unix-socket {sock} -X POST -H 'Content-Type: application/json' -d {shlex.quote(json.dumps(body))} http://lips{path}"

    async def up(self) -> bool:
        code, out, _ = await self.runner(self.ask("/health"), None, 15)
        return code == 0 and out.strip().startswith(b"{")

    async def start(self) -> None:
        """The service started where it is not running: its code sent, the machine's memory asked first."""

        if await self.up():
            return
        need = int(float(self.spec.get("need_gb", 40)))
        if self.host:  # the service's own code, from this repository
            sent = await asyncio.create_subprocess_exec("rsync", "-a", "-e", " ".join(SSH), f"{CODE}/", f"{self.host}:{self.home}/service/",
                                                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await asyncio.wait_for(sent.wait(), 60)
        here = f"{self.home}/service" if self.host else shlex.quote(str(CODE))
        code, _, err = await self.runner(
            f"free=$(awk '/MemAvailable/ {{print int($2/1048576)}}' /proc/meminfo); [ $free -ge {need} ] || {{ echo \"only $free GB of memory free there\" >&2; exit 3; }}; "
            f"LIPSYNC_HOME={self.home} bash {here}/run.sh start", None, 60)
        if code != 0:
            raise RuntimeError(err.decode(errors="replace").strip()[-200:] or "the lip-sync service did not start")
        ends = time.monotonic() + float(self.spec.get("start_s", 120))
        while not await self.up():
            if time.monotonic() > ends:
                raise RuntimeError("the lip-sync service did not come up")
            await asyncio.sleep(1.0)

    def follow(self, loop: Path) -> None:
        """The character's listening loop is the face from now on; in the background."""

        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = asyncio.create_task(self._follow(Path(loop)))

    async def _follow(self, loop: Path) -> None:
        self.avatar = None
        self.say("starting")
        try:
            await self.start()
            self.say("preparing")
            name = await self._prepare(await asyncio.to_thread(loop.read_bytes))
            self.avatar, self.used = name, time.monotonic()
            self.phase = "ready"
            await self.speak(quiet(0.6))  # the first line through the service is slow (25 s measured): this one is nobody's
            self.times = []
            self.say("ready")
            if self.minder is None or self.minder.done():
                self.minder = asyncio.create_task(self._mind())
        except asyncio.CancelledError:
            raise
        except Exception as wrong:  # noqa: BLE001 - the character still talks, with the plain talking loop
            self.say("failed", str(wrong)[:200])

    async def _prepare(self, video: bytes) -> str:
        """A face given to the service (a short film of it): the name it is kept under."""

        self.count += 1
        name, box = f"a{self.tag}{self.count}", f"{self.home}/run/io"
        async with self.lock:
            code, out, err = await self.runner(
                f"mkdir -p {box} && cat > {box}/{name}.mp4 && " + self.ask("/prepare", {"video": f"/run/lipsync/io/{name}.mp4", "id": name}) + f"; rm -f {box}/{name}.mp4", video, 300)
        said = json.loads(out or b"{}")
        if code != 0 or said.get("error") or not said.get("frames"):
            raise RuntimeError(str(said.get("error") or err.decode(errors="replace").strip()[-200:] or "the face could not be prepared"))
        return name

    def face(self, who: str, video: Path) -> None:
        """``who``'s close-up is that character's face from now on (the cast's roster); in the background."""

        self.faces[who] = "preparing"
        self.casting = [t for t in self.casting if not t.done()] + [asyncio.create_task(self._face(who, Path(video)))]

    async def _face(self, who: str, video: Path) -> None:
        try:
            if self.phase in ("off", "failed"):
                self.say("starting")
            async with self.starting:
                await self.start()
            if self.phase == "starting":
                self.say("preparing")
            self.avatars[who] = await self._prepare(await asyncio.to_thread(video.read_bytes))
            first, self.phase, self.used = self.phase != "ready", "ready", time.monotonic()
            if first:
                await self.speak(quiet(0.6), who=who)  # the first line through the service is slow: this one is nobody's
                self.times = []
                if self.minder is None or self.minder.done():
                    self.minder = asyncio.create_task(self._mind())
            self.faces[who] = "ready"
            self.say("ready")
        except asyncio.CancelledError:
            raise
        except Exception as wrong:  # noqa: BLE001 - that character answers in sound only
            self.faces[who] = "failed"
            self.say(self.phase if self.avatars else "failed", str(wrong)[:200])

    def clear(self) -> None:
        """A new scene: the faces of the one before are nobody's."""

        for task in self.casting:
            task.cancel()
        self.casting, self.avatars, self.faces = [], {}, {}

    async def speak(self, wav: bytes, start: int = 0, who: str | None = None) -> tuple[bytes, dict] | None:
        """One line: its film (MP4 with the sound) and what the service said about it; None when it cannot be made."""

        avatar = self.of(who)
        if self.phase != "ready" or avatar is None:
            return None
        began, box = time.monotonic(), f"{self.home}/run/io"
        async with self.lock:
            self.count += 1
            name = f"l{self.tag}{self.count}"
            body = {"id": avatar, "wav": f"/run/lipsync/io/{name}.wav", "out": f"/run/lipsync/io/{name}.mp4", "start": int(start)}
            try:  # one trip: the sound in, the service's answer on standard error, the film on standard output
                code, out, err = await self.runner(
                    f"cat > {box}/{name}.wav && {self.ask('/speak', body)} >&2 && cat {box}/{name}.mp4; rm -f {box}/{name}.wav {box}/{name}.mp4",
                    wav, float(self.spec.get("line_s", 60)))
                said = json.loads(err[err.index(b"{"):] if b"{" in err else b"{}")
            except (asyncio.TimeoutError, OSError, ValueError):
                return None
        self.used = time.monotonic()
        if code != 0 or said.get("error") or len(out) < 1000:
            if not await self.up():  # its guard stopped it (the live stream's engine arrived, or the memory ran short)
                self.avatar, self.avatars, self.faces = None, {}, {}
                self.say("off", "the lip-sync service stopped")
            return None
        self.times = (self.times + [round(time.monotonic() - began, 2)])[-20:]
        return out, said

    async def stream(self, wav: bytes, start: int = 0, who: str | None = None):
        """One line as it is made: what the service said about it ({"next", "seconds", "first_s"}) and the film's
        pieces (a fragmented MP4 that grows, about a second after it was asked). None when it cannot be made."""

        avatar = self.of(who)
        if self.phase != "ready" or avatar is None:
            return None
        await self.lock.acquire()
        process = None
        try:
            self.count += 1
            name, box = f"l{self.tag}{self.count}", f"{self.home}/run/io"
            body = {"id": avatar, "wav": f"/run/lipsync/io/{name}.wav", "out": f"/run/lipsync/io/{name}.mp4", "start": int(start), "stream": True}
            film = f"curl -sN -m 300 --unix-socket {self.home}/run/lipsync.sock 'http://lips/film?out=/run/lipsync/io/{name}.mp4'"
            command = f"cat > {box}/{name}.wav && {self.ask('/speak', body)} >&2 && echo >&2 && {film}; rm -f {box}/{name}.wav {box}/{name}.mp4"
            argv = [*SSH, self.host, command] if self.host else ["bash", "-c", command]
            process = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            process.stdin.write(wav)
            await process.stdin.drain()
            process.stdin.close()
            line = await asyncio.wait_for(process.stderr.readline(), float(self.spec.get("first_s", 30)))
            said = json.loads(line[line.index(b"{"):]) if b"{" in line else {}
            if said.get("error") or not said.get("frames"):
                raise ValueError(str(said.get("error") or "no film"))
        except (asyncio.TimeoutError, OSError, ValueError):
            if process is not None and process.returncode is None:
                process.kill()
            self.lock.release()
            return None
        self.used = time.monotonic()

        async def pieces():
            try:
                while True:
                    piece = await process.stdout.read(65536)
                    if not piece:
                        break
                    yield piece
            finally:
                if process.returncode is None:
                    process.kill()
                self.used = time.monotonic()
                if self.lock.locked():
                    self.lock.release()

        async def unclaimed():  # a page that never came for its film must not hold the service
            await asyncio.sleep(float(self.spec.get("claim_s", 20)))
            if not said.get("claimed") and self.lock.locked():
                if process.returncode is None:
                    process.kill()
                self.lock.release()
        asyncio.get_running_loop().create_task(unclaimed())
        return said, pieces

    async def _mind(self) -> None:
        idle = float(self.spec.get("idle_s", 1800))
        while self.phase == "ready":
            await asyncio.sleep(min(30.0, idle))
            if time.monotonic() - self.used > idle:
                await self.stop()

    async def stop(self) -> None:
        if self.task is not None and not self.task.done() and self.task is not asyncio.current_task():
            self.task.cancel()
        for task in self.casting:
            if task is not asyncio.current_task():
                task.cancel()
        was, self.avatar, self.avatars, self.casting = self.phase, None, {}, []
        self.faces = {who: "off" for who in self.faces}
        self.phase, self.error = "off", ""
        if was != "off":
            here = f"{self.home}/service" if self.host else shlex.quote(str(CODE))
            try:
                await self.runner(f"LIPSYNC_HOME={self.home} bash {here}/run.sh stop", None, 30)
            except Exception:  # noqa: BLE001 - its own guard and its idle machine end it
                pass
