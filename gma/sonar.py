"""The Sonar look (one of gma/look.py's looks): the clip is dark and black-and-white until pulses of echolocation leave
its subject, run over the surfaces of the scene, and leave it in HDR colour, each with a sonar ping.

``finish(service, source, look, spec)`` makes both copies of ``source`` as every look has them (the real HDR copy and an
SDR copy beside it):

1. the HDR look's copies of the clip (made now if they are not there): the colour the pulses reveal;
2. ``plan(seconds)``: when the pulses go and how fast, so the last one has crossed the scene before the clip ends;
3. the picture, by gma/sonar_worker.py in its own process (a depth estimate of every frame, then numpy): a few minutes
   for a 10 s clip, and its memory is the system's again when it ends;
4. the sound: the clip's own, with a ping (a high tone that dies away in echoes, and a low thump) at each pulse.

Options, under ``look.sonar`` in the profile: ``python`` (the interpreter with numpy, torch and transformers; this one by
default), ``model`` (the depth model), ``far`` (metres the scene is clipped to), ``speed`` (the wave's metres a second),
``dist`` (metres from the camera the subject is put at), ``at`` ([x, y] of the picture, 0..1, where the pulses leave from;
by default the nearest thing in the middle of the picture), ``floor_mb`` (free memory under which the worker stops).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

FAR_M, SPEED, DIST_M = 20.0, 5.0, 3.5
EVERY_S = 4.5  # about one pulse for this much clip, four at most
SETTLE_S = 0.8  # the clip goes on this long in full colour after the last wave has crossed it
LIMIT_S = 3 * 3600


def plan(seconds: float, spec: dict | None = None) -> dict:
    """{"pulses": [s], "speed", "far", "dist", "at"} for a clip this long."""

    spec = spec or {}
    far, speed = float(spec.get("far", FAR_M)), float(spec.get("speed", SPEED))
    reach = far * 1.25  # the far corners are further from the subject than `far` is from the camera
    seconds = max(float(seconds), 1.0)
    first = min(1.0, 0.1 * seconds)
    last = max(first, seconds - reach / speed - SETTLE_S)
    count = max(1, min(4, round(seconds / EVERY_S), 1 + int((last - first) / 1.5)))
    tail = seconds - last - SETTLE_S
    if tail * speed < reach:  # a short clip: the wave goes faster, to be across before the end
        speed = reach / max(tail, 0.4)
    pulses = [round(first + (last - first) * i / max(count - 1, 1), 2) for i in range(count)]
    return {"pulses": pulses, "speed": round(speed, 2), "far": far, "dist": float(spec.get("dist", DIST_M)), "at": spec.get("at")}


def measure(ffmpeg: str, clip: Path) -> tuple[float, bool]:
    """(seconds, has sound) of a clip, by the ffprobe beside ffmpeg."""

    tool = str(Path(ffmpeg).with_name("ffprobe")) if Path(ffmpeg).parent != Path(".") else "ffprobe"
    done = subprocess.run([tool, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(clip)],
                          capture_output=True, text=True, timeout=60)
    said = json.loads(done.stdout or "{}")
    seconds = float((said.get("format") or {}).get("duration") or 0)
    if done.returncode or seconds <= 0:
        raise RuntimeError(f"cannot read {clip.name}")
    return seconds, any(s.get("codec_type") == "audio" for s in said.get("streams") or [])


def sound(ffmpeg: str, clip: Path, pulses: list[float], seconds: float, has_sound: bool, out: Path) -> None:
    """The clip's sound with a sonar ping at each pulse, as an AAC file."""

    ins = ["-i", str(clip)] if has_sound else ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    graph, mixed = ["[0:a]aresample=48000,volume=0.75[bed]"], "[bed]"
    for i, at in enumerate(pulses, 1):
        ins += ["-f", "lavfi", "-t", "3", "-i", "sine=f=1280:r=48000", "-f", "lavfi", "-t", "1.2", "-i", "sine=f=52:r=48000"]
        ms = int(at * 1000)
        graph.append(f"[{2 * i - 1}:a]volume='0.55*exp(-5*t)':eval=frame,aecho=0.8:0.75:240|480|760:0.5|0.3|0.18,adelay={ms}:all=1[p{i}]")
        graph.append(f"[{2 * i}:a]volume='0.9*exp(-4*t)':eval=frame,adelay={ms}:all=1[s{i}]")
        mixed += f"[p{i}][s{i}]"
    graph.append(f"{mixed}amix=inputs={2 * len(pulses) + 1}:normalize=0:duration=first,alimiter=limit=0.95[a]")
    done = subprocess.run([ffmpeg, "-y", "-v", "error", *ins, "-filter_complex", ";".join(graph), "-map", "[a]", "-t", f"{seconds:.3f}",
                           "-c:a", "aac", "-b:a", "192k", str(out)], capture_output=True, text=True, timeout=1800)
    if done.returncode != 0 or not out.is_file():
        raise RuntimeError("ffmpeg could not mix the sonar pings: " + done.stderr.strip()[-200:])


def finish(service, source: Path, look: str, spec: dict) -> None:
    """Both Sonar copies of ``source`` (blocking: minutes). ``service`` is the LookService whose HDR copies are revealed."""

    ffmpeg = service.ffmpeg
    if not service.made(source, "hdr"):
        service.finish(source, "hdr")
    seconds, has_sound = measure(ffmpeg, source)
    drawn = plan(seconds, spec)
    targets = [service.copy(source, look, False), service.copy(source, look, True)]  # the SDR copy, the HDR copy
    with tempfile.TemporaryDirectory(prefix="ghost-sonar-") as scratch:
        scratch = Path(scratch)
        silent = [scratch / "sdr.mp4", scratch / "hdr.mp4"]
        order = {**drawn, "source": str(source), "colours": [str(service.copy(source, "hdr", False)), str(service.copy(source, "hdr", True))],
                 "outs": [str(p) for p in silent], "model": spec.get("model"), "depth": spec.get("depth"),
                 "floor_mb": int(spec.get("floor_mb", 6000)), "ffmpeg": ffmpeg}
        (scratch / "plan.json").write_text(json.dumps(order))
        worker = Path(__file__).with_name("sonar_worker.py")
        done = subprocess.run([str(spec.get("python") or sys.executable), str(worker), str(scratch / "plan.json")],
                              capture_output=True, text=True, timeout=float(spec.get("limit_s", LIMIT_S)))
        if done.returncode != 0 or not all(p.is_file() for p in silent):
            said = (done.stderr.strip() or done.stdout.strip() or "the worker ended").splitlines()[-1]
            raise RuntimeError(said[-240:])
        sound(ffmpeg, source, drawn["pulses"], seconds, has_sound, scratch / "sound.m4a")
        for picture, target, hdr in zip(silent, targets, (False, True)):
            part = target.with_name(target.name + ".part.mp4")
            done = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(picture), "-i", str(scratch / "sound.m4a"), "-map", "0:v:0",
                                   "-map", "1:a:0", "-c", "copy", *(["-tag:v", "hvc1"] if hdr else []), "-movflags", "+faststart", str(part)],
                                  capture_output=True, text=True, timeout=1800)
            if done.returncode != 0 or not part.is_file():
                raise RuntimeError("ffmpeg could not join the Sonar look's picture and sound: " + done.stderr.strip()[-200:])
            part.replace(target)
