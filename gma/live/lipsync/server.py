"""A talking face for a picture that already moves: the character's listening loop and a spoken line in, the same
loop with the mouth following the words out. MuseTalk 1.5 (Tencent Music Entertainment Lyra Lab, MIT) does the mouth:
one UNet pass per frame on a 256x256 crop of the face, in the latent space of sd-vae-ft-mse, led by Whisper-tiny's
features of the audio. Everything about the picture that does not change with the words (where the face is, its
latents, the mask the new mouth is blended through) is worked out once per loop (``/prepare``), so a line (``/speak``)
costs only Whisper, the UNet, the VAE decoder and the blend.

    POST /prepare {"video": path, "id"?: name}        -> {"id", "frames", "fps", "size", "face", "prepare_s"}
    POST /speak   {"id", "wav": path, "out": path, "start"?: frame, "stream"?: bool}
                                                       -> {"out", "frames", "seconds", "speak_s", "next", "done"}
    GET  /film?out=path                                the film as it is written (a streamed line), to its end
    GET  /job?out=path                                 -> {"done", "frames_made", "error"}
    GET  /health                                       -> {"avatars", "gpu_mb", "rss_mb"}

``start`` is the loop's frame the line begins on and ``next`` the frame it ends before, so the page can cut from the
listening loop into the line and back with no jump. A streamed line is a fragmented MP4 that grows while it is made
(about 1.3x faster than it plays on a GB10 that has the GPU to itself); ``/speak`` answers when its first pictures
are written, and ``/film`` hands it on as it grows.

It listens on a unix socket (the container has no network). The landmarks come from face-alignment (S3FD + 2D-FAN,
BSD-3) in place of MuseTalk's DWPose, the mask from MuseTalk's BiSeNet face parsing (MIT).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import resource
import subprocess
import sys
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
from aiohttp import web
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, "/work")
from musetalk.models.unet import PositionalEncoding  # noqa: E402
from musetalk.utils.blending import get_image_blending, get_image_prepare_material  # noqa: E402
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402

MODELS = os.environ.get("LIPSYNC_MODELS", "/models")
DEVICE = "cuda"
BATCH = int(os.environ.get("LIPSYNC_BATCH", "8"))
KEEP = 8  # avatars kept in memory (a cast's close-ups are one second each)
EVERY = 3  # the face is looked for in every third frame of a loop, and its box drawn between them
MARGIN = 10  # MuseTalk 1.5's extra margin under the chin, in pixels of a 256-pixel face: scaled to the face's size
LOCK = threading.Lock()


class Parsing(FaceParsing):
    def model_init(self, *_, **__):
        return super().model_init(f"{MODELS}/face-parse-bisent/resnet18-5c106cde.pth", f"{MODELS}/face-parse-bisent/79999_iter.pth")


class Engine:
    def __init__(self) -> None:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import AutoFeatureExtractor, WhisperModel
        import face_alignment

        t0 = time.time()
        config = json.load(open(f"{MODELS}/musetalkV15/musetalk.json"))
        self.unet = UNet2DConditionModel(**{k: v for k, v in config.items() if not k.startswith("_")})
        self.unet.load_state_dict(torch.load(f"{MODELS}/musetalkV15/unet.pth", map_location="cpu", weights_only=True))
        self.unet = self.unet.half().to(DEVICE).eval()
        self.pe = PositionalEncoding(d_model=384)
        self.vae = AutoencoderKL.from_pretrained(f"{MODELS}/sd-vae").half().to(DEVICE).eval()
        self.scale = self.vae.config.scaling_factor
        self.features = AutoFeatureExtractor.from_pretrained(f"{MODELS}/whisper")
        self.whisper = WhisperModel.from_pretrained(f"{MODELS}/whisper").half().to(DEVICE).eval()
        self.parsing = Parsing(left_cheek_width=90, right_cheek_width=90)
        self.marks = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=DEVICE, flip_input=False, compile=False)
        # face-alignment turns cudnn's benchmark on at every detection: its trial of every algorithm took the first
        # one to 14 s and the GPU to 35 GB (measured 2026-11-19). Its detector is given a torch that cannot.
        import types
        from face_alignment.detection.sfd import detect as sfd_detect

        class Quiet:
            backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))

            def __getattr__(self, name):
                return getattr(torch, name)
        sfd_detect.torch = Quiet()
        torch.backends.cudnn.benchmark = False
        self.step = torch.tensor([0], device=DEVICE)
        self.jobs: dict[str, dict] = {}
        with torch.no_grad():  # the first pass of each network is slow: made here, not in the first line
            made = self.unet(torch.zeros(BATCH, 8, 32, 32, device=DEVICE).half(), self.step,
                             encoder_hidden_states=torch.zeros(BATCH, 50, 384, device=DEVICE).half()).sample
            self.vae.decode(made / self.scale)
            self.latents(np.zeros((256, 256, 3), np.uint8))
            self.parsing(__import__("PIL.Image").Image.new("RGB", (256, 256)), mode="jaw")
        self.avatars: OrderedDict[str, dict] = OrderedDict()
        self.load_s = round(time.time() - t0, 1)

    # ----- once per loop -----

    def box(self, marks: np.ndarray, height: int) -> tuple[int, int, int, int]:
        """MuseTalk's crop from 68 landmarks: as wide as the face, from as far above the bridge of the nose as the chin
        is below it, down to the chin and a little under."""

        nose = marks[29, 1]
        chin = marks[:, 1].max()
        top = max(0.0, nose - (chin - nose))
        bottom = min(float(height), chin + MARGIN * (chin - top) / 256)
        return int(marks[:, 0].min()), int(top), int(marks[:, 0].max()), int(bottom)

    @torch.no_grad()
    def latents(self, face: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).to(DEVICE).permute(2, 0, 1).float() / 255
        both = torch.stack([x.clone(), x])
        both[0, :, 128:, :] = 0  # the lower half hidden: what the UNet paints in
        both = ((both - 0.5) / 0.5).half()
        made = self.vae.encode(both).latent_dist.sample() * self.scale
        return torch.cat([made[0:1], made[1:2]], dim=1)

    def prepare(self, video: str, name: str | None) -> dict:
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        reader = cv2.VideoCapture(video)
        fps = reader.get(cv2.CAP_PROP_FPS) or 24.0
        frames = []
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
        reader.release()
        if not frames:
            raise ValueError("no frames could be read from the loop")
        height, width = frames[0].shape[:2]
        seen: dict[int, tuple] = {}
        for at in sorted({*range(0, len(frames), EVERY), len(frames) - 1}):  # the head moves slowly: every third frame is looked at
            found = self.marks.get_landmarks_from_image(cv2.cvtColor(frames[at], cv2.COLOR_BGR2RGB))
            if found:
                best = max(found, key=lambda m: (m[:, 0].max() - m[:, 0].min()) * (m[:, 1].max() - m[:, 1].min()))
                seen[at] = self.box(np.asarray(best)[:, :2], height)
        if not seen:
            raise ValueError("no face was found in the loop")
        at = sorted(seen)
        boxes = np.stack([np.interp(np.arange(len(frames)), at, [seen[a][i] for a in at]) for i in range(4)], axis=1).astype(np.float32)
        # the box follows the head smoothly: a box that jumps a pixel from frame to frame makes the mouth shimmer
        kernel = np.hanning(9) / np.hanning(9).sum()
        padded = np.pad(boxes, ((4, 4), (0, 0)), mode="edge")
        boxes = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(4)], axis=1).round().astype(int)
        boxes[:, 0] = boxes[:, 0].clip(0, width - 2); boxes[:, 2] = boxes[:, 2].clip(1, width)
        boxes[:, 1] = boxes[:, 1].clip(0, height - 2); boxes[:, 3] = boxes[:, 3].clip(1, height)
        latents, masks, crops = [], [], []
        for frame, (x1, y1, x2, y2) in zip(frames, boxes):
            face = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.latents(face))
            mask, crop = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.parsing, mode="jaw")
            masks.append(mask); crops.append(crop)
        name = name or f"a{int(time.time() * 1000):x}"
        self.avatars[name] = {"frames": frames, "boxes": boxes.tolist(), "latents": torch.cat(latents), "masks": masks, "crops": crops,
                              "fps": fps, "size": [width, height]}
        self.avatars.move_to_end(name)
        while len(self.avatars) > KEEP:
            self.avatars.popitem(last=False)
        box = boxes[0].tolist()
        return {"id": name, "frames": len(frames), "fps": fps, "size": [width, height],
                "face": {"box": box, "width": box[2] - box[0], "height": box[3] - box[1]}, "prepare_s": round(time.time() - t0, 2),
                "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}

    # ----- once per line -----

    @torch.no_grad()
    def heard(self, wav: str, fps: float) -> torch.Tensor:
        """Whisper-tiny's features of the line, one window per frame (MuseTalk's get_whisper_chunk at this fps)."""

        rate, audio = wavfile.read(wav)
        audio = audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != 16000:
            g = math.gcd(int(rate), 16000)
            audio = resample_poly(audio, 16000 // g, int(rate) // g).astype(np.float32)
        feats = []
        for at in range(0, len(audio), 30 * 16000):
            mel = self.features(audio[at:at + 30 * 16000], return_tensors="pt", sampling_rate=16000).input_features
            hidden = self.whisper.encoder(mel.to(DEVICE).half(), output_hidden_states=True).hidden_states
            feats.append(torch.stack(hidden, dim=2))
        feats = torch.cat(feats, dim=1)[:, :math.floor(len(audio) / 16000 * 50)]
        per = 50 / fps
        pad = math.ceil(per)
        feats = torch.cat([torch.zeros_like(feats[:, :pad * 2]), feats, torch.zeros_like(feats[:, :pad * 6])], 1)
        count = math.floor(len(audio) / 16000 * fps)
        windows = [feats[:, math.floor(i * per): math.floor(i * per) + 10] for i in range(count)]
        windows = torch.cat([w for w in windows if w.shape[1] == 10], dim=0)  # T, 10, 5, 384
        return windows.reshape(windows.shape[0], 50, 384)

    def speak(self, name: str, wav: str, out: str, start: int = 0, stream: bool = False) -> dict:
        avatar = self.avatars.get(name)
        if avatar is None:
            raise KeyError("no such avatar: prepare its loop first")
        self.avatars.move_to_end(name)
        t0 = time.time()
        with LOCK:
            windows = self.heard(wav, avatar["fps"])
        count, fps, total = len(windows), avatar["fps"], len(avatar["frames"])
        job = {"out": out, "frames": count, "seconds": round(count / fps, 2), "start": start, "next": (start + count) % total,
               "done": False, "frames_made": 0, "error": None, "first": threading.Event()}
        self.jobs[out] = job
        while len(self.jobs) > 16:
            self.jobs.pop(next(iter(self.jobs)))
        worker = threading.Thread(target=self.film, args=(avatar, windows, wav, job, stream, t0), daemon=True)
        worker.start()
        if stream:
            job["first"].wait(120)
            job["first_s"] = round(time.time() - t0, 2)
        else:
            worker.join()
        if job["error"]:
            raise RuntimeError(job["error"])
        return {k: v for k, v in job.items() if k != "first"}

    @torch.no_grad()
    def film(self, avatar: dict, windows: torch.Tensor, wav: str, job: dict, stream: bool, t0: float) -> None:
        fps, (width, height), total, start = avatar["fps"], avatar["size"], len(avatar["frames"]), job["start"]
        # a streamed line is written in pieces a player can begin on (one each half second), with no look-ahead
        how = (["-tune", "zerolatency", "-g", str(round(fps / 2)), "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1"]
               if stream else ["-movflags", "+faststart"])
        ffmpeg = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
             "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-shortest", *how, "-f", "mp4", job["out"]], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_s = 0.0
        faces_made: queue.Queue = queue.Queue(maxsize=4)

        def write() -> None:  # the blend and the film's writing, beside the GPU's work and not after it
            while True:
                got = faces_made.get()
                if got is None:
                    return
                for i, face in zip(*got):
                    x1, y1, x2, y2 = avatar["boxes"][i]
                    face = cv2.resize(np.ascontiguousarray(face), (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                    frame = get_image_blending(avatar["frames"][i], face, [x1, y1, x2, y2], avatar["masks"][i], avatar["crops"][i])
                    try:
                        ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
                    except OSError:
                        return
                job["frames_made"] += len(got[0])
                if job["frames_made"] >= min(len(windows), round(fps)) and os.path.exists(job["out"]) and os.path.getsize(job["out"]) > 2048:
                    job["first"].set()

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        try:
            with LOCK:
                torch.cuda.reset_peak_memory_stats()
                for at in range(0, len(windows), BATCH):
                    t1 = time.time()
                    index = [(start + at + i) % total for i in range(min(BATCH, len(windows) - at))]
                    full = index + [index[-1]] * (BATCH - len(index))  # every pass the same shape: no pause at the last one
                    heard = windows[at:at + len(index)]
                    heard = torch.cat([heard, heard[-1:].expand(BATCH - len(index), -1, -1)]) if len(index) < BATCH else heard
                    made = self.unet(avatar["latents"][full], self.step, encoder_hidden_states=self.pe(heard.float()).half()).sample
                    faces = (self.vae.decode(made / self.scale).sample / 2 + 0.5).clamp(0, 1)
                    faces = (faces.permute(0, 2, 3, 1).float() * 255).round().byte().cpu().numpy()[..., ::-1]
                    gpu_s += time.time() - t1
                    faces_made.put((index, faces))
                job["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            faces_made.put(None)
            writer.join()
            ffmpeg.stdin.close()
            wrong = ffmpeg.stderr.read().decode()[-300:]
            if ffmpeg.wait() != 0:
                job["error"] = f"the film could not be written: {wrong}"
        except Exception as wrong:  # noqa: BLE001 - the caller is told
            job["error"] = f"{type(wrong).__name__}: {str(wrong)[:300]}"
            ffmpeg.kill()
            try:
                faces_made.put_nowait(None)
            except queue.Full:
                pass
        job.update(speak_s=round(time.time() - t0, 2), gpu_s=round(gpu_s, 2), done=True)
        job["first"].set()


def main() -> None:
    global FFMPEG
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    socket = sys.argv[1]
    engine = Engine()

    async def call(work, *args):
        def locked():
            with LOCK:
                return work(*args)
        try:
            return web.json_response(await asyncio.to_thread(locked))
        except (KeyError, ValueError) as wrong:
            return web.json_response({"error": str(wrong).strip("'\"")}, status=409)
        except Exception as wrong:  # noqa: BLE001 - the caller is told; the service stays up
            return web.json_response({"error": f"{type(wrong).__name__}: {str(wrong)[:300]}"}, status=500)

    async def prepare(request):
        said = await request.json()
        return await call(engine.prepare, str(said["video"]), said.get("id"))

    async def speak(request):
        said = await request.json()
        try:
            return web.json_response(await asyncio.to_thread(engine.speak, str(said["id"]), str(said["wav"]), str(said["out"]),
                                                             int(said.get("start") or 0), bool(said.get("stream"))))
        except (KeyError, ValueError) as wrong:
            return web.json_response({"error": str(wrong).strip("'\"")}, status=409)
        except Exception as wrong:  # noqa: BLE001
            return web.json_response({"error": f"{type(wrong).__name__}: {str(wrong)[:300]}"}, status=500)

    async def job(request):
        found = engine.jobs.get(request.query.get("out", ""))
        if found is None:
            return web.json_response({"error": "no such line"}, status=404)
        return web.json_response({k: v for k, v in found.items() if k != "first"})

    async def film(request):
        """The line's film as it grows: what is written is sent, to the end of the film."""

        found = engine.jobs.get(request.query.get("out", ""))
        if found is None or not os.path.exists(found["out"]):
            return web.json_response({"error": "no such line"}, status=404)
        answer = web.StreamResponse(headers={"Content-Type": "video/mp4", "Cache-Control": "no-store"})
        await answer.prepare(request)
        with open(found["out"], "rb") as file:
            while True:
                piece = file.read(65536)
                if piece:
                    await answer.write(piece)
                elif found["done"]:
                    break
                else:
                    await asyncio.sleep(0.03)
        await answer.write_eof()
        return answer

    async def health(_):
        return web.json_response({"avatars": list(engine.avatars), "load_s": engine.load_s,
                                  "gpu_mb": round(torch.cuda.memory_allocated() / 2**20), "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20),
                                  "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                                  "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)})

    app = web.Application(client_max_size=2**20)
    app.add_routes([web.post("/prepare", prepare), web.post("/speak", speak), web.get("/job", job), web.get("/film", film), web.get("/health", health)])
    print("LIPSYNC READY", engine.load_s, "s", flush=True)
    web.run_app(app, path=socket, print=None)


if __name__ == "__main__":
    main()
