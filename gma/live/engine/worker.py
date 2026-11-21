"""The live stage's picture engine: FastFrames (S, or the larger L) as an endless stream on one GB10, directed while it plays.

    worker.py '{"socket": "/run/live/stage.sock", "size": "640x480", "window": 12}'

It runs inside the stage's container (no network) and serves one client, the app, over a Unix socket. Messages both ways
are a 4-byte big-endian length, a JSON header of that length, and then ``sum(header["sizes"])`` bytes of payload.

    app -> worker   {"op": "start", "prompt": "...", "seed": 0}     a new scene begins to play
                    {"op": "prompt", "text": "...", "id": 7}        the scene from the next latent frame on
                    {"op": "stop"}                                  the scene ends; the models stay loaded
                    {"op": "take", "name": "t1", "seed": 0, "frames": 96, "script": [[0, "..."], [24, "..."]]}
                    {"op": "quit"}
    worker -> app   {"type": "ready", "size": [640, 480], "fps": 16, "window": 12, "load_s": ..}
                    {"type": "frame", "f": 12, "ms": 219.0, "late_ms": 0, "prompt": 7, "sizes": [..]} + JPEG pictures
                    {"type": "prompt", "id": 7, "at": 13, "ms": 140.0}   the direction is in from latent frame 13
                    {"type": "stopped", "frames": 96}
                    {"type": "take", "name": "t1", "state": "rendering" | "done" | "failed", ...}
                    (with "send_takes" the finished take's files follow as payload: "files" names them, "sizes" sizes them)
                    {"type": "error", "detail": "..."}

A latent frame is four pictures (a quarter of a second at 16 fps); the first of a scene is one. How the 21-frame clip
model becomes a stream:

* the self-attention cache is a ring of the last ``window`` latent frames: frame f is written to slot f % window and
  every frame attends to the whole ring (the order of the slots does not matter, each key carries its own position);
* positions keep counting past the clip's length: RoPE is relative, so only the distance between frames matters;
* the scene's first ``sink`` frames never leave the cache: their keys are turned in time before every frame so that they
  sit just beyond the ring. Without them one unchanged scene drifts orange in 25 s and is fog in 40 s; with three it
  holds for a minute (measured over 60 s; the face still ages slowly);
* a direction is a new text cache from the next latent frame on; the frames in the ring were made under the old one;
* the pixel decoder is causal and reaches six latent frames back: each new frame is decoded with the six before it.

Measured on a DGX Spark (2026-11-18), 640x480: a ring of 12 makes a latent frame in 219 ms (1.14x playback) and stays
sharp over 24 s; rings of 16 and 21 dissolve into fog after about 10 s, with BF16 attention as well as NVFP4.
FastFrames-L ("model": "ff-l", "lean": true) at the same size: 829 ms a latent frame (0.30x playback), 47.5 GB of GPU, a
far better picture, steady over 40 s.
"""
import dataclasses, io, json, os, queue, socket, struct, sys, threading, time
from collections import deque
from pathlib import Path

SPEC = {"model": "ff-s", "checkpoints": "/ckpt", "socket": "/run/live/stage.sock", "takes": "/run/live/takes", "size": "640x480",
        "window": 12, "sink": 3, "lead_s": 0.5, "jpeg": 82, "attn": 0.15, "fp4": True, "jit": True, "lean": False, "send_takes": False, "alone_s": 90, **json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")}
FRAME_S, REACH = 0.25, 7


def pack(header: dict, payload: bytes = b"") -> bytes:
    head = json.dumps(header).encode()
    return struct.pack(">I", len(head)) + head + payload


def read(conn) -> dict | None:
    """The next header from the app (its messages carry no payload). None when it has gone."""

    def exactly(n):
        data = b""
        while len(data) < n:
            piece = conn.recv(n - len(data))
            if not piece:
                return None
            data += piece
        return data
    size = exactly(4)
    body = exactly(struct.unpack(">I", size)[0]) if size else None
    return json.loads(body) if body else None


def main() -> None:
    began = time.time()
    import torch
    from PIL import Image
    sys.path.insert(0, "/work"); sys.path.insert(0, "/work/scripts")
    from generate import load_model
    from fastframes.pipeline import to_uint8
    from fastframes.video import save_video
    from fastframes.wan.attention import attention, cross_attention
    from fastframes.wan.causal_model import allocate_kv_cache
    import fastframes.wan.causal_model as cm
    import fastpipe as fp

    width, height = (int(v) for v in SPEC["size"].split("x"))
    window = int(SPEC["window"])
    model = load_model(SPEC["model"], SPEC["checkpoints"])
    model.config = cfg = dataclasses.replace(model.config, latent_size=(height // 8, width // 8), num_frames=window)
    model.frame_tokens = tokens = cfg.latent_size[0] * cfg.latent_size[1] // 4
    model.fast = None
    fp.install(model, fp4=bool(SPEC["fp4"]), lean=bool(SPEC["fp4"] and SPEC["lean"]), jit=bool(SPEC["jit"]))
    dit, device = model.dit, model.device
    at = {"pos": 0, "seen": 0}  # the frame being made: its position in time, and how many ring slots hold frames

    class Rope:
        """The (cos, sin) table of any position in time, made when first asked for."""

        def __init__(self):
            self.h, self.w = cfg.latent_size[0] // dit.patch_size[1], cfg.latent_size[1] // dit.patch_size[2]
            self.half = dit.dim // dit.num_heads // 2
            dit.freqs = dit.freqs.to(device)
            self.parts = dit.freqs.split([self.half - 2 * (self.half // 3), self.half // 3, self.half // 3], dim=1)
            self.made = {}

        def __getitem__(self, pos):
            if pos not in self.made:
                t, v, hz = self.parts; h, w = self.h, self.w
                if len(self.made) > 2 * window: self.made.pop(next(iter(self.made)))
                self.made[pos] = torch.view_as_real(torch.cat((
                    t[pos:pos + 1].view(1, 1, 1, -1).expand(1, h, w, -1), v[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                    hz[:w].view(1, 1, w, -1).expand(1, h, w, -1)), dim=-1).reshape(h * w, 1, self.half)).float().contiguous()
            return self.made[pos]

    model.fast.rope = Rope()
    positions = dit.freqs.shape[0]

    def forward_frame(self, x, e, grid_sizes, freqs, cross_kv, kv_cache, frame, fast=None, layer=0):
        """fastpipe.forward_frame over the ring: ``frame`` is the slot, at["pos"] the position, at["seen"] the slots in use."""
        st = self._ff
        count = x.shape[1]
        start, end, seen = frame * count, (frame + 1) * count, at["seen"] * count
        m = st.mod(self, e)
        sa = self.self_attn
        shape = (x.shape[0], count, sa.num_heads, sa.head_dim)
        rope = fast.rope[at["pos"]]
        q, k, v = st.qkv(fp.FN["pre"](x, m[1], m[0], self.norm1.eps)).chunk(3, dim=-1)
        q = fp.FN["rms_rope"](q, st.nq, rope, shape, st.eps)
        kv_cache["k"][:, start:end].copy_(fp.FN["rms_rope"](k, st.nk, rope, shape, st.eps))
        kv_cache["v"][:, start:end].copy_(v.view(shape))
        mode = fp.ATTN.get(layer)
        if mode is None:
            y = attention(q, kv_cache["k"][:, :seen], kv_cache["v"][:, :seen])
        else:
            packed = fp.packed(layer, sa.num_heads, sa.head_dim, kv_cache["k"].shape[1], count, x.device)
            packed.write(kv_cache["k"], kv_cache["v"], start, end)
            y = packed.attend(q, seen)
            if mode == "measure":
                exact = attention(q, kv_cache["k"][:, :seen], kv_cache["v"][:, :seen])
                fp.ERR.setdefault(layer, []).append(((y.float() - exact.float()).norm() / exact.float().norm()))
                y = exact
        x = fp.FN["post"](x, st.o(y.flatten(2)), m[2])
        cq = fp.FN["rms"](st.cq(self.norm3(x)), st.cnq, st.eps).view(shape)
        x = x + st.co(cross_attention(cq, cross_kv["k"], cross_kv["v"]).flatten(2))
        y = st.f2(fp.FN["gelu"](st.f0(fp.FN["pre"](x, m[4], m[3], self.norm2.eps))))
        return fp.FN["post"](x, y, m[5])

    cm.CausalWanAttentionBlock.forward_frame = forward_frame
    sink = max(0, min(int(SPEC.get("sink", 0)), window - 4))  # the scene's first frames, kept in the cache for good
    theta = torch.angle(model.fast.rope.parts[0][1]).double()  # the temporal frequencies (radians a latent frame)

    def turned(k0, delta):
        """Keys ``k0`` ([1, tokens, heads, dim] FP32) as they would be ``delta`` latent frames later in time."""
        n = theta.shape[0]
        cos, sin = torch.cos(theta * delta).float(), torch.sin(theta * delta).float()
        pairs = k0.view(*k0.shape[:-1], k0.shape[-1] // 2, 2)
        re, im = pairs[..., :n, 0], pairs[..., :n, 1]
        out = pairs.clone()
        out[..., :n, 0] = re * cos - im * sin
        out[..., :n, 1] = re * sin + im * cos
        return out.view(k0.shape).to(torch.bfloat16)

    grid_hw = (cfg.latent_size[0] // 2, cfg.latent_size[1] // 2)
    kv = allocate_kv_cache(dit, window, tokens, device=device)  # one ring for every scene: a new scene writes it over

    class Scene:
        """One stream: its noise, its place in the ring, the text it is made under."""

        def __init__(self, seed: int):
            self.generator = torch.Generator(device=device).manual_seed(int(seed))
            self.recent, self.f, self.text, self.kept = deque(maxlen=REACH), 0, None, {}

        def noise(self):
            return torch.randn((1, 1, 16, *cfg.latent_size), generator=self.generator, device=device, dtype=torch.bfloat16)

        def direct(self, prompt: str) -> None:
            self.text = dit.text_cache(model.encode_prompt(prompt))

        @torch.inference_mode()
        def step(self):
            """The next latent frame's pictures, [n, H, W, 3] uint8 on the GPU."""
            f = self.f
            slot = f if f < window else sink + (f - sink) % (window - sink)
            at["pos"], at["seen"] = f % positions, min(f + 1, window)
            if sink and f >= window:  # the kept frames stay just beyond the ring, in time: the anchor that holds the picture
                for i in range(sink):
                    delta = f - (window - sink) - (sink - 1 - i) - i
                    for layer, cache in enumerate(kv):
                        cache["k"][:, i * tokens:(i + 1) * tokens].copy_(turned(self.kept[i][layer], delta))
                        if fp.ATTN.get(layer) is not None:
                            fp.POOL[layer].write(cache["k"], cache["v"], i * tokens, (i + 1) * tokens)
            schedule = cfg.first_frame_timesteps if f == 0 else cfg.timesteps
            x = self.noise()
            for step, timestep in enumerate(schedule):
                x0, feature, mod = model._predict(x, timestep, slot, self.text, kv)
                if step + 1 < len(schedule):
                    t = torch.full((1,), schedule[step + 1], device=device, dtype=torch.float32)
                    x = model.scheduler.add_noise(x0.flatten(0, 1), self.noise().flatten(0, 1), t).unflatten(0, x0.shape[:2])
            if f == 0:
                model._predict(x0, 0.0, slot, self.text, kv)
            else:
                model._refine(x0, slot, kv)
            if f < sink:
                self.kept[f] = [cache["k"][:, f * tokens:(f + 1) * tokens].float().clone() for cache in kv]
            self.recent.append((feature, mod))
            grid = torch.tensor([[len(self.recent), *grid_hw]], device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pixels = model.decoder(torch.cat([r[0] for r in self.recent], dim=1), torch.cat([r[1] for r in self.recent], dim=1), grid)
            self.f += 1
            return to_uint8(pixels[0, -4:] if f else pixels[0, -1:])

    def rehearse(count: int) -> None:
        scene = Scene(99); scene.direct("A corgi surfing a wave at sunset")
        for _ in range(count): scene.step()
        torch.cuda.synchronize()

    # warm: the calibration of the NVFP4 scales and of which attention layers stay BF16, then the compiled pieces
    if SPEC["fp4"] and SPEC.get("attn"): fp.attention_fp4(model)
    rehearse(window + 4)
    if SPEC["fp4"] and SPEC.get("attn"): fp.attention_fp4(model, float(SPEC["attn"]))
    if SPEC["fp4"]:
        import fp4
        fp4.freeze()
    rehearse(window + 4)

    path = Path(SPEC["socket"]); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); server.bind(str(path)); server.listen(1)
    os.chmod(path, 0o600)
    commands: queue.Queue = queue.Queue()
    outbox: queue.Queue = queue.Queue(maxsize=64)
    link = {"conn": None, "since": time.monotonic()}
    ready = {"type": "ready", "size": [width, height], "fps": 16, "window": window, "load_s": round(time.time() - began, 1),
             "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "model": SPEC["model"]}

    def accept() -> None:  # one app at a time; when it goes the scene stops and the next one is waited for
        while True:
            conn, _ = server.accept()
            link["conn"] = conn
            outbox.put((ready, None))
            while (said := read(conn)) is not None:
                commands.put(said)
            link["conn"], link["since"] = None, time.monotonic()
            commands.put({"op": "stop", "gone": True})

    def alone() -> None:  # an app that died without a word must not leave the models loaded for good
        while True:
            time.sleep(20)
            if link["conn"] is None and time.monotonic() - link["since"] > float(SPEC["alone_s"]):
                print("STAGE ALONE: no app for", SPEC["alone_s"], "s, leaving", flush=True)
                os._exit(0)

    def send() -> None:  # pictures to JPEG here, off the GPU loop's thread (the encoder lets the interpreter go)
        while True:
            header, pictures = outbox.get()
            payload = b""
            if isinstance(pictures, list):  # files, as they are
                header["sizes"], payload = [len(p) for p in pictures], b"".join(pictures)
            elif pictures is not None:
                parts = []
                for picture in pictures.numpy():
                    out = io.BytesIO(); Image.fromarray(picture).save(out, "JPEG", quality=int(SPEC["jpeg"])); parts.append(out.getvalue())
                header["sizes"], payload = [len(p) for p in parts], b"".join(parts)
            conn = link["conn"]
            if conn is None:
                continue
            try:
                conn.sendall(pack(header, payload))
            except OSError:
                pass

    threading.Thread(target=accept, daemon=True).start()
    threading.Thread(target=send, daemon=True).start()
    threading.Thread(target=alone, daemon=True).start()
    print("STAGE READY", json.dumps(ready), flush=True)

    def say(header: dict, pictures=None) -> None:
        try:
            outbox.put_nowait((header, pictures))
        except queue.Full:  # the app is not reading: pictures are dropped, never the stream held
            if pictures is None or isinstance(pictures, list): outbox.put((header, pictures))

    def take(order: dict) -> None:
        name = "".join(c for c in str(order.get("name", "take")) if c.isalnum() or c in "-_")[:60] or "take"
        script = {int(f): str(p) for f, p in order.get("script", [])}
        total = max(1, min(int(order.get("frames", 0)), 2400))
        say({"type": "take", "name": name, "state": "rendering", "done": 0, "frames": total})
        started = time.time()
        try:
            if 0 not in script: raise ValueError("the take's script has no opening")
            scene, pictures = Scene(int(order.get("seed", 0))), []
            for f in range(total):
                if f in script: scene.direct(script[f])
                pictures.append(scene.step().cpu())
                if f % 8 == 7: say({"type": "take", "name": name, "state": "rendering", "done": f + 1, "frames": total})
            folder = Path(SPEC["takes"]); folder.mkdir(parents=True, exist_ok=True)
            frames = torch.cat(pictures)
            save_video(folder / f"{name}.mp4", frames)
            Image.fromarray(frames[min(len(frames) - 1, 8)].numpy()).save(folder / f"{name}.jpg", "JPEG", quality=85)
            done = {"type": "take", "name": name, "state": "done", "file": f"{name}.mp4", "seconds": round(len(frames) / 16, 2),
                    "took_s": round(time.time() - started, 1)}
            files = [f"{name}.mp4", f"{name}.jpg"] if SPEC["send_takes"] else []
            say({**done, "files": files} if files else done, [(folder / f).read_bytes() for f in files] if files else None)
        except Exception as wrong:  # noqa: BLE001 - told to the app, the worker stays up
            say({"type": "take", "name": name, "state": "failed", "detail": str(wrong)[-300:]})

    scene, zero, prompt_id = None, 0.0, 0
    while True:
        wait = None
        if scene is not None:
            due = zero + scene.f * FRAME_S - float(SPEC["lead_s"])
            wait = max(0.0, due - time.monotonic())
        try:
            order = commands.get(timeout=wait) if wait is None or wait > 0 else commands.get_nowait()
        except queue.Empty:
            order = None
        if order is not None:
            op = order.get("op")
            try:
                if op == "start":
                    scene = Scene(int(order.get("seed", 0))); scene.direct(str(order["prompt"]))
                    zero, prompt_id = time.monotonic(), int(order.get("id", 0))
                    say({"type": "started", "seed": int(order.get("seed", 0)), "prompt": prompt_id})
                elif op == "prompt" and scene is not None:
                    t = time.perf_counter(); scene.direct(str(order["text"])); torch.cuda.synchronize()
                    prompt_id = int(order.get("id", 0))
                    say({"type": "prompt", "id": prompt_id, "at": scene.f, "ms": round((time.perf_counter() - t) * 1e3, 1)})
                elif op == "stop":
                    if scene is not None: say({"type": "stopped", "frames": scene.f})
                    scene = None
                elif op == "take" and scene is None:
                    take(order)
                elif op == "quit":
                    return
            except Exception as wrong:  # noqa: BLE001
                say({"type": "error", "detail": f"{op}: {str(wrong)[-300:]}"})
            continue
        if scene is None:
            continue
        torch.cuda.synchronize(); t = time.perf_counter()
        pictures = scene.step().cpu()
        ms = (time.perf_counter() - t) * 1e3
        late = time.monotonic() - (zero + scene.f * FRAME_S)
        if late > 0.5:  # slower than it plays: the clock moves, the stream never bursts to catch up
            zero += late
        say({"type": "frame", "f": scene.f - 1, "ms": round(ms, 1), "late_ms": round(max(late, 0.0) * 1e3), "prompt": prompt_id}, pictures)


if __name__ == "__main__":
    main()
