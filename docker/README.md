# Boo in Docker (DGX Spark)

`ghcr.io/drowzeys/keys-ghost-machine-audio-visual-agent` runs the whole Spark stack in one container: Nemotron 3.5 Bolt
on stock TensorFold 1.1.4, Drama 3 NVFP4 with the 14 voices and Cosmos3-Edge on the keys TensorFold builds, Parakeet v3
(NeMo), the agent and page, and optionally Eyes of Kai and the lite voice (Kokoro-82M, `VOICE_ENGINE=lite` or `both`).
Model weights are not in the image; mount them.

`VOICE_ENGINE` needs image 0.2.0 or later (0.1.0 has the full voice, Drama 3, only). Since 0.2.0 the DGX Spark's
Drama 3 engine also has TensorFold 1.1.4's silence guard, which the container turns on.

## Requirements

One NVIDIA DGX Spark (GB10, linux/arm64) with Docker and the NVIDIA Container Toolkit (`--gpus all`), about 64 GB of
free memory, or about 48 GB with the lite voice (see the main README), and the weights below.

## Mounts

| Path in the container | What | Where to get it |
|---|---|---|
| `/models/bolt` | Nemotron 3.5 Bolt 74B-A9B, TensorFold 24 GB layout | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B` |
| `/models/drama` | Drama 3 NVFP4 pruned (not needed with `VOICE_ENGINE=lite`) | `plotarmordev/drama-3-NVFP4-Pruned` (Fish Audio licence) |
| `/models/cosmos3` | Cosmos3-Edge, converted | `nvidia/Cosmos3-Edge` + `tools/cosmos3_edge_to_mlx.py` from the release archive |
| `/models/hf-cache` | Hugging Face cache; Parakeet v3 (and Kokoro-82M for the lite voice) download here on first start | — |

A model folder that is not mounted is skipped. Point at a server elsewhere instead with `BOLT_URL`, `DRAMA_URL` or
`COSMOS_URL` (OpenAI-compatible base URLs ending in `/v1`).

## Run

```sh
docker run --gpus all --rm -p 8095:8095 \
  -v ~/models/bolt:/models/bolt -v ~/models/drama:/models/drama -v ~/models/cosmos3:/models/cosmos3 \
  -v ~/models/hf-cache:/models/hf-cache -e KAI=1 \
  ghcr.io/drowzeys/keys-ghost-machine-audio-visual-agent:0.2.1
```

Lite voice only (Kokoro-82M, no Drama 3 mount), or both with a Full/Lite switch in the page:

```sh
docker run --gpus all --rm -p 8095:8095 -e VOICE_ENGINE=lite \
  -v ~/models/bolt:/models/bolt -v ~/models/cosmos3:/models/cosmos3 -v ~/models/hf-cache:/models/hf-cache \
  ghcr.io/drowzeys/keys-ghost-machine-audio-visual-agent:0.2.1
# both: add -v ~/models/drama:/models/drama and use -e VOICE_ENGINE=both
```

Then open `http://<spark>:8095/`. Browsers allow the microphone and camera only over HTTPS or on localhost; from a phone,
put HTTPS in front of it (for example Tailscale Serve) as the main README describes.

| Setting | Default | Meaning |
|---|---|---|
| `BOO_PORT` | 8095 | the page and API |
| `BOLT_PORT`, `DRAMA_PORT`, `COSMOS_PORT` | 8091, 8093, 8099 | the model servers inside the container |
| `VOICE` | vesper | default Drama 3 voice |
| `VOICE_ENGINE` | full | `full` (Drama 3), `lite` (Kokoro-82M instead) or `both` (a Full/Lite switch in the page) |
| `KOKORO_VOICE`, `KOKORO_PORT` | bf_emma, 8197 | the lite voice's default voice and port; `KOKORO_URL` points at one elsewhere |
| `KAI` | 0 | 1 starts Eyes of Kai for camera memory (kept 2 days; the page can change it) |
| `HERMES_CMD` | — | a Hermes ACP command for the back agent, e.g. `ssh host hermes acp` |
| `DEEP_URL`, `DEEP_MODEL` | — | an OpenAI-compatible model to hand deep questions to instead |

## Build it yourself

The build context is this repository plus three folders beside it: the official `tensorfold-1.1.4-linux-aarch64`
archive, `keys-boo-tensorfold-1.1.4-linux-aarch64` from this repo's release, and `voices/` (the `voices/spark` files and
previews from the Hugging Face repo).

```sh
docker build -f docker/Dockerfile -t keys-ghost-machine .
```

Lite voice only, without Drama 3:

```sh
docker run --gpus all --rm -p 8095:8095 -e VOICE_ENGINE=lite \
  -v ~/models/bolt:/models/bolt -v ~/models/cosmos3:/models/cosmos3 -v ~/models/hf-cache:/models/hf-cache \
  keys-ghost-machine
```
