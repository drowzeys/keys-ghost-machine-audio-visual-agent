# Boo in Docker (DGX Spark)

`ghcr.io/drowzeys/keys-ghost-machine-audio-visual-agent` runs the whole Spark stack in one container: Nemotron 3.5 Bolt
on stock TensorFold 1.1.4, Drama 3 NVFP4 with the 14 voices and Cosmos3-Edge on the keys TensorFold builds, Parakeet v3
(NeMo), the agent and page, and optionally Eyes of Kai. Model weights are not in the image; mount them.

## Requirements

One NVIDIA DGX Spark (GB10, linux/arm64) with Docker and the NVIDIA Container Toolkit (`--gpus all`), about 64 GB of
free memory (see the main README) and the weights below.

## Mounts

| Path in the container | What | Where to get it |
|---|---|---|
| `/models/bolt` | Nemotron 3.5 Bolt 74B-A9B, TensorFold 24 GB layout | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B` |
| `/models/drama` | Drama 3 NVFP4 pruned | `plotarmordev/drama-3-NVFP4-Pruned` (Fish Audio licence) |
| `/models/cosmos3` | Cosmos3-Edge, converted | `nvidia/Cosmos3-Edge` + `tools/cosmos3_edge_to_mlx.py` from the release archive |
| `/models/hf-cache` | Hugging Face cache; Parakeet v3 downloads here on first start | — |

A model folder that is not mounted is skipped. Point at a server elsewhere instead with `BOLT_URL`, `DRAMA_URL` or
`COSMOS_URL` (OpenAI-compatible base URLs ending in `/v1`).

## Run

```sh
docker run --gpus all --rm -p 8095:8095 \
  -v ~/models/bolt:/models/bolt -v ~/models/drama:/models/drama -v ~/models/cosmos3:/models/cosmos3 \
  -v ~/models/hf-cache:/models/hf-cache -e KAI=1 \
  ghcr.io/drowzeys/keys-ghost-machine-audio-visual-agent:0.1.0
```

Then open `http://<spark>:8095/`. Browsers allow the microphone and camera only over HTTPS or on localhost; from a phone,
put HTTPS in front of it (for example Tailscale Serve) as the main README describes.

| Setting | Default | Meaning |
|---|---|---|
| `BOO_PORT` | 8095 | the page and API |
| `BOLT_PORT`, `DRAMA_PORT`, `COSMOS_PORT` | 8091, 8093, 8099 | the model servers inside the container |
| `VOICE` | vesper | default voice |
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
