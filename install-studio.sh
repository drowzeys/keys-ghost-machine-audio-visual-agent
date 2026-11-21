#!/usr/bin/env bash
# One-shot install of Ghost Machine Studio on an NVIDIA DGX Spark (GB10, Linux aarch64, CUDA 13):
#
#   curl -fsSL https://raw.githubusercontent.com/drowzeys/ghost-machine-studio/main/install-studio.sh | bash -s -- --accept-h3-license
#   … | bash -s -- --agent-only       Ghost Agent alone (Boo, the camera, the back agent): no Studio, no clips
#
# Everything goes under one folder (~/ghost-studio unless GMS_HOME says otherwise). Running it again picks up where it
# stopped and skips what is already there. No sudo. When it finishes, `gms start` runs the Studio (it starts it for you
# unless GMS_START=0), and the page is http://localhost:8196/.
#
# What it installs, in order:
#   1. this repository, a Python environment for the app, and one for the lite voice (Kokoro-82M)
#   2. TensorFold: the stock release for the brain, and the keys build for the camera model
#   3. the brain (Qwen3.8-27B MLX 4-bit and its DFlash2 drafter, 19 GB) and the camera model (Cosmos3-Edge, converted)
#   4. Imagination Studio, with its own installer when it is not on this machine yet (keyframes, a film's sound)
#   5. the clip worker: FastVideo, its CUDA kernels, the FastH3 V2 NVFP4 weights (57 GB) and the first-frame table
#   6. Live Action's blocking preview: NVIDIA MotionBricks (needs Docker)
#   7. the profile, the `gms` command and the memory guard
#
# MiniMax H3 (steps 4 and 5: every clip, take and film) is under the MiniMax H3 Community License, which excludes the
# USA, the EU, the UK and South Korea without MiniMax's authorization. Steps 4 and 5 run only with --accept-h3-license
# (or GMS_ACCEPT_H3=1): read that licence first. MotionBricks' sample motions are under the NVIDIA Sample Data Evaluation
# License. Nothing here ships a model: each is downloaded from its owner.
#
# Options (the same as the settings in capitals):
#   --accept-h3-license   you have read the MiniMax H3 Community License and may use H3 where you are   GMS_ACCEPT_H3=1
#   --agent-only          steps 1-3 and 7 only                                                          GMS_AGENT_ONLY=1
#   --no-live             skip step 6                                                                   GMS_LIVE=0
#   --no-start            install only                                                                  GMS_START=0
# Settings, all optional:
#   GMS_HOME           install folder                                   (~/ghost-studio)
#   GMS_MODELS         model folder                                     ($GMS_HOME/models)
#   GMS_QWEN_DIR       an existing Qwen3.8-27B MLX 4-bit folder         ($GMS_MODELS/qwen38-27b-mlx4)
#   GMS_DRAFTER_DIR    an existing Qwen3.8-27B DFlash2 folder           ($GMS_MODELS/qwen38-27b-dflash2)
#   GMS_COSMOS_DIR     an existing converted Cosmos3-Edge folder        ($GMS_MODELS/cosmos3-tf)
#   GMS_FASTH3_DIR     an existing FastH3 V2 NVFP4 Consumer folder      ($GMS_MODELS/fastvideo/FastH3-V2-Consumer)
#   GMS_ADALN_TABLE    an existing first-frame table                    ($GMS_MODELS/fastvideo/adaln_tables_t2v_fl2va.pt)
#   GMS_STUDIO_PREFIX  where Imagination Studio is (or goes)            (~/.local/opt/tensorfold-studio)
#   GMS_STUDIO_FLAGS   flags for Imagination Studio's installer         (--no-render)
#   GMS_BLOCKING_IMAGE a container image with python3, for the blocking preview (python:3.12-slim, pulled)
#   GMS_PORT_BASE      move every port: base+196 the page, +92 brain, +197 voice, +99 camera model, +420 clips,
#                      +890 the Studio, +290 its ComfyUI                (unset: 8196, 8092, 8197, 8099, 8420, 7890, 8290)
#   GMS_REF            the branch or tag of this repository to install  (main)
#   GMS_FORCE=1        install even if the memory or disk check fails
#   HF_TOKEN           a Hugging Face token, if you have one (not required)

set -euo pipefail

REPO=drowzeys/ghost-machine-studio
BOO_RELEASE=v0.3.0            # the release that holds the keys TensorFold build (Cosmos3-Edge on CUDA, its converter)
KEYS_TF=1.1.5
TF_VERSION=1.1.8              # stock TensorFold, for the brain
FASTVIDEO_COMMIT=a01c50a8063386895c6d8895a1e2509d11d55284
FASTH3_REPO=FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer
FASTH3_REV=9d0d0d3a
MOTIONBRICKS_COMMIT=54fc563aeaaf819a753d72cc9fdc6713989cdd43
STUDIO_REPO=Keys-Wasserman/Imagination-Studio-by-keys-Wasserman
BLOCKING_IMAGES="${GMS_BLOCKING_IMAGE:-} python:3.12-slim public.ecr.aws/docker/library/python:3.12-slim"   # any image with python3
BLOCKING_IMAGE=

GMS_HOME=${GMS_HOME:-$HOME/ghost-studio}
GMS_MODELS=${GMS_MODELS:-$GMS_HOME/models}
QWEN_DIR=${GMS_QWEN_DIR:-$GMS_MODELS/qwen38-27b-mlx4}
DRAFTER_DIR=${GMS_DRAFTER_DIR:-$GMS_MODELS/qwen38-27b-dflash2}
COSMOS_DIR=${GMS_COSMOS_DIR:-$GMS_MODELS/cosmos3-tf}
COSMOS_SRC=$GMS_MODELS/cosmos3-edge-src
FASTH3_DIR=${GMS_FASTH3_DIR:-$GMS_MODELS/fastvideo/FastH3-V2-Consumer}
ADALN_TABLE=${GMS_ADALN_TABLE:-$GMS_MODELS/fastvideo/adaln_tables_t2v_fl2va.pt}
STUDIO_PREFIX=${GMS_STUDIO_PREFIX:-$HOME/.local/opt/tensorfold-studio}
STUDIO_FLAGS=${GMS_STUDIO_FLAGS:---no-render}
REF=${GMS_REF:-main}

say()  { printf '\n\033[1;35m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

ACCEPT_H3=${GMS_ACCEPT_H3:-0}; AGENT_ONLY=${GMS_AGENT_ONLY:-0}; LIVE=${GMS_LIVE:-1}; START=${GMS_START:-1}
while [ $# -gt 0 ]; do
  case $1 in
    --accept-h3-license|--accept-h3-licence) ACCEPT_H3=1 ;;
    --agent-only) AGENT_ONLY=1 ;;
    --no-live) LIVE=0 ;;
    --no-start) START=0 ;;
    *) die "unknown option $1 (the installer takes --accept-h3-license, --agent-only, --no-live, --no-start)" ;;
  esac
  shift
done
[ "$AGENT_ONLY" = 1 ] && LIVE=0
if [ "$AGENT_ONLY" != 1 ] && [ "$ACCEPT_H3" != 1 ]; then
  die "The Studio's clips, takes and films are made with MiniMax H3, whose Community License excludes the USA, the EU,
   the UK and South Korea without MiniMax's authorization. Read it (https://huggingface.co/$FASTH3_REPO), then run
   this again with --accept-h3-license. Or install Ghost Agent alone with --agent-only."
fi

PATH="$GMS_HOME/tools/bin:$HOME/.local/bin:$PATH"; export PATH

# ---------------------------------------------------------------------------------------------------------- platform
say "Checking this machine"
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = aarch64 ] && have nvidia-smi || [ -n "${GMS_FORCE:-}" ] || \
  die "Ghost Machine Studio runs on an NVIDIA DGX Spark (Linux aarch64, CUDA 13). This is $(uname -s) $(uname -m)."
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case $GPU in *GB10*|"") ;; *) note "GPU is '$GPU'; the Studio is built and tested for the GB10 (DGX Spark).";; esac
TOTAL_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
FREE_MEM_GB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
MEM_NEED=110; [ "$AGENT_ONLY" = 1 ] && MEM_NEED=48
note "DGX Spark (${GPU:-NVIDIA GPU}), $TOTAL_GB GB of memory, $FREE_MEM_GB GB free now"
if [ "$TOTAL_GB" -lt "$MEM_NEED" ]; then
  [ -n "${GMS_FORCE:-}" ] || die "This needs a machine with at least $MEM_NEED GB of memory; this one has $TOTAL_GB GB. (GMS_FORCE=1 installs anyway.)"
  note "warning: $TOTAL_GB GB is under the $MEM_NEED GB needed (continuing: GMS_FORCE)"
fi
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
ls "$CUDA_HOME"/lib64/libcublasLt.so.13* >/dev/null 2>&1 || note "warning: no CUDA 13 cuBLAS in $CUDA_HOME/lib64 (DGX OS has it there; set CUDA_HOME)"

ready() { [ -f "$1/.gms-ready" ] || { [ -n "${2:-}" ] && [ -e "$1/$2" ]; }; }
NEED_GB=20   # the environments, the engines, the caches
ready "$QWEN_DIR" config.json || NEED_GB=$((NEED_GB + 16))
ready "$DRAFTER_DIR" config.json || NEED_GB=$((NEED_GB + 4))
ready "$COSMOS_DIR" model.safetensors || NEED_GB=$((NEED_GB + 10))
if [ "$AGENT_ONLY" != 1 ]; then
  ready "$FASTH3_DIR" model_index.json || NEED_GB=$((NEED_GB + 58))
  [ -d "$GMS_HOME/fv-venv/lib" ] || NEED_GB=$((NEED_GB + 12))
  [ -x "$STUDIO_PREFIX/venv/bin/python" ] || NEED_GB=$((NEED_GB + 120))   # Imagination Studio's core install, roughly
fi
mkdir -p "$GMS_HOME" "$GMS_MODELS" "$GMS_HOME/logs"
FREE_GB=$(df -Pk "$GMS_MODELS" | awk 'NR==2{print int($4/1048576)}')
note "$FREE_GB GB free on disk; this install still needs about $NEED_GB GB"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  [ -n "${GMS_FORCE:-}" ] || die "Not enough disk: about $NEED_GB GB needed, $FREE_GB GB free in $GMS_MODELS. Free some space or point GMS_MODELS elsewhere."
  note "warning: not enough disk (continuing: GMS_FORCE)"
fi
[ -n "${GMS_CHECK_ONLY:-}" ] && { say "Checks passed"; exit 0; }

# ------------------------------------------------------------------------------------------------------------ tools
say "Tools"
have git || die "git is needed (sudo apt install git)."
have curl || die "curl is needed."
if [ "$LIVE" = 1 ]; then
  have docker && docker info >/dev/null 2>&1 || die "Live Action's blocking preview runs in a Docker container: Docker must be installed and usable without sudo
   (sudo usermod -aG docker \$USER, then log in again). Or run this again with --no-live."
fi
sha256() { sha256sum "$1" | cut -d' ' -f1; }
fetch() {  # url file: download once, resumable
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}
verify() {  # file checksum-file
  local want got; want=$(cut -d' ' -f1 < "$2"); got=$(sha256 "$1")
  [ "$want" = "$got" ] || { rm -f "$1"; die "checksum mismatch for $(basename "$1"); deleted it, run the installer again."; }
  note "verified $(basename "$1")"
}
if ! have uv; then
  note "installing uv (Python manager) into $GMS_HOME/tools"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$GMS_HOME/tools/bin" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
fi
note "uv $(uv --version | cut -d' ' -f2)"

# ---------------------------------------------------------------------------------------------------------- the app
say "Ghost Machine Studio's app"
APP=$GMS_HOME/repo
if [ -d "$APP/.git" ]; then
  git -C "$APP" fetch -q origin "$REF" && git -C "$APP" checkout -q FETCH_HEAD 2>/dev/null || note "warning: could not update $APP (local changes?); using it as it is"
elif [ -d "$APP" ] && [ -f "$APP/voice_agent/server.py" ]; then
  note "using the copy already in $APP (not a git checkout: it is not updated)"
else
  git clone -q "https://github.com/$REPO" "$APP" && git -C "$APP" checkout -q "$REF"
fi
note "$(cat "$APP/VERSION" 2>/dev/null || echo '?') $(git -C "$APP" log -1 --format='%h %s' 2>/dev/null | cut -c1-70)"

say "Python environments"
VENV=$GMS_HOME/venv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
if ! "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  note "PyTorch for CUDA 13 (aarch64), for NeMo's speech recognition"
  uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu130
fi
uv pip install -q --python "$VENV/bin/python" -r "$APP/requirements-spark.txt" "huggingface_hub>=0.34" imageio-ffmpeg
note "the app: $VENV"
KV=$GMS_HOME/kokoro-venv   # its own environment: misaki pins spaCy's tokenizer libraries
[ -x "$KV/bin/python" ] || uv venv -q --python 3.12 "$KV"
if ! "$KV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  uv pip install -q --python "$KV/bin/python" torch --index-url https://download.pytorch.org/whl/cu130
fi
uv pip install -q --python "$KV/bin/python" -r "$APP/requirements-kokoro-spark.txt"
note "the lite voice (Kokoro-82M): $KV"
mkdir -p "$GMS_HOME/tools/bin"
if ! have ffmpeg; then   # the blocking preview and the takes are written with it
  ln -sf "$("$VENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$GMS_HOME/tools/bin/ffmpeg"
  note "ffmpeg: the one that comes with imageio-ffmpeg (none was installed)"
fi

# ---------------------------------------------------------------------------------------------- TensorFold builds
say "TensorFold"
ENG=$GMS_HOME/engines; mkdir -p "$ENG/dl"
KB=keys-boo-tensorfold-$KEYS_TF-linux-aarch64
if [ ! -x "$ENG/$KB/bin/tensorfold-cosmos3" ]; then
  base="https://github.com/$REPO/releases/download/$BOO_RELEASE"
  fetch "$base/$KB.tar.gz.sha256" "$ENG/dl/$KB.tar.gz.sha256"
  fetch "$base/$KB.tar.gz" "$ENG/dl/$KB.tar.gz"
  verify "$ENG/dl/$KB.tar.gz" "$ENG/dl/$KB.tar.gz.sha256"
  LC_ALL=C tar xzf "$ENG/dl/$KB.tar.gz" -C "$ENG"
fi
note "keys build (the camera model's engine; an unofficial community build of TensorFold, Apache-2.0): $ENG/$KB"
TFR=tensorfold-$TF_VERSION-linux-aarch64
if [ ! -x "$ENG/$TFR/bin/tensorfold-native" ]; then
  base="https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION"
  fetch "$base/$TFR.tar.gz.sha256" "$ENG/dl/$TFR.tar.gz.sha256"
  fetch "$base/$TFR.tar.gz" "$ENG/dl/$TFR.tar.gz"
  verify "$ENG/dl/$TFR.tar.gz" "$ENG/dl/$TFR.tar.gz.sha256"
  LC_ALL=C tar xzf "$ENG/dl/$TFR.tar.gz" -C "$ENG"
fi
note "stock TensorFold $TF_VERSION (the brain's engine): $ENG/$TFR"

# ---------------------------------------------------------------------------------------------------------- models
HF=$VENV/bin/hf
export HF_HUB_DISABLE_PROGRESS_BARS=1
download() {  # repo dir [hf download options...]
  local repo=$1 dir=$2; shift 2
  "$HF" download "$repo" --local-dir "$dir" "$@" >/dev/null && touch "$dir/.gms-ready"
}
say "Models (skipped when already there)"
if ready "$QWEN_DIR" config.json; then note "the brain: $QWEN_DIR"; else
  note "Qwen3.8-27B, MLX 4-bit (about 15 GB)…"; download TensorFold/Qwen3.8-27B-MLX-4bit "$QWEN_DIR"; fi
if ready "$DRAFTER_DIR" config.json; then note "its drafter: $DRAFTER_DIR"; else
  note "the DFlash2 drafter for Qwen3.8-27B (about 4 GB)…"; download z-lab/Qwen3.8-27B-DFlash2 "$DRAFTER_DIR"; fi
if ready "$COSMOS_DIR" model.safetensors; then note "the camera model: $COSMOS_DIR"; else
  if ! ready "$COSMOS_SRC"; then
    note "Cosmos3-Edge, the reasoner parts only (about 7.2 GB)…"
    download nvidia/Cosmos3-Edge "$COSMOS_SRC" --include "config.json" --include "model.safetensors.index.json" --include "*.jinja" \
      --include "*_config.json" --include "tokenizer.json" --include "special_tokens_map.json" --include "transformer/*" --include "vision_encoder/*"
  fi
  note "converting Cosmos3-Edge to TensorFold's 8-bit format (a few minutes)…"
  CV=$GMS_HOME/convert-venv
  [ -x "$CV/bin/python" ] || uv venv -q --python 3.12 "$CV"
  uv pip install -q --python "$CV/bin/python" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$CV/bin/python" "mlx[cpu]==0.32.4" "transformers==5.20.2" accelerate safetensors numpy
  "$CV/bin/python" "$ENG/$KB/tools/cosmos3_edge_to_mlx.py" "$COSMOS_SRC" "$COSMOS_DIR.part" 8
  rm -rf "$COSMOS_DIR" && mv "$COSMOS_DIR.part" "$COSMOS_DIR" && touch "$COSMOS_DIR/.gms-ready"
  [ -n "${GMS_KEEP_SRC:-}" ] || { rm -rf "$COSMOS_SRC" "$CV"; note "removed the original download and the converter's environment"; }
fi
"$HF" download hexgrad/Kokoro-82M >/dev/null 2>&1 && note "Kokoro-82M: in the Hugging Face cache (British voice bf_emma is the default)"
note "Parakeet v3 downloads itself the first time the Studio starts (about 2.5 GB)."

# ------------------------------------------------------------------------------------------------ Imagination Studio
STUDIO_SRC=$GMS_HOME/studio-src
if [ "$AGENT_ONLY" != 1 ]; then
  say "Imagination Studio (keyframes and a film's sound; it runs headless beside the app)"
  if [ -d "$STUDIO_SRC/.git" ]; then git -C "$STUDIO_SRC" pull -q --ff-only || note "warning: could not update $STUDIO_SRC; using it as it is"
  elif [ ! -f "$STUDIO_SRC/app/server.py" ]; then git clone -q "https://github.com/$STUDIO_REPO" "$STUDIO_SRC"; fi
  note "its source: $STUDIO_SRC ($(git -C "$STUDIO_SRC" log -1 --format='%h' 2>/dev/null || echo 'a copy'))"
  if [ -x "$STUDIO_PREFIX/venv/bin/python" ]; then
    note "its install is already here: $STUDIO_PREFIX"
  else
    note "installing it with its own installer (oneshot-setup-spark.sh $STUDIO_FLAGS): this is the long step, an hour or more"
    PREFIX="$STUDIO_PREFIX" bash "$STUDIO_SRC/oneshot-setup-spark.sh" $STUDIO_FLAGS
    [ -x "$STUDIO_PREFIX/venv/bin/python" ] || die "Imagination Studio's installer finished without $STUDIO_PREFIX/venv: see its output above."
  fi
fi

# --------------------------------------------------------------------------------------------------- the clip worker
FV=$GMS_HOME/fv-venv
if [ "$AGENT_ONLY" != 1 ]; then
  say "The clip worker (FastH3 V2 on FastVideo's FP4 recipe)"
  [ -x "$FV/bin/python" ] || uv venv -q --python 3.12 "$FV"
  if ! "$FV/bin/python" -c "import torch, flashinfer, diffusers" 2>/dev/null; then
    note "its environment (PyTorch 2.12 for CUDA 13, FlashInfer and the rest)…"
    UV_TORCH_BACKEND=cu130 uv pip install -q --python "$FV/bin/python" torch==2.12.0+cu130 torchvision==0.27.0+cu130 torchaudio==2.11.0+cu130 \
      triton flashinfer-python transformers tokenizers safetensors diffusers accelerate peft einops numpy pillow imageio imageio-ffmpeg av \
      pyyaml huggingface-hub loguru tqdm requests psutil regex cloudpickle remote-pdb ftfy scipy ninja
  fi
  FVS=$GMS_HOME/fastvideo-src
  if [ ! -d "$FVS/fastvideo" ]; then
    note "FastVideo at ${FASTVIDEO_COMMIT:0:8} (hao-ai-lab/FastVideo, Apache-2.0): its package and its kernels' source"
    rm -rf "$FVS.part" && mkdir -p "$FVS.part" && git -C "$FVS.part" init -q && git -C "$FVS.part" remote add origin https://github.com/hao-ai-lab/FastVideo
    git -C "$FVS.part" fetch -q --depth 1 origin "$FASTVIDEO_COMMIT" && git -C "$FVS.part" checkout -q FETCH_HEAD
    # the kernels' build reads CUTLASS and ThunderKittens from two submodules
    git -C "$FVS.part" submodule update -q --init --depth 1 fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk
    rm -rf "$FVS" && mkdir -p "$FVS" && mv "$FVS.part/fastvideo" "$FVS.part/fastvideo-kernel" "$FVS.part/LICENSE" "$FVS/" && rm -rf "$FVS.part"
    echo "FastVideo (hao-ai-lab/FastVideo, Apache-2.0) at commit $FASTVIDEO_COMMIT: the package and fastvideo-kernel only" > "$FVS/SOURCE.txt"
  fi
  SITE=$("$FV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  echo "$FVS" > "$SITE/ghost_fastvideo.pth"
  # the kernels are built from the same commit as the package: the published fastvideo-kernel 0.3.5 is older than it
  # (it has no vsa_tile_mask_to_fp4_blocks, which every clip calls)
  kernels_ok() { "$FV/bin/python" -c "import fastvideo_kernel, fp4attn_cuda, fp4quant_cuda, attn_qat_infer.api as a; a.vsa_tile_mask_to_fp4_blocks" 2>/dev/null; }
  if ! kernels_ok; then
    note "building FastVideo's CUDA kernels for this GPU (needs the CUDA toolkit; a few minutes)…"
    [ -x "$CUDA_HOME/bin/nvcc" ] || die "no nvcc in $CUDA_HOME/bin: the kernels cannot be built (DGX OS has the CUDA toolkit there; set CUDA_HOME)."
    uv pip install -q --python "$FV/bin/python" "scikit-build-core>=0.10" cmake ninja setuptools wheel
    uv pip uninstall -q --python "$FV/bin/python" fastvideo-kernel 2>/dev/null || true
    PATH="$FV/bin:$CUDA_HOME/bin:$PATH" CUDA_HOME="$CUDA_HOME" CUDACXX="$CUDA_HOME/bin/nvcc" uv pip install --python "$FV/bin/python" --no-deps \
      --no-build-isolation "$FVS/fastvideo-kernel" > "$GMS_HOME/logs/kernel-build.log" 2>&1 \
      || { grep -i -m 5 "error" "$GMS_HOME/logs/kernel-build.log" || true; die "FastVideo's kernels did not build: see $GMS_HOME/logs/kernel-build.log"; }
    kernels_ok || die "FastVideo's kernels were built but do not load: see $GMS_HOME/logs/kernel-build.log"
    note "FastVideo's CUDA kernels: built here"
  fi
  if ready "$FASTH3_DIR" model_index.json; then note "the FastH3 V2 weights: $FASTH3_DIR"; else
    note "FastH3 V2 NVFP4 (about 57 GB; MiniMax H3 Community License)…"
    download "$FASTH3_REPO" "$FASTH3_DIR" --revision "$FASTH3_REV" --exclude "vae/minimax_h3_video_vae_int8_convrot.safetensors"
  fi
  mkdir -p "$GMS_HOME/cache"
  if [ ! -d "$GMS_HOME/cache/flashinfer" ] && [ -z "$(ls -A "$GMS_HOME/cache" 2>/dev/null)" ]; then
    note "building FlashInfer's NVFP4 kernels once (about a minute)…"
    HF_HUB_OFFLINE=1 "$FV/bin/python" -I "$APP/gma/fp4_worker.py" --model "$FASTH3_DIR" --cache "$GMS_HOME/cache" --jit-only > "$GMS_HOME/cache/jit.log" 2>&1 \
      || { tail -20 "$GMS_HOME/cache/jit.log"; die "FlashInfer's kernels did not build: the end of $GMS_HOME/cache/jit.log is above."; }
  fi
  if [ -s "$ADALN_TABLE" ]; then note "the first-frame table: $ADALN_TABLE"; else
    NOW_GB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$NOW_GB" -ge 90 ] || [ -n "${GMS_FORCE:-}" ] || die "Making the first-frame table needs about 90 GB of free memory for a few minutes; $NOW_GB GB is free now.
   Stop what else is using the GPU and run the installer again (it picks up here)."
    note "making the first-frame table from the release's own projections (one short clip is rendered; about ten minutes)…"
    "$FV/bin/python" "$APP/tools/adaln_table.py" --model "$FASTH3_DIR" --out "$ADALN_TABLE" --cache "$GMS_HOME/cache"
  fi
fi

# ----------------------------------------------------------------------------------------------- the blocking preview
MB=$GMS_HOME/motionbricks
if [ "$LIVE" = 1 ]; then
  say "Live Action's blocking preview (NVIDIA MotionBricks, on the CPU, in a container with no network)"
  if [ ! -f "$MB/src/motionbricks/assets/onnx/soma/soma_n1.json" ]; then
    rm -rf "$MB/src" && mkdir -p "$MB/src" && git -C "$MB/src" init -q && git -C "$MB/src" remote add origin https://github.com/NVlabs/motionbricks
    GIT_LFS_SKIP_SMUDGE=1 git -C "$MB/src" fetch -q --depth 1 origin "$MOTIONBRICKS_COMMIT" && GIT_LFS_SKIP_SMUDGE=1 git -C "$MB/src" checkout -q FETCH_HEAD
  fi
  mkdir -p "$MB/lfs" "$MB/deps"
  lfs() {  # path in the repository, name here: the file a Git LFS pointer stands for, checked against the pointer
    local pointer="$MB/src/$1" out="$MB/lfs/$2" want
    [ -s "$out" ] && return 0
    want=$(sed -n 's/^oid sha256://p' "$pointer")
    [ -n "$want" ] || { cp "$pointer" "$out"; return 0; }   # not a pointer: the file itself
    fetch "https://media.githubusercontent.com/media/NVlabs/motionbricks/$MOTIONBRICKS_COMMIT/$1" "$out"
    [ "$(sha256 "$out")" = "$want" ] || { rm -f "$out"; die "checksum mismatch for $2; run the installer again."; }
  }
  lfs motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx soma_fp32.onnx
  M=assets/motions/soma
  for clip in locomotion/idle/idle_loop_001__A268.bvh locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh \
      locomotion/bs_1d_linear/run_loop_180_R_003__A326.bvh variations/combat_quickstep_ff_idle_180_R_002__A365.bvh \
      jump/jump_over_obstacle_0_5m_R_001__A415.bvh locomotion/idle/crouch_idle_001__A529.bvh \
      variations/dance_vouge_vogue_sequence_180_R_002__A316.bvh sitting/sit_on_chair_loop.bvh \
      locomotion/idle/neutral_looking_around_R_001__A542.bvh; do
    lfs "$M/$clip" "$(basename "$clip")"
  done
  note "the motion model and nine sample motions (NVIDIA Sample Data Evaluation License): $MB/lfs"
  for image in $BLOCKING_IMAGES; do   # the first that is here already, or that can be pulled (Docker Hub, then its mirror on ECR)
    if docker image inspect "$image" >/dev/null 2>&1 || timeout 300 docker pull -q "$image" >/dev/null 2>&1; then BLOCKING_IMAGE=$image; break; fi
  done
  [ -n "$BLOCKING_IMAGE" ] || die "no container image with python3 could be pulled (tried:$BLOCKING_IMAGES). Pull one yourself and set GMS_BLOCKING_IMAGE to it."
  if [ ! -d "$MB/deps/onnxruntime" ]; then
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MB/deps:/deps" "$BLOCKING_IMAGE" \
      pip install -q --no-cache-dir --target /deps onnxruntime numpy pillow >/dev/null
  fi
  note "its container image ($BLOCKING_IMAGE) and ONNX Runtime, NumPy and Pillow beside it: $MB/deps"
fi

# ---------------------------------------------------------------------------------------------------------- ports
B=${GMS_PORT_BASE:-8000}
APP_PORT=$((B + 196)); BRAIN_PORT=$((B + 92)); KOKORO_PORT=$((B + 197)); COSMOS_PORT=$((B + 99)); FP4_PORT=$((B + 420))
if [ -n "${GMS_PORT_BASE:-}" ]; then STUDIO_PORT=$((B + 890)); COMFY_PORT=$((B + 290)); else STUDIO_PORT=7890; COMFY_PORT=8290; fi

# --------------------------------------------------------------------------------------------------------- profile
say "Profile"
HERMES=$(command -v hermes || true)
if [ -f "$GMS_HOME/profile.json" ] && [ -z "${GMS_NEW_PROFILE:-}" ]; then
  note "keeping $GMS_HOME/profile.json (GMS_NEW_PROFILE=1 writes it again)"
else
"$VENV/bin/python" - "$APP/profiles/spark.example.json" "$GMS_HOME/profile.json" <<PY
import json, sys
p = json.load(open(sys.argv[1]))
agent_only, live = "$AGENT_ONLY" == "1", "$LIVE" == "1"
G, E = "$GMS_HOME", "$ENG"
at = lambda port: f"http://127.0.0.1:{port}/v1"
p["name"] = "ghost-machine-studio"
p["brain"]["url"] = at($BRAIN_PORT)
p["eyes"]["url"] = at($COSMOS_PORT)
p["voice_engine"] = "lite"; p.pop("voice", None)
p["voice_lite"]["url"] = at($KOKORO_PORT)
p.pop("memory", None)   # Eyes of Kai (the camera's memory) is Boo's own installer's; add it by hand to use it here
if "brain_fallback" in p: p["brain_fallback"]["url"] = at($COSMOS_PORT)
e = p["engines"]
for name in ("bolt", "drama3", "gemma", "ghost", "d1"):
    e.pop(name, None)
tf = f"{E}/$TFR"
e["qwen"].update(port=$BRAIN_PORT, argv=[f"{tf}/bin/tensorfold-native", "serve", "$QWEN_DIR", "--drafter", "$DRAFTER_DIR", "--name", "qwen38-a",
    "--host", "127.0.0.1", "--port", "{port}", "--no-thinking", "--context", "16384", "--parallel", "2", "--prompt-cache-gib", "4", "--no-update-check"],
    env={"TENSORFOLD_MEMORY_RESERVE_GIB": "2", "LD_LIBRARY_PATH": f"{tf}/lib", "HF_HUB_OFFLINE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1",
         "CUDA_CACHE_PATH": f"{G}/cache/nv"})
e["kokoro"].update(port=$KOKORO_PORT, argv=["$KV/bin/python", "-m", "voice_agent.kokoro_server", "--backend", "torch", "--device", "cuda",
    "--voice", "bf_emma", "--port", "{port}"], cwd="{repo}")
e["cosmos"].update(port=$COSMOS_PORT, argv=[f"{E}/$KB/bin/tensorfold-cosmos3", "serve", "$COSMOS_DIR", "--name", "cosmos3", "--host", "127.0.0.1",
    "--port", "{port}", "--no-update-check", "--vision", "--context", "16384", "--no-thinking"],
    env={"TENSORFOLD_NO_UPDATE_CHECK": "1", "HF_HUB_OFFLINE": "1", "LD_LIBRARY_PATH": f"{E}/$KB/lib"})
p["supervisor"] = {"home": f"{G}/home/ghost/app"}
if agent_only:
    for key in ("studio", "video", "autopilot", "multiverse", "watch", "live", "governor"):
        p.pop(key, None)
    e.pop("fp4", None)
else:
    p["studio"].update(src="$STUDIO_SRC", home=f"{G}/home", prefix="$STUDIO_PREFIX", port=$STUDIO_PORT, comfy_port=$COMFY_PORT)
    e["fp4"].update(port=$FP4_PORT, argv=["$FV/bin/python", "-I", "{repo}/gma/fp4_worker.py", "--model", "$FASTH3_DIR", "--port", "{port}",
        "--cache", f"{G}/cache"], env={"HF_HUB_OFFLINE": "1", "FASTVIDEO_H3_ADALN_TABLE": "$ADALN_TABLE"})
    p["governor"]["renderer"]["dir"] = f"{G}/home/ghost/renders"
    p["video"]["dir"] = f"{G}/home/ghost/video"
    a = p["autopilot"]
    a["director"] = "qwen"
    a["director_choices"] = {"qwen": {"engine": "qwen", "url": at($BRAIN_PORT), "model": "qwen38-a"}}
    a["eyes"] = "cosmos"
    a["eyes_choices"] = {"cosmos": {"engine": "cosmos", "url": at($COSMOS_PORT), "model": "cosmos3", "registry": "cosmos"}}
    a["judge"] = "eyes"   # d1-2B judges on the machine this was built on; its TensorFold port has no release to install
    a.pop("judge_choices", None)
    a["ears"]["whisper_python"] = "$STUDIO_PREFIX/venv/bin/python"
    p["multiverse"]["dir"] = f"{G}/home/ghost/multiverse"
    p["watch"]["studios"] = [{"url": "http://127.0.0.1:$STUDIO_PORT", "library": f"{G}/home", "name": "Imagination Studio", "skip_titles": ["Ghost Studio"]}]
    stage = {"render": "h3", "roster": True, "auto_take": True, "run": f"{G}/home/ghost/live"}
    if live:
        stage["blocking"] = {"home": "$MB", "image": "$BLOCKING_IMAGE"}
    p["live"] = stage
example = p.pop("agents", {})
p.pop("agent", None)
if "$HERMES":
    hermes = example.get("hermes", {"kind": "acp", "label": "Hermes", "cwd": "~", "permissions": "ask"})
    hermes["command"] = ["$HERMES", "acp"]
    p["agents"], p["agent"] = {"hermes": hermes}, "hermes"
json.dump(p, open(sys.argv[2], "w"), indent=1)
PY
  note "$GMS_HOME/profile.json"
fi
if [ -n "$HERMES" ]; then note "Hermes found: it is the back agent ($HERMES acp)"
else note "no back agent (Hermes not found): Boo answers alone. Add one under \"agents\" in the profile (see profiles/spark.example.json)."; fi

# ------------------------------------------------------------------------------------------------------------- gms
say "The gms command"
mkdir -p "$GMS_HOME/bin" "$GMS_HOME/logs" "$GMS_HOME/home/ghost/app"
cat > "$GMS_HOME/guard.sh" <<'GUARD'
#!/bin/bash
# The Studio's memory guard, outside the app (every 0.3 s): under FLOOR_MB of MemAvailable it SIGKILLs the clip worker's
# whole session first, then the Studio's own ComfyUI, then, if memory is still short, the app and its engines. A DGX
# Spark that runs out of unified memory hangs; a killed render is told on the page and can be started again.
G=$(cd "$(dirname "$0")" && pwd); FLOOR_MB=${FLOOR_MB:-14336}; COMFY_PORT=${COMFY_PORT:-8290}
echo $$ > "$G/guard.pid"
while true; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  psi=$(awk '/^some/{split($2,x,"=");print int(x[2])}' /proc/pressure/memory 2>/dev/null || echo 0)
  if [ "$a" -lt "$FLOOR_MB" ] || { [ "${psi:-0}" -gt 40 ] && [ "$a" -lt 30720 ]; }; then
    hit=""
    for p in $(pgrep -f '[f]p4_worker'); do
      s=$(ps -o sid= -p "$p" | tr -d ' '); [ -n "$s" ] && pkill -9 -s "$s" && hit="$hit clip-worker"
    done
    [ -z "$hit" ] && pkill -9 -f "[m]ain.py --listen 127.0.0.1 --port $COMFY_PORT" && hit=" comfyui"
    sleep 1
    a2=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ -z "$hit" ] || [ "$a2" -lt "$FLOOR_MB" ]; then
      [ -f "$G/app.pid" ] && kill -9 -- -"$(cat "$G/app.pid")" 2>/dev/null && hit="$hit app"
    fi
    echo "$(date -Is) guard: $a MB free (floor $FLOOR_MB), pressure ${psi:-?}: killed$hit" >> "$G/logs/guard.log"
    sleep 3
  fi
  sleep 0.3
done
GUARD
chmod +x "$GMS_HOME/guard.sh"
cat > "$GMS_HOME/bin/gms" <<GMS
#!/bin/bash
# Ghost Machine Studio: gms start | stop | restart | status | logs | url | update
G="$GMS_HOME"; PORT=$APP_PORT; export COMFY_PORT=$COMFY_PORT
PATH="\$G/tools/bin:\$HOME/.local/bin:\$PATH"; export PATH
up() { [ -f "\$G/app.pid" ] && kill -0 "\$(cat "\$G/app.pid")" 2>/dev/null; }
case "\${1:-status}" in
  start)
    up && { echo "already running (pid \$(cat "\$G/app.pid")): http://localhost:\$PORT/"; exit 0; }
    if ! { [ -f "\$G/guard.pid" ] && kill -0 "\$(cat "\$G/guard.pid")" 2>/dev/null; }; then
      setsid nohup bash "\$G/guard.sh" > "\$G/logs/guard.out" 2>&1 < /dev/null &
    fi
    cd "\$G/repo" || exit 1
    GMA_HOME="\$G/home/ghost/app" setsid nohup "\$G/venv/bin/python" -m voice_agent.server --profile "\$G/profile.json" \\
      --port \$PORT > "\$G/logs/app.log" 2>&1 < /dev/null &
    echo \$! > "\$G/app.pid"   # the app itself, the leader of its own session: stop ends it and what it started
    for i in \$(seq 180); do
      curl -sf -m 2 "127.0.0.1:\$PORT/health" >/dev/null && { echo "Ghost Machine Studio is up: http://localhost:\$PORT/  (the first answer waits for the brain to load, about a minute)"; exit 0; }
      up || { echo "it stopped while starting:"; tail -20 "\$G/logs/app.log"; exit 1; }
      sleep 1
    done
    echo "not up after three minutes:"; tail -20 "\$G/logs/app.log"; exit 1 ;;
  stop)
    if [ -f "\$G/app.pid" ]; then P=\$(cat "\$G/app.pid"); kill -TERM "\$P" 2>/dev/null
      for i in \$(seq 90); do kill -0 "\$P" 2>/dev/null || break; sleep 1; done; kill -9 -- -"\$P" 2>/dev/null; rm -f "\$G/app.pid"; fi
    for p in \$(pgrep -f "[f]p4_worker.py --model"); do kill -9 -- -"\$(ps -o sid= -p "\$p" | tr -d ' ')" 2>/dev/null; done
    docker kill ghost-live-stage >/dev/null 2>&1
    [ -f "\$G/guard.pid" ] && kill "\$(cat "\$G/guard.pid")" 2>/dev/null; rm -f "\$G/guard.pid"
    echo "stopped" ;;
  restart) "\$0" stop; "\$0" start ;;
  status)
    if up; then curl -s -m 3 "127.0.0.1:\$PORT/health"; echo; curl -s -m 3 "127.0.0.1:\$PORT/api/engines" | "\$G/venv/bin/python" -c "
import json,sys
for e in json.load(sys.stdin).get('engines', []): print(f\"  {e['name']:8} {e.get('role') or '':7} {'loaded' if e.get('loaded') else 'not loaded':11} {e.get('label','')}\")" 2>/dev/null
    else echo "not running (gms start)"; fi ;;
  logs) tail -n "\${2:-60}" -f "\$G/logs/app.log" ;;
  url) echo "http://localhost:\$PORT/" ;;
  update) curl -fsSL "https://raw.githubusercontent.com/$REPO/main/install-studio.sh" | GMS_HOME="\$G" GMS_START=0 GMS_ACCEPT_H3=$ACCEPT_H3 GMS_AGENT_ONLY=$AGENT_ONLY GMS_LIVE=$LIVE bash && "\$0" restart ;;
  *) echo "gms start | stop | restart | status | logs [lines] | url | update"; exit 2 ;;
esac
GMS
chmod +x "$GMS_HOME/bin/gms"
mkdir -p "$HOME/.local/bin" && ln -sf "$GMS_HOME/bin/gms" "$HOME/.local/bin/gms"
note "$GMS_HOME/bin/gms (linked into ~/.local/bin)"

say "Installed"
note "folder:   $GMS_HOME"
note "profile:  $GMS_HOME/profile.json"
if [ "$AGENT_ONLY" = 1 ]; then note "modes:    Ghost Agent (run again without --agent-only for the Studio's other three)"
else note "modes:    Ghost Agent, Multiverse Timeline, Multiverse Generator, Live Action$([ "$LIVE" = 1 ] || echo ' (without the blocking preview)')"; fi
if [ "$START" = 1 ]; then "$GMS_HOME/bin/gms" start; else note "start it: gms start"; fi
note "The microphone and the camera need http://localhost or https: from another device, tunnel the port or serve it over https."
