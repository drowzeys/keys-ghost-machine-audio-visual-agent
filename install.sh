#!/usr/bin/env bash
# One-shot install of Keys Ghost Machine (Boo) on a DGX Spark or an Apple silicon Mac:
#
#   curl -fsSL https://raw.githubusercontent.com/drowzeys/keys-ghost-machine-audio-visual-agent/main/install.sh | bash
#   … | bash -s -- --voice lite      the lite voice (Kokoro-82M) instead of Drama 3: about 21 GB less disk, less memory
#   … | bash -s -- --voice both      both voices, with a Full/Lite switch in the page
#
# Everything goes under one folder (~/keys-ghost-machine unless KGM_HOME says otherwise): the agent, the TensorFold
# builds, a Python environment, Eyes of Kai, the voices and the models. Running it again picks up where it stopped and
# skips what is already there. No sudo. When it finishes, `kgm start` runs the whole stack (it starts it for you unless
# KGM_START=0). Settings, all optional:
#
#   KGM_VOICE        full (Drama 3), lite (Kokoro-82M) or both  (full; same as --voice; a re-run keeps the last one)
#   KGM_HOME         install folder                          (~/keys-ghost-machine)
#   KGM_MODELS       model folder                            ($KGM_HOME/models)
#   KGM_BOLT_DIR     an existing Bolt folder to use          ($KGM_MODELS/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
#   KGM_DRAMA_DIR    an existing Drama 3 NVFP4 folder        ($KGM_MODELS/drama-3-NVFP4-Pruned)
#   KGM_COSMOS_DIR   an existing converted Cosmos3 folder    ($KGM_MODELS/cosmos3-tf)
#   KGM_PORT_BASE    move every port: base+1 Bolt, +3 or +8 Drama, +9 Cosmos3, +5 agent, +70 Kai, +107 Kokoro (unset: 8091 …)
#   KGM_START=0      install only, don't start
#   KGM_KEEP_SRC=1   keep the original Cosmos3-Edge download after converting it (7.2 GB)
#   KGM_FORCE=1      install even if the memory or disk check fails
#   HF_TOKEN         a Hugging Face token, if you have one (not required)

set -euo pipefail

REPO=drowzeys/keys-ghost-machine-audio-visual-agent
RELEASE=${KGM_RELEASE:-v0.2.0}
TF_VERSION=1.1.4
KAI_COMMIT=7f2f02b
KGM_HOME=${KGM_HOME:-$HOME/keys-ghost-machine}
KGM_MODELS=${KGM_MODELS:-$KGM_HOME/models}
BOLT_DIR=${KGM_BOLT_DIR:-$KGM_MODELS/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB}
DRAMA_DIR=${KGM_DRAMA_DIR:-$KGM_MODELS/drama-3-NVFP4-Pruned}
COSMOS_DIR=${KGM_COSMOS_DIR:-$KGM_MODELS/cosmos3-tf}
COSMOS_SRC=$KGM_MODELS/cosmos3-edge-src

say()  { printf '\n\033[1;35m==>\033[0m \033[1m%s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# the voice: full = Drama 3 (expressive, 14 voices), lite = Kokoro-82M (small and fast, English), both = a page switch
VOICE_MODE=${KGM_VOICE:-}
while [ $# -gt 0 ]; do
  case $1 in
    --voice) VOICE_MODE=${2:-}; shift 2 || shift ;;
    --voice=*) VOICE_MODE=${1#*=}; shift ;;
    *) die "unknown option $1 (the installer takes --voice full|lite|both)" ;;
  esac
done
if [ -z "$VOICE_MODE" ] && [ -f "$KGM_HOME/kgm.env" ]; then  # a re-run keeps the voice it was installed with
  VOICE_MODE=$(sed -n 's/^VOICE_MODE=//p' "$KGM_HOME/kgm.env")
fi
VOICE_MODE=${VOICE_MODE:-full}
case $VOICE_MODE in full|lite|both) ;; *) die "--voice must be full, lite or both (not '$VOICE_MODE')" ;; esac
DRAMA=1; [ "$VOICE_MODE" = lite ] && DRAMA=0
KOKORO=1; [ "$VOICE_MODE" = full ] && KOKORO=0

# Homebrew's tools on a Mac, and anything this installer put in place before
[ -d /opt/homebrew/bin ] && PATH="/opt/homebrew/bin:$PATH"
PATH="$KGM_HOME/tools/bin:$KGM_HOME/tools/node/bin:$HOME/.local/bin:$PATH"
export PATH

# ---------------------------------------------------------------------------------------------------------- platform
say "Checking this machine"
OS=$(uname -s); ARCH=$(uname -m)
PLATFORM=${KGM_PLATFORM:-}
if [ -z "$PLATFORM" ]; then
  if [ "$OS" = Darwin ] && [ "$ARCH" = arm64 ]; then
    PLATFORM=mac
  elif [ "$OS" = Linux ] && [ "$ARCH" = aarch64 ] && have nvidia-smi; then
    PLATFORM=spark
  else
    die "Boo runs on an Apple silicon Mac (macOS 26+) or an NVIDIA DGX Spark (Linux aarch64, CUDA 13). This is $OS $ARCH."
  fi
fi
case $PLATFORM in
  mac)
    MACOS=$(sw_vers -productVersion 2>/dev/null || echo 0)
    [ "${MACOS%%.*}" -ge 26 ] 2>/dev/null || [ -n "${KGM_FORCE:-}" ] || die "macOS 26 or later is needed for Metal 4 (this is $MACOS)."
    MEM_GB=${KGM_MEM_GB:-$(( $(sysctl -n hw.memsize) / 1073741824 ))}
    MEM_NEED=96; MEM_WHAT="unified memory"
    [ "$VOICE_MODE" = lite ] && MEM_NEED=64   # Bolt + Cosmos3 + Parakeet + Kokoro ≈ 55 GB in use: tight on 64 GB
    note "Apple silicon Mac, macOS $MACOS, $MEM_GB GB unified memory"
    if [ "$VOICE_MODE" = lite ] && [ "$MEM_GB" -lt 96 ]; then
      note "warning: on under 96 GB, raise the GPU memory limit before starting (≈ 55 GB needed; macOS allows about 3/4):"
      note "         sudo sysctl iogpu.wired_limit_mb=$(( MEM_GB * 1024 - 8192 ))   and close other apps"
    fi
    ;;
  spark)
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    case $GPU in *GB10*|"") ;; *) note "GPU is '$GPU'; Boo is built and tested for the GB10 (DGX Spark).";; esac
    MEM_GB=${KGM_MEM_GB:-$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)}
    MEM_NEED=64; MEM_WHAT="free memory (MemAvailable)"
    [ "$VOICE_MODE" = lite ] && MEM_NEED=48   # Kokoro ≈ 3.5 GB instead of Drama 3's ≈ 21 GB and more
    [ "$VOICE_MODE" = both ] && MEM_NEED=68
    CUDA_LIBS=${CUDA_LIBS:-/usr/local/cuda/lib64}
    note "DGX Spark (${GPU:-NVIDIA GPU}), $MEM_GB GB free memory"
    [ "$VOICE_MODE" = full ] || note "voice: $VOICE_MODE"
    ls "$CUDA_LIBS"/libcublasLt.so.13* >/dev/null 2>&1 || \
      note "warning: no CUDA 13 cuBLAS in $CUDA_LIBS (DGX OS has it there; set CUDA_LIBS to its folder)"
    ;;
  *) die "KGM_PLATFORM must be mac or spark" ;;
esac
RUNNING=0   # a re-run while Boo is up: its own models hold the memory the check would look for
for f in "$KGM_HOME"/run/*.pid; do [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && RUNNING=1; done
if [ "$RUNNING" = 1 ]; then
  note "Boo is running from this folder; skipping the memory check (kgm stop first for a clean check)"
elif [ "$MEM_GB" -lt "$MEM_NEED" ]; then
  [ -n "${KGM_FORCE:-}" ] || die "Boo needs at least $MEM_NEED GB of $MEM_WHAT; this machine has $MEM_GB GB. (KGM_FORCE=1 installs anyway.)"
  note "warning: $MEM_GB GB of $MEM_WHAT is under the $MEM_NEED GB Boo needs (continuing: KGM_FORCE)"
fi

# disk: only what is still to download
ready() { [ -f "$1/.kgm-ready" ] || { [ -n "${2:-}" ] && [ -f "$1/$2" ]; }; }
NEED_GB=4
ready "$BOLT_DIR" config.json || NEED_GB=$((NEED_GB + 24))
[ "$DRAMA" = 1 ] && { ready "$DRAMA_DIR" config.json || NEED_GB=$((NEED_GB + 22)); }
if [ "$KOKORO" = 1 ] && [ ! -d "$KGM_HOME/kokoro-venv/lib" ]; then   # Kokoro, spaCy and its environment (CUDA: + PyTorch)
  if [ "$PLATFORM" = spark ]; then NEED_GB=$((NEED_GB + 7)); else NEED_GB=$((NEED_GB + 2)); fi
fi
ready "$COSMOS_DIR" model.safetensors || NEED_GB=$((NEED_GB + 10))
[ "$PLATFORM" = spark ] && [ ! -d "$KGM_HOME/venv/lib" ] && NEED_GB=$((NEED_GB + 8))   # PyTorch (CUDA) and NeMo
mkdir -p "$KGM_HOME" "$KGM_MODELS"
FREE_GB=${KGM_FREE_GB:-$(df -Pk "$KGM_MODELS" | awk 'NR==2{print int($4/1048576)}')}
note "$FREE_GB GB free on disk; this install still needs about $NEED_GB GB"
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  [ -n "${KGM_FORCE:-}" ] || die "Not enough disk: $NEED_GB GB needed, $FREE_GB GB free in $KGM_MODELS. Free some space or point KGM_MODELS elsewhere."
  note "warning: not enough disk (continuing: KGM_FORCE)"
fi
[ -n "${KGM_CHECK_ONLY:-}" ] && { say "Checks passed ($PLATFORM)"; exit 0; }

# ------------------------------------------------------------------------------------------------------------ tools
say "Tools"
have git || die "git is needed (Mac: xcode-select --install; Linux: your package manager)."
have curl || die "curl is needed."
sha256() { if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
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

# uv for Python (installed into $KGM_HOME/tools if missing; it also brings Python 3.12 when the system has none)
if ! have uv; then
  note "installing uv (Python manager) into $KGM_HOME/tools"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$KGM_HOME/tools/bin" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
fi
note "uv $(uv --version | cut -d' ' -f2)"

# Node 22+ for Eyes of Kai
node_ok() { have node && [ "$(node -p 'process.versions.node.split(".")[0]')" -ge 22 ]; }
if ! node_ok; then
  if [ "$PLATFORM" = mac ]; then
    have brew && { note "installing Node with Homebrew"; brew install node >/dev/null; } || \
      die "Node 22 or later is needed for Eyes of Kai: install Homebrew (https://brew.sh) then run: brew install node"
  else
    NODE_V=v22.14.0
    note "downloading Node $NODE_V into $KGM_HOME/tools/node"
    fetch "https://nodejs.org/dist/$NODE_V/node-$NODE_V-linux-arm64.tar.xz" "$KGM_HOME/tools/node.tar.xz"
    mkdir -p "$KGM_HOME/tools/node" && LC_ALL=C tar xJf "$KGM_HOME/tools/node.tar.xz" -C "$KGM_HOME/tools/node" --strip-components 1
  fi
fi
note "node $(node --version)"

# -------------------------------------------------------------------------------------------------------- the agent
say "Boo's agent and page"
APP=$KGM_HOME/app
if [ -d "$APP/.git" ]; then
  git -C "$APP" pull -q --ff-only || note "warning: could not update $APP (local changes?); using it as it is"
else
  git clone -q "https://github.com/$REPO" "$APP"
fi
note "$(git -C "$APP" log -1 --format='%h %s' | cut -c1-80)"

say "Python environment"
VENV=$KGM_HOME/venv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
if [ "$PLATFORM" = spark ] && ! "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  note "PyTorch for CUDA 13 (aarch64), for NeMo's speech recognition"
  uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cu130
fi
uv pip install -q --python "$VENV/bin/python" -r "$APP/requirements-$PLATFORM.txt" "huggingface_hub>=0.34"
note "installed $(basename "$APP")/requirements-$PLATFORM.txt"

# ---------------------------------------------------------------------------------------------- TensorFold builds
say "TensorFold $TF_VERSION builds"
ENG=$KGM_HOME/engines; mkdir -p "$ENG/dl"
if [ "$PLATFORM" = mac ]; then KB=keys-boo-tensorfold-$TF_VERSION-macos-arm64; else KB=keys-boo-tensorfold-$TF_VERSION-linux-aarch64; fi
if [ -d "$ENG/$KB" ] && [ "$(cat "$ENG/$KB/.kgm-release" 2>/dev/null || echo v0.1.0)" != "$RELEASE" ]; then
  note "replacing the keys build with the one from $RELEASE"   # an update: a release can ship a newer build
  rm -rf "$ENG/$KB" "$ENG/dl/$KB.tar.gz" "$ENG/dl/$KB.tar.gz.sha256"
fi
if [ ! -d "$ENG/$KB" ]; then
  base="https://github.com/$REPO/releases/download/$RELEASE"
  fetch "$base/$KB.tar.gz.sha256" "$ENG/dl/$KB.tar.gz.sha256"
  fetch "$base/$KB.tar.gz" "$ENG/dl/$KB.tar.gz"
  verify "$ENG/dl/$KB.tar.gz" "$ENG/dl/$KB.tar.gz.sha256"
  LC_ALL=C tar xzf "$ENG/dl/$KB.tar.gz" -C "$ENG"
  echo "$RELEASE" > "$ENG/$KB/.kgm-release"
  [ "$PLATFORM" = mac ] && xattr -dr com.apple.quarantine "$ENG/$KB" 2>/dev/null || true
fi
note "keys build: $ENG/$KB (unofficial community build of TensorFold, Apache-2.0)"
if [ "$PLATFORM" = spark ]; then   # Bolt runs on the stock CUDA release
  TFR=tensorfold-$TF_VERSION-linux-aarch64
  if [ ! -x "$ENG/$TFR/bin/tensorfold-native" ]; then
    base="https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION"
    fetch "$base/$TFR.tar.gz.sha256" "$ENG/dl/$TFR.tar.gz.sha256"
    fetch "$base/$TFR.tar.gz" "$ENG/dl/$TFR.tar.gz"
    verify "$ENG/dl/$TFR.tar.gz" "$ENG/dl/$TFR.tar.gz.sha256"
    LC_ALL=C tar xzf "$ENG/dl/$TFR.tar.gz" -C "$ENG"
  fi
  note "stock TensorFold $TF_VERSION for Bolt: $ENG/$TFR"
fi

# ---------------------------------------------------------------------------------------------------------- models
HF=$VENV/bin/hf
download() {  # repo dir [include patterns...]
  local repo=$1 dir=$2; shift 2
  local args=(); for p in "$@"; do args+=(--include "$p"); done
  "$HF" download "$repo" --local-dir "$dir" "${args[@]}" >/dev/null && touch "$dir/.kgm-ready"
}
say "Models (skipped when already there)"
if ready "$BOLT_DIR" config.json; then note "Bolt: $BOLT_DIR"; else
  note "Nemotron 3.5 Bolt 74B-A9B (about 24 GB)…"; download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB "$BOLT_DIR"; fi
if [ "$DRAMA" = 0 ]; then note "Drama 3: not needed (lite voice)"
elif ready "$DRAMA_DIR" config.json; then note "Drama 3: $DRAMA_DIR"; else
  note "Drama 3 NVFP4 pruned (about 22 GB; Fish Audio's Drama 3 licence applies)…"
  download plotarmordev/drama-3-NVFP4-Pruned "$DRAMA_DIR"; fi
if ready "$COSMOS_DIR" model.safetensors; then note "Cosmos3-Edge: $COSMOS_DIR"; else
  if ! ready "$COSMOS_SRC"; then
    note "Cosmos3-Edge, the reasoner parts only (about 7.2 GB)…"
    download nvidia/Cosmos3-Edge "$COSMOS_SRC" "config.json" "model.safetensors.index.json" "*.jinja" "*_config.json" \
      "tokenizer.json" "special_tokens_map.json" "transformer/*" "vision_encoder/*"
  fi
  note "converting Cosmos3-Edge to TensorFold's 8-bit format (a few minutes)…"
  CV=$KGM_HOME/convert-venv
  [ -x "$CV/bin/python" ] || uv venv -q --python 3.12 "$CV"
  if [ "$PLATFORM" = mac ]; then
    uv pip install -q --python "$CV/bin/python" "mlx==0.32.4" "torch==2.14.1"
  else  # the conversion runs on the CPU: CPU builds of PyTorch and MLX
    uv pip install -q --python "$CV/bin/python" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
    uv pip install -q --python "$CV/bin/python" "mlx[cpu]==0.32.4"
  fi
  uv pip install -q --python "$CV/bin/python" "transformers==5.20.2" accelerate safetensors numpy
  "$CV/bin/python" "$ENG/$KB/tools/cosmos3_edge_to_mlx.py" "$COSMOS_SRC" "$COSMOS_DIR.part" 8
  rm -rf "$COSMOS_DIR" && mv "$COSMOS_DIR.part" "$COSMOS_DIR" && touch "$COSMOS_DIR/.kgm-ready"
  [ -n "${KGM_KEEP_SRC:-}" ] || { rm -rf "$COSMOS_SRC" "$CV"; note "removed the original download and the converter's environment"; }
fi
note "Parakeet v3 downloads itself the first time the agent starts (about 2.5 GB)."

VOICES=$KGM_HOME/voices
if [ "$DRAMA" = 1 ]; then
say "Voices"
if [ ! -f "$VOICES/.kgm-ready" ]; then
  "$HF" download "$REPO" --local-dir "$KGM_HOME/hf-voices" --include "voices/$PLATFORM/*" --include "previews/*" >/dev/null
  mkdir -p "$VOICES" && cp "$KGM_HOME/hf-voices/voices/$PLATFORM/"*.json "$VOICES/" && cp "$KGM_HOME/hf-voices/previews/"*.wav "$VOICES/"
  rm -rf "$KGM_HOME/hf-voices" && touch "$VOICES/.kgm-ready"
fi
note "$(ls "$VOICES"/*.json | wc -l | tr -d ' ') voices in $VOICES (Vesper is the default)"
fi

if [ "$KOKORO" = 1 ]; then
  say "Lite voice: Kokoro-82M"
  KV=$KGM_HOME/kokoro-venv   # its own environment: misaki pins spaCy's tokenizer libraries
  [ -x "$KV/bin/python" ] || uv venv -q --python 3.12 "$KV"
  if [ "$PLATFORM" = spark ] && ! "$KV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    note "PyTorch for CUDA 13 (aarch64), for Kokoro"
    uv pip install -q --python "$KV/bin/python" torch --index-url https://download.pytorch.org/whl/cu130
  fi
  uv pip install -q --python "$KV/bin/python" -r "$APP/requirements-kokoro-$PLATFORM.txt"
  if [ "$PLATFORM" = mac ]; then KOKORO_REPO=mlx-community/Kokoro-82M-bf16; else KOKORO_REPO=hexgrad/Kokoro-82M; fi
  "$HF" download "$KOKORO_REPO" >/dev/null   # about 0.4 GB into the Hugging Face cache, with its voices
  note "Kokoro-82M ($KOKORO_REPO) in $KV; British voice bf_emma is the default"
fi

# ---------------------------------------------------------------------------------------------------------- ports
if [ -n "${KGM_PORT_BASE:-}" ]; then
  B=$KGM_PORT_BASE
  BOLT_PORT=$((B + 1)); COSMOS_PORT=$((B + 9)); AGENT_PORT=$((B + 5)); KAI_PORT=$((B + 70)); KOKORO_PORT=$((B + 107))
  if [ "$PLATFORM" = mac ]; then DRAMA_PORT=$((B + 8)); else DRAMA_PORT=$((B + 3)); fi
else
  BOLT_PORT=8091; COSMOS_PORT=8099; AGENT_PORT=8095; KAI_PORT=3870; KOKORO_PORT=8197
  if [ "$PLATFORM" = mac ]; then DRAMA_PORT=8098; else DRAMA_PORT=8093; fi
fi

# ---------------------------------------------------------------------------------------------------- Eyes of Kai
say "Eyes of Kai (camera memory)"
KAI=$KGM_HOME/eyes-of-kai
[ -d "$KAI/.git" ] || git clone -q https://github.com/tonyd2wild/eyes-of-kai "$KAI"
git -C "$KAI" checkout -q "$KAI_COMMIT" 2>/dev/null || { git -C "$KAI" fetch -q origin && git -C "$KAI" checkout -q "$KAI_COMMIT"; }
if git -C "$KAI" apply --check "$APP/patches/eyes-of-kai-retention.patch" 2>/dev/null; then
  git -C "$KAI" apply "$APP/patches/eyes-of-kai-retention.patch"; note "applied the memory-length patch"
fi
(cd "$KAI" && npm ci --omit=dev --no-audit --no-fund --loglevel=error >/dev/null)
cat > "$KAI/.env" <<EOF
PORT=$KAI_PORT
BIND_HOST=127.0.0.1
TLS_ENABLED=false
VISION_BASE_URL=http://127.0.0.1:$COSMOS_PORT
VISION_MODEL=cosmos3
VISION_INTERVAL_MS=1000
VISION_RETENTION_DAYS=2
STT_PROVIDER=local
TTS_PROVIDER=none
AGENT_PROVIDER=direct
AGENT_BASE_URL=http://127.0.0.1:$BOLT_PORT/v1
AGENT_MODEL=nvidia-nemotron-3.5-bolt
ASSISTANT_NAME=Boo
USER_NAME=friend
EOF
note "Eyes of Kai $KAI_COMMIT with the patch; memory kept 2 days (the page can change it)"

# --------------------------------------------------------------------------------------------------------- profile
say "Profile"
HERMES=$(command -v hermes || true)
"$VENV/bin/python" - "$APP/profiles/$PLATFORM.example.json" "$KGM_HOME/profile.json" <<PY
import json, sys
p = json.load(open(sys.argv[1]))
p["brain"]["url"] = "http://127.0.0.1:$BOLT_PORT/v1"
p["voice"]["url"] = "http://127.0.0.1:$DRAMA_PORT/v1"
p["voice"]["samples"] = "$VOICES"
p["voice"]["voice"] = "vesper"
p["voice_engine"] = "$VOICE_MODE"   # full, lite or both; kgm voice changes it
p["voice_lite"] = {"url": "http://127.0.0.1:$KOKORO_PORT/v1", "voice": "bf_emma"}   # Kokoro-82M (see voice_agent/server.py)
p["eyes"] = {"url": "http://127.0.0.1:$COSMOS_PORT/v1", "model": "cosmos3"}
p["memory"]["url"] = "http://127.0.0.1:$KAI_PORT"
p["memory"]["captioner"] = {"url": "http://127.0.0.1:$COSMOS_PORT/v1", "model": "cosmos3"}
example = p.pop("agents", {})
if "$HERMES":
    hermes = example.get("hermes", {"kind": "acp", "label": "Hermes", "cwd": "~", "permissions": "ask"})
    hermes["command"] = ["$HERMES", "acp"]
    p["agents"], p["agent"] = {"hermes": hermes}, "hermes"
else:
    p.pop("agent", None)
    json.dump({"agents": example, "agent": "hermes"}, open("$KGM_HOME/agents.example.json", "w"), indent=2)
json.dump(p, open(sys.argv[2], "w"), indent=2)
PY
if [ -n "$HERMES" ]; then note "Hermes found: wired as the back agent ($HERMES acp)"
else note "no back agent (Hermes not found). To add one, copy the \"agents\" and \"agent\" keys from"
     note "$KGM_HOME/agents.example.json into $KGM_HOME/profile.json and run: kgm restart agent"; fi
note "$KGM_HOME/profile.json"

# ------------------------------------------------------------------------------------------------------------- kgm
say "The kgm command"
mkdir -p "$KGM_HOME/bin" "$KGM_HOME/run" "$KGM_HOME/logs"
cat > "$KGM_HOME/kgm.env" <<EOF
KGM_HOME="$KGM_HOME"
PLATFORM=$PLATFORM
BOLT_DIR="$BOLT_DIR"
DRAMA_DIR="$DRAMA_DIR"
COSMOS_DIR="$COSMOS_DIR"
KB="$ENG/$KB"
TF_STOCK="$ENG/tensorfold-$TF_VERSION-linux-aarch64"
CUDA_LIBS="${CUDA_LIBS:-}"
BOLT_PORT=$BOLT_PORT
DRAMA_PORT=$DRAMA_PORT
COSMOS_PORT=$COSMOS_PORT
KAI_PORT=$KAI_PORT
AGENT_PORT=$AGENT_PORT
KOKORO_PORT=$KOKORO_PORT
VOICE_MODE=$VOICE_MODE
EOF
cat > "$KGM_HOME/bin/kgm" <<'KGM'
#!/usr/bin/env bash
# kgm start|stop|restart [service] · status · logs [service] · url · voice full|lite|both — Boo's stack:
# bolt, drama (full voice) and/or kokoro (lite voice), cosmos, kai, agent
set -uo pipefail
H="$(cd "$(dirname "$0")/.." && pwd)"
. "$H/kgm.env"
[ -d /opt/homebrew/bin ] && PATH="/opt/homebrew/bin:$PATH"
PATH="$KGM_HOME/tools/bin:$KGM_HOME/tools/node/bin:$PATH"
[ -n "${CUDA_LIBS:-}" ] && export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
VOICE_MODE=${VOICE_MODE:-full}; KOKORO_PORT=${KOKORO_PORT:-8197}
services() { case $1 in lite) echo "bolt kokoro cosmos kai agent";; both) echo "bolt drama kokoro cosmos kai agent";;
                        *) echo "bolt drama cosmos kai agent";; esac; }
ALL=$(services "$VOICE_MODE")
port() { case $1 in bolt) echo "$BOLT_PORT";; drama) echo "$DRAMA_PORT";; cosmos) echo "$COSMOS_PORT";; kai) echo "$KAI_PORT";;
                    agent) echo "$AGENT_PORT";; kokoro) echo "$KOKORO_PORT";; esac; }
check() { case $1 in
  kai)   curl -sf -m 2 -o /dev/null "127.0.0.1:$KAI_PORT/" ;;
  agent) curl -sf -m 2 -o /dev/null "127.0.0.1:$AGENT_PORT/health" ;;
  *)     curl -sf -m 2 -o /dev/null "127.0.0.1:$(port "$1")/v1/models" ;;
esac; }
# Drama 3 sometimes goes quiet after a sentence until the frame cap; TensorFold 1.1.4's silence guard redraws such a reply
# and ends such a stream (Boo then asks for the rest). Only builds that list the flags get them (the Mac build, and the
# CUDA NVFP4 engine since v0.2.0); Boo's own check covers any other.
GUARD="--silence-floor-dbfs=-30 --silence-seconds=4 --silence-mid-s=2 --silence-mid-min-voiced-s=1 --silence-redraws=3 --silence-redraw-budget-s=90"
guard() { case "$("$1" serve --help 2>&1)" in *--silence-mid-min-voiced-s*) echo "$GUARD" ;; esac; }
pidof_() { local f="$KGM_HOME/run/$1.pid"; [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && cat "$f"; }
launch() {
  local s=$1 log="$KGM_HOME/logs/$1.log"
  case $s in
    bolt) if [ "$PLATFORM" = mac ]; then
            exec env MODEL="$BOLT_DIR" PORT="$BOLT_PORT" CONTEXT=65536 PARALLEL=2 "$KB/scripts/serve-bolt.sh"
          else
            exec "$TF_STOCK/bin/tensorfold-native" serve "$BOLT_DIR" --name nvidia-nemotron-3.5-bolt --host 127.0.0.1 \
              --port "$BOLT_PORT" --no-thinking --context 32768 --parallel 2 --no-update-check
          fi ;;
    drama)  if [ "$PLATFORM" = mac ]; then d="$KB/bin/tensorfold-native"; else d="$KB/bin/tensorfold-drama3-nvfp4"; fi
            # shellcheck disable=SC2046 # the guard's flags are separate words
            exec env MODEL="$DRAMA_DIR" VOICES="$KGM_HOME/voices" VOICE=vesper PORT="$DRAMA_PORT" "$KB/scripts/serve-drama.sh" $(guard "$d") ;;
    cosmos) exec env MODEL="$COSMOS_DIR" PORT="$COSMOS_PORT" "$KB/scripts/serve-cosmos.sh" ;;
    kai)    cd "$KGM_HOME/eyes-of-kai" && exec node src/server.js ;;
    kokoro) if [ "$PLATFORM" = mac ]; then b=mlx; else b=torch; fi
            cd "$KGM_HOME/app" && exec "$KGM_HOME/kokoro-venv/bin/python" -m voice_agent.kokoro_server --backend "$b" --port "$KOKORO_PORT" ;;
    agent)  cd "$KGM_HOME/app" && exec "$KGM_HOME/venv/bin/python" -m voice_agent.server --profile "$KGM_HOME/profile.json" \
              --port "$AGENT_PORT" --voice "$VOICE_MODE" ;;
  esac > "$log" 2>&1 < /dev/null
}
# On a DGX Spark the GPU allocates from the same memory as the page cache, and CUDA cannot take memory the cache holds:
# right after a 50 GB download a model can fail to load with free memory "available". Dropping the cached pages of
# our own model files (no root needed) gives that memory back.
uncache() {
  [ "$PLATFORM" = spark ] || return 0
  "$KGM_HOME/venv/bin/python" - "$BOLT_DIR" "${DRAMA_DIR:-}" "$COSMOS_DIR" <<'PY' 2>/dev/null || true
import os, sys
for d in sys.argv[1:]:
    for root, _, files in os.walk(d):
        for f in files:
            try:
                fd = os.open(os.path.join(root, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            except OSError:
                pass
PY
}
start1() {
  local s=$1
  if [ -n "$(pidof_ "$s")" ]; then echo "  $s already running (pid $(pidof_ "$s"))"; return 0; fi
  if check "$s"; then echo "  $s: port $(port "$s") is already answering (another server?) - not starting a second one"; return 1; fi
  case $s in bolt|drama|cosmos|kokoro) uncache ;; esac
  ( launch "$s" ) & echo $! > "$KGM_HOME/run/$s.pid"
  printf '  %-7s starting on :%s ' "$s" "$(port "$s")"
  for _ in $(seq 900); do
    check "$s" && { echo "ready"; return 0; }
    kill -0 "$(cat "$KGM_HOME/run/$s.pid")" 2>/dev/null || { echo "EXITED - last lines of $KGM_HOME/logs/$s.log:"; tail -15 "$KGM_HOME/logs/$s.log"; return 1; }
    sleep 1; printf '.'
  done
  echo "timed out (see $KGM_HOME/logs/$s.log)"; return 1
}
stop1() {
  local s=$1 p; p=$(pidof_ "$s") || true
  [ -n "$p" ] || { echo "  $s not running"; return 0; }
  kill -TERM "$p" 2>/dev/null
  for _ in $(seq 25); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  kill -KILL "$p" 2>/dev/null; rm -f "$KGM_HOME/run/$s.pid"; echo "  $s stopped"
}
url() {
  echo "Open:  http://127.0.0.1:$AGENT_PORT/"
  echo "Phone: with Tailscale on this machine and your phone, run once"
  echo "         tailscale serve --bg --https=8443 http://127.0.0.1:$AGENT_PORT"
  echo "       then open https://<this-machine>.<your-tailnet>.ts.net:8443 in Safari or Chrome."
}
# kgm voice full|lite|both: switch the voice engine (the one it needs must be installed: install.sh --voice …)
voice() {
  local m=${1:-} f
  case $m in full|lite|both) ;; *) echo "usage: kgm voice full|lite|both (now: $VOICE_MODE)"; return 2 ;; esac
  if [ "$m" != full ] && [ ! -x "$KGM_HOME/kokoro-venv/bin/python" ]; then
    echo "The lite voice is not installed yet: run the installer again with --voice $m"; return 1; fi
  if [ "$m" != lite ] && [ ! -f "$KGM_HOME/voices/.kgm-ready" ]; then
    echo "Drama 3 is not installed yet: run the installer again with --voice $m"; return 1; fi
  for s in $(services "$VOICE_MODE"); do case " $(services "$m") " in *" $s "*) ;; *) stop1 "$s" ;; esac; done
  f="$KGM_HOME/kgm.env"; grep -v '^VOICE_MODE=' "$f" > "$f.new"; echo "VOICE_MODE=$m" >> "$f.new"; mv "$f.new" "$f"
  "$KGM_HOME/venv/bin/python" - "$KGM_HOME/profile.json" "$m" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); p["voice_engine"] = sys.argv[2]; json.dump(p, open(sys.argv[1], "w"), indent=2)
PY
  VOICE_MODE=$m; ALL=$(services "$m")
  stop1 agent; for s in $ALL; do start1 "$s" || return 1; done
  echo "Voice: $m"
}
cmd=${1:-status}; shift || true
svcs=${*:-$ALL}
case $cmd in
  start)   echo "Starting Boo ($PLATFORM)"; ok=1; for s in $svcs; do start1 "$s" || ok=0; done
           [ $ok = 1 ] && { echo; url; } || { echo "Some parts did not start; see the logs above."; exit 1; } ;;
  stop)    for s in $(echo "$svcs" | awk '{for (i = NF; i > 0; i--) print $i}'); do stop1 "$s"; done ;;
  restart) for s in $svcs; do stop1 "$s"; done; for s in $svcs; do start1 "$s"; done ;;
  status)  for s in $ALL; do p=$(pidof_ "$s") || true
             if check "$s"; then st=up; elif [ -n "$p" ]; then st=starting; else st=down; fi
             printf '  %-7s %-9s :%-5s %s\n' "$s" "$st" "$(port "$s")" "${p:+pid $p}"; done ;;
  logs)    s=${1:-agent}; tail -n 60 -f "$KGM_HOME/logs/$s.log" ;;
  url)     url ;;
  voice)   voice "$@" ;;
  *)       echo "usage: kgm start|stop|restart [$ALL] | status | logs [service] | url | voice full|lite|both"; exit 2 ;;
esac
KGM
chmod +x "$KGM_HOME/bin/kgm"
if [ -d "$HOME/.local/bin" ] && [ "$KGM_HOME" = "$HOME/keys-ghost-machine" ]; then
  ln -sf "$KGM_HOME/bin/kgm" "$HOME/.local/bin/kgm"; note "kgm is on your PATH (~/.local/bin/kgm)"
else
  note "run it as $KGM_HOME/bin/kgm (or add $KGM_HOME/bin to your PATH)"
fi

# ------------------------------------------------------------------------------------------------------------ done
say "Installed"
note "folder:   $KGM_HOME"
note "models:   $BOLT_DIR"
[ "$DRAMA" = 1 ] && note "          $DRAMA_DIR"
note "          $COSMOS_DIR"
case $VOICE_MODE in
  lite) VP="Kokoro $KOKORO_PORT" ;; both) VP="Drama $DRAMA_PORT · Kokoro $KOKORO_PORT" ;; *) VP="Drama $DRAMA_PORT" ;; esac
note "voice:    $VOICE_MODE"
note "ports:    Bolt $BOLT_PORT · $VP · Cosmos3 $COSMOS_PORT · Kai $KAI_PORT · page $AGENT_PORT"
note "commands: kgm start | stop | status | logs [service] | url | voice full|lite|both"
if [ "${KGM_START:-1}" = 1 ]; then
  say "Starting Boo (the first start loads up to 50 GB of models; give it a few minutes)"
  "$KGM_HOME/bin/kgm" start
else
  echo; "$KGM_HOME/bin/kgm" url
fi
