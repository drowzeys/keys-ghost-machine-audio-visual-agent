#!/usr/bin/env bash
# Starts the mounted models on TensorFold, then Boo's agent on 0.0.0.0:${BOO_PORT}.
#   /models/bolt     Nemotron 3.5 Bolt 74B-A9B (TensorFold 24 GB layout)       -> stock TensorFold 1.1.4
#   /models/drama    Drama 3 NVFP4 pruned (plotarmordev/drama-3-NVFP4-Pruned)  -> keys build (voices, NVFP4)
#   /models/cosmos3  Cosmos3-Edge converted with tools/cosmos3_edge_to_mlx.py  -> keys build (Cosmos3 on CUDA)
# A model whose folder is missing is skipped; its URL can point at another server instead (BOLT_URL, DRAMA_URL,
# COSMOS_URL). KAI=1 also starts Eyes of Kai for camera memory. HERMES_CMD (e.g. "ssh host hermes acp") or
# DEEP_URL + DEEP_MODEL add a back agent. VOICE_ENGINE=lite speaks with Kokoro-82M instead of Drama 3 (no /models/drama
# needed), both runs the two with a Full/Lite switch in the page; full (Drama 3) is the default.
set -euo pipefail
BOLT_PORT=${BOLT_PORT:-8091} DRAMA_PORT=${DRAMA_PORT:-8093} COSMOS_PORT=${COSMOS_PORT:-8099} BOO_PORT=${BOO_PORT:-8095}
KOKORO_PORT=${KOKORO_PORT:-8197} VOICE_ENGINE=${VOICE_ENGINE:-full}
case $VOICE_ENGINE in full|lite|both) ;; *) echo "[boo] VOICE_ENGINE must be full, lite or both"; exit 2 ;; esac
export VOICE_ENGINE
TF=/opt/tensorfold/bin/tensorfold-native
KT=/opt/keys-tensorfold/bin
pids=()
trap 'kill "${pids[@]}" 2>/dev/null; wait' EXIT INT TERM

start() {  # name port command...
  local name=$1 port=$2; shift 2
  echo "[boo] starting $name on :$port"
  "$@" > "/tmp/$name.log" 2>&1 &
  pids+=($!)
  for _ in $(seq 600); do
    curl -sf "127.0.0.1:$port/v1/models" >/dev/null && { echo "[boo] $name ready"; return; }
    kill -0 "${pids[-1]}" 2>/dev/null || { echo "[boo] $name exited:"; tail -20 "/tmp/$name.log"; exit 1; }
    sleep 1
  done
  echo "[boo] $name did not come up"; tail -20 "/tmp/$name.log"; exit 1
}

if [ -d /models/bolt ]; then
  start bolt "$BOLT_PORT" "$TF" serve /models/bolt --name nvidia-nemotron-3.5-bolt --host 127.0.0.1 --port "$BOLT_PORT" \
    --no-thinking --context "${BOLT_CONTEXT:-32768}" --parallel 2 --no-update-check
fi
# Drama 3's silence guard (TensorFold 1.1.4) when the build has it (the CUDA NVFP4 engine since v0.2.0); Boo's own
# check also drops a stalled take's quiet tail and asks for the rest
GUARD="--silence-floor-dbfs=-30 --silence-seconds=4 --silence-mid-s=2 --silence-mid-min-voiced-s=1 --silence-redraws=3 --silence-redraw-budget-s=90"
guard() { case "$("$1" serve --help 2>&1 || true)" in *--silence-mid-min-voiced-s*) echo "$GUARD" ;; esac; }
if [ -d /models/drama ] && [ "$VOICE_ENGINE" != lite ]; then
  # shellcheck disable=SC2046 # the guard's flags are separate words
  start drama "$DRAMA_PORT" "$KT/tensorfold-drama3-nvfp4" serve /models/drama --name drama-3 --host 127.0.0.1 \
    --port "$DRAMA_PORT" --voices /voices --default-voice "${VOICE:-vesper}" --no-update-check $(guard "$KT/tensorfold-drama3-nvfp4")
fi
if [ "$VOICE_ENGINE" != full ]; then
  start kokoro "$KOKORO_PORT" /opt/kokoro-venv/bin/python -m voice_agent.kokoro_server --backend torch \
    --voice "${KOKORO_VOICE:-bf_emma}" --port "$KOKORO_PORT"
fi
if [ -d /models/cosmos3 ]; then
  start cosmos3 "$COSMOS_PORT" "$KT/tensorfold-cosmos3" serve /models/cosmos3 --name cosmos3 --host 127.0.0.1 \
    --port "$COSMOS_PORT" --vision --context 16384 --parallel 2 --keep-warm 0 --no-update-check
fi
if [ "${KAI:-0}" = 1 ]; then
  (cd /opt/eyes-of-kai && PORT=3870 BIND_HOST=127.0.0.1 VISION_RETENTION_DAYS=2 node src/server.js > /tmp/kai.log 2>&1) &
  pids+=($!)
fi

python3 - <<PY
import json, os
p = json.load(open("/opt/boo/profiles/spark.example.json"))
p["brain"]["url"] = os.environ.get("BOLT_URL", "http://127.0.0.1:$BOLT_PORT/v1")
p["voice"]["url"] = os.environ.get("DRAMA_URL", "http://127.0.0.1:$DRAMA_PORT/v1")
p["voice"]["samples"] = "/voices"
p["voice_engine"] = os.environ["VOICE_ENGINE"]
p["voice_lite"] = {"url": os.environ.get("KOKORO_URL", "http://127.0.0.1:$KOKORO_PORT/v1"), "voice": os.environ.get("KOKORO_VOICE", "bf_emma")}
cosmos = os.environ.get("COSMOS_URL", "http://127.0.0.1:$COSMOS_PORT/v1")
if os.path.isdir("/models/cosmos3") or "COSMOS_URL" in os.environ:
    p["eyes"]["url"] = cosmos
    p["memory"]["captioner"]["url"] = cosmos
else:
    p.pop("eyes", None)
if os.environ.get("KAI") != "1":
    p.pop("memory", None)
agents = {}
if os.environ.get("HERMES_CMD"):
    agents["hermes"] = dict(p["agents"]["hermes"], command=os.environ["HERMES_CMD"].split())
if os.environ.get("DEEP_URL"):
    agents["deep"] = dict(p["agents"]["deep"], url=os.environ["DEEP_URL"], model=os.environ.get("DEEP_MODEL", "your-model"))
p["agents"] = agents
p["agent"] = next(iter(agents), None)
json.dump(p, open("/tmp/boo-profile.json", "w"), indent=2)
PY
echo "[boo] agent on 0.0.0.0:$BOO_PORT"
cd /opt/boo && VOICE_HOST=0.0.0.0 python3 -m voice_agent.server --profile /tmp/boo-profile.json --port "$BOO_PORT" &
pids+=($!)
wait -n
