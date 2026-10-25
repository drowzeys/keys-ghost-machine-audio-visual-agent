#!/usr/bin/env bash
# Starts the lite voice, Kokoro-82M, on 127.0.0.1:${KOKORO_PORT:-8197} from its own environment (kokoro-venv; see
# requirements-kokoro-mac.txt / requirements-kokoro-spark.txt). Then start the agent with the lite voice:
#   ./serve-kokoro.sh && ./serve.sh profiles/mac.json lite        (lite: Kokoro only; both: Drama 3 + a Full/Lite switch)
set -euo pipefail
cd "$(dirname "$0")"
PY="${KOKORO_PYTHON:-kokoro-venv/bin/python}"
if [ "$(uname -s)" = Darwin ]; then BACKEND=mlx; else BACKEND=torch; fi
[ -f kokoro.pid ] && kill "$(cat kokoro.pid)" 2>/dev/null && sleep 1 || true
nohup "$PY" -m voice_agent.kokoro_server --backend "${KOKORO_BACKEND:-$BACKEND}" --voice "${KOKORO_VOICE:-bf_emma}" \
  --port "${KOKORO_PORT:-8197}" > kokoro.log 2>&1 < /dev/null &
echo $! > kokoro.pid
for _ in $(seq 120); do curl -sf -m 2 -o /dev/null "127.0.0.1:${KOKORO_PORT:-8197}/v1/models" && break; sleep 1; done
echo "kokoro pid $(cat kokoro.pid); log: kokoro.log; voice: ${KOKORO_VOICE:-bf_emma}"
