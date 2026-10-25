#!/usr/bin/env bash
# Starts the agent (page + API) in the background on 127.0.0.1:${PORT:-8095}.
#   ./serve.sh profiles/mac.json             (a copy of profiles/mac.example.json, edited)
#   ./serve.sh profiles/mac.json lite        the lite voice: Kokoro-82M (./serve-kokoro.sh) instead of Drama 3
#   ./serve.sh profiles/mac.json both        Drama 3 and Kokoro-82M, with a Full/Lite switch in the page
# Bolt, the voice engine, Cosmos3-Edge and Eyes of Kai must already be up at the URLs the profile names.
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:?usage: ./serve.sh PROFILE.json [full|lite|both]}"
VOICE_ENGINE="${2:-${VOICE_ENGINE:-}}"
export VOICE_ENGINE
PY="${PYTHON:-venv/bin/python}"
[ -f agent.pid ] && kill "$(cat agent.pid)" 2>/dev/null && sleep 1 || true
nohup "$PY" -m voice_agent.server --profile "$PROFILE" --port "${PORT:-8095}" > agent.log 2>&1 < /dev/null &
echo $! > agent.pid
echo "agent pid $(cat agent.pid); voice: ${VOICE_ENGINE:-from the profile}; log: agent.log; page: http://127.0.0.1:${PORT:-8095}/"
