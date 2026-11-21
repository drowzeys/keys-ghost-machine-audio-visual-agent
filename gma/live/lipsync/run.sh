#!/bin/bash
# The lip-sync service (MuseTalk 1.5) in a container with no network, a read-only root and no capabilities.
#   run.sh start | stop | log        socket: $HOME/lipsync-try/run/lipsync.sock; files it reads and writes: .../run/io
# Its machine keeps FLOOR_GB of available memory: under it the container is stopped (a guard beside it).
set -u
HOME_DIR=${LIPSYNC_HOME:-$HOME/lipsync-try}
NAME=ghost-lipsync
IMAGE=${LIPSYNC_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}
MEMORY=${LIPSYNC_MEMORY:-45g}
FLOOR_GB=${LIPSYNC_FLOOR_GB:-30}
CLAIM=${LIPSYNC_CLAIM:-$HOME/NODE_CLAIM.txt}
case "${1:-start}" in
start)
  mkdir -p "$HOME_DIR/run/io"; rm -f "$HOME_DIR/run/lipsync.sock"
  echo "$(date -Is) ghost-lipsync: Live Action lip-sync (MuseTalk 1.5, container $NAME, cap $MEMORY, ~8 GB) START" >> "$CLAIM"
  docker run -d --rm --name $NAME --network none --read-only --tmpfs /tmp:exec,size=2g --cap-drop ALL \
    --security-opt no-new-privileges --user "$(id -u):$(id -g)" --pids-limit 1024 --memory "$MEMORY" --gpus all --ipc private \
    -v "$HOME_DIR/src/MuseTalk:/work:ro" -v "$HOME_DIR/models:/models:ro" -v "$HOME_DIR/deps:/deps:ro" \
    -v "$HOME_DIR/service:/service:ro" -v "$HOME_DIR/run:/run/lipsync" \
    -e PYTHONPATH=/deps -e HOME=/tmp -e TORCH_HOME=/models/torchhome -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e LIPSYNC_BATCH="${LIPSYNC_BATCH:-16}" -w /work --entrypoint python3 "$IMAGE" \
    /service/server.py /run/lipsync/lipsync.sock >/dev/null || exit 1
  # the guard: the container goes when the machine is short of memory, or when the live stage's engine arrives
  nohup bash -c "while docker inspect $NAME >/dev/null 2>&1; do
      free_gb=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)
      if [ \"\$free_gb\" -lt $FLOOR_GB ] || docker inspect ghost-live-stage >/dev/null 2>&1; then
        echo \"\$(date -Is) ghost-lipsync: stopped by its guard (\${free_gb} GB available)\" >> '$CLAIM'; docker kill $NAME >/dev/null 2>&1; break
      fi; sleep 2; done; echo \"\$(date -Is) ghost-lipsync: Live Action lip-sync DONE\" >> '$CLAIM'" >/dev/null 2>&1 &
  ;;
stop) docker kill $NAME >/dev/null 2>&1; echo stopped ;;
log) docker logs --tail "${2:-40}" $NAME 2>&1 ;;
esac
