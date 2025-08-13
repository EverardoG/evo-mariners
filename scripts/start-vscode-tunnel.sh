#!/usr/bin/env bash
set -euo pipefail

SUBMIT="submit-a"
ACCOUNT="kt-lab"
PARTITION="preempt"
TIME="10:00:00"
CPUS=12
MEM="16G"
NODES=1
NODELIST="cn-v-[1-9]"     # leave empty to omit
LOCAL_FOLDER="/nfs/stak/users/gonzaeve/evo-mariners"
LOG_FILE=".vscode_tunnel.log"

# preflight: 'code' CLI on your Mac
command -v code >/dev/null 2>&1 || { echo "Install 'code' CLI on your Mac."; exit 1; }

echo "[1/2] Starting Slurm job and tunnel on $SUBMIT"

ssh "$SUBMIT" ACCOUNT="$ACCOUNT" PARTITION="$PARTITION" TIME="$TIME" CPUS="$CPUS" MEM="$MEM" NODES="$NODES" NODELIST="$NODELIST" LOG_FILE="$LOG_FILE" bash -s -- <<'REMOTE'
set -e
NODEFLAG=""
if [ -n "${NODELIST}" ]; then NODEFLAG="-w ${NODELIST}"; fi
# fire-and-forget Slurm step; write logs on the remote side
nohup srun -A "$ACCOUNT" -p "$PARTITION" -t "$TIME" --cpus-per-task="$CPUS" --mem="$MEM" -N "$NODES" $NODEFLAG \
  --job-name=vscode-tunnel \
  bash -lc '
    set -e
    TUN=hpc
    echo "[INFO] Starting tunnel $TUN on $(hostname) at $(date)" >> "$HOME/'"$LOG_FILE"'"
    code tunnel --accept-server-license-terms --name "$TUN"
  ' >> "$HOME/$LOG_FILE" 2>&1 &
disown
echo started
REMOTE

echo "[2/3] Waiting for tunnel 'hpc' to become available..."

# Wait for tunnel to be available on the remote side
i=0
while [[ $i -lt 60 ]]; do
  if ssh "$SUBMIT" "grep -q 'Open this link in your browser' \$HOME/$LOG_FILE 2>/dev/null || grep -q 'tunnel.*hpc.*ready' \$HOME/$LOG_FILE 2>/dev/null" 2>/dev/null; then
    echo "Tunnel 'hpc' is now available!"
    break
  fi
  sleep 3
  i=$((i+1))
  echo "Waiting... ($((i*3))s)"
done

if [[ $i -ge 60 ]]; then
  echo "Timed out waiting for tunnel. Check logs:"
  echo "  ssh submit-a 'tail -n 200 ~/.vscode_tunnel.log'"
  echo "You can try connecting manually once the tunnel is ready:"
  echo "  code --new-window --remote tunnel+hpc '$LOCAL_FOLDER'"
  exit 1
fi

echo "[3/3] Connecting to tunnel 'hpc'..."
code --new-window --remote "tunnel+hpc" "$LOCAL_FOLDER"
# i=0
# while [[ -z "$JOBID" && $i -lt 60 ]]; do
#   JOBID=$(ssh submit-a bash -lc "squeue -u \$USER -n vscode-tunnel -h -o '%i' | awk 'NF{print \$1; exit}'" 2>/dev/null || true)
#   [[ -n "$JOBID" ]] && break
#   sleep 2
#   i=$((i+1))
# done

# if [[ -z "$JOBID" ]]; then
#   echo "Timed out waiting for job id. Check logs:"
#   echo "  ssh submit-a 'tail -n 200 ~/.vscode_tunnel.log'"
#   exit 1
# fi

# TUN="vs-$JOBID"
# echo "[3/3] Tunnel should be available as: $TUN"
# code --new-window --remote "tunnel+$TUN" "$LOCAL_FOLDER"
