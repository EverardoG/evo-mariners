#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
SUBMIT_HOST="submit-a"
ACCOUNT="kt-lab"
PARTITION="preempt"
TIME="10:00:00"
CPUS=12
NODES=1
MEM="16G"
NODELIST="cn-v-[1-9]"   # node constraint; adjust as needed
# ---------------

echo "[1/3] Requesting Slurm allocation on $SUBMIT_HOST..."

# Request allocation in background and grab the first node name
NODE=$(ssh -t "$SUBMIT_HOST" \
  "salloc -A $ACCOUNT --partition=$PARTITION --time=$TIME -c $CPUS --nodes=$NODES --mem=$MEM --nodelist=$NODELIST bash -lc 'echo \$SLURM_NODELIST' " \
  | tr -d '\r' | head -n1)

NODE=$(ssh "$SUBMIT_HOST" "scontrol show hostnames $NODE | head -n1")

if [[ -z "$NODE" ]]; then
    echo "Error: Failed to get compute node name."
    exit 1
fi

echo "[2/3] Allocated compute node: $NODE"

echo "[3/3] Launching vscode:"

# Launch VS Code immediately
echo "Launching VS Code on $NODE..."
code --new-window --remote "ssh-remote+$NODE"
