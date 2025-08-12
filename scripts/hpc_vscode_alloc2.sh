#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
SUBMIT_HOST="submit-a"
ACCOUNT="kt-lab"
PARTITION="preempt"
TIME="01:00:00"
CPUS=12
NODES=1
MEM="16G"
NODELIST="cn-v-[1-9]"      # adjust or leave empty to omit
JOB_NAME="vs-code"
REMOTE_BOOT_DIR=".vsalloc"
REMOTE_NODE_FILE="$REMOTE_BOOT_DIR/node"
REMOTE_LOG="$REMOTE_BOOT_DIR/log"
# ---------------

echo "[1/5] Starting Slurm allocation on $SUBMIT_HOST"

# Kick off salloc remotely in the background and have it write the first hostname to a file
ssh -t "$SUBMIT_HOST" bash -lc "'
  set -e
  mkdir -p ~/$REMOTE_BOOT_DIR
  : > ~/$REMOTE_LOG
  rm -f ~/$REMOTE_NODE_FILE
  nohup salloc -A $ACCOUNT --partition=$PARTITION --time=$TIME -c $CPUS --nodes=$NODES --mem=$MEM ${NODELIST:+--nodelist=$NODELIST} \
    bash -lc \"scontrol show hostnames \\\"\\\$SLURM_NODELIST\\\" | head -n1 > ~/$REMOTE_NODE_FILE; exec sleep infinity\" \
    >> ~/$REMOTE_LOG 2>&1 &
  disown
  echo started
'"

echo "[2/5] Waiting for node assignment…"
NODE=""
for i in {1..120}; do
  NODE=$(ssh "$SUBMIT_HOST" "bash -lc 'test -s ~/$REMOTE_NODE_FILE && cat ~/$REMOTE_NODE_FILE'") || true
  [[ -n "$NODE" ]] && break
  sleep 2
done

if [[ -z "$NODE" ]]; then
  echo "Timed out waiting for allocation. Check remote log: ssh $SUBMIT_HOST 'tail -n 200 ~/$REMOTE_LOG'"
  return 1 2>/dev/null || exit 1
fi

echo "[3/5] Allocated node: $NODE"

# Define local aliases for this shell session
alias hpc-node="ssh $NODE"
alias hpc-code="code --new-window --remote ssh-remote+$NODE"

echo "[4/5] Aliases set:"
echo "  hpc-node → ssh $NODE"
echo "  hpc-code → VS Code Remote-SSH to $NODE"

echo "[5/5] Launching VS Code on $NODE…"
code --new-window --remote "ssh-remote+$NODE"

echo "Tip: to end the allocation later:"
echo "  ssh $SUBMIT_HOST 'scancel -u \$USER -n $JOB_NAME'  # or kill the sleep in the job"
