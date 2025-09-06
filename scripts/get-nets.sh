#!/bin/bash

# Check for required arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <generation_id> <team_id> [subdir]"
    exit 1
fi

GEN_ID="$1"
TEAM_ID="$2"
SUBDIR="${3:-}"

REMOTE_BASE="/nfs/stak/users/gonzaeve/evo-mariners/results/2025-08-13/mike/multiagent/four-agents/trial_0"
REMOTE_SUB="gen_${GEN_ID}/team_${TEAM_ID}_inds*/rollout_0"
REMOTE_PATH="${REMOTE_BASE}/${REMOTE_SUB}/neural_network_abe*.csv"
REMOTE_HOST="submit-a"

DEST_BASE="$HOME/moos-ivp-learn/missions/alpha_learn/net-bv"
DEST_DIR="$DEST_BASE"
if [ -n "$SUBDIR" ]; then
    DEST_DIR="$DEST_BASE/$SUBDIR"
fi

# Ensure destination exists
mkdir -p "$DEST_DIR"

# Rsync files from remote, skip existing
rsync -a --ignore-existing "${REMOTE_HOST}:${REMOTE_PATH}" "$DEST_DIR/"
if [ $? -ne 0 ]; then
    echo "rsync failed. Check connection and paths."
    exit 2
fi

# Append metadata line to each CSV file
for f in "$DEST_DIR"/neural_network_abe*.csv; do
    [ -e "$f" ] || continue
    # Try to get the actual source path for metadata
    SRC_PATH=$(ssh "$REMOTE_HOST" "find ${REMOTE_BASE}/gen_${GEN_ID}/team_${TEAM_ID}_inds*/rollout_0 -name $(basename "$f") 2>/dev/null | head -n 1")
    echo "# Source: ${SRC_PATH:-${REMOTE_BASE}/gen_${GEN_ID}/team_${TEAM_ID}_inds*/rollout_0}" >> "$f"
done
