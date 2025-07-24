#!/bin/bash
set -e  # Exit on error

# Constants
IMAGE="apptainer/learn-container.sif"
OVERLAY_SIZE_MB=1024
WORKDIR="/home/moos"
OVERLAY_PATH="/tmp/overlay-$(uuidgen).ext3"

# Create overlay
echo "Creating overlay: $OVERLAY_PATH"
apptainer overlay create --size $OVERLAY_SIZE_MB "$OVERLAY_PATH"

# Clean up overlay on script exit
trap "echo 'Cleaning up overlay...'; rm -f \"$OVERLAY_PATH\"" EXIT

# Launch Apptainer shell
apptainer shell \
  --overlay "$OVERLAY_PATH" \
  --pwd "$WORKDIR" \
  --no-home \
  --cleanenv \
  --containall \
  "$IMAGE"

