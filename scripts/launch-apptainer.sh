#!/bin/bash
set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_OVERLAY="$SCRIPT_DIR/../overlays/overlay-template.ext3"
INSTANCE_OVERLAY="/tmp/overlay-$(uuidgen).ext3"
SIF_PATH="$SCRIPT_DIR/../apptainer/learn-container.sif"

# Copy overlay from template
echo "Copying overlay template to $INSTANCE_OVERLAY..."
cp "$TEMPLATE_OVERLAY" "$INSTANCE_OVERLAY"

# Clean up overlay on exit
cleanup() {
    echo "Cleaning up overlay..."
    rm -f "$INSTANCE_OVERLAY"
}
trap cleanup EXIT

# Launch Apptainer shell
# apptainer shell \
#     --overlay "$INSTANCE_OVERLAY" \
#     --pwd /tmp \
#     --no-home \
#     --cleanenv \
#     --containall \
#     "$SIF_PATH"

# apptainer shell \
#   --overlay "$(pwd)/overlays/overlay-template.ext3:rw" \
#   --pwd /home/moos \
#   --no-home \
#   --cleanenv \
#   apptainer/learn-container.sif

apptainer shell \
  --no-home \
  --cleanenv \
  --pwd /home/moos \
  --overlay "$(pwd)/overlays/overlay-template.ext3:rw" \
  apptainer/learn-container.sif
