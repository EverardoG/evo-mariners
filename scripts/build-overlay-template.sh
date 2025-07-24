#!/bin/bash
set -e

TEMPLATE_DIR="$(dirname "$0")/../overlays"
TEMPLATE_PATH="$TEMPLATE_DIR/overlay-template.ext3"

mkdir -p "$TEMPLATE_DIR"  # Ensure the overlays directory exists

if [ -f "$TEMPLATE_PATH" ]; then
    echo "Overlay template already exists at $TEMPLATE_PATH"
    exit 0
fi

echo "Creating overlay template at $TEMPLATE_PATH..."
apptainer overlay create --size 1024 "$TEMPLATE_PATH"
echo "Done."
