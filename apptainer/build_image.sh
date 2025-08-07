#!/bin/bash

# Build script for MOOS-IvP learning environment Apptainer image
# This script builds the complete environment in a single .sif file

set -e  # Exit on any error

# Configuration
DEF_FILE="ubuntu_20.04_ivp_2680_learn.def"
SIF_FILE="ubuntu_20.04_ivp_2680_learn.sif"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Building MOOS-IvP Learning Environment"
echo "=================================================="
echo "Definition file: $DEF_FILE"
echo "Output file: $SIF_FILE"
echo "Build directory: $SCRIPT_DIR"
echo ""

# Check if definition file exists
if [ ! -f "$SCRIPT_DIR/$DEF_FILE" ]; then
    echo "Error: Definition file $DEF_FILE not found in $SCRIPT_DIR"
    exit 1
fi

# Change to the script directory
cd "$SCRIPT_DIR"

# Remove existing SIF file if it exists
if [ -f "$SIF_FILE" ]; then
    echo "Removing existing $SIF_FILE..."
    rm -f "$SIF_FILE"
fi

echo "Building Apptainer image..."
echo "This may take 10-20 minutes depending on your system..."
echo ""

# Build the image
if apptainer build --force --fakeroot "$SIF_FILE" "$DEF_FILE"; then
    echo ""
    echo "=================================================="
    echo "Build completed successfully!"
    echo "=================================================="
    echo "Image file: $SCRIPT_DIR/$SIF_FILE"
    echo "Size: $(du -h "$SIF_FILE" | cut -f1)"
    echo ""
    echo "To test the image:"
    echo "  apptainer test $SIF_FILE"
    echo ""
    echo "To run interactively:"
    echo "  apptainer shell --fakeroot $SIF_FILE"
    echo ""
    echo "To execute commands:"
    echo "  apptainer exec $SIF_FILE <command>"
    echo ""
    echo "To bind your workspace:"
    echo "  apptainer shell --fakeroot --bind $(dirname "$SCRIPT_DIR"):/workspace $SIF_FILE"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "Build failed!"
    echo "=================================================="
    echo "Check the output above for errors."
    exit 1
fi
