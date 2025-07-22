#!/bin/bash

INSTANCE=ubuntu-x86
PROJECT_PATH="$HOME/evo-mariners"

# echo "Starting Lima instance '$INSTANCE'..."
limactl start "$INSTANCE"

echo "Lima VM started."

echo "Launching VS Code connected to '$INSTANCE'..."
code --remote ssh-remote+lima-$INSTANCE $PROJECT_PATH
