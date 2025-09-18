#!/bin/bash

# Usage: ./remove-cbf-circles.sh input.alog output.alog

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.alog output.alog"
    exit 1
fi

input_file="$1"
output_file="$2"

grep -v 'VIEW_CIRCLE[[:space:]]*pCBF' "$input_file" > "$output_file"
