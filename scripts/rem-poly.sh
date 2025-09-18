#!/bin/bash

# Usage: ./rem-blue.sh input.alog output.alog

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.alog output.alog"
    exit 1
fi

input_file="$1"
output_file="$2"

grep -Ev '^\s*[0-9]+\.[0-9]+[[:space:]]+VIEW_POLYGON[[:space:]]' "$input_file" > "$output_file"
