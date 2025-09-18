#!/bin/bash

# Usage: ./green-edges.sh input.alog output.alog

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.alog output.alog"
    exit 1
fi

input_file="$1"
output_file="$2"

sed 's/fill_color=0:0:1/fill_color=1:0:0/g' "$input_file" > "$output_file"
