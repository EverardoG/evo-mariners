#!/bin/bash

# Usage: ./rem-blue.sh input.alog output.alog

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.alog output.alog"
    exit 1
fi

input_file="$1"
output_file="$2"

sed -e 's/edge_color=0:0:1/edge_color=1:0:0/g' \
    -e 's/vertex_color=0:0:1/vertex_color=1:0:0/g' \
    -e 's/fill_color=0:0:1/fill_color=1:0:0/g' \
    "$input_file" > "$output_file"
