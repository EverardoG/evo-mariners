#!/bin/bash

# Usage: ./green-edges.sh input.alog output.alog

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input.alog output.alog"
    exit 1
fi

input_file="$1"
output_file="$2"

awk '
  /^\s*[0-9]+\.[0-9]+[[:space:]]+VIEW_POLYGON[[:space:]]/ {
    gsub(/edge_color=[^,]*/, "edge_color=0:1:0");
    gsub(/vertex_color=[^,]*/, "vertex_color=0:1:0");
    print;
    next;
  }
  { print }
' "$input_file" > "$output_file"
