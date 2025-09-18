#!/bin/bash

# Usage: ./append-log.sh base.alog append.alog output.alog

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 base.alog append.alog output.alog"
    exit 1
fi

base_file="$1"
append_file="$2"
output_file="$3"

cat "$base_file" "$append_file" | sort -n -k1 > "$output_file"
