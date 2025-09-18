#!/bin/bash

# Usage: ./untar-everything.sh /path/to/directory

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

dir="$1"

for file in "$dir"/*.tgz; do
    [ -e "$file" ] || continue
    tar xzvf "$file" -C "$dir"
done