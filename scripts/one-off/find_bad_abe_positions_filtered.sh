#!/bin/bash

search_dir="${1:-.}"

find "$search_dir" -type f -name 'abe_positions_filtered.csv' | while read -r file; do
    # Print filename if any row after header has '-' as the second column
    if awk -F, 'NR>1 && $2=="-" {print FILENAME; exit}' "$file" | grep -q .; then
        echo "$file"
    fi
done
