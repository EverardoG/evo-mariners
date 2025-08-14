#!/bin/bash
# filepath: /nfs/stak/users/gonzaeve/evo-mariners/check_timestamps.sh

# Function to check timestamp in a file
check_timestamp() {
    local file="$1"
    
    # Get the second to last line (skip the last newline)
    local second_last_line=$(tail -n 2 "$file" | head -n 1)
    
    # Extract the timestamp (first field before whitespace)
    local timestamp=$(echo "$second_last_line" | awk '{print $1}')
    
    # Check if timestamp is a valid number and greater than 200
    if [[ "$timestamp" =~ ^[0-9]+\.?[0-9]*$ ]] && (( $(echo "$timestamp > 200" | bc -l) )); then
        echo "$(realpath "$file")"
    fi
}

# Default to current directory if no argument provided
search_dir="${1:-.}"

# Find all XLOG_SHORESIDE.alog files and check their timestamps
find "$search_dir" -name "XLOG_SHORESIDE.alog" -type f | while read -r file; do
    check_timestamp "$file"
done