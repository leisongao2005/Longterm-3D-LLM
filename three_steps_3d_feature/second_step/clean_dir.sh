#!/bin/bash

# Directory you want to clean up
TARGET_DIR="/local1/leisongao/data/3dllm/blip_features"

# Size threshold in kilobytes (14 GB = 14 * 1024 * 1024 KB)
THRESHOLD_KB=14418000

# Go through each subdirectory
for dir in "$TARGET_DIR"/*/; do
    # Check if it is a directory
    if [ -d "$dir" ]; then
        # Get size in KB
        size_kb=$(du -s "$dir" | awk '{print $1}')
        
        if (( size_kb < THRESHOLD_KB )); then
            echo "Deleting $dir (size: ${size_kb} KB)"
            rm -rf "$dir"
        else
            echo "Keeping $dir (size: ${size_kb} KB)"
        fi
    fi
done