#!/bin/bash
# Script format fixing tool
# Features:
# 1. Convert Windows (CRLF) line endings to Unix (LF)
# 2. Remove trailing whitespace at the end of lines (fixes syntax errors caused by spaces after line continuation `\`)
# 3. Grant execute permissions to scripts (+x)

TARGET_DIR="train_eval_scripts"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist. Please run this script from the project root."
    exit 1
fi

echo "Scanning and fixing .sh scripts in $TARGET_DIR..."

# Find and process all .sh files
find "$TARGET_DIR" -name "multi_opti_*.sh" -o -name "opti_*.sh" | while read -r file; do
    if [ -f "$file" ]; then
        echo "Processing: $file"

        # 1. Remove Windows line endings (\r)
        sed -i 's/\r$//' "$file"

        # 2. Remove trailing whitespace (including spaces after line continuation \)
        # Note: [[:space:]] includes space and tab
        sed -i 's/[[:space:]]*$//' "$file"

        # 3. Grant execute permissions
        chmod +x "$file"
    fi
done

echo "========================================"
echo "Fix complete! All scripts have been converted to Unix format and granted execute permissions."
echo "You can now run directly: ./train_eval_scripts/multi_opti_mlp.sh"
