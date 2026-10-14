#!/bin/bash
DOWNLOAD_URL="https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64"
OUTPUT_FILE="$HOME/Downloads/vscode_install.deb"

if wget -O "$OUTPUT_FILE" "$DOWNLOAD_URL"; then 
 echo "Download Complele ✅" 
else 
 echo "Download Failed ❌"
        exit 1
fi
sudo dpkg -i "$OUTPUT_FILE" 