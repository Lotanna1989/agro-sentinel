##!/bin/bash
# ADTC 2026 Submission - Model Download Script
# Downloads the Qwen2.5-3B-Instruct GGUF (Q4_K_M quantization) into model/
# Idempotent: safe to run multiple times, skips download if file already exists.

set -e  # exit immediately if any command fails

MODEL_DIR="model"
MODEL_FILE="$MODEL_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already downloaded at $MODEL_FILE - skipping."
else
    echo "Downloading Qwen2.5-3B-Instruct (Q4_K_M) to $MODEL_FILE ..."
    curl -L --retry 5 --retry-delay 5 -o "$MODEL_FILE" "$MODEL_URL"
    echo "Download complete."
fi

# Basic sanity check: confirm the file is non-trivial in size (not an error page)
FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
if [ "$FILE_SIZE" -lt 1000000 ]; then
    echo "ERROR: Downloaded file is suspiciously small ($FILE_SIZE bytes). Download may have failed."
    exit 1
fi

echo "Model ready at $MODEL_FILE ($FILE_SIZE bytes)."
