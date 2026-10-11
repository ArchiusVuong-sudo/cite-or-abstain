#!/usr/bin/env bash
# Fetch the pinned llama.cpp GGUF converter (sparse, shallow; about 3 MB) into var/llama.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${LLAMA_CPP_TAG:-v0.6.0}"
DEST="var/llama.cpp"
if [ -f "$DEST/convert_hf_to_gguf.py" ]; then
  echo "converter already at $DEST"
  exit 0
fi
mkdir -p var
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp "$DEST"
git -C "$DEST" sparse-checkout set --no-cone '/convert_hf_to_gguf.py' '/gguf-py/' '/conversion/' '/LICENSE'
echo "converter $TAG ready at $DEST"
