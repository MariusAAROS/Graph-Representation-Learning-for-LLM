#!/bin/bash
# Pre-download every backbone in configs/model/ into the shared HF cache.
#
# Compute nodes run with HF_HUB_OFFLINE=1, so this MUST be run on a login node
# before submitting any job. Gated repos (Llama, Gemma, Mistral) additionally
# require accepting the licence on huggingface.co and `hf auth login`.
#
# Usage:
#   bash scripts/jeanzay/fetch_models.sh                       # all models
#   bash scripts/jeanzay/fetch_models.sh qwen2_5_3b mistral7b  # a subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/../../configs/model" && pwd)"

export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
echo "HF_HOME=$HF_HOME"

if [[ $# -gt 0 ]]; then
  files=()
  for name in "$@"; do files+=("$MODEL_DIR/$name.yaml"); done
else
  files=("$MODEL_DIR"/*.yaml)
fi

failed=()
for f in "${files[@]}"; do
  repo=$(grep -oP '^\s+name:\s*"\K[^"]+' "$f" | head -1)
  if [[ -z "$repo" ]]; then
    echo "!! no model.name found in $f" >&2
    failed+=("$(basename "$f")")
    continue
  fi
  echo "==> $(basename "$f" .yaml): $repo"
  # Skip the .pth/.msgpack/.h5 mirrors of the same weights.
  if ! hf download "$repo" --exclude "*.pth" "*.msgpack" "*.h5"; then
    echo "!! download failed for $repo (gated repo? run 'hf auth login')" >&2
    failed+=("$repo")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo
echo "All models cached under $HF_HOME"
