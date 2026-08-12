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
repos=()
for f in "${files[@]}"; do
  repo=$(grep -oP '^\s+name:\s*"\K[^"]+' "$f" | head -1)
  if [[ -z "$repo" ]]; then
    echo "!! no model.name found in $f" >&2
    failed+=("$(basename "$f")")
    continue
  fi
  repos+=("$repo")
done

# The `hf download` CLI silently treats extra --exclude patterns as an explicit
# filename list and fetches nothing, so use the Python API instead.
rc=0
python - "${repos[@]}" <<'PY' || rc=$?
import sys, os, glob
from huggingface_hub import snapshot_download

# Weight mirrors of the same tensors: .pth/.h5/.msgpack, plus Mistral's
# consolidated.safetensors and Llama's original/ folder.
IGNORE = ["*.pth", "*.msgpack", "*.h5", "consolidated*", "original/*"]

failed = []
for repo in sys.argv[1:]:
    print(f"==> {repo}", flush=True)
    try:
        path = snapshot_download(repo, ignore_patterns=IGNORE)
    except Exception as exc:
        print(f"!! {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
        failed.append(repo)
        continue

    missing = []
    if not os.path.exists(os.path.join(path, "config.json")):
        missing.append("config.json")
    # SentencePiece repos may ship tokenizer.model instead of tokenizer.json.
    if not any(os.path.exists(os.path.join(path, f))
               for f in ("tokenizer.json", "tokenizer.model")):
        missing.append("tokenizer.json|tokenizer.model")
    if not glob.glob(os.path.join(path, "*.safetensors")):
        missing.append("*.safetensors")
    if missing:
        print(f"!! {repo}: incomplete snapshot, missing {missing}", file=sys.stderr)
        failed.append(repo)
        continue

    size = sum(os.path.getsize(os.path.realpath(p))
               for p in glob.glob(os.path.join(path, "*.safetensors")))
    print(f"    ok: {size / 2**30:.1f} GiB of weights", flush=True)

if failed:
    print("\nFailed: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
PY

if [[ ${#failed[@]} -gt 0 || $rc -ne 0 ]]; then
  echo
  [[ ${#failed[@]} -gt 0 ]] && echo "Unreadable config files: ${failed[*]}" >&2
  echo "Some models were not cached (gated repo? run 'hf auth login' and accept" >&2
  echo "the licence on the model page)." >&2
  exit 1
fi
echo
echo "All models cached under $HF_HOME"
