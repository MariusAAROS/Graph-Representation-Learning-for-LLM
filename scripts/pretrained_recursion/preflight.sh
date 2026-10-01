#!/usr/bin/env bash
# Checks everything the queues need before the window starts. Prints OK / FAIL per item and
# exits 1 if anything failed. Also finds the GraphQA reference checkpoint and records it in
# $ROOT/graphqa_ckpt.txt (override: GRAPHQA_CKPT=<path> bash preflight.sh).
#
#   bash scripts/pretrained_recursion/preflight.sh            # expects 2 GPUs
#   MIN_GPUS=1 bash scripts/pretrained_recursion/preflight.sh
set -uo pipefail
source "$(dirname "$0")/common.sh"
MIN_GPUS=${MIN_GPUS:-2}

failed=0
ok()   { echo "  OK    $*"; }
fail() { echo "  FAIL  $*"; failed=1; }

echo "== machine"
n_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
(( n_gpus >= MIN_GPUS )) && ok "$n_gpus GPU(s)" || fail "$n_gpus GPU(s), need $MIN_GPUS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/        /'
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
(( busy == 0 )) && ok "no process on the GPUs" || echo "  WARN  $busy process(es) already on the GPUs"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
(( free_gb >= 300 )) && ok "${free_gb} GB free under $ROOT" || fail "${free_gb} GB free under $ROOT, need 300"

echo "== code"
[[ $(git -C "$GRL" branch --show-current) == cross-repo ]] && ok "GRL on branch cross-repo ($GRL_COMMIT)" \
  || fail "GRL not on branch cross-repo"
[[ -z $(git -C "$GRL" status --porcelain --untracked-files=no) ]] && ok "GRL tracked files clean" \
  || echo "  WARN  GRL has uncommitted changes to tracked files"

echo "== python ($PYTHON)"
if [[ -x $PYTHON ]]; then
  ok "interpreter found"
  if (cd "$GRL" && "$PYTHON" - <<'EOF'); then
import importlib, transformers
for mod in ("torch", "pytorch_lightning", "hydra", "wandb", "numpy", "peft", "transformers.models.hrm_text"):
    importlib.import_module(mod)
import torch
assert torch.cuda.is_available(), "CUDA not available"
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("sapientinc/HRM-Text-1B")
assert (cfg.H_cycles, cfg.L_cycles, cfg.num_layers_per_stack) == (2, 3, 16), cfg
print("        torch", torch.__version__, "| transformers", transformers.__version__)
EOF
    ok "imports, CUDA, HRM-Text-1B config in the HF cache (HF_HOME=${HF_HOME:-default})"
  else
    fail "imports / CUDA / HF cache (see the traceback above)"
  fi
else
  fail "no graph-hrm interpreter (tried \$HOME/miniforge3 and /work/dfm/.home/miniforge3)"
fi

echo "== W&B (project $PROJECT, mode $WANDB_MODE)"
if viewer=$("$PYTHON" -c "import wandb; print(wandb.Api(timeout=30).viewer.username)" 2>/dev/null); then
  ok "logged in as $viewer"
else
  fail "not logged in or no network: run '$PYTHON -m wandb login' (never use WANDB_MODE=disabled)"
fi

echo "== data"
for ds in "${FT_ORDER[@]}"; do
  for split in train val test; do
    f=$GRL/data/$ds/baseline/standard/$split.json
    [[ -f $f ]] || fail "missing $f"
  done
done
ok "train/val/test for ${FT_ORDER[*]} (missing ones listed above)"

echo "== checkpoints"
if [[ -z $GRAPHQA_CKPT ]]; then
  if GRAPHQA_CKPT=$(cd "$GRL" && "$PYTHON" scripts/pretrained_recursion/find_graphqa_ckpt.py); then
    echo "$GRAPHQA_CKPT" > "$ROOT/graphqa_ckpt.txt"
    FT_CKPT[graphqa]=$GRAPHQA_CKPT
    ok "GraphQA reference found, written to $ROOT/graphqa_ckpt.txt"
  else
    fail "GraphQA reference checkpoint not found: set GRAPHQA_CKPT=<.../last.ckpt>"
  fi
fi
if pgrep -f "runner.py.*dataset.name=graphqa" > /dev/null; then
  echo "  WARN  a GraphQA runner.py is still running: its last.ckpt may not be final"
fi
for ds in "${FT_ORDER[@]}"; do
  [[ -f ${FT_CKPT[$ds]} ]] && ok "$ds: ${FT_CKPT[$ds]}" || fail "$ds: missing ${FT_CKPT[$ds]:-<unset>}"
done

echo
if (( failed )); then echo "PREFLIGHT FAILED"; exit 1; fi
echo "PREFLIGHT OK"
