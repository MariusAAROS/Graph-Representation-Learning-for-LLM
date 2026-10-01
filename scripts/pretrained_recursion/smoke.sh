#!/usr/bin/env bash
# End-to-end smoke test of every job type with tiny limits (~10 min on one GPU), logged to a
# separate W&B project so the real one stays clean. Checkpoints are deleted at the end.
#
#   GPU=0 SMOKE_BATCH=8 bash scripts/pretrained_recursion/smoke.sh
#
# SMOKE_BATCH is the fine-tune batch size (default 2; 8 = the real one, to check memory).
# Uses the GraphQA reference as the e warm start if it is known, else the MetaQA Gold-1
# fine-tune (same architecture, so the plumbing is identical).
set -uo pipefail

# Overrides go through the environment: every job re-sources common.sh in a fresh bash.
REAL_ROOT=${ROOT:-/work/dfm/marius-ortega/pretrained_recursion}
GOLD1_CKPT=/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM/Meta-ICL/kbxwtebz/checkpoints/last.ckpt
graphqa_ckpt=${GRAPHQA_CKPT:-$(cat "$REAL_ROOT/graphqa_ckpt.txt" 2>/dev/null || true)}
[[ -f $graphqa_ckpt ]] || graphqa_ckpt=$GOLD1_CKPT
export ROOT=$REAL_ROOT/smoke PR_WANDB_PROJECT=HRM-Pretrained-Recursion-smoke SMOKE=1 POST_EVAL_MIN=0
export GRAPHQA_CKPT=$graphqa_ckpt
export SMOKE_BATCH=${SMOKE_BATCH:-2}

rm -rf "$ROOT"/{done,status,ckpts}
mkdir -p "$ROOT"
echo $(( $(date +%s) + 3 * 3600 )) > "$ROOT/deadline"
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
log "smoke on GPU $CUDA_VISIBLE_DEVICES, batch $SMOKE_BATCH, e warm start: $GRAPHQA_CKPT"

run_job smoke-probe-pretrained 1 10 probe --source pretrained --dataset graphqa --name pretrained
run_job smoke-probe-fresh 1 10 probe --source pretrained --init fresh --dataset graphqa --name pretrained
run_job smoke-probe-ft 1 10 probe --source "${FT_CKPT[metaqa-gold-1]}" --dataset metaqa-gold-1 --name ft-metaqa-gold-1
run_job smoke-b 1 20 depth_override metaqa-gold-1
run_job smoke-d 1 20 finetune d smoke-d 1 6 pretrained
run_job smoke-d-eval 1 15 post_eval d smoke-d 1 6
run_job smoke-e 1 20 finetune e smoke-e 1 1 "$GRAPHQA_CKPT" trainer.val_check_interval=0.5 \
  trainer.early_stopping_patience=10
run_job smoke-e-eval 1 15 post_eval e smoke-e 1 1

echo
echo "== smoke results (status per job; logs in $ROOT/logs)"
for f in "$ROOT"/status/*; do printf "  %-28s %s\n" "$(basename "$f")" "$(cat "$f")"; done
ls -la "$ROOT"/ckpts/* 2>/dev/null
n_bad=$(grep -lE "FAILED|KILLED|skipped" "$ROOT"/status/* 2>/dev/null | wc -l)
rm -rf "$ROOT/ckpts"
(( n_bad == 0 )) && echo "SMOKE OK" || { echo "SMOKE FAILED ($n_bad job(s))"; exit 1; }
