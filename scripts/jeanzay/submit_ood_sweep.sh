#!/bin/bash
# Throttled LOTO sweep across several backbones.
#
# Submits model x held-out-task as individual single-GPU jobs, keeping at most
# MAX_INFLIGHT of them queued at any time. The cap is global across all models
# of the condition, unlike looping submit_ood_throttled.sh per model.
#
# Usage (from repo root on a Jean Zay login node):
#   export MODELS="qwen3_0_6b qwen2_5_1_5b llama3_2_1b smollm2_1_7b gemma3_1b qwen2_5_3b qwen3_4b mistral7b"
#   bash scripts/jeanzay/submit_ood_sweep.sh scripts/jeanzay/baseline-1b-ood.slurm
#   bash scripts/jeanzay/submit_ood_sweep.sh scripts/jeanzay/meta-icl-1b-ood.slurm
#
#   MAX_INFLIGHT=6 bash scripts/jeanzay/submit_ood_sweep.sh scripts/jeanzay/baseline-1b-ood.slurm
#   TASKS="ShortestPath MaximumFlow" bash scripts/jeanzay/submit_ood_sweep.sh ...
#   DRY_RUN=1 bash scripts/jeanzay/submit_ood_sweep.sh ...
#
# With MODELS unset it sweeps only the script's built-in backbone, which is how
# the HRM scripts (they do not read $MODEL) should be launched.
set -euo pipefail

SCRIPT="${1:?usage: submit_ood_sweep.sh <path/to/xxx-ood.slurm>}"
[[ -f "$SCRIPT" ]] || { echo "No such script: $SCRIPT" >&2; exit 1; }

MODELS="${MODELS:-}"
MAX_INFLIGHT="${MAX_INFLIGHT:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"

ALL_TASKS=(CycleCheck EdgeExistence NodeCount NodeDegree EdgeCount ConnectedNodes DisconnectedNodes Reachability ShortestPath TriangleCounting MaximumFlow)
read -r -a WANT <<< "${TASKS:-${ALL_TASKS[*]}}"

indices=()
for t in "${WANT[@]}"; do
  found=-1
  for j in "${!ALL_TASKS[@]}"; do [[ "${ALL_TASKS[$j]}" == "$t" ]] && { found=$j; break; }; done
  (( found >= 0 )) || { echo "Unknown task: $t (valid: ${ALL_TASKS[*]})" >&2; exit 1; }
  indices+=("$found")
done

# Every job of this condition shares this name prefix, so one squeue query caps them all.
JOBPREFIX="$(sed -n 's/^#SBATCH --job-name=\(.*\)/\1/p' "$SCRIPT" | head -1)"
[[ -n "$JOBPREFIX" ]] || { echo "Could not read --job-name from $SCRIPT" >&2; exit 1; }

if [[ -n "$MODELS" ]]; then
  grep -q 'MODEL=${MODEL:-' "$SCRIPT" || { echo "$SCRIPT does not read \$MODEL" >&2; exit 1; }
  read -r -a MODEL_LIST <<< "$MODELS"
else
  MODEL_LIST=("")
fi

mkdir -p logs

inflight() {
  command -v squeue >/dev/null 2>&1 || { echo 0; return; }   # allow DRY_RUN off-cluster
  squeue -u "$USER" -h -r -o "%j" 2>/dev/null | grep -c "^${JOBPREFIX}" || true
}

total=$(( ${#MODEL_LIST[@]} * ${#indices[@]} ))
echo "Script      : $SCRIPT  (job-name prefix: $JOBPREFIX)"
echo "Models      : ${MODELS:-<script default>}"
echo "Tasks       : ${WANT[*]}"
echo "Jobs        : $total   (max $MAX_INFLIGHT in flight, poll ${POLL_SECONDS}s)"
echo

n=0
for model in "${MODEL_LIST[@]}"; do
  if [[ -n "$model" ]]; then
    extra=(--export=ALL,MODEL="$model" --job-name="${JOBPREFIX}_${model}")
    label="$model"
  else
    extra=()
    label="default"
  fi

  for idx in "${indices[@]}"; do
    while (( $(inflight) >= MAX_INFLIGHT )); do
      echo "[throttle] $(inflight)/${MAX_INFLIGHT} '${JOBPREFIX}*' jobs queued; waiting ${POLL_SECONDS}s ..."
      sleep "$POLL_SECONDS"
    done
    n=$(( n + 1 ))
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[dry-run $n/$total] sbatch ${extra[*]:-} --array=$idx $SCRIPT   # $label / ${ALL_TASKS[$idx]}"
    else
      sbatch "${extra[@]}" --array="$idx" "$SCRIPT" >/dev/null
      echo "[submit $n/$total] $label / ${ALL_TASKS[$idx]} (array index $idx)"
    fi
  done
done

echo
echo "All $total submissions issued."
