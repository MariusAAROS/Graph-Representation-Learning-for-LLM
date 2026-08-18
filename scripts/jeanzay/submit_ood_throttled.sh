#!/bin/bash
# Throttled OOD launcher.
#
# Submits each LOTO held-out task as its own single-GPU job, keeping at most
# MAX_INFLIGHT jobs in the queue at a time. This stays under the qos_gpu-dev
# submit cap that rejects a full 11-element `#SBATCH --array=0-10` (all 11
# elements count as queued jobs at submit time, so a `%N` throttle can't help).
#
# No edits to the *-ood.slurm scripts are needed: `sbatch --array=<i>` overrides
# the in-file array directive and submits a single job, and the script's own
# `TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}` resolves the task from that index.
#
# Usage (from repo root on a Jean Zay login node):
#   bash scripts/jeanzay/submit_ood_throttled.sh scripts/jeanzay/hrm-ood.slurm
#   MAX_INFLIGHT=3 bash scripts/jeanzay/submit_ood_throttled.sh scripts/jeanzay/hrm-ood.slurm
#   TASKS="MaximumFlow ShortestPath" bash scripts/jeanzay/submit_ood_throttled.sh scripts/jeanzay/hrm-ood.slurm
#   MODEL=mistral7b bash scripts/jeanzay/submit_ood_throttled.sh scripts/jeanzay/baseline-1b-ood.slurm
#   DRY_RUN=1 bash scripts/jeanzay/submit_ood_throttled.sh scripts/jeanzay/hrm-ood.slurm
set -euo pipefail

SCRIPT="${1:?usage: submit_ood_throttled.sh <path/to/xxx-ood.slurm>}"
[[ -f "$SCRIPT" ]] || { echo "No such script: $SCRIPT" >&2; exit 1; }

MAX_INFLIGHT="${MAX_INFLIGHT:-2}"    # max jobs (pending+running) kept in the queue
POLL_SECONDS="${POLL_SECONDS:-30}"   # how often to re-check the queue while throttled
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-}"                   # backbone from configs/model/ (empty = script default)

# Canonical LOTO task order; the index is the SLURM array id the .slurm scripts expect.
ALL_TASKS=(CycleCheck EdgeExistence NodeCount NodeDegree EdgeCount ConnectedNodes DisconnectedNodes Reachability ShortestPath TriangleCounting MaximumFlow)

# Which tasks to submit (names). Default = all 11.
read -r -a WANT <<< "${TASKS:-${ALL_TASKS[*]}}"

# Map each requested task name to its array index.
indices=()
for t in "${WANT[@]}"; do
  found=-1
  for j in "${!ALL_TASKS[@]}"; do [[ "${ALL_TASKS[$j]}" == "$t" ]] && { found=$j; break; }; done
  (( found >= 0 )) || { echo "Unknown task: $t (valid: ${ALL_TASKS[*]})" >&2; exit 1; }
  indices+=("$found")
done

# Read the job name declared in the .slurm so we only count our own jobs.
JOBNAME="$(sed -n 's/^#SBATCH --job-name=\(.*\)/\1/p' "$SCRIPT" | head -1)"
[[ -n "$JOBNAME" ]] || { echo "Could not read --job-name from $SCRIPT" >&2; exit 1; }

# Per-model job name so concurrent sweeps of the same script throttle separately.
sbatch_extra=()
if [[ -n "$MODEL" ]]; then
  grep -q 'MODEL=${MODEL:-' "$SCRIPT" || { echo "$SCRIPT does not read \$MODEL" >&2; exit 1; }
  JOBNAME="${JOBNAME}_${MODEL}"
  sbatch_extra=(--export=ALL,MODEL="$MODEL" --job-name="$JOBNAME")
fi

mkdir -p logs

inflight() {
  command -v squeue >/dev/null 2>&1 || { echo 0; return; }   # allow DRY_RUN off-cluster
  squeue -u "$USER" -h -r -n "$JOBNAME" | wc -l
}

echo "Script      : $SCRIPT  (job-name: $JOBNAME)"
echo "Tasks       : ${WANT[*]}"
[[ -n "$MODEL" ]] && echo "Model       : $MODEL"
echo "Max inflight: $MAX_INFLIGHT  (poll every ${POLL_SECONDS}s)"
echo

for idx in "${indices[@]}"; do
  while (( $(inflight) >= MAX_INFLIGHT )); do
    echo "[throttle] $(inflight)/${MAX_INFLIGHT} '${JOBNAME}' jobs in queue; waiting ${POLL_SECONDS}s ..."
    sleep "$POLL_SECONDS"
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] sbatch ${sbatch_extra[*]:-} --array=$idx $SCRIPT   # ${ALL_TASKS[$idx]}"
  else
    sbatch "${sbatch_extra[@]}" --array="$idx" "$SCRIPT"
    echo "[submit] ${ALL_TASKS[$idx]} (index $idx)"
  fi
done

echo "All submissions issued."
