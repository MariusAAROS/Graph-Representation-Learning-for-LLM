#!/usr/bin/env bash
# GraphQA leave-one-task-out (LOTO) sweeps for the late baselines, one run after another on
# the local GPU. Each group repeats that model's in-distribution recipe with
# dataset.test_type=ood and one held-out task:
#   mimir-baseline   --config-name=mimir1, standard SFT         (as the mimir_v1-id runs)
#   mimir-meta-icl   --config-name=mimir1, meta-icl format      (logged as mimir_v1-meta-icl)
#   ouro-baseline    --config-name=baseline_1b model=ouro_1_4b  (as run_ouro_baselines.sh)
#   ouro-meta-icl    --config-name=meta_icl_1b model=ouro_1_4b
#
#   scripts/run_graphqa_ood_local.sh mimir-baseline ouro-baseline mimir-meta-icl ouro-meta-icl
#   TASKS="ShortestPath MaximumFlow" scripts/run_graphqa_ood_local.sh mimir-baseline
#   DRY_RUN=1 scripts/run_graphqa_ood_local.sh mimir-baseline
#
# A finished run leaves $LOG_DIR/<name>.done and is skipped on relaunch, so an interrupted
# queue resumes where it stopped. A failed run is logged and the queue moves on; a summary
# is printed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

GROUPS_=${*:?usage: run_graphqa_ood_local.sh <group>... (mimir-baseline | mimir-meta-icl | ouro-baseline | ouro-meta-icl)}
PYTHON=${PYTHON:-/home/ucloud/miniforge3/envs/graph-hrm/bin/python}
LOG_DIR=${LOG_DIR:-logs/graphqa_ood}
DRY_RUN=${DRY_RUN:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-online}

ALL_TASKS="CycleCheck EdgeExistence NodeCount NodeDegree EdgeCount ConnectedNodes DisconnectedNodes Reachability ShortestPath TriangleCounting MaximumFlow"
read -r -a OOD_TASKS <<< "${TASKS:-$ALL_TASKS}"

# Each queue entry is "<log name>|<runner.py args>".
queue=()
for group in $GROUPS_; do
  case "$group" in
    mimir-baseline) base="--config-name=mimir1 dataset.dataset_config=baseline" ;;
    mimir-meta-icl) base="--config-name=mimir1 dataset.dataset_config=meta-icl logger.name=mimir_v1-meta-icl" ;;
    ouro-baseline)  base="--config-name=baseline_1b model=ouro_1_4b" ;;
    ouro-meta-icl)  base="--config-name=meta_icl_1b model=ouro_1_4b" ;;
    *) echo "unknown group: $group" >&2; exit 1 ;;
  esac
  for task in "${OOD_TASKS[@]}"; do
    [[ " $ALL_TASKS " == *" $task "* ]] || { echo "unknown task: $task" >&2; exit 1; }
    queue+=("$group-ood-$task|$base dataset.test_type=ood dataset.ood_task=$task")
  done
done

mkdir -p "$LOG_DIR"
failed=()
skipped=0
for i in "${!queue[@]}"; do
  name=${queue[$i]%%|*}
  args=${queue[$i]#*|}
  log="$LOG_DIR/$name.log"
  if [[ -f "$LOG_DIR/$name.done" ]]; then
    echo "[$((i + 1))/${#queue[@]}] $name already done, skipping"
    skipped=$((skipped + 1))
    continue
  fi
  echo "[$((i + 1))/${#queue[@]}] $(date '+%F %T') $name -> $log"
  if [[ $DRY_RUN == 1 ]]; then
    echo "    $PYTHON runner.py $args"
    continue
  fi
  # shellcheck disable=SC2086
  if $PYTHON runner.py $args > "$log" 2>&1; then
    touch "$LOG_DIR/$name.done"
  else
    echo "    FAILED (see $log)"
    failed+=("$name")
  fi
done

echo "$(date '+%F %T') Done: $(( ${#queue[@]} - ${#failed[@]} ))/${#queue[@]} succeeded ($skipped already done)."
[[ ${#failed[@]} -eq 0 ]] || { echo "Failed: ${failed[*]}"; exit 1; }
