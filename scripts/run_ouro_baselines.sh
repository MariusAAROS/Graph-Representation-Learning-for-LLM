#!/usr/bin/env bash
# Ouro-1.4B (configs/model/ouro_1_4b.yaml) through the causal_lm baseline configs,
# one run after another on the local GPU. Build the MetaQA / KQA Pro data first
# with scripts/prepare_local_data.sh.
#
#   scripts/run_ouro_baselines.sh                  # id: every in-distribution run
#   CONFIGS="kqapro_baseline_1b" scripts/run_ouro_baselines.sh   # a subset of id
#   scripts/run_ouro_baselines.sh ood-baseline     # graphqa LOTO sweep, standard SFT
#   scripts/run_ouro_baselines.sh ood-meta-icl     # graphqa LOTO sweep, meta-icl
#   TASKS="ShortestPath MaximumFlow" scripts/run_ouro_baselines.sh ood-baseline
#   DRY_RUN=1 scripts/run_ouro_baselines.sh id ood-baseline
#
# A failed run is logged and the queue moves on; a summary is printed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

GROUPS_=${*:-id}
MODEL=${MODEL:-ouro_1_4b}
PYTHON=${PYTHON:-/home/ucloud/miniforge3/envs/graph-hrm/bin/python}
LOG_DIR=${LOG_DIR:-logs/ouro}
DRY_RUN=${DRY_RUN:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Meta-icl only on graphqa; MetaQA and KQA Pro run the standard SFT format only,
# and metaqa-multi is skipped.
# CONFIGS="a b" overrides the list, e.g. to resume a partially finished queue.
ID_CONFIGS=(
  baseline_1b meta_icl_1b
  metaqa_baseline_1b
  metaqa_baseline_gold1_1b metaqa_baseline_gold5_1b metaqa_baseline_gold10_1b
  metaqa_baseline_retrieved1_1b
  kqapro_baseline_1b
)
[[ -n ${CONFIGS:-} ]] && read -r -a ID_CONFIGS <<< "$CONFIGS"
ALL_TASKS="CycleCheck EdgeExistence NodeCount NodeDegree EdgeCount ConnectedNodes DisconnectedNodes Reachability ShortestPath TriangleCounting MaximumFlow"
read -r -a OOD_TASKS <<< "${TASKS:-$ALL_TASKS}"

# Each queue entry is "<log name>|<runner.py args>".
queue=()
for group in $GROUPS_; do
  case "$group" in
    id)
      for cfg in "${ID_CONFIGS[@]}"; do
        queue+=("$cfg-id|--config-name=$cfg dataset.test_type=standard")
      done ;;
    ood-baseline|ood-meta-icl)
      cfg=$([[ $group == ood-baseline ]] && echo baseline_1b || echo meta_icl_1b)
      for task in "${OOD_TASKS[@]}"; do
        [[ " $ALL_TASKS " == *" $task "* ]] || { echo "unknown task: $task" >&2; exit 1; }
        queue+=("$cfg-ood-$task|--config-name=$cfg dataset.test_type=ood dataset.ood_task=$task")
      done ;;
    *) echo "unknown group: $group (id | ood-baseline | ood-meta-icl)" >&2; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
failed=()
for i in "${!queue[@]}"; do
  name=${queue[$i]%%|*}
  args=${queue[$i]#*|}
  log="$LOG_DIR/$MODEL-$name.log"
  echo "[$((i + 1))/${#queue[@]}] $(date '+%F %T') $name -> $log"
  if [[ $DRY_RUN == 1 ]]; then
    echo "    $PYTHON runner.py $args model=$MODEL"
    continue
  fi
  # shellcheck disable=SC2086
  if ! $PYTHON runner.py $args model="$MODEL" > "$log" 2>&1; then
    echo "    FAILED (see $log)"
    failed+=("$name")
  fi
done

echo "Done: $(( ${#queue[@]} - ${#failed[@]} ))/${#queue[@]} succeeded."
[[ ${#failed[@]} -eq 0 ]] || { echo "Failed: ${failed[*]}"; exit 1; }
