#!/usr/bin/env bash
# An HRM-text-arch model (configs/hrm_text.yaml by default) on the MetaQA variants and
# KQA Pro with gradient clipping at 1.0, to match the Ouro baseline
# (configs/model/ouro_1_4b.yaml). Same commands as scripts/jeanzay/*-hrm-id.slurm, run
# one after another on the local GPU. Runs are named <dataset>-<TAG>-clip1 so they sit
# next to the unclipped runs in wandb.
#
#   scripts/run_hrm_clip1.sh                                     # HRM-Text-1B
#   CONFIG=mimir1 TAG=mimir-v1 scripts/run_hrm_clip1.sh          # DFM-Mimir v1
#   DATASETS="metaqa-gold-1 kqapro" scripts/run_hrm_clip1.sh
#   DRY_RUN=1 scripts/run_hrm_clip1.sh
#
# A failed run is logged and the queue moves on; a summary is printed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/ucloud/miniforge3/envs/graph-hrm/bin/python}
CONFIG=${CONFIG:-hrm_text}
TAG=${TAG:-hrm-text}
LOG_DIR=${LOG_DIR:-logs/$([[ $TAG == hrm-text ]] && echo hrm || echo "$TAG")_clip1}
DRY_RUN=${DRY_RUN:-0}
export PYTHONPATH=.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=online

# Plain metaqa and metaqa-multi are skipped: paper MetaQA results average these four variants.
DATASETS_=(metaqa-gold-1 metaqa-gold-5 metaqa-gold-10 metaqa-retrieved-1 kqapro)
[[ -n ${DATASETS:-} ]] && read -r -a DATASETS_ <<< "$DATASETS"

mkdir -p "$LOG_DIR"
failed=()
for i in "${!DATASETS_[@]}"; do
  ds=${DATASETS_[$i]}
  log="$LOG_DIR/$ds.log"
  args="--config-name=$CONFIG dataset.name=$ds dataset.dataset_config=baseline
    dataset.test_type=standard trainer.gradient_clip_val=1.0 logger.name=$ds-$TAG-clip1"
  echo "[$((i + 1))/${#DATASETS_[@]}] $(date '+%F %T') $ds -> $log"
  if [[ $DRY_RUN == 1 ]]; then
    echo "    $PYTHON runner.py" $args
    continue
  fi
  # shellcheck disable=SC2086
  if ! $PYTHON runner.py $args > "$log" 2>&1; then
    echo "    FAILED (see $log)"
    failed+=("$ds")
  fi
done

echo "Done: $(( ${#DATASETS_[@]} - ${#failed[@]} ))/${#DATASETS_[@]} succeeded."
[[ ${#failed[@]} -eq 0 ]] || { echo "Failed: ${failed[*]}"; exit 1; }
