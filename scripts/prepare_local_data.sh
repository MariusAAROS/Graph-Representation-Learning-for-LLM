#!/usr/bin/env bash
# Build the MetaQA variants and KQA Pro files that configs/{metaqa,kqapro}_*.yaml read.
# Each step is skipped when its output already exists, so re-running is cheap.
# KQA Pro's QA splits are downloaded from the HF Hub, so that step needs network.
#
#   scripts/prepare_local_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/home/ucloud/miniforge3/envs/graph-hrm/bin/python}

metaqa() {  # <output-dir> [generator args...]
  local out=$1; shift
  if [[ -f "$out/meta-icl/standard/test.json" ]]; then
    echo "[skip] $out"
  else
    echo "[build] $out $*"
    $PYTHON -m src.datasets.metaqa.dataset_generator --output-dir "$out" "$@"
  fi
}

# Arguments match the config headers and exploration/metaqa_variants.ipynb.
metaqa data/metaqa
metaqa data/metaqa-multi       --answer-len 5  --interv 2
metaqa data/metaqa-gold-1      --answer-len 1  --interv 0 --strategies gold
metaqa data/metaqa-gold-5      --answer-len 5  --interv 0 --strategies gold
metaqa data/metaqa-gold-10     --answer-len 10 --interv 0 --strategies gold
metaqa data/metaqa-retrieved-1 --answer-len 1  --interv 0 --strategies retrieved

if [[ -f data/kqapro/train.json && -f data/kqapro/val.json ]]; then
  echo "[skip] data/kqapro/{train,val}.json"
else
  echo "[build] data/kqapro/{train,val}.json"
  $PYTHON -m scripts.prepare_kqapro_qa
fi
for model_type in baseline meta-icl; do
  if [[ -f "data/kqapro/$model_type/standard/test.json" ]]; then
    echo "[skip] data/kqapro/$model_type"
  else
    echo "[build] data/kqapro/$model_type"
    $PYTHON -m src.datasets.kqapro.dataset_generator --model-type "$model_type"
  fi
done
