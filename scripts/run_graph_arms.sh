#!/usr/bin/env bash
# Structure-injection arms for HRM-Text-1B on graphqa/baseline-* (configs/hrm_graph.yaml).
# Needs data/graphqa/{baseline-struct,baseline-wl,baseline-wlshuf}/standard (see the
# config header for the generator command).
#
#   scripts/run_graph_arms.sh "A0 A2-node" "0 1 2"     # arms, seeds
#
# Arms run sequentially; start several invocations to share the GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

ARMS=${1:-"A0 A1-tok A1-shuf A2-node A2-shuf A2-frozen A2-wl"}
SEEDS=${2:-"0 1 2"}
PYTHON=${PYTHON:-python}

overrides_for() {
  case "$1" in
    A0)        echo "model.arch=hrm_text" ;;
    A1-tok)    echo "model.arch=hrm_text dataset.dataset_config=baseline-wl" ;;
    A1-shuf)   echo "model.arch=hrm_text dataset.dataset_config=baseline-wlshuf" ;;
    A2-node)   echo "" ;;
    A2-shuf)   echo "structure.node_mode=shuffled" ;;
    A2-frozen) echo "structure.train_codes=false" ;;
    A2-wl)     echo "structure.node=false structure.wl=true" ;;
    A2-wlshuf) echo "structure.node=false structure.wl=true structure.wl_mode=shuffled" ;;
    *) echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

for arm in $ARMS; do overrides_for "$arm" > /dev/null; done  # fail fast on typos

for seed in $SEEDS; do
  for arm in $ARMS; do
    # shellcheck disable=SC2046
    $PYTHON runner.py --config-name hrm_graph $(overrides_for "$arm") \
      +seed="$seed" logger.name="hrm-graph-$arm-s$seed"
  done
done
