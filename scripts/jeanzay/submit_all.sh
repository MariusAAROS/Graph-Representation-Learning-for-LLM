#!/bin/bash
# Submit all Jean Zay training jobs at once.
#
# Usage (run from the repo root on a Jean Zay login node):
#   bash scripts/jeanzay/submit_all.sh              # submit every *.slurm here
#   bash scripts/jeanzay/submit_all.sh '*-ood'      # only OOD jobs
#   bash scripts/jeanzay/submit_all.sh 'hrm-*'      # only HRM jobs
#   DRY_RUN=1 bash scripts/jeanzay/submit_all.sh    # validate only, don't submit
#
#   # sweep several backbones (configs/model/*.yaml) over the 1b scripts:
#   MODELS="qwen2_5_1_5b llama3_2_1b mistral7b" bash scripts/jeanzay/submit_all.sh '*-1b-*'
#
# Each .slurm becomes its own SLURM job (own job ID + own log under logs/).

set -euo pipefail

# Directory containing this script (so it works from any CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional glob to filter which scripts to submit (default: all).
PATTERN="${1:-*}"

# Backbones to sweep. Empty = submit each script once with its built-in default.
MODELS="${MODELS:-}"

# Logs are written to logs/%x_%j.out relative to the submission CWD.
mkdir -p logs

shopt -s nullglob
scripts=("$SCRIPT_DIR"/${PATTERN}.slurm)
shopt -u nullglob

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "No .slurm files matching '${PATTERN}.slurm' in $SCRIPT_DIR" >&2
  exit 1
fi

echo "Found ${#scripts[@]} job script(s):"
for f in "${scripts[@]}"; do echo "  - $(basename "$f")"; done
[[ -n "$MODELS" ]] && echo "Sweeping models: $MODELS"
echo

submit() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    sbatch --test-only "$@"
  else
    sbatch "$@"
  fi
}

for f in "${scripts[@]}"; do
  base="$(basename "$f" .slurm)"
  # Only scripts that read $MODEL can be swept; the rest run once as-is.
  if [[ -n "$MODELS" ]] && grep -q 'MODEL=${MODEL:-' "$f"; then
    for m in $MODELS; do
      submit --export=ALL,MODEL="$m" --job-name="${base}_${m}" "$f"
    done
  else
    submit "$f"
  fi
done

