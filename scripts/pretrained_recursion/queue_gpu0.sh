#!/usr/bin/env bash
# GPU0 queue: c (gain probes) -> b (depth override, 6 checkpoints) -> d runs.
# Started by launch.sh. Each job is skipped if already done or if too little time is left.
set -o pipefail
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
log "queue gpu0 on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, $(minutes_left) min left"

# c: raw pretrained model, its random-init twin, every fine-tuned checkpoint (~2 min each).
run_job c-pretrained 5 20 probe --source pretrained --dataset graphqa --name pretrained
run_job c-fresh 5 20 probe --source pretrained --init fresh --dataset graphqa --name pretrained
for ds in "${FT_ORDER[@]}"; do
  run_job "c-ft-$ds" 5 20 probe --source "${FT_CKPT[$ds]}" --dataset "$ds" --name "ft-$ds"
done

# b: 9 inference depths per fine-tuned checkpoint (~5-15 min each).
for ds in "${FT_ORDER[@]}"; do
  run_job "b-$ds" 20 75 depth_override "$ds"
done

# d: fine-tune the pretrained model on GraphQA at another depth. <job> <H> <L> <min minutes>
# The last line is a stretch run: it only starts with >= 180 min left.
while read -r job H L need; do
  run_job "$job" "$need" 0 finetune d "$job" "$H" "$L" pretrained trainer.max_epochs=15
  [[ -f $ROOT/done/$job ]] && run_job "$job-eval" 15 45 post_eval d "$job" "$H" "$L"
done <<'EOF'
d-graphqa-H1L1 1 1 60
d-graphqa-H2L6 2 6 90
d-graphqa-H2L1 2 1 60
d-graphqa-H1L6 1 6 75
d-graphqa-H4L3 4 3 180
EOF

log "queue gpu0 finished"
