#!/usr/bin/env bash
# GPU1 queue: e runs (continue the GraphQA H2L3 fine-tune at another depth) -> d H3L3.
# Started by launch.sh. Each job is skipped if already done or if too little time is left.
set -o pipefail
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-1}
log "queue gpu1 on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, $(minutes_left) min left"

# e: warm start from the GraphQA reference, validated at step 0 and every half epoch.
# Patience 10 checks = 5 epochs, the same early-stopping horizon as the reference run.
while read -r job H L need; do
  run_job "$job" "$need" 0 finetune e "$job" "$H" "$L" "$GRAPHQA_CKPT" \
    trainer.max_epochs=10 trainer.val_check_interval=0.5 trainer.early_stopping_patience=10
  [[ -f $ROOT/done/$job ]] && run_job "$job-eval" 15 45 post_eval e "$job" "$H" "$L"
done <<'EOF'
e-graphqa-to-H1L1 1 1 45
e-graphqa-to-H2L6 2 6 75
e-graphqa-to-H3L3 3 3 75
EOF

# d: H3L3 from the pretrained model; the last line (a repeat of d H1L1) is a stretch run.
while read -r job H L need; do
  run_job "$job" "$need" 0 finetune d "$job" "$H" "$L" pretrained trainer.max_epochs=15
  [[ -f $ROOT/done/$job ]] && run_job "$job-eval" 15 45 post_eval d "$job" "$H" "$L"
done <<'EOF'
d-graphqa-H3L3 3 3 90
d-graphqa-H1L1-rep2 1 1 60
EOF

log "queue gpu1 finished"
