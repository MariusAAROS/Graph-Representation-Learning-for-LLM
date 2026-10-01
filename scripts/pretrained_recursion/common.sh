#!/usr/bin/env bash
# Shared settings and job helpers for the pretrained-recursion runs (b, c, d, e).
# Sourced by preflight.sh, smoke.sh and the queue scripts; not meant to be executed.
#
# Overridable from the environment: ROOT, PYTHON, PR_WANDB_PROJECT, GRAPHQA_CKPT,
# POST_EVAL_MIN, SMOKE, SMOKE_BATCH. Every job re-sources this file in a fresh bash (see
# run_job), so overrides must come from the environment, not from edits after sourcing.

COMMON_SH=$(readlink -f "${BASH_SOURCE[0]}")
GRL=/work/dfm/marius-ortega/Graph-Representation-Learning-for-LLM
export ROOT=${ROOT:-/work/dfm/marius-ortega/pretrained_recursion}
PROJECT=${PR_WANDB_PROJECT:-HRM-Pretrained-Recursion}
export PR_WANDB_PROJECT=$PROJECT

# Python: the graph-hrm env from $HOME, else the copy on /work (a fresh machine may lack $HOME).
for candidate in "${PYTHON:-}" "$HOME/miniforge3/envs/graph-hrm/bin/python" \
                 /work/dfm/.home/miniforge3/envs/graph-hrm/bin/python; do
  if [[ -n $candidate && -x $candidate ]]; then PYTHON=$candidate; break; fi
done

# HF cache: default location, else the copy on /work. Offline: every model is cached.
if [[ -z ${HF_HOME:-} && ! -d $HOME/.cache/huggingface/hub/models--sapientinc--HRM-Text-1B ]]; then
  export HF_HOME=/work/dfm/.home/.cache/huggingface
fi
export PYTHONPATH=$GRL HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online  # never "disabled": the machine is gone after the window, W&B is the record
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$ROOT"/{logs,done,status,ckpts,results}

# Fine-tuned HRM-Text-1B checkpoints (H2L3, clip 1.0), one per dataset. The GraphQA one is the
# reference of d/e; preflight.sh finds it and writes its path to $ROOT/graphqa_ckpt.txt.
META=$GRL/Meta-ICL
export GRAPHQA_CKPT=${GRAPHQA_CKPT:-$(cat "$ROOT/graphqa_ckpt.txt" 2>/dev/null || true)}
declare -A FT_CKPT=(
  [graphqa]=$GRAPHQA_CKPT
  [metaqa-gold-1]=$META/kbxwtebz/checkpoints/last.ckpt
  [metaqa-gold-5]=$META/rw7yujy0/checkpoints/last.ckpt
  [metaqa-gold-10]=$META/yjtrlv9m/checkpoints/last.ckpt
  [metaqa-retrieved-1]=$META/k38t16mn/checkpoints/last.ckpt
  [kqapro]=$META/0pm24tug/checkpoints/last.ckpt
)
FT_ORDER=(graphqa metaqa-gold-1 metaqa-gold-5 metaqa-gold-10 metaqa-retrieved-1 kqapro)

GRL_COMMIT=$(git -C "$GRL" rev-parse --short HEAD 2>/dev/null || echo unknown)

# Recipe of the GraphQA reference run (scripts/run_hrm_clip1.sh with dataset.name=graphqa).
GRAPHQA_ARGS=(--config-name=hrm_text dataset.name=graphqa dataset.dataset_config=baseline
              dataset.test_type=standard trainer.gradient_clip_val=1.0)

# b: inference depths, (2,3) = trained.
B_CONFIGS="[[1,1],[1,3],[2,1],[2,2],[2,3],[2,4],[2,6],[3,3],[4,3]]"

# Minutes kept free after a training run for its test eval + probe.
export POST_EVAL_MIN=${POST_EVAL_MIN:-25}

# Smoke mode (smoke.sh): tiny limits everywhere.
export SMOKE=${SMOKE:-0}

# ---------------------------------------------------------------- deadline / bookkeeping

DEADLINE=$(cat "$ROOT/deadline" 2>/dev/null || echo 0)

log() { echo "[$(date '+%F %T')] $*"; }

set_status() {  # <job> <text>: one file per job, so both queues can write without locking
  echo "$(date '+%F %T') $2" > "$ROOT/status/$1"
  log "$1: $2"
}

minutes_left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }

max_time_str() {  # <minutes> -> Lightning max_time "DD:HH:MM:SS"
  local m=$1
  printf "%02d:%02d:%02d:00" $((m / 1440)) $(((m % 1440) / 60)) $((m % 60))
}

bp_list() {  # <H> <L> -> pretraining BPTT rule [0,...,0,L]: only the last H cycle's L steps get gradient
  local H=$1 L=$2 out="[" i
  for ((i = 1; i < H; i++)); do out+="0,"; done
  echo "${out}${L}]"
}

# run_job <job> <min_minutes> <max_minutes> <command...>
# Skips a job already marked done, or when fewer than min_minutes remain before DEADLINE.
# Kills the job (its whole process group) after max_minutes, or at DEADLINE if sooner;
# max_minutes 0 = until DEADLINE. A hung job can therefore never block its queue.
# The command (a function of this file) runs in a fresh bash that re-sources this file,
# from $GRL, with output in $ROOT/logs/<job>.log.
run_job() {
  local job=$1 need=$2 cap=$3 left status
  shift 3
  if [[ -f $ROOT/done/$job ]]; then log "$job: already done, skipped"; return 0; fi
  left=$(minutes_left)
  if (( left < need )); then set_status "$job" "skipped (needs ${need} min, ${left} left)"; return 0; fi
  (( cap == 0 || cap > left )) && cap=$left
  set_status "$job" "running (${left} min left, killed after ${cap} min)"
  timeout --kill-after=120 "${cap}m" \
    bash -c 'source "$1"; shift; cd "$GRL" && "$@"' _ "$COMMON_SH" "$@" \
    < /dev/null > "$ROOT/logs/$job.log" 2>&1
  status=$?
  if (( status == 0 )); then
    touch "$ROOT/done/$job"
    set_status "$job" done
  elif (( status == 124 || status == 137 )); then
    set_status "$job" "KILLED after ${cap} min (timeout, see logs/$job.log)"
  else
    set_status "$job" "FAILED (exit $status, see logs/$job.log)"
  fi
  return 0  # a failed job never stops the queue
}

# ---------------------------------------------------------------- jobs

# c: gain probe. probe <probe_gain_hf.py args...>
probe() {
  local extra=()
  (( SMOKE )) && extra=(--n_samples 2)
  "$PYTHON" scripts/pretrained_recursion/probe_gain_hf.py --exp c --project "$PROJECT" \
    --csv "$ROOT/results/probe_hf.csv" "${extra[@]}" "$@"
}

probe_all() {  # raw pretrained model, its random-init twin, and every fine-tuned checkpoint
  probe --source pretrained --dataset graphqa --name pretrained || return 1
  probe --source pretrained --init fresh --dataset graphqa --name pretrained || return 1
  local ds
  for ds in "${FT_ORDER[@]}"; do
    probe --source "${FT_CKPT[$ds]}" --dataset "$ds" --name "ft-$ds" || return 1
  done
}

# b: inference depth override of one fine-tuned checkpoint. depth_override <dataset>
depth_override() {
  local ds=$1 extra=()
  (( SMOKE )) && extra=(+depth.limit_val_batches=2)
  "$PYTHON" depth_runner.py --config-name=hrm_text dataset.name="$ds" dataset.dataset_config=baseline \
    dataset.test_type=standard logger.project="$PROJECT" \
    +depth.ckpt="${FT_CKPT[$ds]}" "+depth.configs=${B_CONFIGS}" +depth.exp=b +depth.tag="$ds" \
    "+depth.trained=[2,3]" "+depth.extra={grl_commit:'${GRL_COMMIT}'}" "${extra[@]}"
}

# d / e: GraphQA fine-tune at depth (H, L), stopped by max_time before the deadline.
# finetune <exp> <job> <H> <L> <init: pretrained | path to a Lightning ckpt> [hydra overrides...]
# A non-pretrained init is a weights-only warm start, validated once before the first step.
# On CUDA OOM it retries once with batch 4 x accumulation 8 (same effective batch of 32).
finetune() {
  local exp=$1 job=$2 H=$3 L=$4 init=$5
  shift 5
  local init_tag=pretrained
  [[ $init != pretrained ]] && init_tag=graphqa-H2L3
  local args=("${GRAPHQA_ARGS[@]}"
    logger.project="$PROJECT" logger.name="$job" +logger.group="$exp"
    "+logger.config={exp:${exp},H:${H},L:${L},init:${init_tag},grl_commit:'${GRL_COMMIT}'}"
    "+model.hrm_overrides={H_cycles:${H},L_cycles:${L},L_bp_cycles:$(bp_list "$H" "$L")}"
    +trainer.checkpoint_dir="$ROOT/ckpts/$job")
  if [[ $init != pretrained ]]; then
    args+=(+model.init_from_ckpt="$init" +trainer.validate_before_fit=true)
  fi
  if (( SMOKE )); then
    args+=(+trainer.limit_train_batches=8 +trainer.limit_val_batches=2 trainer.max_epochs=1
           dataset.batch_size="${SMOKE_BATCH:-2}" trainer.gradient_accumulation=1)
  fi

  local budget attempt batch_args=()
  for attempt in 1 2; do
    budget=$(( $(minutes_left) - POST_EVAL_MIN ))
    log "finetune $job (attempt $attempt): H=$H L=$L init=$init_tag budget=${budget} min ${batch_args[*]}"
    "$PYTHON" runner.py "${args[@]}" "+trainer.max_time='$(max_time_str "$budget")'" "${batch_args[@]}" "$@" \
      && return 0
    (( attempt == 1 )) && grep -q "OutOfMemoryError\|CUDA out of memory" "$ROOT/logs/$job.log" || return 1
    log "finetune $job: CUDA OOM, retrying with batch 4 x accumulation 8"
    rm -rf "$ROOT/ckpts/$job"
    batch_args=(dataset.batch_size=4 trainer.gradient_accumulation=8)
  done
}

# After a d/e run: test exact match of its last.ckpt at the trained depth, at L=1 and at the
# native (2,3), then the gain probe at the trained depth. post_eval <exp> <job> <H> <L>
post_eval() {
  local exp=$1 job=$2 H=$3 L=$4
  local ckpt=$ROOT/ckpts/$job/last.ckpt extra=()
  [[ -f $ckpt ]] || { echo "no checkpoint at $ckpt"; return 1; }
  (( SMOKE )) && extra=(+depth.limit_val_batches=2)
  "$PYTHON" depth_runner.py "${GRAPHQA_ARGS[@]}" logger.project="$PROJECT" \
    "+model.hrm_overrides={H_cycles:${H},L_cycles:${L}}" +depth.ckpt="$ckpt" \
    "+depth.configs=[[${H},${L}],[${H},1],[2,3]]" +depth.exp="${exp}-test" +depth.tag="$job" \
    "+depth.trained=[${H},${L}]" "+depth.extra={grl_commit:'${GRL_COMMIT}'}" "${extra[@]}" || return 1
  probe --source "$ckpt" --dataset graphqa --name "$job" --H "$H" --L "$L"
}
