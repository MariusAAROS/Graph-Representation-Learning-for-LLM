#!/bin/bash
#SBATCH --job-name=metaqa_hrm_id
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=axa@a100
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0
source $WORK/Graph-Representation-Learning-for-LLM/.venv/bin/activate

export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TMPDIR=$JOBSCRATCH
export NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include

cd $WORK/Graph-Representation-Learning-for-LLM

## HRM-Text 1B | MetaQA | baseline | ID (standard)
# MetaQA has no LOTO variant, so test_type is always standard here.
# max_seq_len stays at the config's 2048: every MetaQA baseline prompt fits (p95 = 1232).
srun python runner.py --config-name=hrm_text \
  dataset.name=metaqa dataset.dataset_config=baseline dataset.test_type=standard \
  logger.name=metaqa-hrm-text
