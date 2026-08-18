#!/bin/bash
#SBATCH --job-name=kqapro_base1b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=axa@a100
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3

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

## ~1B baseline | KQA Pro data | ID (standard); KQA Pro has no LOTO variant

# Backbone to finetune; any file name in configs/model/ works.
#   MODEL=llama3_2_1b sbatch scripts/jeanzay/kqapro-baseline-1b-id.slurm
MODEL=${MODEL:-qwen2_5_1_5b}

srun python runner.py --config-name=kqapro_baseline_1b model=$MODEL dataset.test_type=standard
