#!/bin/bash
#SBATCH --job-name=grl_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=105511@v100   # <-- replace, run `idracct` to find it

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # 1 task = 1 GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-dev                # dev queue: max 2h, fast scheduling

module purge
module load pytorch-gpu/py3/2.3.0
source $WORK/venvs/grl/bin/activate

# --- Offline mode (compute nodes have NO internet) ---
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                # sync later from a login node with `wandb sync`

export TMPDIR=$JOBSCRATCH

cd $WORK/repositories/graph-representation-learning-for-llm
srun python runner.py --config-name=baseline_1b