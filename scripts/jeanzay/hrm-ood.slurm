#!/bin/bash
#SBATCH --job-name=hrm_ood
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=axa@a100
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --array=0-10   # one job per LOTO held-out task

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

## HRM-Text 1B | baseline | OOD (LOTO sweep over held-out tasks)

# Each array index finetunes with one task held out; runs in parallel, one GPU each.
TASKS=(CycleCheck EdgeExistence NodeCount NodeDegree EdgeCount ConnectedNodes DisconnectedNodes Reachability ShortestPath TriangleCounting MaximumFlow)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

srun python runner.py --config-name=hrm_text dataset.dataset_config=baseline dataset.test_type=ood dataset.ood_task=$TASK