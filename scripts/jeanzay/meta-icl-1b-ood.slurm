#!/bin/bash
#SBATCH --job-name=micl1b_ood
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=axa@a100
#SBATCH -C a100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread

#SBATCH --time=02:00:00
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

## Qwen2.5-1.5B baseline | meta-icl data | OOD (LOTO sweep over held-out tasks)
## Top 3 hardest tasks (based on CL competency score)
srun python runner.py -m --config-name=meta_icl_1b dataset.test_type=ood dataset.ood_task=EdgeCount,DisconnectedNodes,MaximumFlow

## OOD - Complete (all 11 tasks)
## python runner.py -m --config-name=meta_icl_1b dataset.test_type=ood dataset.ood_task=CycleCheck,EdgeExistence,NodeCount,NodeDegree,EdgeCount,ConnectedNodes,DisconnectedNodes,Reachability,ShortestPath,TriangleCounting,MaximumFlow
