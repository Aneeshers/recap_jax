#!/bin/bash
#SBATCH --job-name=bc_recap                   # Default job name (sweep overrides)
#SBATCH -c 2                               # CPU cores per task
#SBATCH -t 2-17:10                         # Runtime (D-HH:MM)
#SBATCH -p kempner_h100                    # Partition
#SBATCH --account=kempner_gershman_lab
#SBATCH --gres=gpu:1                       # 4 GPUs per task
#SBATCH --mem=80G                          # RAM for the job
#SBATCH -o slurm-%x-%A_%a_az_ckpt.out                 # STDOUT (%x=jobname, %A=array_job_id, %a=task_id)
#SBATCH -e slurm-%x-%A_%a_az_ckpt.err                 # STDERR

# Load modules / env
module load python/3.10.9-fasrc01

# If you rely on conda commands:
# source ~/.bashrc || true
# conda activate torch || true
cd /n/home04/amuppidi/recap
# Use explicit Python path from your torch env (as in your example)
# Pass SLURM_ARRAY_TASK_ID to read hyperparameters from CSV
~/.conda/envs/torch/bin/python BC_policy.py
