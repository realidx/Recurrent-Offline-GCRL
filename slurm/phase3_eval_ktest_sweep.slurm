#!/bin/bash
# Backwards-compatible wrapper for the old filename.
# The actual sweep logic lives in `slurm/phase3_eval_ktest.slurm`.
#SBATCH --job-name=p3_ktest
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-8

set -euo pipefail

exec bash slurm/phase3_eval_ktest.slurm

