#!/bin/bash
# Backwards-compatible wrapper for the old filename.
# The actual eval logic lives in `slurm/phase3_eval_sinusoidal.slurm`.
#SBATCH --job-name=p3_sin_eval
#SBATCH --partition=gpu-long
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

exec bash slurm/phase3_eval_sinusoidal.slurm

