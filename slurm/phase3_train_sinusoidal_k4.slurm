#!/bin/bash
# Backwards-compatible wrapper for the old filename.
# The actual training logic lives in `slurm/phase3_train_sinusoidal.slurm`.
#
# This wrapper sets K_train=4 to match the old intent.
#SBATCH --job-name=p3_sin_k4
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

export KTRAIN="${KTRAIN:-4}"

exec bash slurm/phase3_train_sinusoidal.slurm

