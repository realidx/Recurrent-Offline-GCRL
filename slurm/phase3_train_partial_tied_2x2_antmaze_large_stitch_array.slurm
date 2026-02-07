#!/bin/bash
# Backwards-compatible wrapper for the old filename.
# The actual training logic lives in `slurm/phase3_train_partial_tied.slurm`.
#
# This wrapper preserves the old defaults (2×2, discrete encoding).
#SBATCH --job-name=p3_partial_2x2
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

export RUN_GROUP="${RUN_GROUP:-P3_PartialTied_2x2}"
export PARTIAL_GROUPS="${PARTIAL_GROUPS:-2}"
export PARTIAL_ITERS_PER_GROUP="${PARTIAL_ITERS_PER_GROUP:-2}"
export PARTIAL_PER_GROUP_FILM="${PARTIAL_PER_GROUP_FILM:-0}"
export RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-16}"
export RECUR_SINUSOIDAL="${RECUR_SINUSOIDAL:-0}"

exec bash slurm/phase3_train_partial_tied.slurm

