#!/bin/bash
#SBATCH --job-name=p4_actordeep_recurcmp
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_4/slurm-%x-%A_%a.err
#SBATCH --array=0-5

# Compares recurrent critic variants with a deeper actor:
# - recur_tied 1x4 (num_blocks=1, ktrain=4)
# - recur_tied 2x4 (num_blocks=2, ktrain=4)
# Array mapping:
#   0..2 -> 1x4 seeds 0,1,2
#   3..5 -> 2x4 seeds 0,1,2
#
# Example:
# sbatch --export="ALL,ENV_NAME=humanoidmaze-large-stitch-v0,RUN_GROUP_PREFIX=P4_HumanoidLargeStitch_ActorDeep,ACTOR_HIDDEN_DIMS=512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" \
#   slurm/phase4_train_actor_deeper_recur_compare_array.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(0 1 2)
NUM_SEEDS="${#SEEDS[@]}"
VARIANT_IDX=$(( TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( TASK_ID % NUM_SEEDS ))

if (( VARIANT_IDX < 0 || VARIANT_IDX > 1 )); then
  echo "Error: SLURM_ARRAY_TASK_ID=${TASK_ID} out of expected range 0-5." >&2
  exit 2
fi

SEED="${SEEDS[SEED_IDX]}"
ENV_NAME="${ENV_NAME:-humanoidmaze-large-stitch-v0}"
RUN_GROUP_PREFIX="${RUN_GROUP_PREFIX:-P4_ActorDeep_RecurCompare}"
ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-512x512x512x512x512}"

case "${VARIANT_IDX}" in
  0)
    VARIANT_TAG="recur1x4"
    KTRAIN=4
    RECUR_NUM_BLOCKS=1
    ;;
  1)
    VARIANT_TAG="recur2x4"
    KTRAIN=4
    RECUR_NUM_BLOCKS=2
    ;;
esac

RUN_GROUP="${RUN_GROUP_PREFIX}_${VARIANT_TAG}"
EXP_NAME="${EXP_NAME:-sd$(printf %03d "${SEED}")_${ENV_NAME%-v0}_${VARIANT_TAG}_actdeep}"

# Re-map for the generic script's per-seed array logic.
export SLURM_ARRAY_TASK_ID=0
export SEEDS="${SEED}"

export ENV_NAME
export RUN_GROUP
export EXP_NAME
export CRITIC_BACKBONE="recur_tied"
export KTRAIN
export RECUR_NUM_BLOCKS
export ACTOR_HIDDEN_DIMS

# Defaults can still be overridden via sbatch --export.
export RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-24}"
export SINUSOIDAL="${SINUSOIDAL:-1}"
export TIED_LN="${TIED_LN:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
export EVAL_EPISODES="${EVAL_EPISODES:-20}"
export VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
export LOG_INTERVAL="${LOG_INTERVAL:-20000}"
export VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
export DISABLE_TQDM="${DISABLE_TQDM:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Launching ${VARIANT_TAG} seed=${SEED} env=${ENV_NAME} actor_hidden_dims=${ACTOR_HIDDEN_DIMS}"

bash slurm/phase4_train_crl_generic_array.slurm
