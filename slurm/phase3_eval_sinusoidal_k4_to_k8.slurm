#!/bin/bash
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

# Phase 3 — Sinusoidal Encoding Extrapolation Test
#
# Evaluate sinusoidal K=4 checkpoint at K_test ∈ {4, 8}.
# - K=4: Parity check (should match discrete baseline ~0.37)
# - K=8: Extrapolation test (should be ≥90% of K=4 if it works)
#
# Prerequisites:
#   1. Training completed: sbatch slurm/phase3_train_sinusoidal_k4.slurm
#   2. Checkpoints exist: exp/OGBench/P3_Sinusoidal_Incremental/sd{000,001,002}_sin_k4_lrd1000000/
#
# Array layout:
#   idx = seed (fastest) + K_test (slowest)
#   6 jobs = 3 seeds × 2 K_test

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# Array sweep
SEEDS=(2)
K_TESTS=(4 8 16)

idx="${SLURM_ARRAY_TASK_ID}"
seed_idx=$(( idx % ${#SEEDS[@]} ))
ktest_idx=$(( idx / ${#SEEDS[@]} ))

SEED="${SEEDS[seed_idx]}"
KTEST="${K_TESTS[ktest_idx]}"

# Source checkpoint from sinusoidal K=4 training
TRAIN_RUN_GROUP="P3_Sinusoidal_Incremental"
TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_sin_k4_lrd1000000"
RESTORE_PATH="${SLURM_SUBMIT_DIR}/exp/OGBench/${TRAIN_RUN_GROUP}/${TRAIN_BASE_NAME}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"

# Eval settings
IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
REFINE_STEPS="${REFINE_STEPS:-10}"

RUN_GROUP="P3_Sinusoidal_Eval_K4toK8"
EXP_NAME="sd$(printf %03d ${SEED})_sin_ktrain4_ktest${KTEST}_ref${REFINE_STEPS}"

export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="allow"
export PYTHONUNBUFFERED=1

# Platform
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
mkdir -p logs/phase_3

# Build command
cd "${IMPL_DIR}"

cmd=(
  python main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--env_name=${ENV_NAME}"
  "--seed=${SEED}"
  "--train_steps=0"
  "--eval_only=1"
  "--restore_path=${RESTORE_PATH}"
  "--restore_epoch=${RESTORE_EPOCH}"
  "--exp_name=${EXP_NAME}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--eval_tasks=5"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--eval_refine_steps=${REFINE_STEPS}"
  "--agent=agents/crl.py"
  # Architecture flags — must match training exactly
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=1"         # Match training config
  "--agent.critic_layerscale_init=1e-2"
  "--agent.critic_eval_num_iters=${KTEST}"   # Override K at eval (KEY!)
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
)

echo "SINUSOIDAL EVAL: seed=${SEED} ktest=${KTEST} refine=${REFINE_STEPS} exp=${EXP_NAME} restore=${RESTORE_PATH}@${RESTORE_EPOCH}"
echo "${cmd[*]}"

# Conda bootstrap
if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  for conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${conda_sh}" ]]; then
      source "${conda_sh}"
      break
    fi
  done
fi

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  echo "Error: 'conda' not found in this job environment." >&2
  exit 3
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" python - <<'PY'
import jax
print("jax_devices:", jax.devices())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
