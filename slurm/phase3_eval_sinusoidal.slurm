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
#SBATCH --array=0-2

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

# -----------------------------------------------------------------------------
# Sweep controls (override via `sbatch --export=ALL,...`)
# - Set `REFINE_STEPS_LIST="2 5 10"` to sweep T with K fixed.
# - Set `K_TESTS_LIST="4 8 12 16"` to sweep K (meaningful only if refine>0).
# -----------------------------------------------------------------------------
EVAL_SEEDS="${EVAL_SEEDS:-2}"
K_TESTS_LIST="${K_TESTS_LIST:-4}"
REFINE_STEPS_LIST="${REFINE_STEPS_LIST:-0}"

read -r -a SEEDS <<< "${EVAL_SEEDS}"
read -r -a K_TESTS <<< "${K_TESTS_LIST}"
read -r -a REFINE_STEPS <<< "${REFINE_STEPS_LIST}"

idx="${SLURM_ARRAY_TASK_ID}"
total=$(( ${#SEEDS[@]} * ${#K_TESTS[@]} * ${#REFINE_STEPS[@]} ))
if (( idx < 0 || idx >= total )); then
  echo "Error: SLURM_ARRAY_TASK_ID=${idx} out of range for total=${total} (seeds=${#SEEDS[@]} ktests=${#K_TESTS[@]} refine_steps=${#REFINE_STEPS[@]})." >&2
  echo "Fix: resubmit with --array=0-$((total - 1)) or adjust *_LIST variables." >&2
  exit 2
fi

# Layout: refine (fastest) → ktest → seed (slowest)
refine_idx=$(( idx % ${#REFINE_STEPS[@]} ))
ktest_idx=$(( (idx / ${#REFINE_STEPS[@]}) % ${#K_TESTS[@]} ))
seed_idx=$(( idx / (${#REFINE_STEPS[@]} * ${#K_TESTS[@]}) ))

SEED="${SEEDS[seed_idx]}"
KTEST="${K_TESTS[ktest_idx]}"
REFINE="${REFINE_STEPS[refine_idx]}"

# Source checkpoint from sinusoidal K=4 training
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-P3_Sinusoidal}"
KTRAIN="${KTRAIN:-4}"
TRAIN_NAME_SUFFIX="${TRAIN_NAME_SUFFIX:-}"
TRAIN_BASE_NAME="${TRAIN_BASE_NAME:-sd$(printf %03d ${SEED})_sin_k${KTRAIN}${TRAIN_NAME_SUFFIX}}"
RESTORE_PATH="${SLURM_SUBMIT_DIR}/exp/OGBench/${TRAIN_RUN_GROUP}/${TRAIN_BASE_NAME}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"

# Eval settings
IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"
RESTORE_USE_FLAGS="${RESTORE_USE_FLAGS:-1}"

ENV_NAME="${ENV_NAME:-antmaze-large-navigate-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
REFINE_LR="${REFINE_LR:-0.01}"
REFINE_L2="${REFINE_L2:-0.01}"

RUN_GROUP="${RUN_GROUP:-P3_Sinusoidal_Eval}"
EXP_NAME="${EXP_NAME:-sd$(printf %03d ${SEED})_sin_ktrain${KTRAIN}_ktest${KTEST}_ref${REFINE}}"

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
  "--restore_use_flags=${RESTORE_USE_FLAGS}"
  "--exp_name=${EXP_NAME}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--eval_tasks=5"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--eval_refine_steps=${REFINE}"
  "--eval_refine_lr=${REFINE_LR}"
  "--eval_refine_l2=${REFINE_L2}"
  "--agent=agents/crl.py"
  # Architecture flags — must match training exactly
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=${KTRAIN}"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=1"         # Match training config
  "--agent.critic_layerscale_init=1e-2"
  "--agent.critic_eval_num_iters=${KTEST}"   # Override K at eval (KEY!)
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
)

echo "SINUSOIDAL EVAL: seed=${SEED} ktest=${KTEST} refine=${REFINE} exp=${EXP_NAME} restore=${RESTORE_PATH}@${RESTORE_EPOCH}"
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
