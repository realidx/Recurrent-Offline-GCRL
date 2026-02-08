#!/bin/bash
#SBATCH --job-name=p3_partial_eval
#SBATCH --partition=gpu-long
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

# Phase 3 — evaluate ANY partial_tied checkpoint (no K-test sweep)
#
# This script is architecture-agnostic: it restores the exact agent config from the checkpoint's `flags.json`
# (restore_use_flags=1) so you don't need to pass `--agent.critic_partial_*` flags here.
#
# Usage examples:
#   # Evaluate 1×2 partial tying (3 seeds)
#   sbatch --array=0-2 --export=ALL,TRAIN_RUN_GROUP=P3_PartialTied_1x2,PARTIAL_GROUPS=1,PARTIAL_ITERS_PER_GROUP=2 slurm/phase3_eval_partial.slurm
#
#   # Evaluate 2×1 partial tying (3 seeds)
#   sbatch --array=0-2 --export=ALL,TRAIN_RUN_GROUP=P3_PartialTied_2x1,PARTIAL_GROUPS=2,PARTIAL_ITERS_PER_GROUP=1 slurm/phase3_eval_partial.slurm
#
#   # Evaluate 1×4 (3 seeds)
#   sbatch --array=0-2 --export=ALL,TRAIN_RUN_GROUP=P3_PartialTied_1x4,PARTIAL_GROUPS=1,PARTIAL_ITERS_PER_GROUP=4 slurm/phase3_eval_partial.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# -----------------------------------------------------------------------------
# Seed selection
# -----------------------------------------------------------------------------
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2}"
read -r -a SEEDS <<< "${EVAL_SEEDS}"
SEED="${SEEDS[SLURM_ARRAY_TASK_ID]}"

# -----------------------------------------------------------------------------
# Which run to restore
# -----------------------------------------------------------------------------
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-P3_PartialTied}"
PARTIAL_GROUPS="${PARTIAL_GROUPS:-2}"
PARTIAL_ITERS_PER_GROUP="${PARTIAL_ITERS_PER_GROUP:-2}"
TRAIN_NAME_SUFFIX="${TRAIN_NAME_SUFFIX:-}"

TRAIN_BASE_NAME="${TRAIN_BASE_NAME:-sd$(printf %03d ${SEED})_partial${PARTIAL_GROUPS}x${PARTIAL_ITERS_PER_GROUP}${TRAIN_NAME_SUFFIX}}"
RESTORE_PATH="${RESTORE_PATH:-${SLURM_SUBMIT_DIR}/exp/OGBench/${TRAIN_RUN_GROUP}/${TRAIN_BASE_NAME}}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"
RESTORE_USE_FLAGS="${RESTORE_USE_FLAGS:-1}"

# Eval settings
IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_TASKS="${EVAL_TASKS:-5}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
EVAL_REFINE_STEPS="${EVAL_REFINE_STEPS:-0}"
EVAL_REFINE_LR="${EVAL_REFINE_LR:-0.05}"
EVAL_REFINE_L2="${EVAL_REFINE_L2:-0.0}"

RUN_GROUP="${RUN_GROUP:-P3_PartialTied_Eval}"
EXP_NAME="${EXP_NAME:-sd$(printf %03d ${SEED})_partial${PARTIAL_GROUPS}x${PARTIAL_ITERS_PER_GROUP}_eval}"

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
  "--eval_tasks=${EVAL_TASKS}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--eval_refine_steps=${EVAL_REFINE_STEPS}"
  "--eval_refine_lr=${EVAL_REFINE_LR}"
  "--eval_refine_l2=${EVAL_REFINE_L2}"
  "--agent=agents/crl.py"
)

echo "PARTIAL EVAL: seed=${SEED} exp=${EXP_NAME} restore=${RESTORE_PATH}@${RESTORE_EPOCH} env=${ENV_NAME} refine=${EVAL_REFINE_STEPS}"
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
