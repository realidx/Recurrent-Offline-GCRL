#!/bin/bash
#SBATCH --job-name=p2_eval
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_2/slurm-%x-%A.out
#SBATCH --error=logs/phase_2/slurm-%x-%A.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P2_EvalOnly}"
EXP_NAME="${EXP_NAME:-}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

RESTORE_PATH="${RESTORE_PATH:?Set RESTORE_PATH to the training run directory glob (must match exactly one).}"
RESTORE_EPOCH="${RESTORE_EPOCH:?Set RESTORE_EPOCH (e.g., 1000000).}"

EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_2

if [[ -n "${RESTORE_PATH}" && "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi

cd "${IMPL_DIR}"

cmd=(
  python main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--seed=0"
  "--env_name=${ENV_NAME}"
  "--train_steps=0"
  "--eval_only=1"
  "--restore_path=${RESTORE_PATH}"
  "--restore_epoch=${RESTORE_EPOCH}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--agent=agents/crl.py"
)

if [[ -n "${EXP_NAME}" ]]; then
  cmd+=("--exp_name=${EXP_NAME}")
fi

if [[ -n "${EVAL_TASKS}" ]]; then
  cmd+=("--eval_tasks=${EVAL_TASKS}")
fi

echo "cwd=$(pwd) RESTORE_PATH=${RESTORE_PATH} RESTORE_EPOCH=${RESTORE_EPOCH} ${cmd[*]}"

if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  for conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck source=/dev/null
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
  nvidia-smi || true
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
