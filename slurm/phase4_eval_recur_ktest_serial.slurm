#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 4 — recurrent critic test-time K sweep (serial, single GPU job)
#
# Evaluates one trained recurrent-critic checkpoint multiple times, sweeping
# `K_test` serially inside a single SLURM allocation.
#
# IMPORTANT:
# - `--agent.critic_eval_num_iters=K_test` only affects evaluation if
#   `--eval_refine_steps > 0`, because the actor does not query the critic
#   during standard action sampling.
# - Architecture/training flags are loaded from the restored checkpoint
#   (`restore_use_flags=True` in OGBench main), so this script only overrides
#   eval-time knobs plus `critic_eval_num_iters`.
#
# Default checkpoint:
#   exp/OGBench/P4_Ablation_AntMazeLargeStitch_2x4_act8_hideFiLM/
#     sd000_antmaze_large_stitch_recur_tied
#
# Submit:
#   sbatch --export=ALL,WANDB_MODE=online \
#     slurm/phase4_eval_recur_ktest_serial.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=p4_recur_ktest
#SBATCH --partition=gpu-long
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%j.out
#SBATCH --error=logs/phase_4/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh first." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

SEED="${SEED:-0}"
RUN_GROUP="${RUN_GROUP:-P4_KTestSweep_2x4_Act8}"
RESTORE_PATH="${RESTORE_PATH:-${SLURM_SUBMIT_DIR}/exp/OGBench/P4_Ablation_AntMazeLargeStitch_2x4_act8_hideFiLM/sd000_antmaze_large_stitch_recur_tied}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"

# Space-separated K sweep, evaluated serially in this single job.
K_TESTS_STR="${K_TESTS:-1 2 3 4 6 8}"
read -r -a K_TESTS <<< "${K_TESTS_STR}"
if [[ "${#K_TESTS[@]}" -eq 0 ]]; then
  echo "Error: K_TESTS is empty." >&2
  exit 2
fi

EVAL_REFINE_STEPS="${EVAL_REFINE_STEPS:-10}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

if [[ "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi
if [[ ! -d "${RESTORE_PATH}" ]]; then
  echo "Error: RESTORE_PATH does not exist: ${RESTORE_PATH}" >&2
  exit 2
fi

if [[ ! "${RESTORE_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "Error: RESTORE_EPOCH must be an integer. Got '${RESTORE_EPOCH}'." >&2
  exit 2
fi

if [[ ! "${EVAL_REFINE_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "Error: EVAL_REFINE_STEPS must be a non-negative integer. Got '${EVAL_REFINE_STEPS}'." >&2
  exit 2
fi

for k in "${K_TESTS[@]}"; do
  if [[ ! "${k}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: K_TESTS contains non-positive integer '${k}'." >&2
    exit 2
  fi
done

export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
mkdir -p logs/phase_4

cd "${IMPL_DIR}"

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
fi

for KTEST in "${K_TESTS[@]}"; do
  EXP_NAME="sd$(printf %03d "${SEED}")_ktest${KTEST}_ref${EVAL_REFINE_STEPS}"
  export WANDB_RUN_ID="${WANDB_RUN_ID_PREFIX:-ogbench}_${RUN_GROUP}_${EXP_NAME}"
  export WANDB_RESUME="allow"

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
    "--eval_tasks=${EVAL_TASKS}"
    "--video_episodes=${VIDEO_EPISODES}"
    "--disable_tqdm=${DISABLE_TQDM}"
    "--eval_on_cpu=${EVAL_ON_CPU}"
    "--eval_refine_steps=${EVAL_REFINE_STEPS}"
    "--agent=agents/crl.py"
    "--agent.critic_eval_num_iters=${KTEST}"
  )

  echo "P4 KTEST: seed=${SEED} ktest=${KTEST} refine=${EVAL_REFINE_STEPS}"
  echo "restore=${RESTORE_PATH}@${RESTORE_EPOCH}"
  echo "${cmd[*]}"

  "${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
done
