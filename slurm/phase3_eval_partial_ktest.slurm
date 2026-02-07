#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 3 — partial-tied test-time compute sweep: K_test × refinement × seed
#
# Evaluates checkpoints produced by `slurm/phase3_train_partial_tied.slurm`.
#
# IMPORTANT:
# - K_test only affects performance if action refinement is enabled
#   (`--eval_refine_steps > 0`), because the actor does not call the critic.
# - `--agent.critic_eval_num_iters=K_test` changes the critic unroll ONLY.
#
# Usage (2×4 model):
#   sbatch --export=ALL,TRAIN_RUN_GROUP=P3_PartialTied,PARTIAL_GROUPS=2,PARTIAL_ITERS_PER_GROUP=4 slurm/phase3_eval_partial_ktest.slurm
#
# Usage (1×8 model):
#   sbatch --export=ALL,TRAIN_RUN_GROUP=P3_PartialTied,PARTIAL_GROUPS=1,PARTIAL_ITERS_PER_GROUP=8 slurm/phase3_eval_partial_ktest.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=p3_partial_ktest
#SBATCH --partition=gpu-long
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-17

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ---------------------------------------------------------------------------
# Sweep layout
#   idx = refine_idx + R * (ktest_idx + K * seed_idx)
# ---------------------------------------------------------------------------
SEEDS=(0 1 2)
K_TESTS=(8 12 16)
REFINE_STEPS=(0 10)

idx="${SLURM_ARRAY_TASK_ID}"
refine_idx=$(( idx % ${#REFINE_STEPS[@]} ))
ktest_idx=$(( (idx / ${#REFINE_STEPS[@]}) % ${#K_TESTS[@]} ))
seed_idx=$(( idx / (${#REFINE_STEPS[@]} * ${#K_TESTS[@]}) ))

SEED="${SEEDS[seed_idx]}"
KTEST="${K_TESTS[ktest_idx]}"
REFINE="${REFINE_STEPS[refine_idx]}"

# ---------------------------------------------------------------------------
# Which partial-tied checkpoint to load (must exist)
# ---------------------------------------------------------------------------
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-P3_PartialTied}"
PARTIAL_GROUPS="${PARTIAL_GROUPS:-2}"
PARTIAL_ITERS_PER_GROUP="${PARTIAL_ITERS_PER_GROUP:-4}"
TRAIN_EXP_NAME_SUFFIX="${TRAIN_EXP_NAME_SUFFIX:-}"  # e.g. "_lrd1000000"

TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_partial${PARTIAL_GROUPS}x${PARTIAL_ITERS_PER_GROUP}${TRAIN_EXP_NAME_SUFFIX}"
RESTORE_PATH="${SLURM_SUBMIT_DIR}/exp/OGBench/${TRAIN_RUN_GROUP}/${TRAIN_BASE_NAME}"
RESTORE_EPOCH="${RESTORE_EPOCH:-1000000}"

# ---------------------------------------------------------------------------
# Eval settings
# ---------------------------------------------------------------------------
IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

RUN_GROUP="${RUN_GROUP:-P3_PartialTied_TestTimeScaling}"
EXP_NAME="sd$(printf %03d ${SEED})_partial${PARTIAL_GROUPS}x${PARTIAL_ITERS_PER_GROUP}_ktest${KTEST}_ref${REFINE}"

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
  "--eval_tasks=${EVAL_TASKS}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--eval_refine_steps=${REFINE}"
  "--agent=agents/crl.py"
  # Only override K at eval time; all other arch flags are loaded from flags.json (restore_use_flags=True).
  "--agent.critic_eval_num_iters=${KTEST}"
)

echo "PARTIAL KTEST: seed=${SEED} partial=${PARTIAL_GROUPS}x${PARTIAL_ITERS_PER_GROUP} ktest=${KTEST} refine=${REFINE}"
echo "restore=${RESTORE_PATH}@${RESTORE_EPOCH}"
echo "${cmd[*]}"

# Conda bootstrap
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

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"

