#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 3 — test-time compute sweep: K_test × action-refinement × seed
#
# Sources the K_train=4 LR-decay checkpoints from P3_RecurTied_DynFiLM.
# Sweeps:
#   REFINE_STEPS  0 (no refinement — control)  |  10 (gradient-ascent refinement)
#   K_TESTS       4  |  8  |  16               (critic recurrent iterations at eval)
#   SEEDS         0  |  1  |  2                (one checkpoint per seed)
#
# NOTE: without refinement (refine=0) K_test has NO effect on success rate —
#       the actor network π(s,g) is fixed and never calls the critic.
#       The refine=0 row exists only as a sanity / Q-statistic check.
#       The meaningful comparison is within refine=10.
#
# Usage:
#   sbatch slurm/phase3_eval_ktest_sweep.slurm
#
# Overrides (set before sbatch or via --export):
#   TRAIN_RUN_GROUP   source run-group        (default: P3_RecurTied_DynFiLM)
#   RESTORE_EPOCH     checkpoint step         (default: 1000000)
#   EVAL_EPISODES     episodes per task       (default: 50)
#   EVAL_ON_CPU       0=GPU / 1=CPU           (default: 0; GPU needed for speed with refinement)
# ---------------------------------------------------------------------------
#SBATCH --job-name=p3_ktest
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ---------------------------------------------------------------------------
# Array sweep layout
#   idx = refine_idx + REFINE_STEPS * (ktest_idx + K_TESTS * seed_idx)
#   refine (fastest) → ktest → seed (slowest)
# ---------------------------------------------------------------------------
SEEDS=(0)
K_TESTS=(4 8 16)
REFINE_STEPS=(10)

idx="${SLURM_ARRAY_TASK_ID}"
refine_idx=$(( idx % ${#REFINE_STEPS[@]} ))
ktest_idx=$(( (idx / ${#REFINE_STEPS[@]}) % ${#K_TESTS[@]} ))
seed_idx=$(( idx / (${#REFINE_STEPS[@]} * ${#K_TESTS[@]}) ))

SEED="${SEEDS[seed_idx]}"
KTEST="${K_TESTS[ktest_idx]}"
REFINE="${REFINE_STEPS[refine_idx]}"

# ---------------------------------------------------------------------------
# Source checkpoint (K_train=4, LR-decay run)
# ---------------------------------------------------------------------------
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-P3_RecurTied_DynFiLM}"
TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_ktrain4_a01_lrd1000000"
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
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

RUN_GROUP="${RUN_GROUP:-P3_TestTimeScaling}"
EXP_NAME="sd$(printf %03d ${SEED})_ktrain4_ktest${KTEST}_ref${REFINE}"

export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="allow"

# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"
mkdir -p logs/phase_3

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------
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
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--eval_refine_steps=${REFINE}"
  "--agent=agents/crl.py"
  # Architecture flags — must match training exactly.
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_layerscale_init=1e-2"
  "--agent.critic_eval_num_iters=${KTEST}"
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
)

echo "seed=${SEED} ktest=${KTEST} refine=${REFINE} exp=${EXP_NAME} restore=${RESTORE_PATH}@${RESTORE_EPOCH}"
echo "${cmd[*]}"

# ---------------------------------------------------------------------------
# Conda bootstrap
# ---------------------------------------------------------------------------
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
