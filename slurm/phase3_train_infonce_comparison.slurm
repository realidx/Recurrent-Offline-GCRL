#!/bin/bash
#SBATCH --job-name=p3_infonce
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

# Phase 3 — InfoNCE vs Sigmoid BCE Comparison
#
# Tests whether InfoNCE (softmax cross-entropy) improves over sigmoid BCE
# for the contrastive loss. Uses the same 1×4 discrete recurrent config as
# the successful P3_RecurTied_DynFiLM baseline.
#
# Array: 3 seeds × 2 loss types = 6 jobs
#
# Usage:
#   sbatch slurm/phase3_train_infonce_comparison.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh on the login node first." >&2
  exit 2
fi

if [[ ! -f "third_party/ogbench/impls/agents/crl.py" ]] || ! grep -q "critic_backbone" "third_party/ogbench/impls/agents/crl.py" 2>/dev/null; then
  echo "Error: OGBench checkout is missing patches (agent.critic_backbone not found)." >&2
  echo "Fix: run ./scripts/bootstrap_ogbench.sh CLEAN=1 on the login node, then resubmit." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P3_InfoNCE_Comparison}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
AUTO_RESUME="${AUTO_RESUME:-1}"
EXP_NAME_SUFFIX="${EXP_NAME_SUFFIX:-}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

# Train settings
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# LR decay (cosine schedule)
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"
LR_MIN="${LR_MIN:-1e-5}"

# Evaluation during training
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_3

SEEDS=(0 1 2)
LOSS_TYPES=(infonce)
idx="${SLURM_ARRAY_TASK_ID}"
loss_idx=$(( idx % ${#LOSS_TYPES[@]} ))
seed_idx=$(( idx / ${#LOSS_TYPES[@]} ))

SEED="${SEEDS[seed_idx]}"
LOSS_TYPE="${LOSS_TYPES[loss_idx]}"

# InfoNCE settings
INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.1}"

# Deterministic exp_name
EXP_NAME_BASE="sd$(printf %03d ${SEED})_${LOSS_TYPE}_ktrain4_a01_lrd1000000"
if [[ -n "${EXP_NAME_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME:-${EXP_NAME_BASE}_${EXP_NAME_SUFFIX}}"
else
  EXP_NAME="${EXP_NAME:-${EXP_NAME_BASE}}"
fi

export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
  "--agent.batch_size=1024"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=0"
  "--agent.critic_layerscale_init=1e-2"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
  # Loss selection (default in code: bce).
  "--agent.contrastive_loss_type=${LOSS_TYPE}"
)

if [[ "${LOSS_TYPE}" == "infonce" ]]; then
  AGENT_FLAGS+=("--agent.infonce_temperature=${INFONCE_TEMPERATURE}")
fi

cd "${IMPL_DIR}"

RUN_DIR="${SAVE_DIR}/OGBench/${RUN_GROUP}/${EXP_NAME}"
if [[ -z "${RESTORE_PATH}" && "${AUTO_RESUME}" == "1" && -d "${RUN_DIR}" ]]; then
  # Resume from latest checkpoint in-place
  latest_epoch="$(
    for p in "${RUN_DIR}"/params_*.pkl; do
      [[ -e "${p}" ]] || continue
      b="$(basename "${p}")"
      b="${b#params_}"
      b="${b%.pkl}"
      if [[ "${b}" =~ ^[0-9]+$ ]]; then
        echo "${b}"
      fi
    done | sort -n | tail -1
  )"
  if [[ -n "${latest_epoch}" ]]; then
    RESTORE_PATH="${RUN_DIR}"
    RESTORE_EPOCH="${RESTORE_EPOCH:-${latest_epoch}}"
    echo "Auto-resume: RESTORE_PATH=${RESTORE_PATH} RESTORE_EPOCH=${RESTORE_EPOCH}"
  fi
fi

if [[ -n "${RESTORE_EPOCH}" && ! "${RESTORE_EPOCH}" =~ ^[0-9]+$ ]]; then
  if [[ "${RESTORE_EPOCH}" =~ params_([0-9]+)\\.pkl$ ]]; then
    echo "Warning: RESTORE_EPOCH looks like a checkpoint path; extracting epoch." >&2
    RESTORE_EPOCH="${BASH_REMATCH[1]}"
  else
    echo "Error: RESTORE_EPOCH must be an integer (e.g., 600000). Got: ${RESTORE_EPOCH}" >&2
    echo "Tip: run 'unset RESTORE_EPOCH RESTORE_PATH' before sbatch, or submit with --export=ALL,RESTORE_EPOCH=,RESTORE_PATH=" >&2
    exit 2
  fi
fi

cmd=(
  python main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--seed=${SEED}"
  "--env_name=${ENV_NAME}"
  "--train_steps=${TRAIN_STEPS}"
  "--log_interval=${LOG_INTERVAL}"
  "--validation_log_interval=${VALIDATION_LOG_INTERVAL}"
  "--eval_interval=${EVAL_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=agents/crl.py"
  "--exp_name=${EXP_NAME}"
  "${AGENT_FLAGS[@]}"
)

if [[ -n "${EVAL_TASKS}" ]]; then
  cmd+=("--eval_tasks=${EVAL_TASKS}")
fi

if [[ -n "${RESTORE_PATH}" && "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi
if [[ -n "${RESTORE_PATH}" ]]; then
  cmd+=("--restore_path=${RESTORE_PATH}")
fi
if [[ -n "${RESTORE_EPOCH}" ]]; then
  cmd+=("--restore_epoch=${RESTORE_EPOCH}")
fi

echo "Loss Test: seed=${SEED} loss=${LOSS_TYPE} temp=${INFONCE_TEMPERATURE} exp_name=${EXP_NAME} cwd=$(pwd) WANDB_MODE=${WANDB_MODE}"
echo "${cmd[*]}"

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
