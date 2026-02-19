#!/bin/bash
# Phase 3 — Actor depth scaling under fixed critic (recur_tied 1×4)
#
# Purpose: test whether increasing actor depth improves performance when using the tied-iterated critic.
# This intentionally changes ONLY the actor MLP depth; critic + training budget stay fixed.
#
# Defaults:
# - Env: antmaze-large-stitch-v0
# - Critic: recur_tied, K=4, sinusoidal=1, tied LN
# - Actor widths: 512; depths: {3, 4}; seeds: {0,1,2}
#
# Usage:
#   sbatch slurm/phase3_train_actor_scale_recur_tied_1x4_antmaze_large_stitch_array.slurm
#
# Override examples:
#   # Just seed=1 and depth=4
#   sbatch --array=0 --export=ALL,SEEDS_LIST="1",ACTOR_DEPTHS_LIST="4" slurm/phase3_train_actor_scale_recur_tied_1x4_antmaze_large_stitch_array.slurm

#SBATCH --job-name=p3_actor_scale
#SBATCH --partition=gpu-long
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_3/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_3/slurm-%x-%A_%a.err
#SBATCH --array=0-2

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
RUN_GROUP="${RUN_GROUP:-P3_ActorScale_Recur1x4}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

# Sweep controls
SEEDS_LIST="${SEEDS_LIST:-0 1 2}"
ACTOR_DEPTHS_LIST="${ACTOR_DEPTHS_LIST:-4}"
ACTOR_WIDTH="${ACTOR_WIDTH:-512}"

read -r -a SEEDS <<< "${SEEDS_LIST}"
read -r -a DEPTHS <<< "${ACTOR_DEPTHS_LIST}"

idx="${SLURM_ARRAY_TASK_ID}"
total=$(( ${#SEEDS[@]} * ${#DEPTHS[@]} ))
if (( idx < 0 || idx >= total )); then
  echo "Error: SLURM_ARRAY_TASK_ID=${idx} out of range for total=${total} (seeds=${#SEEDS[@]} depths=${#DEPTHS[@]})." >&2
  echo "Fix: resubmit with --array=0-$((total - 1)) or adjust *_LIST variables." >&2
  exit 2
fi

# Layout: depth (fastest) → seed (slowest)
depth_idx=$(( idx % ${#DEPTHS[@]} ))
seed_idx=$(( idx / ${#DEPTHS[@]} ))

SEED="${SEEDS[seed_idx]}"
ACTOR_DEPTH="${DEPTHS[depth_idx]}"

# Critic config (fixed for this experiment)
KTRAIN="${KTRAIN:-4}"
RECUR_SINUSOIDAL="${RECUR_SINUSOIDAL:-1}"
RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-16}"

# Train settings
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# LR decay (cosine schedule)
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"
LR_MIN="${LR_MIN:-1e-5}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_3

# Build actor hidden_dims tuple string without spaces so config override parsing is robust.
ACTOR_HIDDEN_DIMS="("
for ((i=0; i<ACTOR_DEPTH; i++)); do
  if (( i > 0 )); then
    ACTOR_HIDDEN_DIMS+=","
  fi
  ACTOR_HIDDEN_DIMS+="${ACTOR_WIDTH}"
done
ACTOR_HIDDEN_DIMS+=")"

EXP_NAME="${EXP_NAME:-sd$(printf %03d ${SEED})_actorD${ACTOR_DEPTH}_w${ACTOR_WIDTH}_recur1x${KTRAIN}_sin${RECUR_SINUSOIDAL}}"

export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.actor_hidden_dims=${ACTOR_HIDDEN_DIMS}"
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
  "--agent.batch_size=1024"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=${KTRAIN}"
  "--agent.critic_recur_max_iters=${RECUR_MAX_ITERS}"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=${RECUR_SINUSOIDAL}"
  "--agent.critic_layerscale_init=1e-2"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
)

cd "${IMPL_DIR}"

RUN_DIR="${SAVE_DIR}/OGBench/${RUN_GROUP}/${EXP_NAME}"
if [[ -z "${RESTORE_PATH}" && "${AUTO_RESUME}" == "1" && -d "${RUN_DIR}" ]]; then
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
  "--eval_tasks=${EVAL_TASKS}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=agents/crl.py"
  "--exp_name=${EXP_NAME}"
  "${AGENT_FLAGS[@]}"
)

if [[ -n "${RESTORE_PATH}" && "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi
if [[ -n "${RESTORE_PATH}" ]]; then
  cmd+=("--restore_path=${RESTORE_PATH}")
fi
if [[ -n "${RESTORE_EPOCH}" ]]; then
  cmd+=("--restore_epoch=${RESTORE_EPOCH}")
fi

echo "ACTOR-SCALE: seed=${SEED} actor_hidden_dims=${ACTOR_HIDDEN_DIMS} exp_name=${EXP_NAME} env=${ENV_NAME} WANDB_MODE=${WANDB_MODE} ${cmd[*]}"

# Conda bootstrap for non-interactive jobs
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

