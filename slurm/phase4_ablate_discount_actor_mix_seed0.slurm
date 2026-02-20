#!/bin/bash
#SBATCH --job-name=p4_ablate_d_actor
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_4/slurm-%x-%A_%a.err
#SBATCH --array=0-1

# Phase 4 — Ablation (seed 0 only)
#
# Compare two settings while keeping everything else hard-coded:
#   0) discount=0.995, actor goals: random=0.5, traj=0.5
#   1) discount=0.99,  actor goals: random=0.0, traj=1.0
#
# Default ENV_NAME is antmaze-giant-navigate-v0 (override via --export if needed).
#
# Submit:
#   sbatch --export=ALL,WANDB_MODE=online slurm/phase4_ablate_discount_actor_mix_seed0.slurm
#
# Optional overrides:
#   sbatch --export=ALL,ENV_NAME=pointmaze-giant-navigate-v0,WANDB_MODE=online slurm/phase4_ablate_discount_actor_mix_seed0.slurm

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

ENV_NAME="${ENV_NAME:-antmaze-giant-navigate-v0}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

# Hard-coded training/eval settings (keep comparisons clean).
SEED=0
RUN_GROUP="${RUN_GROUP:-P4_Ablate_DiscountActorMix}"

TRAIN_STEPS=1000000
LOG_INTERVAL=20000
VALIDATION_LOG_INTERVAL=100000
SAVE_INTERVAL=200000
DISABLE_TQDM=1

EVAL_INTERVAL=200000
EVAL_TASKS=5
EVAL_EPISODES=20
VIDEO_EPISODES=0
EVAL_ON_CPU=0

# Hard-coded optimizer schedule (match your earlier "sin_k4" run style).
LR_DECAY_STEPS=1000000
LR_MIN=1e-5

# Hard-coded critic backbone (your current best recurrent 1×4).
ALPHA=0.1
KTRAIN=4
RECUR_MAX_ITERS=24
SINUSOIDAL=1
TIED_LN=1
LAYERSCALE_INIT=1e-2

# Two ablation settings (selected by SLURM_ARRAY_TASK_ID).
DISCOUNTS=(0.995 0.99)
ACTOR_P_RANDOMGOALS=(0.5 0.0)
ACTOR_P_TRAJGOALS=(0.5 1.0)

cfg="${SLURM_ARRAY_TASK_ID}"
DISCOUNT="${DISCOUNTS[cfg]}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOALS[cfg]}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOALS[cfg]}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_4

ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"
DISC_TAG="${DISCOUNT//./}"
AR_TAG="${ACTOR_P_RANDOMGOAL//./}"
AT_TAG="${ACTOR_P_TRAJGOAL//./}"
EXP_NAME="${EXP_NAME:-sd000_${ENV_TAG}_d${DISC_TAG}_ar${AR_TAG}_at${AT_TAG}_recur1x4}"

export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.alpha=${ALPHA}"
  "--agent.discount=${DISCOUNT}"
  "--agent.batch_size=1024"
  "--agent.actor_p_curgoal=0.0"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=${KTRAIN}"
  "--agent.critic_recur_max_iters=${RECUR_MAX_ITERS}"
  "--agent.critic_recur_tied_ln=${TIED_LN}"
  "--agent.critic_recur_sinusoidal=${SINUSOIDAL}"
  "--agent.critic_layerscale_init=${LAYERSCALE_INIT}"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
)

cd "${IMPL_DIR}"

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
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=agents/crl.py"
  "--exp_name=${EXP_NAME}"
  "${AGENT_FLAGS[@]}"
)

echo "PHASE4 ABLATE: env=${ENV_NAME} seed=${SEED} discount=${DISCOUNT} actor_p_randomgoal=${ACTOR_P_RANDOMGOAL} actor_p_trajgoal=${ACTOR_P_TRAJGOAL} exp_name=${EXP_NAME} WANDB_MODE=${WANDB_MODE}"
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

