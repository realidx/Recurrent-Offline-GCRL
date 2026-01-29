#!/bin/bash
#SBATCH --job-name=p2_critics
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_2/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_2/slurm-%x-%A_%a.err
#SBATCH --array=0-20

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh on the login node first." >&2
  exit 2
fi

if [[ ! -f "third_party/ogbench/impls/agents/crl.py" ]] || ! grep -q "critic_backbone" "third_party/ogbench/impls/agents/crl.py" 2>/dev/null; then
  echo "Error: OGBench checkout is missing patches (agent.critic_backbone not found)." >&2
  echo "Fix: run ./scripts/bootstrap_ogbench.sh (CLEAN=1) on the login node, then resubmit." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P2_CriticVariants}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_2

SEEDS=(0 1 2)

# Configs:
#   0: baseline mlp
#   1-3: resnet depth 3/6/12
#   4-6: recur_tied iters 3/6/12
CONFIGS=(
  "mlp"
  "resnet:3"
  "resnet:6"
  "resnet:12"
  "recur_tied:3"
  "recur_tied:6"
  "recur_tied:12"
)

idx="${SLURM_ARRAY_TASK_ID}"
cfg_idx=$(( idx % ${#CONFIGS[@]} ))
seed_idx=$(( idx / ${#CONFIGS[@]} ))

CFG="${CONFIGS[cfg_idx]}"
SEED="${SEEDS[seed_idx]}"

CRITIC_BACKBONE="${CFG%%:*}"
CRITIC_DEPTH_OR_ITERS="${CFG#*:}"

AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
  "--agent.critic_recur_max_iters=32"
  "--agent.critic_layerscale_init=1e-2"
)

if [[ "${CRITIC_BACKBONE}" == "resnet" ]]; then
  AGENT_FLAGS+=("--agent.critic_resnet_depth=${CRITIC_DEPTH_OR_ITERS}")
elif [[ "${CRITIC_BACKBONE}" == "recur_tied" ]]; then
  AGENT_FLAGS+=("--agent.critic_recur_iters=${CRITIC_DEPTH_OR_ITERS}")
fi

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
  "--save_interval=${SAVE_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=agents/crl.py"
  "${AGENT_FLAGS[@]}"
)

echo "cfg=${CFG} seed=${SEED} cwd=$(pwd) WANDB_MODE=${WANDB_MODE} OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR} ${cmd[*]}"

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

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"

