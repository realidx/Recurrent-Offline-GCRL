#!/bin/bash
#SBATCH --job-name=crl_stitch
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=logs/phase_0/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_0/slurm-%x-%A_%a.err
#SBATCH --array=0-11

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# The training script lives under `third_party/ogbench/impls`. Ensure it exists before submitting jobs.
if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh on the login node first." >&2
  exit 2
fi

if [[ ! -f "third_party/ogbench/impls/utils/log_utils.py" ]] || ! rg -q "WANDB_MODE" "third_party/ogbench/impls/utils/log_utils.py" 2>/dev/null; then
  echo "Error: OGBench checkout is missing local patches (WANDB_MODE support not found)." >&2
  echo "Fix: run ./scripts/bootstrap_ogbench.sh (CLEAN=1) on the login node, then resubmit." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

# ---- User knobs ----
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"               # e.g., /home/$USER/miniconda3/bin/conda
WANDB_MODE="${WANDB_MODE:-offline}"           # set online if compute nodes can reach wandb and you have WANDB_API_KEY
RUN_GROUP="${RUN_GROUP:-P0_CRL_AntMazeStitch}"
LOG_ROOT="${LOG_ROOT:-${SLURM_SUBMIT_DIR}/logs/phase_0}"
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

export WANDB_SILENT="${WANDB_SILENT:-true}"

# Headless MuJoCo on servers is often more reliable with EGL.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Keep datasets in a shared path (so array jobs don't re-download to $HOME).
mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

ENVS=(
  "antmaze-medium-stitch-v0"
  "antmaze-large-stitch-v0"
  "antmaze-giant-stitch-v0"
  "antmaze-teleport-stitch-v0"
)
SEEDS=(0 1 2)

idx="${SLURM_ARRAY_TASK_ID}"
env_idx=$(( idx % ${#ENVS[@]} ))
seed_idx=$(( idx / ${#ENVS[@]} ))

ENV_NAME="${ENVS[env_idx]}"
SEED="${SEEDS[seed_idx]}"

EXTRA_AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
)
if [[ "${ENV_NAME}" == "antmaze-giant-stitch-v0" ]]; then
  EXTRA_AGENT_FLAGS+=("--agent.discount=0.995")
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
  "--save_interval=${SAVE_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=agents/crl.py"
  "${EXTRA_AGENT_FLAGS[@]}"
)

cd "${IMPL_DIR}"
echo "cwd=$(pwd) WANDB_MODE=${WANDB_MODE} OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR} ${cmd[*]}"
if ! command -v "${CONDA_EXE}" >/dev/null 2>&1; then
  # Non-interactive SLURM jobs often don't have conda initialized; try common conda.sh locations.
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
  echo "Fix: load your anaconda/miniconda module, or pass CONDA_EXE=/full/path/to/conda via --export." >&2
  exit 3
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
