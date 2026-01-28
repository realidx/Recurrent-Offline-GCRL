#!/bin/bash
#SBATCH --job-name=crl_stitch
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-11

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

# ---- User knobs ----
CONDA_ENV="${CONDA_ENV:-recurrent}"
WANDB_MODE="${WANDB_MODE:-offline}"           # set online if compute nodes can reach wandb and you have WANDB_API_KEY
RUN_GROUP="${RUN_GROUP:-P0_CRL_AntMazeStitch}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"

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
  python third_party/ogbench/impls/main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--seed=${SEED}"
  "--env_name=${ENV_NAME}"
  "--train_steps=${TRAIN_STEPS}"
  "--log_interval=${LOG_INTERVAL}"
  "--eval_interval=${EVAL_INTERVAL}"
  "--save_interval=${SAVE_INTERVAL}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--agent=agents/crl.py"
  "${EXTRA_AGENT_FLAGS[@]}"
)

echo "WANDB_MODE=${WANDB_MODE} OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR} ${cmd[*]}"
conda run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"

