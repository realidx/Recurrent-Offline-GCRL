#!/usr/bin/env bash
set -euo pipefail

# Reproduce OGBench's CRL baseline on the 4 AntMaze stitching datasets.
#
# Defaults are chosen to be "safe" locally (offline W&B). For server runs, set:
#   WANDB_MODE=online WANDB_API_KEY=... SEEDS="0 1 2 3 4" ./scripts/repro_crl_antmaze_stitch.sh
#
# You can also override training/eval cadence for quick sanity checks, e.g.:
#   TRAIN_STEPS=20000 EVAL_INTERVAL=5000 LOG_INTERVAL=1000 ./scripts/repro_crl_antmaze_stitch.sh

RUN_GROUP="${RUN_GROUP:-P0_CRL_AntMazeStitch}"
SAVE_DIR="${SAVE_DIR:-exp/}"
SEEDS="${SEEDS:-0}"

WANDB_MODE="${WANDB_MODE:-offline}"
DATASET_DIR="${DATASET_DIR:-}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_TASKS="${EVAL_TASKS:-}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

DRY_RUN="${DRY_RUN:-0}"
DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-}"

DATASETS=(
  "antmaze-medium-stitch-v0"
  "antmaze-large-stitch-v0"
  "antmaze-giant-stitch-v0"
  "antmaze-teleport-stitch-v0"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${REPO_ROOT}/third_party/ogbench/impls/main.py" ]]; then
  echo "Missing third_party/ogbench/impls; bootstrapping OGBench checkout..." >&2
  "${REPO_ROOT}/scripts/bootstrap_ogbench.sh"
fi

if [[ "${SAVE_DIR}" == "exp/" ]]; then
  # Default to a repo-root output directory even though we run from `third_party/ogbench/impls`.
  SAVE_DIR="${REPO_ROOT}/exp/"
fi

if [[ -n "${DATASETS_OVERRIDE}" ]]; then
  read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi

cd "${REPO_ROOT}/third_party/ogbench/impls"

for seed in ${SEEDS}; do
  for env_name in "${DATASETS[@]}"; do
    extra_agent_flags=(
      "--agent.actor_p_randomgoal=0.5"
      "--agent.actor_p_trajgoal=0.5"
      "--agent.alpha=0.1"
    )
    # Matches `hyperparameters.sh`: giant uses a different discount.
    if [[ "${env_name}" == "antmaze-giant-stitch-v0" ]]; then
      extra_agent_flags+=("--agent.discount=0.995")
    fi

    cmd=(
      python main.py
      "--run_group=${RUN_GROUP}"
      "--save_dir=${SAVE_DIR}"
      "--seed=${seed}"
      "--env_name=${env_name}"
      "--train_steps=${TRAIN_STEPS}"
      "--log_interval=${LOG_INTERVAL}"
      "--eval_interval=${EVAL_INTERVAL}"
      "--save_interval=${SAVE_INTERVAL}"
      "--eval_episodes=${EVAL_EPISODES}"
      "--video_episodes=${VIDEO_EPISODES}"
      "--agent=agents/crl.py"
      "${extra_agent_flags[@]}"
    )
    if [[ -n "${EVAL_TASKS}" ]]; then
      cmd+=("--eval_tasks=${EVAL_TASKS}")
    fi

    echo "WANDB_MODE=${WANDB_MODE} ${cmd[*]}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      env_prefix=(WANDB_MODE="${WANDB_MODE}")
      if [[ -n "${DATASET_DIR}" ]]; then
        env_prefix+=(OGBENCH_DATASET_DIR="${DATASET_DIR}")
      fi
      "${env_prefix[@]}" "${cmd[@]}"
    fi
  done
done
