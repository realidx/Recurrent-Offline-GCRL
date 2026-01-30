#!/bin/bash
#SBATCH --job-name=p2_eval
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/phase_2/slurm-%x-%A.out
#SBATCH --error=logs/phase_2/slurm-%x-%A.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-offline}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P2_EvalOnly}"
EXP_NAME="${EXP_NAME:-}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
CRITIC_EVAL_NUM_ITERS="${CRITIC_EVAL_NUM_ITERS:-}"

RESTORE_PATH="${RESTORE_PATH:?Set RESTORE_PATH to the training run directory glob (must match exactly one).}"
RESTORE_EPOCH="${RESTORE_EPOCH:?Set RESTORE_EPOCH (e.g., 1000000).}"

EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"

export WANDB_SILENT="${WANDB_SILENT:-true}"

# Evaluation should be robust to flaky GPU nodes. If evaluating on CPU, force CPU backend so JAX
# never attempts CUDA initialization (some nodes can report "No visible GPU devices").
#
# Note: OGBench AntMaze environments require MuJoCo. On many clusters OSMesa is not available, so
# we default to EGL even for CPU-eval; you still submit the job to a GPU partition for MuJoCo.
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_2

if [[ -n "${RESTORE_PATH}" && "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi

cd "${IMPL_DIR}"

cmd=(
  python main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--env_name=${ENV_NAME}"
  "--train_steps=0"
  "--eval_only=1"
  "--restore_path=${RESTORE_PATH}"
  "--restore_epoch=${RESTORE_EPOCH}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--agent=agents/crl.py"
)

if [[ -n "${EXP_NAME}" ]]; then
  cmd+=("--exp_name=${EXP_NAME}")
fi

if [[ -n "${EVAL_TASKS}" ]]; then
  cmd+=("--eval_tasks=${EVAL_TASKS}")
fi

echo "cwd=$(pwd) RESTORE_PATH=${RESTORE_PATH} RESTORE_EPOCH=${RESTORE_EPOCH} ${cmd[*]}"

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

export RESTORE_PATH
"${CONDA_EXE}" run -n "${CONDA_ENV}" python - <<'PY'
import json
import os
import shlex
from pathlib import Path

restore_path = Path(os.environ["RESTORE_PATH"]).expanduser()
flags_path = restore_path / "flags.json"
if not flags_path.exists():
    raise SystemExit(f"flags.json not found at: {flags_path}")

flags = json.loads(flags_path.read_text())
agent = flags.get("agent", {}) if isinstance(flags, dict) else {}

def emit(k, v):
    if v is None:
        return
    print(f"{k}={shlex.quote(str(v))}")

emit("RESUME_SEED", flags.get("seed"))
emit("RESUME_ENV_NAME", flags.get("env_name"))
emit("RESUME_CRITIC_BACKBONE", agent.get("critic_backbone"))
emit("RESUME_CRITIC_RESNET_DEPTH", agent.get("critic_resnet_depth"))
emit("RESUME_CRITIC_RECUR_ITERS", agent.get("critic_recur_iters"))
emit("RESUME_CRITIC_BACKBONE_HIDDEN_DIM", agent.get("critic_backbone_hidden_dim"))
emit("RESUME_CRITIC_EVAL_NUM_ITERS", agent.get("critic_eval_num_iters"))
emit("RESUME_ACTOR_P_RANDOMGOAL", agent.get("actor_p_randomgoal"))
emit("RESUME_ACTOR_P_TRAJGOAL", agent.get("actor_p_trajgoal"))
emit("RESUME_ALPHA", agent.get("alpha"))
PY > /tmp/ogbench_eval_resume.env

# shellcheck disable=SC1091
source /tmp/ogbench_eval_resume.env

if [[ -n "${RESUME_ENV_NAME:-}" ]]; then
  ENV_NAME="${RESUME_ENV_NAME}"
fi

AGENT_FLAGS=()
if [[ -n "${RESUME_ACTOR_P_RANDOMGOAL:-}" ]]; then
  AGENT_FLAGS+=("--agent.actor_p_randomgoal=${RESUME_ACTOR_P_RANDOMGOAL}")
fi
if [[ -n "${RESUME_ACTOR_P_TRAJGOAL:-}" ]]; then
  AGENT_FLAGS+=("--agent.actor_p_trajgoal=${RESUME_ACTOR_P_TRAJGOAL}")
fi
if [[ -n "${RESUME_ALPHA:-}" ]]; then
  AGENT_FLAGS+=("--agent.alpha=${RESUME_ALPHA}")
fi

if [[ -n "${RESUME_CRITIC_BACKBONE:-}" ]]; then
  AGENT_FLAGS+=("--agent.critic_backbone=${RESUME_CRITIC_BACKBONE}")
fi
if [[ -n "${RESUME_CRITIC_RESNET_DEPTH:-}" ]]; then
  AGENT_FLAGS+=("--agent.critic_resnet_depth=${RESUME_CRITIC_RESNET_DEPTH}")
fi
if [[ -n "${RESUME_CRITIC_RECUR_ITERS:-}" ]]; then
  AGENT_FLAGS+=("--agent.critic_recur_iters=${RESUME_CRITIC_RECUR_ITERS}")
fi
if [[ -n "${RESUME_CRITIC_BACKBONE_HIDDEN_DIM:-}" ]]; then
  AGENT_FLAGS+=("--agent.critic_backbone_hidden_dim=${RESUME_CRITIC_BACKBONE_HIDDEN_DIM}")
fi

if [[ -n "${CRITIC_EVAL_NUM_ITERS}" ]]; then
  AGENT_FLAGS+=("--agent.critic_eval_num_iters=${CRITIC_EVAL_NUM_ITERS}")
elif [[ -n "${RESUME_CRITIC_EVAL_NUM_ITERS:-}" ]]; then
  AGENT_FLAGS+=("--agent.critic_eval_num_iters=${RESUME_CRITIC_EVAL_NUM_ITERS}")
fi

if [[ -n "${RESUME_SEED:-}" ]]; then
  # For eval logging randomness and exp naming; model params still come from restore_path.
  cmd+=("--seed=${RESUME_SEED}")
else
  cmd+=("--seed=0")
fi

cmd+=("${AGENT_FLAGS[@]}")

"${CONDA_EXE}" run -n "${CONDA_ENV}" python - <<'PY'
import jax
print("jax_devices:", jax.devices())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
