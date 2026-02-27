#!/bin/bash
#SBATCH --job-name=p4_crl_generic
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_4/slurm-%x-%A_%a.err
#SBATCH --array=0-2

# Generic OGBench CRL trainer (array by seed).
#
# Examples:
# 1) CRL-MLP baseline on humanoidmaze-large-navigate (seed 0 only):
#    sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=humanoidmaze-large-navigate-v0,RUN_GROUP=P4_HumanoidLargeNav_MLP,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,WANDB_MODE=online" \
#      slurm/phase4_train_crl_generic_array.slurm
#
# 2) CRL recurrent 1x4 on humanoidmaze-large-stitch (3 seeds):
#    sbatch --array=0-2 --export="ALL,SEEDS='0 1 2',ENV_NAME=humanoidmaze-large-stitch-v0,RUN_GROUP=P4_HumanoidLargeStitch_Recur1x4,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_MAX_ITERS=24,SINUSOIDAL=1,TIED_LN=1,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" \
#      slurm/phase4_train_crl_generic_array.slurm
#
# 3) CRL recurrent stacked 4-block x 4-iter:
#    sbatch --array=0-2 --export="ALL,SEEDS='0 1 2',ENV_NAME=humanoidmaze-large-stitch-v0,RUN_GROUP=P4_HumanoidLargeStitch_Recur4x4,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_BLOCKS=4,RECUR_MAX_ITERS=24,SINUSOIDAL=1,TIED_LN=1,ALPHA=0.1,DISCOUNT=0.995,WANDB_MODE=online" \
#      slurm/phase4_train_crl_generic_array.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh first." >&2
  exit 2
fi

if [[ ! -f "third_party/ogbench/impls/agents/crl.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/agents/crl.py." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P4_GenericCRL}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

# Train and eval cadence.
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# Core CRL defaults.
ALPHA="${ALPHA:-0.1}"
DISCOUNT="${DISCOUNT:-0.99}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
ACTOR_P_CURGOAL="${ACTOR_P_CURGOAL:-0.0}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-1.0}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.0}"
ACTOR_GEOM_SAMPLE="${ACTOR_GEOM_SAMPLE:-}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"
LR_MIN="${LR_MIN:-1e-5}"

# Critic backbone selection.
CRITIC_BACKBONE="${CRITIC_BACKBONE:-mlp}"   # mlp | resnet | recur_tied | partial_tied
CRITIC_BACKBONE_HIDDEN_DIM="${CRITIC_BACKBONE_HIDDEN_DIM:-0}"
LAYERSCALE_INIT="${LAYERSCALE_INIT:-1e-2}"

# ResNet options.
RESNET_DEPTH="${RESNET_DEPTH:-3}"

# recurrent tied options.
KTRAIN="${KTRAIN:-4}"
RECUR_NUM_BLOCKS="${RECUR_NUM_BLOCKS:-1}"
RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-24}"
SINUSOIDAL="${SINUSOIDAL:-1}"
TIED_LN="${TIED_LN:-1}"

# partial tied options.
PARTIAL_GROUPS="${PARTIAL_GROUPS:-2}"
PARTIAL_ITERS_PER_GROUP="${PARTIAL_ITERS_PER_GROUP:-2}"
PARTIAL_PER_GROUP_FILM="${PARTIAL_PER_GROUP_FILM:-0}"
PARTIAL_CYCLE_STEP_PARAMS="${PARTIAL_CYCLE_STEP_PARAMS:-1}"
PARTIAL_BLOCK_SCHEDULE="${PARTIAL_BLOCK_SCHEDULE:-grouped}"   # grouped | stack

# Seeds controlled via env var: SEEDS="0 1 2"
SEEDS="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "${SEEDS}"
if [[ "${#SEED_LIST[@]}" -eq 0 ]]; then
  echo "Error: SEEDS is empty." >&2
  exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#SEED_LIST[@]} )); then
  echo "Error: SLURM_ARRAY_TASK_ID=${IDX} out of range for SEEDS='${SEEDS}'." >&2
  echo "Set --array to 0-$(( ${#SEED_LIST[@]} - 1 )) or change SEEDS." >&2
  exit 2
fi
SEED="${SEED_LIST[IDX]}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
mkdir -p "${DATASET_DIR}" logs/phase_4
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"
BB_TAG="${CRITIC_BACKBONE}"
EXP_NAME="${EXP_NAME:-sd$(printf %03d "${SEED}")_${ENV_TAG}_${BB_TAG}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.alpha=${ALPHA}"
  "--agent.discount=${DISCOUNT}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.actor_p_curgoal=${ACTOR_P_CURGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
  "--agent.critic_layerscale_init=${LAYERSCALE_INIT}"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
)

if [[ -n "${ACTOR_GEOM_SAMPLE}" ]]; then
  AGENT_FLAGS+=("--agent.actor_geom_sample=${ACTOR_GEOM_SAMPLE}")
fi

if [[ "${CRITIC_BACKBONE_HIDDEN_DIM}" != "0" ]]; then
  AGENT_FLAGS+=("--agent.critic_backbone_hidden_dim=${CRITIC_BACKBONE_HIDDEN_DIM}")
fi

case "${CRITIC_BACKBONE}" in
  mlp)
    ;;
  resnet)
    AGENT_FLAGS+=("--agent.critic_resnet_depth=${RESNET_DEPTH}")
    ;;
  recur_tied)
    AGENT_FLAGS+=(
      "--agent.critic_recur_iters=${KTRAIN}"
      "--agent.critic_recur_num_blocks=${RECUR_NUM_BLOCKS}"
      "--agent.critic_recur_max_iters=${RECUR_MAX_ITERS}"
      "--agent.critic_recur_sinusoidal=${SINUSOIDAL}"
      "--agent.critic_recur_tied_ln=${TIED_LN}"
    )
    ;;
  partial_tied)
    AGENT_FLAGS+=(
      "--agent.critic_partial_groups=${PARTIAL_GROUPS}"
      "--agent.critic_partial_iters_per_group=${PARTIAL_ITERS_PER_GROUP}"
      "--agent.critic_partial_per_group_film=${PARTIAL_PER_GROUP_FILM}"
      "--agent.critic_partial_cycle_step_params=${PARTIAL_CYCLE_STEP_PARAMS}"
      "--agent.critic_partial_block_schedule=${PARTIAL_BLOCK_SCHEDULE}"
      "--agent.critic_recur_max_iters=${RECUR_MAX_ITERS}"
      "--agent.critic_recur_sinusoidal=${SINUSOIDAL}"
      "--agent.critic_recur_tied_ln=${TIED_LN}"
    )
    ;;
  *)
    echo "Error: unsupported CRITIC_BACKBONE='${CRITIC_BACKBONE}'." >&2
    exit 2
    ;;
esac

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

if [[ -n "${RESTORE_EPOCH}" && ! "${RESTORE_EPOCH}" =~ ^[0-9]+$ ]]; then
  if [[ "${RESTORE_EPOCH}" =~ params_([0-9]+)\.pkl$ ]]; then
    RESTORE_EPOCH="${BASH_REMATCH[1]}"
  else
    echo "Error: RESTORE_EPOCH must be an integer. Got '${RESTORE_EPOCH}'." >&2
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

echo "GENERIC CRL: env=${ENV_NAME} seed=${SEED} backbone=${CRITIC_BACKBONE} alpha=${ALPHA} discount=${DISCOUNT} run_group=${RUN_GROUP} exp=${EXP_NAME}"
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
  echo "Error: '${CONDA_EXE}' not found." >&2
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
