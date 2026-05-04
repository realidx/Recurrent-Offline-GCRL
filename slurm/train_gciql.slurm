#!/bin/bash
#SBATCH --job-name=p4_gciql
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=logs/phase_5/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_5/slurm-%x-%A_%a.err
#SBATCH --array=0

# Generic OGBench GCIQL trainer (array by seed).
#
# Examples:
#   sbatch --array=0-2 --export="ALL,SEEDS='0 1 2',ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=GCIQL_ALS_MLP,WANDB_MODE=online" \
#     slurm/train_gciql.slurm
#
#   sbatch --array=0-2 --export="ALL,SEEDS='0 1 2',ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=GCIQL_ALS_Recur,VALUE_BACKBONE=recur_tied,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_USE_STEP_INFO=1,WANDB_MODE=online" \
#     slurm/train_gciql.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh first." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
AGENT_PATH="agents/gciql.py"
if [[ ! -f "${IMPL_DIR}/${AGENT_PATH}" ]]; then
  echo "Error: missing ${IMPL_DIR}/${AGENT_PATH}." >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-/app/apps/miniforge3/25.3.1/bin/conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P4_GCIQL}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

TRAIN_STEPS="${TRAIN_STEPS:-}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SAVE_STEPS="${SAVE_STEPS:-600000:800000:900000:1000000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# GCIQL defaults. If left empty, task-family defaults below follow OGBench hyperparameters.sh.
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-}"
DISCOUNT="${DISCOUNT:-}"
TAU="${TAU:-0.005}"
EXPECTILE="${EXPECTILE:-0.9}"
ALPHA="${ALPHA:-}"
ACTOR_LOSS="${ACTOR_LOSS:-}"
CONST_STD="${CONST_STD:-1}"
DISCRETE="${DISCRETE:-}"
ENCODER="${ENCODER:-}"
P_AUG="${P_AUG:-}"
LAYER_NORM="${LAYER_NORM:-1}"

VALUE_P_CURGOAL="${VALUE_P_CURGOAL:-0.2}"
VALUE_P_TRAJGOAL="${VALUE_P_TRAJGOAL:-0.5}"
VALUE_P_RANDOMGOAL="${VALUE_P_RANDOMGOAL:-0.3}"
ACTOR_P_CURGOAL="${ACTOR_P_CURGOAL:-0.0}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-}"

ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-512x512x512}"
VALUE_HIDDEN_DIMS="${VALUE_HIDDEN_DIMS:-512x512x512}"

# Recurrent scalar value/critic options.
VALUE_BACKBONE="${VALUE_BACKBONE:-${BACKBONE:-mlp}}"
CRITIC_BACKBONE="${CRITIC_BACKBONE:-${BACKBONE:-mlp}}"
KTRAIN="${KTRAIN:-4}"
RECUR_NUM_DENSE_LAYERS="${RECUR_NUM_DENSE_LAYERS:-2}"
RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-${KTRAIN}}"
RECUR_LN_MODE="${RECUR_LN_MODE:-per_layer_final}"
RECUR_BLOCK_TYPE="${RECUR_BLOCK_TYPE:-dense}"
RECUR_USE_STEP_INFO="${RECUR_USE_STEP_INFO:-1}"
RECUR_USE_LAYERSCALE="${RECUR_USE_LAYERSCALE:-1}"
RECUR_SWIGLU_PRE_LN="${RECUR_SWIGLU_PRE_LN:-1}"
LAYERSCALE_INIT="${LAYERSCALE_INIT:-1e-2}"

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

# OGBench GCIQL task-family defaults. Explicit exported values always win.
if [[ -z "${TRAIN_STEPS}" ]]; then
  if [[ "${ENV_NAME}" == visual-* || "${ENV_NAME}" == powderworld-* ]]; then
    TRAIN_STEPS="500000"
  else
    TRAIN_STEPS="1000000"
  fi
fi

if [[ -z "${BATCH_SIZE}" ]]; then
  if [[ "${ENV_NAME}" == visual-* || "${ENV_NAME}" == powderworld-* ]]; then
    BATCH_SIZE="256"
  else
    BATCH_SIZE="1024"
  fi
fi

if [[ -z "${ALPHA}" ]]; then
  case "${ENV_NAME}" in
    pointmaze-*) ALPHA="0.003" ;;
    antmaze-*-explore-v0|visual-antmaze-*-explore-v0) ALPHA="0.01" ;;
    antmaze-*|visual-antmaze-*) ALPHA="0.3" ;;
    humanoidmaze-*|visual-humanoidmaze-*) ALPHA="0.1" ;;
    antsoccer-*) ALPHA="0.1" ;;
    cube-*-play-v0|scene-play-v0|puzzle-*-play-v0|visual-cube-*-play-v0|visual-scene-play-v0|visual-puzzle-*-play-v0) ALPHA="1.0" ;;
    cube-*-noisy-v0|scene-noisy-v0|puzzle-*-noisy-v0|visual-cube-*-noisy-v0|visual-scene-noisy-v0|visual-puzzle-*-noisy-v0) ALPHA="0.03" ;;
    powderworld-*) ALPHA="3.0"; ACTOR_LOSS="${ACTOR_LOSS:-awr}"; DISCRETE="${DISCRETE:-1}" ;;
    *) ALPHA="0.3" ;;
  esac
fi

if [[ -z "${DISCOUNT}" ]]; then
  if [[ "${ENV_NAME}" == *giant* || "${ENV_NAME}" == humanoidmaze-* || "${ENV_NAME}" == visual-humanoidmaze-* ]]; then
    DISCOUNT="0.995"
  else
    DISCOUNT="0.99"
  fi
fi

if [[ -z "${ACTOR_P_TRAJGOAL}" || -z "${ACTOR_P_RANDOMGOAL}" ]]; then
  if [[ "${ENV_NAME}" == *-stitch-v0 ]]; then
    ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-0.5}"
    ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.5}"
  elif [[ "${ENV_NAME}" == *-explore-v0 ]]; then
    ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-0.0}"
    ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-1.0}"
  else
    ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-1.0}"
    ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.0}"
  fi
fi

if [[ "${ENV_NAME}" == visual-* && -z "${ENCODER}" ]]; then
  ENCODER="impala_small"
fi
if [[ "${ENV_NAME}" == visual-cube-* || "${ENV_NAME}" == visual-scene-* || "${ENV_NAME}" == visual-puzzle-* ]]; then
  P_AUG="${P_AUG:-0.5}"
fi
if [[ "${ENV_NAME}" == powderworld-* && -z "${ENCODER}" ]]; then
  ENCODER="impala_small"
fi
if [[ "${ENV_NAME}" == powderworld-* ]]; then
  EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.3}"
else
  EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-}"
fi
ACTOR_LOSS="${ACTOR_LOSS:-ddpgbc}"
DISCRETE="${DISCRETE:-0}"
P_AUG="${P_AUG:-0.0}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
mkdir -p "${DATASET_DIR}" "${SAVE_DIR}" logs/phase_5
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"
BB_TAG="v${VALUE_BACKBONE}_q${CRITIC_BACKBONE}"
DEFAULT_EXP_NAME="sd$(printf %03d "${SEED}")_${ENV_TAG}_gciql_${BB_TAG}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX:-${DEFAULT_EXP_NAME}}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.lr=${LR}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.discount=${DISCOUNT}"
  "--agent.tau=${TAU}"
  "--agent.expectile=${EXPECTILE}"
  "--agent.alpha=${ALPHA}"
  "--agent.actor_loss=${ACTOR_LOSS}"
  "--agent.const_std=${CONST_STD}"
  "--agent.discrete=${DISCRETE}"
  "--agent.layer_norm=${LAYER_NORM}"
  "--agent.p_aug=${P_AUG}"
  "--agent.value_p_curgoal=${VALUE_P_CURGOAL}"
  "--agent.value_p_trajgoal=${VALUE_P_TRAJGOAL}"
  "--agent.value_p_randomgoal=${VALUE_P_RANDOMGOAL}"
  "--agent.actor_p_curgoal=${ACTOR_P_CURGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.value_backbone=${VALUE_BACKBONE}"
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
)

append_hidden_dims() {
  local flag_name="$1"
  local dims_spec="$2"
  if [[ "${dims_spec}" == *","* || "${dims_spec}" == *"("* || "${dims_spec}" == *")"* ]]; then
    echo "Error: ${flag_name}='${dims_spec}' uses commas/parentheses. Use x-separated format like 512x512x512." >&2
    exit 2
  fi
  local dims=()
  IFS='x' read -r -a dims <<< "${dims_spec}"
  for dim in "${dims[@]}"; do
    if [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
      echo "Error: ${flag_name} contains non-integer token '${dim}'." >&2
      exit 2
    fi
    AGENT_FLAGS+=("--agent.${flag_name}=${dim}")
  done
}

append_recurrent_flags() {
  local module_name="$1"
  local backbone="$2"
  case "${backbone}" in
    mlp)
      ;;
    recur_tied)
      if [[ ! "${RECUR_LN_MODE}" =~ ^(pre_loop|per_layer|per_layer_final|pre_loop_per_layer)$ ]]; then
        echo "Error: RECUR_LN_MODE='${RECUR_LN_MODE}' is invalid." >&2
        exit 2
      fi
      if [[ ! "${RECUR_BLOCK_TYPE}" =~ ^(dense|swiglu)$ ]]; then
        echo "Error: RECUR_BLOCK_TYPE='${RECUR_BLOCK_TYPE}' is invalid." >&2
        exit 2
      fi
      if [[ "${RECUR_BLOCK_TYPE}" == "swiglu" && "${RECUR_NUM_DENSE_LAYERS}" != "2" ]]; then
        echo "Error: RECUR_BLOCK_TYPE=swiglu requires RECUR_NUM_DENSE_LAYERS=2." >&2
        exit 2
      fi
      AGENT_FLAGS+=(
        "--agent.${module_name}_recur_iters=${KTRAIN}"
        "--agent.${module_name}_recur_num_dense_layers=${RECUR_NUM_DENSE_LAYERS}"
        "--agent.${module_name}_recur_block_type=${RECUR_BLOCK_TYPE}"
        "--agent.${module_name}_recur_max_iters=${RECUR_MAX_ITERS}"
        "--agent.${module_name}_recur_ln_mode=${RECUR_LN_MODE}"
        "--agent.${module_name}_recur_use_step_info=${RECUR_USE_STEP_INFO}"
        "--agent.${module_name}_recur_use_layerscale=${RECUR_USE_LAYERSCALE}"
        "--agent.${module_name}_recur_swiglu_pre_ln=${RECUR_SWIGLU_PRE_LN}"
        "--agent.${module_name}_layerscale_init=${LAYERSCALE_INIT}"
      )
      ;;
    *)
      echo "Error: ${module_name^^}_BACKBONE='${backbone}' is invalid. Use mlp or recur_tied." >&2
      exit 2
      ;;
  esac
}

append_hidden_dims "actor_hidden_dims" "${ACTOR_HIDDEN_DIMS}"
append_hidden_dims "value_hidden_dims" "${VALUE_HIDDEN_DIMS}"
append_recurrent_flags "value" "${VALUE_BACKBONE}"
append_recurrent_flags "critic" "${CRITIC_BACKBONE}"

if [[ -n "${ENCODER}" ]]; then
  AGENT_FLAGS+=("--agent.encoder=${ENCODER}")
fi

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
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=${AGENT_PATH}"
  "--exp_name=${EXP_NAME}"
  "${AGENT_FLAGS[@]}"
)

if [[ -n "${EVAL_TASKS}" ]]; then
  cmd+=("--eval_tasks=${EVAL_TASKS}")
fi
if [[ -n "${SAVE_STEPS}" ]]; then
  cmd+=("--save_steps=${SAVE_STEPS}")
fi
if [[ -n "${EVAL_TEMPERATURE}" ]]; then
  cmd+=("--eval_temperature=${EVAL_TEMPERATURE}")
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

echo "GCIQL: env=${ENV_NAME} seed=${SEED} value_backbone=${VALUE_BACKBONE} critic_backbone=${CRITIC_BACKBONE} alpha=${ALPHA} discount=${DISCOUNT} actor_mix=${ACTOR_P_TRAJGOAL}/${ACTOR_P_RANDOMGOAL} run_group=${RUN_GROUP} exp=${EXP_NAME}"
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
