#!/bin/bash
#SBATCH --job-name=p4_crl_generic
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --output=logs/phase_5/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_5/slurm-%x-%A_%a.err
#SBATCH --array=0

# OGBench CRL trainer (array by seed).
#
# Examples:
# 1) CRL-MLP baseline on humanoidmaze-large-navigate (seed 0 only):
#    sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=humanoidmaze-large-navigate-v0,RUN_GROUP=P4_HumanoidLargeNav_MLP,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,WANDB_MODE=online" \
#      slurm/phase4_train_crl_generic_array.slurm
#
# 2) CRL recurrent critic on humanoidmaze-large-stitch (m=2 SwiGLU layers, K=4 iterations):
#    sbatch --array=0-2 --export="ALL,SEEDS='0 1 2',ENV_NAME=humanoidmaze-large-stitch-v0,RUN_GROUP=P4_HumanoidLargeStitch_Recur,CRITIC_BACKBONE=recur,M=2,K=4,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" \
#      slurm/train_crl.slurm
#
# 4) MLP critic + 32-layer actor without typing 32 copies of 512:
#    sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=P4_AntMaze_MLP_Actor32L,CRITIC_BACKBONE=mlp,ACTOR_NUM_LAYERS=32,ACTOR_LAYER_WIDTH=512,ALPHA=0.1,DISCOUNT=0.99,WANDB_MODE=online" \
#      slurm/phase4_train_crl_generic_array.slurm
#
# Explicit x-separated dims still work and take precedence if both forms are set.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "impls/main.py" ]]; then
  echo "Error: missing impls/main.py." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/impls"
AGENT_PATH="${AGENT_PATH:-agents/crl.py}"
if [[ ! -f "${IMPL_DIR}/${AGENT_PATH}" ]]; then
  echo "Error: missing ${IMPL_DIR}/${AGENT_PATH}." >&2
  exit 2
fi

AGENT_NAME="$(basename "${AGENT_PATH}" .py)"
if [[ "${AGENT_NAME}" != "crl" ]]; then
  echo "Error: AGENT_PATH='${AGENT_PATH}' is unsupported by this submission artifact." >&2
  exit 2
fi

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
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DUMP_EVAL_TRAJS="${DUMP_EVAL_TRAJS:-0}"
EVAL_TRAJ_DIR="${EVAL_TRAJ_DIR:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SAVE_STEPS="${SAVE_STEPS:-600000:800000:900000:1000000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# Core CRL defaults.
LR="${LR:-3e-4}"
ALPHA="${ALPHA:-0.1}"
DISCOUNT="${DISCOUNT:-0.99}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
CONTRASTIVE_LOSS_TYPE="${CONTRASTIVE_LOSS_TYPE:-bce}" # bce | infonce
INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.1}"
ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-}"   # Safe format for sbatch --export: e.g., 512x512x512x512
VALUE_HIDDEN_DIMS="${VALUE_HIDDEN_DIMS:-}"   # Safe format for critic/value MLP dims: e.g., 512x512x512x512x512
ACTOR_NUM_LAYERS="${ACTOR_NUM_LAYERS:-}"     # Shortcut: e.g., 32
ACTOR_LAYER_WIDTH="${ACTOR_LAYER_WIDTH:-512}" # Shortcut width used with ACTOR_NUM_LAYERS
VALUE_NUM_LAYERS="${VALUE_NUM_LAYERS:-}"     # Shortcut: e.g., 32
VALUE_LAYER_WIDTH="${VALUE_LAYER_WIDTH:-512}" # Shortcut width used with VALUE_NUM_LAYERS
ACTOR_BACKBONE="${ACTOR_BACKBONE:-mlp}"      # mlp | residual_mlp
ACTOR_P_CURGOAL="${ACTOR_P_CURGOAL:-0.0}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-1.0}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.0}"
ACTOR_GEOM_SAMPLE="${ACTOR_GEOM_SAMPLE:-}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"
LR_MIN="${LR_MIN:-1e-5}"
LAYER_NORM="${LAYER_NORM:-1}"
LATENT_DIM="${LATENT_DIM:-512}"
ACTOR_LOSS="${ACTOR_LOSS:-}"
CONST_STD="${CONST_STD:-1}"
DISCRETE="${DISCRETE:-0}"
ENCODER="${ENCODER:-}"

# Critic backbone selection.
CRITIC_BACKBONE="${CRITIC_BACKBONE:-mlp}"   # mlp | recur
LAYERSCALE_INIT="${LAYERSCALE_INIT:-1e-2}"

# Recurrent critic options: m residual SwiGLU layers per iteration, K iterations.
K="${K:-4}"
M="${M:-2}"

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
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi
mkdir -p "${DATASET_DIR}" "${SAVE_DIR}" logs/phase_5
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"
BB_TAG="${CRITIC_BACKBONE}"
if [[ "${CRITIC_BACKBONE}" == "recur" ]]; then
  BB_TAG="recur_m${M}_K${K}"
fi
DEFAULT_EXP_NAME="sd$(printf %03d "${SEED}")_${ENV_TAG}_${BB_TAG}"
EXP_NAME="${EXP_NAME:-${EXP_PREFIX:-${DEFAULT_EXP_NAME}}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.lr=${LR}"
  "--agent.alpha=${ALPHA}"
  "--agent.discount=${DISCOUNT}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.actor_p_curgoal=${ACTOR_P_CURGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
  "--agent.critic_layerscale_init=${LAYERSCALE_INIT}"
  "--agent.layer_norm=${LAYER_NORM}"
  "--agent.latent_dim=${LATENT_DIM}"
  "--agent.const_std=${CONST_STD}"
  "--agent.discrete=${DISCRETE}"
)

if [[ "${AGENT_NAME}" == "crl" ]]; then
  if [[ ! "${CONTRASTIVE_LOSS_TYPE}" =~ ^(bce|infonce)$ ]]; then
    echo "Error: CONTRASTIVE_LOSS_TYPE='${CONTRASTIVE_LOSS_TYPE}' is invalid. Use bce or infonce." >&2
    exit 2
  fi
  if ! [[ "${INFONCE_TEMPERATURE}" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "Error: INFONCE_TEMPERATURE='${INFONCE_TEMPERATURE}' is not a positive number." >&2
    exit 2
  fi
  AGENT_FLAGS+=(
    "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
    "--agent.lr_min=${LR_MIN}"
    "--agent.actor_backbone=${ACTOR_BACKBONE}"
    "--agent.contrastive_loss_type=${CONTRASTIVE_LOSS_TYPE}"
    "--agent.infonce_temperature=${INFONCE_TEMPERATURE}"
  )
fi

if [[ -n "${ACTOR_LOSS}" ]]; then
  AGENT_FLAGS+=("--agent.actor_loss=${ACTOR_LOSS}")
fi

if [[ -n "${ENCODER}" ]]; then
  AGENT_FLAGS+=("--agent.encoder=${ENCODER}")
fi

if [[ -n "${ACTOR_NUM_LAYERS}" && ! "${ACTOR_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ACTOR_NUM_LAYERS='${ACTOR_NUM_LAYERS}' is not a positive integer." >&2
  exit 2
fi
if [[ ! "${ACTOR_LAYER_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ACTOR_LAYER_WIDTH='${ACTOR_LAYER_WIDTH}' is not a positive integer." >&2
  exit 2
fi

if [[ -z "${ACTOR_HIDDEN_DIMS}" && -n "${ACTOR_NUM_LAYERS}" ]]; then
  ACTOR_DIMS=()
  for ((i = 0; i < ACTOR_NUM_LAYERS; i++)); do
    ACTOR_DIMS+=("${ACTOR_LAYER_WIDTH}")
  done
  ACTOR_HIDDEN_DIMS="$(IFS=x; echo "${ACTOR_DIMS[*]}")"
fi

if [[ -n "${ACTOR_HIDDEN_DIMS}" ]]; then
  if [[ "${ACTOR_HIDDEN_DIMS}" == *","* || "${ACTOR_HIDDEN_DIMS}" == *"("* || "${ACTOR_HIDDEN_DIMS}" == *")"* ]]; then
    echo "Error: ACTOR_HIDDEN_DIMS='${ACTOR_HIDDEN_DIMS}' uses commas/parentheses, which breaks sbatch --export parsing." >&2
    echo "Use 'x'-separated format instead, e.g. ACTOR_HIDDEN_DIMS=512x512x512x512" >&2
    exit 2
  fi
  IFS='x' read -r -a ACTOR_DIMS <<< "${ACTOR_HIDDEN_DIMS}"
  if [[ "${#ACTOR_DIMS[@]}" -eq 0 ]]; then
    echo "Error: ACTOR_HIDDEN_DIMS is set but no dimensions were parsed." >&2
    exit 2
  fi
  for dim in "${ACTOR_DIMS[@]}"; do
    if [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
      echo "Error: ACTOR_HIDDEN_DIMS contains non-integer token '${dim}'." >&2
      echo "Use format like 512x512x512x512" >&2
      exit 2
    fi
    AGENT_FLAGS+=("--agent.actor_hidden_dims=${dim}")
  done
fi

if [[ -z "${VALUE_HIDDEN_DIMS}" && -n "${VALUE_NUM_LAYERS}" ]]; then
  if [[ ! "${VALUE_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: VALUE_NUM_LAYERS='${VALUE_NUM_LAYERS}' is not a positive integer." >&2
    exit 2
  fi
  if [[ ! "${VALUE_LAYER_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: VALUE_LAYER_WIDTH='${VALUE_LAYER_WIDTH}' is not a positive integer." >&2
    exit 2
  fi
  VALUE_DIMS=()
  for ((i = 0; i < VALUE_NUM_LAYERS; i++)); do
    VALUE_DIMS+=("${VALUE_LAYER_WIDTH}")
  done
  VALUE_HIDDEN_DIMS="$(IFS=x; echo "${VALUE_DIMS[*]}")"
fi

if [[ -n "${VALUE_HIDDEN_DIMS}" ]]; then
  if [[ "${VALUE_HIDDEN_DIMS}" == *","* || "${VALUE_HIDDEN_DIMS}" == *"("* || "${VALUE_HIDDEN_DIMS}" == *")"* ]]; then
    echo "Error: VALUE_HIDDEN_DIMS='${VALUE_HIDDEN_DIMS}' uses commas/parentheses, which breaks sbatch --export parsing." >&2
    echo "Use 'x'-separated format instead, e.g. VALUE_HIDDEN_DIMS=512x512x512x512" >&2
    exit 2
  fi
  IFS='x' read -r -a VALUE_DIMS <<< "${VALUE_HIDDEN_DIMS}"
  if [[ "${#VALUE_DIMS[@]}" -eq 0 ]]; then
    echo "Error: VALUE_HIDDEN_DIMS is set but no dimensions were parsed." >&2
    exit 2
  fi
  for dim in "${VALUE_DIMS[@]}"; do
    if [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
      echo "Error: VALUE_HIDDEN_DIMS contains non-integer token '${dim}'." >&2
      echo "Use format like 512x512x512x512" >&2
      exit 2
    fi
    AGENT_FLAGS+=("--agent.value_hidden_dims=${dim}")
  done
fi

if [[ -n "${ACTOR_GEOM_SAMPLE}" ]]; then
  AGENT_FLAGS+=("--agent.actor_geom_sample=${ACTOR_GEOM_SAMPLE}")
fi

if [[ "${AGENT_NAME}" == "crl" ]]; then
  case "${ACTOR_BACKBONE}" in
    mlp)
      ;;
    residual_mlp)
      ;;
    *)
      echo "Error: unsupported ACTOR_BACKBONE='${ACTOR_BACKBONE}'." >&2
      exit 2
      ;;
  esac
fi

case "${CRITIC_BACKBONE}" in
  mlp)
    ;;
  recur)
    if [[ ! "${K}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Error: K='${K}' is not a positive integer." >&2
      exit 2
    fi
    if [[ ! "${M}" =~ ^[1-9][0-9]*$ ]]; then
      echo "Error: M='${M}' is not a positive integer." >&2
      exit 2
    fi
    AGENT_FLAGS+=(
      "--agent.critic_k=${K}"
      "--agent.critic_m=${M}"
    )
    ;;
  *)
    echo "Error: unsupported CRITIC_BACKBONE='${CRITIC_BACKBONE}'. Use mlp or recur." >&2
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
  "--dump_eval_trajs=${DUMP_EVAL_TRAJS}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--agent=${AGENT_PATH}"
  "--exp_name=${EXP_NAME}"
  "${AGENT_FLAGS[@]}"
)

if [[ -n "${EVAL_TASKS}" ]]; then
  cmd+=("--eval_tasks=${EVAL_TASKS}")
fi

if [[ -n "${EVAL_TRAJ_DIR}" ]]; then
  cmd+=("--eval_traj_dir=${EVAL_TRAJ_DIR}")
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

echo "CRL: env=${ENV_NAME} seed=${SEED} critic_backbone=${CRITIC_BACKBONE} M=${M} K=${K} actor_backbone=${ACTOR_BACKBONE} alpha=${ALPHA} discount=${DISCOUNT} run_group=${RUN_GROUP} exp=${EXP_NAME}"
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
