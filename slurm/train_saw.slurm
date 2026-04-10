#!/bin/bash
#SBATCH --job-name=p4_saw
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_4/slurm-%x-%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh first." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
if [[ ! -f "${IMPL_DIR}/agents/saw.py" ]]; then
  echo "Error: missing ${IMPL_DIR}/agents/saw.py." >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-SAW_AntLargeStitch}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
EVAL_TASKS="${EVAL_TASKS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SAVE_STEPS="${SAVE_STEPS:-600000:800000:900000:1000000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DISCOUNT="${DISCOUNT:-0.99}"
TAU="${TAU:-0.005}"
EXPECTILE="${EXPECTILE:-0.7}"
LOW_ALPHA="${LOW_ALPHA:-3.0}"
AWR_ALPHA="${AWR_ALPHA:-${HIGH_ALPHA:-3.0}}"
KL_ALPHA="${KL_ALPHA:-3.0}"
SUBGOAL_STEPS="${SUBGOAL_STEPS:-25}"
REP_DIM="${REP_DIM:-10}"
LAYER_NORM="${LAYER_NORM:-1}"
CONST_STD="${CONST_STD:-1}"
DISCRETE="${DISCRETE:-0}"
SHARE_GOAL_REP="${SHARE_GOAL_REP:-0}"
GC_NEGATIVE="${GC_NEGATIVE:-1}"
VALUE_GEOM_SAMPLE="${VALUE_GEOM_SAMPLE:-1}"
ACTOR_GEOM_SAMPLE="${ACTOR_GEOM_SAMPLE:-0}"
ACTOR_GEOM_DISCOUNT="${ACTOR_GEOM_DISCOUNT:-0.99}"
P_AUG="${P_AUG:-0.0}"
ENCODER="${ENCODER:-}"
FRAME_STACK="${FRAME_STACK:-}"

ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-512x512x512}"
LOW_ACTOR_HIDDEN_DIMS="${LOW_ACTOR_HIDDEN_DIMS:-512x512x512}"
VALUE_HIDDEN_DIMS="${VALUE_HIDDEN_DIMS:-512x512x512}"
VALUE_BACKBONE="${VALUE_BACKBONE:-mlp}"
VALUE_BACKBONE_HIDDEN_DIM="${VALUE_BACKBONE_HIDDEN_DIM:-0}"
VALUE_NUM_DENSE_LAYERS="${VALUE_NUM_DENSE_LAYERS:-4}"
VALUE_RESIDUAL_SPAN="${VALUE_RESIDUAL_SPAN:-4}"
VALUE_RESIDUAL_USE_SKIP="${VALUE_RESIDUAL_USE_SKIP:-1}"
VALUE_RECUR_ITERS="${VALUE_RECUR_ITERS:-4}"
VALUE_RECUR_NUM_DENSE_LAYERS="${VALUE_RECUR_NUM_DENSE_LAYERS:-2}"
VALUE_RECUR_MAX_ITERS="${VALUE_RECUR_MAX_ITERS:-16}"
VALUE_RECUR_LN_MODE="${VALUE_RECUR_LN_MODE:-per_layer_final}"
VALUE_RECUR_USE_STEP_INFO="${VALUE_RECUR_USE_STEP_INFO:-0}"
VALUE_RECUR_USE_FILM_CONTEXT="${VALUE_RECUR_USE_FILM_CONTEXT:-0}"
VALUE_RECUR_FILM_MODE="${VALUE_RECUR_FILM_MODE:-}"
VALUE_RECUR_USE_ACT="${VALUE_RECUR_USE_ACT:-0}"
VALUE_RECUR_ACT_EPSILON="${VALUE_RECUR_ACT_EPSILON:-0.01}"
VALUE_RECUR_ACT_MIN_ITERS="${VALUE_RECUR_ACT_MIN_ITERS:-1}"
VALUE_RECUR_ACT_PONDER_WEIGHT="${VALUE_RECUR_ACT_PONDER_WEIGHT:-0.0}"
VALUE_LAYERSCALE_INIT="${VALUE_LAYERSCALE_INIT:-1e-2}"
ACTOR_BACKBONE="${ACTOR_BACKBONE:-mlp}"
ACTOR_BACKBONE_HIDDEN_DIM="${ACTOR_BACKBONE_HIDDEN_DIM:-0}"
ACTOR_RESNET_NUM_BLOCKS="${ACTOR_RESNET_NUM_BLOCKS:-3}"
ACTOR_NUM_DENSE_LAYERS="${ACTOR_NUM_DENSE_LAYERS:-4}"
ACTOR_RESIDUAL_SPAN="${ACTOR_RESIDUAL_SPAN:-4}"
ACTOR_RESIDUAL_USE_SKIP="${ACTOR_RESIDUAL_USE_SKIP:-1}"
ACTOR_NUM_BLOCKS="${ACTOR_NUM_BLOCKS:-2}"
ACTOR_BLOCK_NUM_LAYERS="${ACTOR_BLOCK_NUM_LAYERS:-4}"
ACTOR_USE_INNER_LAYER_NORM="${ACTOR_USE_INNER_LAYER_NORM:-1}"
ACTOR_PRENORM_NORM_MODE="${ACTOR_PRENORM_NORM_MODE:-pre_block}"
ACTOR_RECUR_ITERS="${ACTOR_RECUR_ITERS:-4}"
ACTOR_RECUR_NUM_DENSE_LAYERS="${ACTOR_RECUR_NUM_DENSE_LAYERS:-2}"
ACTOR_RECUR_MAX_ITERS="${ACTOR_RECUR_MAX_ITERS:-16}"
ACTOR_RECUR_USE_STEP_INFO="${ACTOR_RECUR_USE_STEP_INFO:-0}"
ACTOR_RECUR_USE_FILM_CONTEXT="${ACTOR_RECUR_USE_FILM_CONTEXT:-0}"
ACTOR_PARTIAL_NUM_GROUPS="${ACTOR_PARTIAL_NUM_GROUPS:-2}"
ACTOR_PARTIAL_ITERS_PER_GROUP="${ACTOR_PARTIAL_ITERS_PER_GROUP:-2}"
ACTOR_PARTIAL_PER_GROUP_FILM="${ACTOR_PARTIAL_PER_GROUP_FILM:-0}"
ACTOR_PARTIAL_CYCLE_STEP_PARAMS="${ACTOR_PARTIAL_CYCLE_STEP_PARAMS:-1}"
ACTOR_PARTIAL_BLOCK_SCHEDULE="${ACTOR_PARTIAL_BLOCK_SCHEDULE:-grouped}"
ACTOR_LAYERSCALE_INIT="${ACTOR_LAYERSCALE_INIT:-1e-2}"
LOW_ACTOR_BACKBONE="${LOW_ACTOR_BACKBONE:-mlp}"
LOW_ACTOR_BACKBONE_HIDDEN_DIM="${LOW_ACTOR_BACKBONE_HIDDEN_DIM:-0}"
LOW_ACTOR_RESNET_NUM_BLOCKS="${LOW_ACTOR_RESNET_NUM_BLOCKS:-3}"
LOW_ACTOR_NUM_DENSE_LAYERS="${LOW_ACTOR_NUM_DENSE_LAYERS:-4}"
LOW_ACTOR_RESIDUAL_SPAN="${LOW_ACTOR_RESIDUAL_SPAN:-4}"
LOW_ACTOR_RESIDUAL_USE_SKIP="${LOW_ACTOR_RESIDUAL_USE_SKIP:-1}"
LOW_ACTOR_NUM_BLOCKS="${LOW_ACTOR_NUM_BLOCKS:-2}"
LOW_ACTOR_BLOCK_NUM_LAYERS="${LOW_ACTOR_BLOCK_NUM_LAYERS:-4}"
LOW_ACTOR_USE_INNER_LAYER_NORM="${LOW_ACTOR_USE_INNER_LAYER_NORM:-1}"
LOW_ACTOR_PRENORM_NORM_MODE="${LOW_ACTOR_PRENORM_NORM_MODE:-pre_block}"
LOW_ACTOR_RECUR_ITERS="${LOW_ACTOR_RECUR_ITERS:-4}"
LOW_ACTOR_RECUR_NUM_DENSE_LAYERS="${LOW_ACTOR_RECUR_NUM_DENSE_LAYERS:-2}"
LOW_ACTOR_RECUR_MAX_ITERS="${LOW_ACTOR_RECUR_MAX_ITERS:-16}"
LOW_ACTOR_RECUR_USE_STEP_INFO="${LOW_ACTOR_RECUR_USE_STEP_INFO:-0}"
LOW_ACTOR_RECUR_USE_FILM_CONTEXT="${LOW_ACTOR_RECUR_USE_FILM_CONTEXT:-0}"
LOW_ACTOR_PARTIAL_NUM_GROUPS="${LOW_ACTOR_PARTIAL_NUM_GROUPS:-2}"
LOW_ACTOR_PARTIAL_ITERS_PER_GROUP="${LOW_ACTOR_PARTIAL_ITERS_PER_GROUP:-2}"
LOW_ACTOR_PARTIAL_PER_GROUP_FILM="${LOW_ACTOR_PARTIAL_PER_GROUP_FILM:-0}"
LOW_ACTOR_PARTIAL_CYCLE_STEP_PARAMS="${LOW_ACTOR_PARTIAL_CYCLE_STEP_PARAMS:-1}"
LOW_ACTOR_PARTIAL_BLOCK_SCHEDULE="${LOW_ACTOR_PARTIAL_BLOCK_SCHEDULE:-grouped}"
LOW_ACTOR_LAYERSCALE_INIT="${LOW_ACTOR_LAYERSCALE_INIT:-1e-2}"

VALUE_P_CURGOAL="${VALUE_P_CURGOAL:-0.2}"
VALUE_P_TRAJGOAL="${VALUE_P_TRAJGOAL:-0.5}"
VALUE_P_RANDOMGOAL="${VALUE_P_RANDOMGOAL:-0.3}"
ACTOR_P_CURGOAL="${ACTOR_P_CURGOAL:-0.0}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-1.0}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.0}"

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
mkdir -p "${DATASET_DIR}" logs/phase_4
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

if [[ -n "${RESTORE_PATH}" && "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi

ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"
EXP_CORE="${EXP_PREFIX:-${ENV_TAG}_saw}"
if [[ "${EXP_CORE}" =~ ^sd[0-9][0-9][0-9]_(.*)$ ]]; then
  EXP_CORE="${BASH_REMATCH[1]}"
fi
DEFAULT_EXP_NAME="sd$(printf %03d "${SEED}")_${EXP_CORE}"
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"

append_hidden_dims() {
  local flag_name="$1"
  local dims_spec="$2"
  if [[ "${dims_spec}" == *","* || "${dims_spec}" == *"("* || "${dims_spec}" == *")"* ]]; then
    echo "Error: ${flag_name}='${dims_spec}' uses commas/parentheses. Use x-separated format like 512x512x512." >&2
    exit 2
  fi

  local dims=()
  IFS='x' read -r -a dims <<< "${dims_spec}"
  if [[ "${#dims[@]}" -eq 0 ]]; then
    echo "Error: ${flag_name} did not parse any dimensions." >&2
    exit 2
  fi

  for dim in "${dims[@]}"; do
    if [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
      echo "Error: ${flag_name} contains non-integer token '${dim}'." >&2
      exit 2
    fi
    AGENT_FLAGS+=("--agent.${flag_name}=${dim}")
  done
}

AGENT_FLAGS=(
  "--agent.lr=${LR}"
  "--agent.batch_size=${BATCH_SIZE}"
  "--agent.discount=${DISCOUNT}"
  "--agent.tau=${TAU}"
  "--agent.expectile=${EXPECTILE}"
  "--agent.low_alpha=${LOW_ALPHA}"
  "--agent.awr_alpha=${AWR_ALPHA}"
  "--agent.kl_alpha=${KL_ALPHA}"
  "--agent.subgoal_steps=${SUBGOAL_STEPS}"
  "--agent.rep_dim=${REP_DIM}"
  "--agent.layer_norm=${LAYER_NORM}"
  "--agent.const_std=${CONST_STD}"
  "--agent.discrete=${DISCRETE}"
  "--agent.share_goal_rep=${SHARE_GOAL_REP}"
  "--agent.gc_negative=${GC_NEGATIVE}"
  "--agent.value_geom_sample=${VALUE_GEOM_SAMPLE}"
  "--agent.actor_geom_sample=${ACTOR_GEOM_SAMPLE}"
  "--agent.actor_geom_discount=${ACTOR_GEOM_DISCOUNT}"
  "--agent.p_aug=${P_AUG}"
  "--agent.value_p_curgoal=${VALUE_P_CURGOAL}"
  "--agent.value_p_trajgoal=${VALUE_P_TRAJGOAL}"
  "--agent.value_p_randomgoal=${VALUE_P_RANDOMGOAL}"
  "--agent.actor_p_curgoal=${ACTOR_P_CURGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.value_backbone=${VALUE_BACKBONE}"
  "--agent.actor_backbone=${ACTOR_BACKBONE}"
  "--agent.low_actor_backbone=${LOW_ACTOR_BACKBONE}"
)

append_hidden_dims "actor_hidden_dims" "${ACTOR_HIDDEN_DIMS}"
append_hidden_dims "low_actor_hidden_dims" "${LOW_ACTOR_HIDDEN_DIMS}"
append_hidden_dims "value_hidden_dims" "${VALUE_HIDDEN_DIMS}"

if [[ -n "${ENCODER}" ]]; then
  AGENT_FLAGS+=("--agent.encoder=${ENCODER}")
fi

if [[ "${VALUE_BACKBONE_HIDDEN_DIM}" != "0" ]]; then
  if [[ ! "${VALUE_BACKBONE_HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: VALUE_BACKBONE_HIDDEN_DIM='${VALUE_BACKBONE_HIDDEN_DIM}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.value_backbone_hidden_dim=${VALUE_BACKBONE_HIDDEN_DIM}")
fi

if [[ "${ACTOR_BACKBONE_HIDDEN_DIM}" != "0" ]]; then
  if [[ ! "${ACTOR_BACKBONE_HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ACTOR_BACKBONE_HIDDEN_DIM='${ACTOR_BACKBONE_HIDDEN_DIM}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.actor_backbone_hidden_dim=${ACTOR_BACKBONE_HIDDEN_DIM}")
fi

if [[ "${LOW_ACTOR_BACKBONE_HIDDEN_DIM}" != "0" ]]; then
  if [[ ! "${LOW_ACTOR_BACKBONE_HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: LOW_ACTOR_BACKBONE_HIDDEN_DIM='${LOW_ACTOR_BACKBONE_HIDDEN_DIM}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.low_actor_backbone_hidden_dim=${LOW_ACTOR_BACKBONE_HIDDEN_DIM}")
fi

case "${VALUE_BACKBONE}" in
  mlp)
    ;;
  residual_dense|dense_ln)
    AGENT_FLAGS+=(
      "--agent.value_num_dense_layers=${VALUE_NUM_DENSE_LAYERS}"
      "--agent.value_residual_span=${VALUE_RESIDUAL_SPAN}"
      "--agent.value_residual_use_skip=${VALUE_RESIDUAL_USE_SKIP}"
    )
    ;;
  recur_tied)
    if [[ ! "${VALUE_RECUR_LN_MODE}" =~ ^(pre_loop|per_layer|per_layer_final|pre_loop_per_layer)$ ]]; then
      echo "Error: VALUE_RECUR_LN_MODE='${VALUE_RECUR_LN_MODE}' is invalid. Use pre_loop, per_layer, per_layer_final, or pre_loop_per_layer." >&2
      exit 2
    fi
    if [[ -n "${VALUE_RECUR_FILM_MODE}" && ! "${VALUE_RECUR_FILM_MODE}" =~ ^(hidden|context|hidden_context)$ ]]; then
      echo "Error: VALUE_RECUR_FILM_MODE='${VALUE_RECUR_FILM_MODE}' is invalid. Use hidden, context, or hidden_context." >&2
      exit 2
    fi
    AGENT_FLAGS+=(
      "--agent.value_recur_iters=${VALUE_RECUR_ITERS}"
      "--agent.value_recur_num_dense_layers=${VALUE_RECUR_NUM_DENSE_LAYERS}"
      "--agent.value_recur_max_iters=${VALUE_RECUR_MAX_ITERS}"
      "--agent.value_recur_ln_mode=${VALUE_RECUR_LN_MODE}"
      "--agent.value_recur_use_step_info=${VALUE_RECUR_USE_STEP_INFO}"
      "--agent.value_recur_use_film_context=${VALUE_RECUR_USE_FILM_CONTEXT}"
      "--agent.value_recur_use_act=${VALUE_RECUR_USE_ACT}"
      "--agent.value_recur_act_epsilon=${VALUE_RECUR_ACT_EPSILON}"
      "--agent.value_recur_act_min_iters=${VALUE_RECUR_ACT_MIN_ITERS}"
      "--agent.value_recur_act_ponder_weight=${VALUE_RECUR_ACT_PONDER_WEIGHT}"
      "--agent.value_layerscale_init=${VALUE_LAYERSCALE_INIT}"
    )
    if [[ -n "${VALUE_RECUR_FILM_MODE}" ]]; then
      AGENT_FLAGS+=("--agent.value_recur_film_mode=${VALUE_RECUR_FILM_MODE}")
    fi
    ;;
  *)
    echo "Error: VALUE_BACKBONE='${VALUE_BACKBONE}' is invalid. Use mlp, residual_dense, dense_ln, or recur_tied." >&2
    exit 2
    ;;
esac

case "${ACTOR_BACKBONE}" in
  mlp)
    ;;
  resnet)
    AGENT_FLAGS+=("--agent.actor_resnet_num_blocks=${ACTOR_RESNET_NUM_BLOCKS}")
    ;;
  residual_dense|dense_ln)
    AGENT_FLAGS+=(
      "--agent.actor_num_dense_layers=${ACTOR_NUM_DENSE_LAYERS}"
      "--agent.actor_residual_span=${ACTOR_RESIDUAL_SPAN}"
      "--agent.actor_residual_use_skip=${ACTOR_RESIDUAL_USE_SKIP}"
    )
    ;;
  prenorm_residual)
    AGENT_FLAGS+=(
      "--agent.actor_num_blocks=${ACTOR_NUM_BLOCKS}"
      "--agent.actor_block_num_layers=${ACTOR_BLOCK_NUM_LAYERS}"
      "--agent.actor_use_inner_layer_norm=${ACTOR_USE_INNER_LAYER_NORM}"
      "--agent.actor_prenorm_norm_mode=${ACTOR_PRENORM_NORM_MODE}"
    )
    ;;
  recur_tied)
    AGENT_FLAGS+=(
      "--agent.actor_recur_iters=${ACTOR_RECUR_ITERS}"
      "--agent.actor_recur_num_dense_layers=${ACTOR_RECUR_NUM_DENSE_LAYERS}"
      "--agent.actor_recur_max_iters=${ACTOR_RECUR_MAX_ITERS}"
      "--agent.actor_recur_use_step_info=${ACTOR_RECUR_USE_STEP_INFO}"
      "--agent.actor_recur_use_film_context=${ACTOR_RECUR_USE_FILM_CONTEXT}"
      "--agent.actor_layerscale_init=${ACTOR_LAYERSCALE_INIT}"
    )
    ;;
  partial_tied)
    AGENT_FLAGS+=(
      "--agent.actor_recur_max_iters=${ACTOR_RECUR_MAX_ITERS}"
      "--agent.actor_partial_num_groups=${ACTOR_PARTIAL_NUM_GROUPS}"
      "--agent.actor_partial_iters_per_group=${ACTOR_PARTIAL_ITERS_PER_GROUP}"
      "--agent.actor_partial_per_group_film=${ACTOR_PARTIAL_PER_GROUP_FILM}"
      "--agent.actor_partial_cycle_step_params=${ACTOR_PARTIAL_CYCLE_STEP_PARAMS}"
      "--agent.actor_partial_block_schedule=${ACTOR_PARTIAL_BLOCK_SCHEDULE}"
      "--agent.actor_layerscale_init=${ACTOR_LAYERSCALE_INIT}"
    )
    ;;
  *)
    echo "Error: ACTOR_BACKBONE='${ACTOR_BACKBONE}' is invalid. Use mlp, resnet, residual_dense, dense_ln, prenorm_residual, recur_tied, or partial_tied." >&2
    exit 2
    ;;
esac

case "${LOW_ACTOR_BACKBONE}" in
  mlp)
    ;;
  resnet)
    AGENT_FLAGS+=("--agent.low_actor_resnet_num_blocks=${LOW_ACTOR_RESNET_NUM_BLOCKS}")
    ;;
  residual_dense|dense_ln)
    AGENT_FLAGS+=(
      "--agent.low_actor_num_dense_layers=${LOW_ACTOR_NUM_DENSE_LAYERS}"
      "--agent.low_actor_residual_span=${LOW_ACTOR_RESIDUAL_SPAN}"
      "--agent.low_actor_residual_use_skip=${LOW_ACTOR_RESIDUAL_USE_SKIP}"
    )
    ;;
  prenorm_residual)
    AGENT_FLAGS+=(
      "--agent.low_actor_num_blocks=${LOW_ACTOR_NUM_BLOCKS}"
      "--agent.low_actor_block_num_layers=${LOW_ACTOR_BLOCK_NUM_LAYERS}"
      "--agent.low_actor_use_inner_layer_norm=${LOW_ACTOR_USE_INNER_LAYER_NORM}"
      "--agent.low_actor_prenorm_norm_mode=${LOW_ACTOR_PRENORM_NORM_MODE}"
    )
    ;;
  recur_tied)
    AGENT_FLAGS+=(
      "--agent.low_actor_recur_iters=${LOW_ACTOR_RECUR_ITERS}"
      "--agent.low_actor_recur_num_dense_layers=${LOW_ACTOR_RECUR_NUM_DENSE_LAYERS}"
      "--agent.low_actor_recur_max_iters=${LOW_ACTOR_RECUR_MAX_ITERS}"
      "--agent.low_actor_recur_use_step_info=${LOW_ACTOR_RECUR_USE_STEP_INFO}"
      "--agent.low_actor_recur_use_film_context=${LOW_ACTOR_RECUR_USE_FILM_CONTEXT}"
      "--agent.low_actor_layerscale_init=${LOW_ACTOR_LAYERSCALE_INIT}"
    )
    ;;
  partial_tied)
    AGENT_FLAGS+=(
      "--agent.low_actor_recur_max_iters=${LOW_ACTOR_RECUR_MAX_ITERS}"
      "--agent.low_actor_partial_num_groups=${LOW_ACTOR_PARTIAL_NUM_GROUPS}"
      "--agent.low_actor_partial_iters_per_group=${LOW_ACTOR_PARTIAL_ITERS_PER_GROUP}"
      "--agent.low_actor_partial_per_group_film=${LOW_ACTOR_PARTIAL_PER_GROUP_FILM}"
      "--agent.low_actor_partial_cycle_step_params=${LOW_ACTOR_PARTIAL_CYCLE_STEP_PARAMS}"
      "--agent.low_actor_partial_block_schedule=${LOW_ACTOR_PARTIAL_BLOCK_SCHEDULE}"
      "--agent.low_actor_layerscale_init=${LOW_ACTOR_LAYERSCALE_INIT}"
    )
    ;;
  *)
    echo "Error: LOW_ACTOR_BACKBONE='${LOW_ACTOR_BACKBONE}' is invalid. Use mlp, resnet, residual_dense, dense_ln, prenorm_residual, recur_tied, or partial_tied." >&2
    exit 2
    ;;
esac

if [[ -n "${FRAME_STACK}" ]]; then
  if [[ ! "${FRAME_STACK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: FRAME_STACK='${FRAME_STACK}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.frame_stack=${FRAME_STACK}")
fi

CMD=(
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
  "--exp_name=${EXP_NAME}"
  "--agent=agents/saw.py"
  "${AGENT_FLAGS[@]}"
)

if [[ -n "${EVAL_TASKS}" ]]; then
  CMD+=("--eval_tasks=${EVAL_TASKS}")
fi

if [[ -n "${SAVE_STEPS}" ]]; then
  CMD+=("--save_steps=${SAVE_STEPS}")
fi

if [[ -n "${RESTORE_PATH}" ]]; then
  CMD+=("--restore_path=${RESTORE_PATH}")
fi

if [[ -n "${RESTORE_EPOCH}" ]]; then
  CMD+=("--restore_epoch=${RESTORE_EPOCH}")
fi

cd "${IMPL_DIR}"
echo "cwd=$(pwd) WANDB_MODE=${WANDB_MODE} OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR} ${CMD[*]}"

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
  echo "Fix: load your anaconda/miniconda module, or pass CONDA_EXE=/full/path/to/conda via --export." >&2
  exit 3
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-} HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} JAX_PLATFORMS=${JAX_PLATFORMS:-<unset>}"

"${CONDA_EXE}" run -n "${CONDA_ENV}" python - <<'PY'
import os
import jax

print("env CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("jax default backend:", jax.default_backend())
print("jax devices:", jax.devices())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${CMD[@]}"
