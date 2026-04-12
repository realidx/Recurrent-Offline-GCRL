#!/bin/bash
#SBATCH --job-name=p4_cgivl
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
if [[ ! -f "${IMPL_DIR}/agents/cgivl.py" ]]; then
  echo "Error: missing ${IMPL_DIR}/agents/cgivl.py." >&2
  echo "This usually means third_party/ogbench was cloned but local patches were not re-applied on this machine." >&2
  echo "Run on the login node after pulling:" >&2
  echo "  ./scripts/bootstrap_ogbench.sh CLEAN=1" >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-CGIVL_AntLargeStitch}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100000}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-0}"
EVAL_WARMUP_STEPS="${EVAL_WARMUP_STEPS:-0}"
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
HIGH_ALPHA="${HIGH_ALPHA:-3.0}"
SUBGOAL_STEPS="${SUBGOAL_STEPS:-25}"
REP_DIM="${REP_DIM:-10}"
LAYER_NORM="${LAYER_NORM:-1}"
CONST_STD="${CONST_STD:-1}"
DISCRETE="${DISCRETE:-0}"
LOW_ACTOR_REP_GRAD="${LOW_ACTOR_REP_GRAD:-0}"
GC_NEGATIVE="${GC_NEGATIVE:-1}"
P_AUG="${P_AUG:-0.0}"
INIT_GOAL_REP="${INIT_GOAL_REP:-0}"
USE_SUBGOALS="${USE_SUBGOALS:-1}"
MIN_D="${MIN_D:-0.0}"
NEG_EPS="${NEG_EPS:-0.001}"
V_DISTILL_EPS="${V_DISTILL_EPS:-0.01}"
ENCODER="${ENCODER:-}"
GD_ENCODER="${GD_ENCODER:-}"
FRAME_STACK="${FRAME_STACK:-}"

ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-512x512x512}"
VALUE_HIDDEN_DIMS="${VALUE_HIDDEN_DIMS:-512x512x512}"
VALUE_BACKBONE="${VALUE_BACKBONE:-mlp}"
VALUE_RECUR_ITERS="${VALUE_RECUR_ITERS:-4}"
VALUE_RECUR_NUM_DENSE_LAYERS="${VALUE_RECUR_NUM_DENSE_LAYERS:-2}"
VALUE_RECUR_MAX_ITERS="${VALUE_RECUR_MAX_ITERS:-4}"
VALUE_RECUR_LN_MODE="${VALUE_RECUR_LN_MODE:-per_layer_final}"
VALUE_RECUR_USE_STEP_INFO="${VALUE_RECUR_USE_STEP_INFO:-0}"
VALUE_RECUR_SINUSOIDAL="${VALUE_RECUR_SINUSOIDAL:-0}"
VALUE_RECUR_USE_FILM="${VALUE_RECUR_USE_FILM:-1}"
VALUE_RECUR_FILM_MODE="${VALUE_RECUR_FILM_MODE:-hidden}"
VALUE_RECUR_USE_LAYERSCALE="${VALUE_RECUR_USE_LAYERSCALE:-1}"
VALUE_RECUR_USE_ACT="${VALUE_RECUR_USE_ACT:-0}"
VALUE_RECUR_ACT_EPSILON="${VALUE_RECUR_ACT_EPSILON:-0.01}"
VALUE_RECUR_ACT_MIN_ITERS="${VALUE_RECUR_ACT_MIN_ITERS:-1}"
VALUE_LAYERSCALE_INIT="${VALUE_LAYERSCALE_INIT:-1e-2}"
GOALDISTANCE_HIDDEN_DIMS="${GOALDISTANCE_HIDDEN_DIMS:-512x512x512}"
GOALDISTANCE_LATENT_DIM="${GOALDISTANCE_LATENT_DIM:-512}"
GOALDISTANCE_LAYER_NORM="${GOALDISTANCE_LAYER_NORM:-1}"

VALUE_P_CURGOAL="${VALUE_P_CURGOAL:-0.2}"
VALUE_P_TRAJGOAL="${VALUE_P_TRAJGOAL:-0.5}"
VALUE_P_RANDOMGOAL="${VALUE_P_RANDOMGOAL:-0.3}"
VALUE_GEOM_SAMPLE="${VALUE_GEOM_SAMPLE:-1}"
GD_P_CURGOAL="${GD_P_CURGOAL:-0.0}"
GD_P_TRAJGOAL="${GD_P_TRAJGOAL:-0.8}"
GD_P_RANDOMGOAL="${GD_P_RANDOMGOAL:-0.2}"
GD_GEOM_SAMPLE="${GD_GEOM_SAMPLE:-1}"
ACTOR_P_CURGOAL="${ACTOR_P_CURGOAL:-0.0}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-1.0}"
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.0}"
ACTOR_GEOM_SAMPLE="${ACTOR_GEOM_SAMPLE:-0}"

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
EXP_CORE="${EXP_PREFIX:-${ENV_TAG}_cgivl}"
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
  "--agent.high_alpha=${HIGH_ALPHA}"
  "--agent.subgoal_steps=${SUBGOAL_STEPS}"
  "--agent.rep_dim=${REP_DIM}"
  "--agent.value_backbone=${VALUE_BACKBONE}"
  "--agent.layer_norm=${LAYER_NORM}"
  "--agent.const_std=${CONST_STD}"
  "--agent.discrete=${DISCRETE}"
  "--agent.low_actor_rep_grad=${LOW_ACTOR_REP_GRAD}"
  "--agent.gc_negative=${GC_NEGATIVE}"
  "--agent.p_aug=${P_AUG}"
  "--agent.init_goal_rep=${INIT_GOAL_REP}"
  "--agent.use_subgoals=${USE_SUBGOALS}"
  "--agent.min_d=${MIN_D}"
  "--agent.neg_eps=${NEG_EPS}"
  "--agent.v_distill_eps=${V_DISTILL_EPS}"
  "--agent.goaldistance_latent_dim=${GOALDISTANCE_LATENT_DIM}"
  "--agent.goaldistance_layer_norm=${GOALDISTANCE_LAYER_NORM}"
  "--agent.value_p_curgoal=${VALUE_P_CURGOAL}"
  "--agent.value_p_trajgoal=${VALUE_P_TRAJGOAL}"
  "--agent.value_p_randomgoal=${VALUE_P_RANDOMGOAL}"
  "--agent.value_geom_sample=${VALUE_GEOM_SAMPLE}"
  "--agent.gd_p_curgoal=${GD_P_CURGOAL}"
  "--agent.gd_p_trajgoal=${GD_P_TRAJGOAL}"
  "--agent.gd_p_randomgoal=${GD_P_RANDOMGOAL}"
  "--agent.gd_geom_sample=${GD_GEOM_SAMPLE}"
  "--agent.actor_p_curgoal=${ACTOR_P_CURGOAL}"
  "--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}"
  "--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}"
  "--agent.actor_geom_sample=${ACTOR_GEOM_SAMPLE}"
)

append_hidden_dims "actor_hidden_dims" "${ACTOR_HIDDEN_DIMS}"
append_hidden_dims "value_hidden_dims" "${VALUE_HIDDEN_DIMS}"
append_hidden_dims "goaldistance_hidden_dims" "${GOALDISTANCE_HIDDEN_DIMS}"

case "${VALUE_BACKBONE}" in
  mlp)
    ;;
  recur_tied)
    if [[ ! "${VALUE_RECUR_LN_MODE}" =~ ^(pre_loop|per_layer|per_layer_final|pre_loop_per_layer)$ ]]; then
      echo "Error: VALUE_RECUR_LN_MODE='${VALUE_RECUR_LN_MODE}' is invalid. Use pre_loop, per_layer, per_layer_final, or pre_loop_per_layer." >&2
      exit 2
    fi
    if [[ ! "${VALUE_RECUR_FILM_MODE}" =~ ^(hidden|context|hidden_context)$ ]]; then
      echo "Error: VALUE_RECUR_FILM_MODE='${VALUE_RECUR_FILM_MODE}' is invalid. Use hidden, context, or hidden_context." >&2
      exit 2
    fi
    AGENT_FLAGS+=(
      "--agent.value_recur_iters=${VALUE_RECUR_ITERS}"
      "--agent.value_recur_num_dense_layers=${VALUE_RECUR_NUM_DENSE_LAYERS}"
      "--agent.value_recur_max_iters=${VALUE_RECUR_MAX_ITERS}"
      "--agent.value_recur_ln_mode=${VALUE_RECUR_LN_MODE}"
      "--agent.value_recur_use_step_info=${VALUE_RECUR_USE_STEP_INFO}"
      "--agent.value_recur_sinusoidal=${VALUE_RECUR_SINUSOIDAL}"
      "--agent.value_recur_use_film=${VALUE_RECUR_USE_FILM}"
      "--agent.value_recur_film_mode=${VALUE_RECUR_FILM_MODE}"
      "--agent.value_recur_use_layerscale=${VALUE_RECUR_USE_LAYERSCALE}"
      "--agent.value_recur_use_act=${VALUE_RECUR_USE_ACT}"
      "--agent.value_recur_act_epsilon=${VALUE_RECUR_ACT_EPSILON}"
      "--agent.value_recur_act_min_iters=${VALUE_RECUR_ACT_MIN_ITERS}"
      "--agent.value_layerscale_init=${VALUE_LAYERSCALE_INIT}"
    )
    ;;
  *)
    echo "Error: VALUE_BACKBONE='${VALUE_BACKBONE}' is invalid. Use mlp or recur_tied." >&2
    exit 2
    ;;
esac

if [[ -n "${ENCODER}" ]]; then
  AGENT_FLAGS+=("--agent.encoder=${ENCODER}")
fi

if [[ -n "${GD_ENCODER}" ]]; then
  AGENT_FLAGS+=("--agent.gd_encoder=${GD_ENCODER}")
fi

if [[ -n "${FRAME_STACK}" ]]; then
  if [[ ! "${FRAME_STACK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: FRAME_STACK='${FRAME_STACK}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.frame_stack=${FRAME_STACK}")
fi

if [[ ! "${GOALDISTANCE_LATENT_DIM}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: GOALDISTANCE_LATENT_DIM='${GOALDISTANCE_LATENT_DIM}' is not a positive integer." >&2
  exit 2
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
  "--value_warmup_steps=${VALUE_WARMUP_STEPS}"
  "--eval_warmup_steps=${EVAL_WARMUP_STEPS}"
  "--eval_episodes=${EVAL_EPISODES}"
  "--video_episodes=${VIDEO_EPISODES}"
  "--eval_on_cpu=${EVAL_ON_CPU}"
  "--save_interval=${SAVE_INTERVAL}"
  "--disable_tqdm=${DISABLE_TQDM}"
  "--exp_name=${EXP_NAME}"
  "--agent=agents/cgivl.py"
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
