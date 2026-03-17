#!/bin/bash
#SBATCH --job-name=p4_nav_giant_1x4
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%A_%a.out
#SBATCH --error=logs/phase_4/slurm-%x-%A_%a.err
#SBATCH --array=0-2

# Phase 4 — Giant Navigate (CRL + recurrent critic 1×4)
#
# This script is intended to run the SAME CRL algorithm/hparams as OGBench defaults
# for navigate tasks, swapping ONLY the critic backbone to recur_tied (K_train=4).
#
# Use it twice (two separate jobs) for:
#   - antmaze-giant-navigate-v0
#   - pointmaze-giant-navigate-v0
#
# Example submissions:
#   sbatch --export=ALL,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_AntMazeGiantNav_Recur1x4,WANDB_MODE=online slurm/phase4_train_recur_tied_1x4_giant_navigate_array.slurm
#   sbatch --export=ALL,ENV_NAME=pointmaze-giant-navigate-v0,RUN_GROUP=P4_PointMazeGiantNav_Recur1x4,WANDB_MODE=online slurm/phase4_train_recur_tied_1x4_giant_navigate_array.slurm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh on the login node first." >&2
  exit 2
fi

if [[ ! -f "third_party/ogbench/impls/agents/crl.py" ]] || ! grep -q "critic_backbone" "third_party/ogbench/impls/agents/crl.py" 2>/dev/null; then
  echo "Error: OGBench checkout is missing patches (agent.critic_backbone not found)." >&2
  echo "Fix: run ./scripts/bootstrap_ogbench.sh CLEAN=1 on the login node, then resubmit." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"

CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-giant-navigate-v0}"
RUN_GROUP="${RUN_GROUP:-P4_GiantNavigate}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

# Train settings.
TRAIN_STEPS="${TRAIN_STEPS:-1000000}"
LOG_INTERVAL="${LOG_INTERVAL:-20000}"
VALIDATION_LOG_INTERVAL="${VALIDATION_LOG_INTERVAL:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200000}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"

# Evaluation during training (diagnostics). For "paper-style" final numbers, run a separate eval-only job later.
EVAL_INTERVAL="${EVAL_INTERVAL:-200000}"
EVAL_TASKS="${EVAL_TASKS:-}"     # empty => evaluate all tasks in the env
EVAL_EPISODES="${EVAL_EPISODES:-20}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"

# Critic backbone settings.
KTRAIN="${KTRAIN:-4}"                      # K in NxK: recurrent iterations at training time
RECUR_NUM_DENSE_LAYERS="${RECUR_NUM_DENSE_LAYERS:-}" # N in NxK: dense layers inside each tied recurrent residual update
RECUR_NUM_BLOCKS="${RECUR_NUM_BLOCKS:-1}"  # Deprecated fallback: number of 2-layer MLP pairs inside each tied recurrent residual update.
RECUR_MAX_ITERS="${RECUR_MAX_ITERS:-24}"  # allocated step-params (alpha/LN/step-embed) slots (keep consistent across runs)
SINUSOIDAL="${SINUSOIDAL:-1}"             # 1 => sinusoidal encoding, 0 => discrete lookup
TIED_LN="${TIED_LN:-1}"                   # 1 => single LN shared across iterations
RECUR_LN_MODE="${RECUR_LN_MODE:-per_layer}" # pre_loop | per_layer
RECUR_USE_STEP_INFO="${RECUR_USE_STEP_INFO:-1}" # 1 => use loop-step conditioning
RECUR_USE_FILM="${RECUR_USE_FILM:-1}"     # 1 => FiLM modulation, 0 => additive no-FiLM ablation
RECUR_USE_LAYERSCALE="${RECUR_USE_LAYERSCALE:-1}" # 1 => learnable residual scaling alpha_k
LAYERSCALE_INIT="${LAYERSCALE_INIT:-1e-2}"

# LR schedule. (Set LR_DECAY_STEPS=0 to disable and use constant LR.)
LR_DECAY_STEPS="${LR_DECAY_STEPS:-${TRAIN_STEPS}}"
LR_MIN="${LR_MIN:-1e-5}"

# Navigate task defaults (from OGBench impls/hyperparameters.sh):
# - antmaze-giant-navigate-v0: alpha=0.1, discount=0.995
# - pointmaze-giant-navigate-v0: alpha=0.03, discount=0.995
ALPHA="${ALPHA:-}"
DISCOUNT="${DISCOUNT:-}"
if [[ -z "${DISCOUNT}" ]]; then
  if [[ "${ENV_NAME}" == *"-giant-"* ]]; then
    DISCOUNT="0.995"
  else
    DISCOUNT="0.99"
  fi
fi
if [[ -z "${ALPHA}" ]]; then
  if [[ "${ENV_NAME}" == pointmaze-* ]]; then
    ALPHA="0.1"
  else
    ALPHA="0.1"
  fi
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

mkdir -p "${DATASET_DIR}"
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

mkdir -p logs/phase_4

SEEDS=(0 1 2)
idx="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[idx]}"

# Deterministic exp_name makes resume trivial (re-submitting array resumes into the same dir).
ENV_TAG="${ENV_NAME%-v0}"
ENV_TAG="${ENV_TAG//-/_}"

# Optional actor goal-mix overrides.
# Defaults to 0.5/0.5 (your current preferred setting); override via sbatch --export.
ACTOR_P_RANDOMGOAL="${ACTOR_P_RANDOMGOAL:-0.5}"
ACTOR_P_TRAJGOAL="${ACTOR_P_TRAJGOAL:-0.5}"
# Optional: geometric sampling for actor future (traj) goals (0/1). Default is CRL config default.
ACTOR_GEOM_SAMPLE="${ACTOR_GEOM_SAMPLE:-}"
ACTOR_BACKBONE="${ACTOR_BACKBONE:-mlp}"            # mlp | residual_dense
ACTOR_BACKBONE_HIDDEN_DIM="${ACTOR_BACKBONE_HIDDEN_DIM:-0}"
ACTOR_NUM_LAYERS="${ACTOR_NUM_LAYERS:-}"           # MLP hidden layers or residual actor dense-layer count
ACTOR_LAYER_WIDTH="${ACTOR_LAYER_WIDTH:-512}"
ACTOR_HIDDEN_DIMS="${ACTOR_HIDDEN_DIMS:-}"         # Optional explicit x-separated MLP dims
ACTOR_RESIDUAL_NUM_DENSE_LAYERS="${ACTOR_RESIDUAL_NUM_DENSE_LAYERS:-}"
ACTOR_RESIDUAL_SPAN="${ACTOR_RESIDUAL_SPAN:-4}"

if [[ -n "${RECUR_NUM_DENSE_LAYERS}" ]]; then
  if [[ ! "${RECUR_NUM_DENSE_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: RECUR_NUM_DENSE_LAYERS='${RECUR_NUM_DENSE_LAYERS}' is not a positive integer." >&2
    exit 2
  fi
else
  RECUR_NUM_DENSE_LAYERS="$(( RECUR_NUM_BLOCKS * 2 ))"
fi

EFFECTIVE_DENSE_LAYERS="${RECUR_NUM_DENSE_LAYERS}"
EXP_NAME="${EXP_NAME:-sd$(printf %03d ${SEED})_${ENV_TAG}_recur${EFFECTIVE_DENSE_LAYERS}x${KTRAIN}}"

export WANDB_RUN_ID="${WANDB_RUN_ID:-ogbench_${RUN_GROUP}_${EXP_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

AGENT_FLAGS=(
  "--agent.alpha=${ALPHA}"
  "--agent.discount=${DISCOUNT}"
  "--agent.batch_size=1024"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=${KTRAIN}"
  "--agent.critic_recur_num_dense_layers=${RECUR_NUM_DENSE_LAYERS}"
  "--agent.critic_recur_max_iters=${RECUR_MAX_ITERS}"
  "--agent.critic_recur_use_step_info=${RECUR_USE_STEP_INFO}"
  "--agent.critic_recur_tied_ln=${TIED_LN}"
  "--agent.critic_recur_ln_mode=${RECUR_LN_MODE}"
  "--agent.critic_recur_sinusoidal=${SINUSOIDAL}"
  "--agent.critic_recur_use_film=${RECUR_USE_FILM}"
  "--agent.critic_recur_use_layerscale=${RECUR_USE_LAYERSCALE}"
  "--agent.critic_layerscale_init=${LAYERSCALE_INIT}"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
  "--agent.actor_backbone=${ACTOR_BACKBONE}"
)

if [[ ! "${ACTOR_BACKBONE}" =~ ^(mlp|residual_dense)$ ]]; then
  echo "Error: ACTOR_BACKBONE='${ACTOR_BACKBONE}' is invalid. Use mlp or residual_dense." >&2
  exit 2
fi

if [[ -n "${ACTOR_NUM_LAYERS}" && ! "${ACTOR_NUM_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ACTOR_NUM_LAYERS='${ACTOR_NUM_LAYERS}' is not a positive integer." >&2
  exit 2
fi
if [[ ! "${ACTOR_LAYER_WIDTH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: ACTOR_LAYER_WIDTH='${ACTOR_LAYER_WIDTH}' is not a positive integer." >&2
  exit 2
fi

if [[ "${ACTOR_BACKBONE}" == "mlp" ]]; then
  if [[ -z "${ACTOR_HIDDEN_DIMS}" && -n "${ACTOR_NUM_LAYERS}" ]]; then
    ACTOR_DIMS=()
    for ((i = 0; i < ACTOR_NUM_LAYERS; i++)); do
      ACTOR_DIMS+=("${ACTOR_LAYER_WIDTH}")
    done
    ACTOR_HIDDEN_DIMS="$(IFS=x; echo "${ACTOR_DIMS[*]}")"
  fi
  if [[ -n "${ACTOR_HIDDEN_DIMS}" ]]; then
    if [[ "${ACTOR_HIDDEN_DIMS}" == *","* || "${ACTOR_HIDDEN_DIMS}" == *"("* || "${ACTOR_HIDDEN_DIMS}" == *")"* ]]; then
      echo "Error: ACTOR_HIDDEN_DIMS='${ACTOR_HIDDEN_DIMS}' must use x-separated format, e.g. 512x512x512." >&2
      exit 2
    fi
    IFS='x' read -r -a ACTOR_DIMS <<< "${ACTOR_HIDDEN_DIMS}"
    for dim in "${ACTOR_DIMS[@]}"; do
      if [[ ! "${dim}" =~ ^[0-9]+$ ]]; then
        echo "Error: ACTOR_HIDDEN_DIMS contains non-integer token '${dim}'." >&2
        exit 2
      fi
      AGENT_FLAGS+=("--agent.actor_hidden_dims=${dim}")
    done
  fi
else
  if [[ -z "${ACTOR_RESIDUAL_NUM_DENSE_LAYERS}" && -n "${ACTOR_NUM_LAYERS}" ]]; then
    ACTOR_RESIDUAL_NUM_DENSE_LAYERS="${ACTOR_NUM_LAYERS}"
  fi
  if [[ -z "${ACTOR_RESIDUAL_NUM_DENSE_LAYERS}" ]]; then
    ACTOR_RESIDUAL_NUM_DENSE_LAYERS="4"
  fi
  if [[ ! "${ACTOR_RESIDUAL_NUM_DENSE_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ACTOR_RESIDUAL_NUM_DENSE_LAYERS='${ACTOR_RESIDUAL_NUM_DENSE_LAYERS}' is not a positive integer." >&2
    exit 2
  fi
  if [[ ! "${ACTOR_RESIDUAL_SPAN}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ACTOR_RESIDUAL_SPAN='${ACTOR_RESIDUAL_SPAN}' is not a positive integer." >&2
    exit 2
  fi
  if (( ACTOR_BACKBONE_HIDDEN_DIM == 0 )); then
    ACTOR_BACKBONE_HIDDEN_DIM="${ACTOR_LAYER_WIDTH}"
  fi
  AGENT_FLAGS+=(
    "--agent.actor_num_dense_layers=${ACTOR_RESIDUAL_NUM_DENSE_LAYERS}"
    "--agent.actor_residual_span=${ACTOR_RESIDUAL_SPAN}"
  )
fi

if [[ "${ACTOR_BACKBONE_HIDDEN_DIM}" != "0" ]]; then
  if [[ ! "${ACTOR_BACKBONE_HIDDEN_DIM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ACTOR_BACKBONE_HIDDEN_DIM='${ACTOR_BACKBONE_HIDDEN_DIM}' is not a positive integer." >&2
    exit 2
  fi
  AGENT_FLAGS+=("--agent.actor_backbone_hidden_dim=${ACTOR_BACKBONE_HIDDEN_DIM}")
fi

if [[ -n "${ACTOR_P_RANDOMGOAL}" ]]; then
  AGENT_FLAGS+=("--agent.actor_p_randomgoal=${ACTOR_P_RANDOMGOAL}")
fi
if [[ -n "${ACTOR_P_TRAJGOAL}" ]]; then
  AGENT_FLAGS+=("--agent.actor_p_trajgoal=${ACTOR_P_TRAJGOAL}")
fi
if [[ -n "${ACTOR_GEOM_SAMPLE}" ]]; then
  AGENT_FLAGS+=("--agent.actor_geom_sample=${ACTOR_GEOM_SAMPLE}")
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

if [[ -n "${RESTORE_EPOCH}" && ! "${RESTORE_EPOCH}" =~ ^[0-9]+$ ]]; then
  if [[ "${RESTORE_EPOCH}" =~ params_([0-9]+)\.pkl$ ]]; then
    echo "Warning: RESTORE_EPOCH looks like a checkpoint path; extracting epoch." >&2
    RESTORE_EPOCH="${BASH_REMATCH[1]}"
  else
    echo "Error: RESTORE_EPOCH must be an integer (e.g., 600000). Got: ${RESTORE_EPOCH}" >&2
    echo "Tip: run 'unset RESTORE_EPOCH RESTORE_PATH' before sbatch, or submit with --export=ALL,RESTORE_EPOCH=,RESTORE_PATH=" >&2
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

echo "PHASE4: env=${ENV_NAME} seed=${SEED} nxk=${EFFECTIVE_DENSE_LAYERS}x${KTRAIN} alpha=${ALPHA} discount=${DISCOUNT} exp_name=${EXP_NAME} cwd=$(pwd) WANDB_MODE=${WANDB_MODE}"
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
  echo "Error: 'conda' not found in this job environment." >&2
  exit 3
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" python - <<'PY'
import jax
print("jax_devices:", jax.devices())
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
fi

"${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
