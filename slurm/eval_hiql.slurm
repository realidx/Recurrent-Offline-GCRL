#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 4 — HIQL checkpoint evaluation
#
# Restores one trained HIQL checkpoint and runs evaluation only.
#
# Submit:
#   sbatch --export=ALL,WANDB_MODE=online,\
#RESTORE_PATH=/abs/path/to/run_dir,RESTORE_EPOCH=800000,EVAL_REPEATS=20 \
#     slurm/eval_hiql.slurm
# ---------------------------------------------------------------------------
#SBATCH --job-name=p4_hiql_eval
#SBATCH --partition=gpu-long
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=h100-47
#SBATCH --output=logs/phase_4/slurm-%x-%j.out
#SBATCH --error=logs/phase_4/slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ ! -f "third_party/ogbench/impls/main.py" ]]; then
  echo "Error: missing third_party/ogbench/impls/main.py. Run ./scripts/bootstrap_ogbench.sh first." >&2
  exit 2
fi

IMPL_DIR="${SLURM_SUBMIT_DIR}/third_party/ogbench/impls"
CONDA_ENV="${CONDA_ENV:-recurrent}"
CONDA_EXE="${CONDA_EXE:-conda}"
WANDB_MODE="${WANDB_MODE:-online}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P4_HIQL_Eval}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

SEED="${SEED:-0}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_TASKS="${EVAL_TASKS:-5}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-0}"
DISABLE_TQDM="${DISABLE_TQDM:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-20}"

if [[ -z "${RESTORE_PATH}" ]]; then
  echo "Error: RESTORE_PATH is required." >&2
  exit 2
fi

if [[ -z "${RESTORE_EPOCH}" ]]; then
  echo "Error: RESTORE_EPOCH is required." >&2
  exit 2
fi

if [[ "${RESTORE_PATH}" != /* ]]; then
  RESTORE_PATH="${SLURM_SUBMIT_DIR}/${RESTORE_PATH}"
fi

if [[ ! -d "${RESTORE_PATH}" ]]; then
  echo "Error: RESTORE_PATH does not exist: ${RESTORE_PATH}" >&2
  exit 2
fi

if [[ ! "${RESTORE_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "Error: RESTORE_EPOCH must be an integer. Got '${RESTORE_EPOCH}'." >&2
  exit 2
fi

if [[ ! "${EVAL_EPISODES}" =~ ^[0-9]+$ ]]; then
  echo "Error: EVAL_EPISODES must be a non-negative integer. Got '${EVAL_EPISODES}'." >&2
  exit 2
fi

if [[ ! "${EVAL_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: EVAL_REPEATS must be a positive integer. Got '${EVAL_REPEATS}'." >&2
  exit 2
fi

if [[ -n "${EVAL_TASKS}" && ! "${EVAL_TASKS}" =~ ^[0-9]+$ ]]; then
  echo "Error: EVAL_TASKS must be an integer or empty. Got '${EVAL_TASKS}'." >&2
  exit 2
fi

RESTORE_BASENAME="$(basename "${RESTORE_PATH}")"
DEFAULT_EXP_NAME="sd$(printf %03d "${SEED}")_${RESTORE_BASENAME}_eval${RESTORE_EPOCH}"
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONUNBUFFERED=1
if [[ "${EVAL_ON_CPU}" == "1" ]]; then
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
else
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
fi

mkdir -p "${DATASET_DIR}" logs/phase_4
export OGBENCH_DATASET_DIR="${DATASET_DIR}"

cd "${IMPL_DIR}"
echo "cwd=$(pwd) WANDB_MODE=${WANDB_MODE} OGBENCH_DATASET_DIR=${OGBENCH_DATASET_DIR}"

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

SCORES_FILE="$(mktemp)"
trap 'rm -f "${SCORES_FILE}"' EXIT

for (( rep = 1; rep <= EVAL_REPEATS; rep++ )); do
  REP_EXP_NAME="${EXP_NAME}_r$(printf %02d "${rep}")"
  REP_SEED="${SEED}"

  CMD=(
    python main.py
    "--run_group=${RUN_GROUP}"
    "--save_dir=${SAVE_DIR}"
    "--seed=${REP_SEED}"
    "--env_name=${ENV_NAME}"
    "--train_steps=0"
    "--eval_only=1"
    "--restore_path=${RESTORE_PATH}"
    "--restore_epoch=${RESTORE_EPOCH}"
    "--exp_name=${REP_EXP_NAME}"
    "--eval_episodes=${EVAL_EPISODES}"
    "--video_episodes=${VIDEO_EPISODES}"
    "--eval_on_cpu=${EVAL_ON_CPU}"
    "--disable_tqdm=${DISABLE_TQDM}"
    "--agent=agents/hiql.py"
  )

  if [[ -n "${EVAL_TASKS}" ]]; then
    CMD+=("--eval_tasks=${EVAL_TASKS}")
  fi

  echo "repeat=${rep}/${EVAL_REPEATS} seed=${REP_SEED} exp_name=${REP_EXP_NAME}"
  echo "${CMD[*]}"

  "${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${CMD[@]}"

  EVAL_CSV="${SAVE_DIR}/OGBench/${RUN_GROUP}/${REP_EXP_NAME}/eval.csv"
  if [[ ! -f "${EVAL_CSV}" ]]; then
    echo "Error: missing eval.csv for repeat ${rep}: ${EVAL_CSV}" >&2
    exit 4
  fi

  SCORE="$("${CONDA_EXE}" run -n "${CONDA_ENV}" python - "${EVAL_CSV}" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("empty eval.csv")
score = rows[-1].get('evaluation/overall_success')
if score is None or score == '':
    raise SystemExit("missing evaluation/overall_success")
print(score)
PY
)"
  echo "${SCORE}" >> "${SCORES_FILE}"
  echo "repeat=${rep} overall_success=${SCORE}"
done

echo "Summary over ${EVAL_REPEATS} repeats:"
"${CONDA_EXE}" run -n "${CONDA_ENV}" python - "${SCORES_FILE}" <<'PY'
import math, sys
path = sys.argv[1]
scores = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            scores.append(float(line))
if not scores:
    raise SystemExit("no scores collected")
mean = sum(scores) / len(scores)
var = sum((x - mean) ** 2 for x in scores) / len(scores)
std = math.sqrt(var)
best = max(scores)
best_idx = max(range(len(scores)), key=lambda i: scores[i]) + 1
print(f"overall_success_best={best:.6f}")
print(f"overall_success_best_repeat={best_idx}")
print(f"overall_success_mean={mean:.6f}")
print(f"overall_success_std={std:.6f}")
print("overall_success_all=" + ",".join(f"{x:.6f}" for x in scores))
PY
