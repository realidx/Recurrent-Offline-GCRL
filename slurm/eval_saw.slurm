#!/bin/bash
# ---------------------------------------------------------------------------
# SAW checkpoint evaluation with repeated runs into one eval.csv.
# ---------------------------------------------------------------------------
#SBATCH --job-name=p4_saw_eval
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
WANDB_MODE="${WANDB_MODE:-disabled}"

ENV_NAME="${ENV_NAME:-antmaze-large-stitch-v0}"
RUN_GROUP="${RUN_GROUP:-P4_SAW_Eval}"
SAVE_DIR="${SAVE_DIR:-${SLURM_SUBMIT_DIR}/exp}"
DATASET_DIR="${DATASET_DIR:-${SLURM_SUBMIT_DIR}/.ogbench_data}"

SEED="${SEED:-0}"
RESTORE_PATH="${RESTORE_PATH:-}"
RESTORE_EPOCH="${RESTORE_EPOCH:-}"

EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_TASKS="${EVAL_TASKS:-5}"
VIDEO_EPISODES="${VIDEO_EPISODES:-0}"
EVAL_ON_CPU="${EVAL_ON_CPU:-1}"
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
TARGET_DIR="${SAVE_DIR}/OGBench/${RUN_GROUP}/${EXP_NAME}"
EVAL_CSV="${TARGET_DIR}/eval.csv"
SUMMARY_JSON="${TARGET_DIR}/eval_summary.json"
SUMMARY_TXT="${TARGET_DIR}/eval_summary.txt"

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
echo "cwd=$(pwd)"
echo "target_dir=${TARGET_DIR}"
echo "eval_csv=${EVAL_CSV}"
echo "summary_json=${SUMMARY_JSON}"
echo "summary_txt=${SUMMARY_TXT}"

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

rm -rf "${TARGET_DIR}"

for (( rep = 1; rep <= EVAL_REPEATS; rep++ )); do
  CMD=(
    python main.py
    "--run_group=${RUN_GROUP}"
    "--save_dir=${SAVE_DIR}"
    "--seed=${SEED}"
    "--env_name=${ENV_NAME}"
    "--train_steps=0"
    "--eval_only=1"
    "--restore_path=${RESTORE_PATH}"
    "--restore_epoch=${RESTORE_EPOCH}"
    "--exp_name=${EXP_NAME}"
    "--eval_episodes=${EVAL_EPISODES}"
    "--video_episodes=${VIDEO_EPISODES}"
    "--eval_on_cpu=${EVAL_ON_CPU}"
    "--disable_tqdm=${DISABLE_TQDM}"
    "--agent=agents/saw.py"
  )

  if [[ -n "${EVAL_TASKS}" ]]; then
    CMD+=("--eval_tasks=${EVAL_TASKS}")
  fi

  echo "repeat=${rep}/${EVAL_REPEATS} seed=${SEED} exp_name=${EXP_NAME}"
  echo "${CMD[*]}"

  "${CONDA_EXE}" run -n "${CONDA_ENV}" env WANDB_MODE="${WANDB_MODE}" "${CMD[@]}"
done

if [[ ! -f "${EVAL_CSV}" ]]; then
  echo "Error: missing eval.csv after repeated evaluation: ${EVAL_CSV}" >&2
  exit 4
fi

echo "Summary over ${EVAL_REPEATS} repeats:"
"${CONDA_EXE}" run -n "${CONDA_ENV}" python - "${EVAL_CSV}" "${SUMMARY_JSON}" "${SUMMARY_TXT}" "${EVAL_REPEATS}" <<'PY'
import csv
import json
import math
import sys

csv_path, json_path, txt_path, repeats = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

with open(csv_path, newline='') as f:
    rows = list(csv.DictReader(f))

if len(rows) < repeats:
    raise SystemExit(f"expected at least {repeats} rows in eval.csv, found {len(rows)}")

rows = rows[-repeats:]
scores = [float(r['evaluation/overall_success']) for r in rows]
mean = sum(scores) / len(scores)
var = sum((x - mean) ** 2 for x in scores) / len(scores)
std = math.sqrt(var)
best = max(scores)
best_idx = max(range(len(scores)), key=lambda i: scores[i]) + 1

summary = {
    'overall_success_best': best,
    'overall_success_best_repeat': best_idx,
    'overall_success_mean': mean,
    'overall_success_std': std,
    'overall_success_all': scores,
}

metric_names = [
    'evaluation/task1_success',
    'evaluation/task2_success',
    'evaluation/task3_success',
    'evaluation/task4_success',
    'evaluation/task5_success',
    'evaluation/overall_success',
    'evaluation/value_mean',
    'evaluation/value_std',
    'evaluation/value_abs_max',
]
for name in metric_names:
    vals = []
    for row in rows:
        raw = row.get(name, '')
        if raw not in ('', None):
            vals.append(float(raw))
    if vals:
        summary[name] = {
            'mean': sum(vals) / len(vals),
            'std': math.sqrt(sum((x - (sum(vals) / len(vals))) ** 2 for x in vals) / len(vals)),
            'all': vals,
        }

with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2, sort_keys=True)

lines = [
    f"overall_success_best={best:.6f} repeat={best_idx}",
    f"overall_success_mean={mean:.6f}",
    f"overall_success_std={std:.6f}",
]
for name in metric_names:
    if name in summary and isinstance(summary[name], dict):
        lines.append(f"{name}: mean={summary[name]['mean']:.6f} std={summary[name]['std']:.6f}")

text = "\n".join(lines) + "\n"
with open(txt_path, 'w') as f:
    f.write(text)
print(text, end='')
PY
