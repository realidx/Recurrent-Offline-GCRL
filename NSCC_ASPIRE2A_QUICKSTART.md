# NSCC ASPIRE2A Quickstart

This repo was originally set up around Slurm launchers, but NSCC ASPIRE2A uses PBS Pro queues. To make the existing training logic reusable, use the PBS wrappers in `pbs/`, which forward into the existing launcher scripts after exporting compatible environment variables.

## What Changed In This Repo

- `pbs/train_crl.pbs`
- `pbs/train_hiql.pbs`
- `pbs/train_saw.pbs`

These wrappers let you keep using the existing environment variables such as `ENV_NAME`, `RUN_GROUP`, `EXP_NAME`, `TRAIN_STEPS`, `WANDB_MODE`, and `CONDA_ENV`.

The Slurm launchers were also relaxed from a hardcoded `h100-47` request to a generic `--gpus=1`, which makes them more portable on non-H100 systems.

The PBS wrappers now default to `CONDA_MODULE=none` and `CONDA_EXE=/app/apps/miniforge3/25.3.1/bin/conda`, which avoids the broken `miniforge3/25.3.1` modulefile in batch jobs.

The PBS wrappers now default to:

- `CONDA_ENV=recurrent`
- `CONDA_EXE=/app/apps/miniforge3/25.3.1/bin/conda`
- `WANDB_MODE=online`
- `SAVE_DIR=${SCRATCH_ROOT}/OGBench/exp`
- `DATASET_DIR=${SCRATCH_ROOT}/OGBench/data`

where `SCRATCH_ROOT` resolves to `${SCRATCH}` if available, else `/scratch/users/nus/${USER}`.

The shared Slurm launchers still default to repo-local paths (`exp` and `.ogbench_data`) so they remain portable to non-NSCC clusters.

## Cluster Notes

- ASPIRE2A uses PBS Pro queues, not Slurm.
- NSCC documents `normal` and `ai` as the user-facing queues.
- GPU jobs on ASPIRE2A use A100 GPUs.
- NSCC enforces scheduler use for computational and pre/post-processing jobs.
- For strict compliance with the cluster notice, do the one-time setup steps inside an interactive PBS session instead of on the login node.

Official references:

- ASPIRE2A overview: <https://help.nscc.sg/aspire2a/about/>
- ASPIRE2A quickstart / queues: <https://help.nscc.sg/wp-content/uploads/2024/06/ASPIRE2A-General-Quickstart-Guide-1.pdf>
- ASPIRE2A FAQs: <https://help.nscc.sg/aspire2a/faqs/>
- ASPIRE2A software list: <https://help.nscc.sg/wp-content/uploads/2512-SoftwareList-2A.pdf>
- JAX installation guide: <https://docs.jax.dev/en/latest/installation.html>

## Fastest Safe Setup

### 0. Open an interactive setup session

If your site enforces the "all computational jobs via scheduler" rule strictly, start here and do the rest from inside the allocated shell:

```bash
qsub -I -P <project_id> -q normal -l select=1:ncpus=4:mem=16G -l walltime=01:00:00
```

If your project has access to the `ai` queue and you specifically want node-local NVMe at `/raid`, request that queue instead.

### 1. Start from a clean module environment

```bash
module purge
. /app/apps/resetmodule
module avail 2>&1 | egrep -i 'miniforge|miniconda|anaconda|python'
```

On ASPIRE2A, prefer the newest available `miniforge3` module. From the observed module list on this cluster, `miniforge3/25.3.1` is the best default.

Do not use `cray-python` for this repo setup. The launchers are written around `conda run -n ...`, so a Miniforge/Conda module is the path of least resistance.

Example:

```bash
module load miniforge3/25.3.1
```

### 2. Create the Python environment

```bash
cd /path/to/Recurrent-Offline-RL
conda create -n recurrent python=3.10 -y
conda activate recurrent
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install GPU-enabled JAX first, then repo requirements

For Linux NVIDIA GPUs, the current JAX docs recommend the pip CUDA wheels. CUDA 12 is the conservative choice for this repo and cluster class:

```bash
python -m pip install --upgrade "jax[cuda12]>=0.4.26"
python -m pip install -r requirements.txt
```

If your node image has a new enough driver and you explicitly want CUDA 13 wheels, check the JAX docs first and change that one line.

### 4. Rebuild the patched OGBench checkout

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```

Sanity-check the bootstrap result:

```bash
test -f third_party/ogbench/impls/main.py
test -f third_party/ogbench/impls/agents/crl.py
test -f third_party/ogbench/impls/agents/saw.py
test -f third_party/ogbench/impls/agents/hiql.py
```

## First Sanity Run

Use a short single-seed CRL job to verify PBS submission, Conda, JAX GPU visibility, dataset download, and output paths.

```bash
mkdir -p logs/phase_4 logs/phase_5
mkdir -p "${SCRATCH:-/scratch/users/nus/${USER}}/OGBench/exp"
mkdir -p "${SCRATCH:-/scratch/users/nus/${USER}}/OGBench/data"

qsub -P personal \
  -q normal \
  -v WANDB_MODE=offline,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=NSCC_Sanity,EXP_NAME=sd000_nscc_sanity,TRAIN_STEPS=2000,LOG_INTERVAL=200,EVAL_INTERVAL=1000,SAVE_INTERVAL=2000,EVAL_TASKS=1,EVAL_EPISODES=2,VIDEO_EPISODES=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5 \
  pbs/train_crl.pbs
```

The wrapper writes its combined stdout/stderr to `logs/phase_5/pbs-crl_train-<jobid>_<array>.log`.

## First Real Runs

### CRL single seed

```bash
qsub -P <project_id> \
  -q normal \
  -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_SWIGLU_PRE_LN=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 \
  pbs/train_crl.pbs
```

### SAW single seed

```bash
qsub -P <project_id> \
  -q normal \
  -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd000_AGN_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 \
  pbs/train_saw.pbs
```

### HIQL single seed

```bash
qsub -P <project_id> \
  -q normal \
  -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch,EXP_NAME=sd000_ALS_hiql_recur,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_USE_STEP_INFO=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25 \
  pbs/train_hiql.pbs
```

## Multi-Seed Runs

The PBS wrappers default to the same 3-way seed array pattern as the Slurm scripts.

Use:

```bash
qsub -P <project_id> -q normal -J 0-2 pbs/train_crl.pbs
```

and add the same `ENV_NAME=...`, `RUN_GROUP=...`, `EXP_NAME=...`, and model hyperparameters you would normally pass.

If you keep the default seed list in the launcher, `-J 0-2` maps to seeds `0 1 2`.

## What To Check If A Job Fails Fast

### `conda: command not found`

If the wrapper still does not find Conda, either override the module name or pass the full executable path explicitly:

```bash
qsub -P personal -q normal -v CONDA_MODULE=none,CONDA_ENV=recurrent,CONDA_EXE=/full/path/to/conda ...
```

### JAX only sees CPU

- make sure you installed `jax[cuda12]`, not CPU-only `jax`
- inspect the job output for the `jax devices:` line printed by the launcher
- avoid setting `LD_LIBRARY_PATH` unless you need it

### `third_party/ogbench/impls/main.py` missing

Run:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```

again on the repo checkout used for submission.

### Dataset download / storage issues

By default the launchers use:

```bash
SAVE_DIR=${SCRATCH_ROOT}/OGBench/exp
DATASET_DIR=${SCRATCH_ROOT}/OGBench/data
```

Override them with `-v SAVE_DIR=...` and `-v DATASET_DIR=...` if you want a different location.

## Weights & Biases

The training launchers default to `WANDB_MODE=online`.

Before your first online run, authenticate once inside the `recurrent` environment:

```bash
module load miniforge3/25.3.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate recurrent
wandb login
```

If you prefer not to paste the API key interactively, set:

```bash
export WANDB_API_KEY=...
```

before submitting, or put it in your shell startup file on the cluster.
