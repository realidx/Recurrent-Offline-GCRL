# SLURM

## CRL baseline repro (AntMaze stitch)

One run per array task: 4 envs × 5 seeds = 20 tasks.

```bash
# Create log directory for SLURM stdout/stderr.
mkdir -p logs/phase_0

# Ensure the upstream OGBench repo is present locally.
./scripts/bootstrap_ogbench.sh CLEAN=1

# (Optional) prefetch datasets on login node (recommended if compute nodes have no internet)
conda run -n recurrent python scripts/prefetch_ogbench_datasets.py --dataset-dir .ogbench_data

sbatch slurm/repro_crl_antmaze_stitch_array.slurm
```

Common overrides:

```bash
sbatch \
  --export=ALL,WANDB_MODE=online,CONDA_ENV=recurrent,DATASET_DIR=/scratch/$USER/ogbench_data,LOG_ROOT=/scratch/$USER/logs/phase_0,SAVE_DIR=/scratch/$USER/exp,DISABLE_TQDM=1,LOG_INTERVAL=20000,EVAL_INTERVAL=200000 \
  slurm/repro_crl_antmaze_stitch_array.slurm
```

If your compute nodes say `conda: command not found`, either:

- Add your cluster’s `module load anaconda` / `module load miniconda` near the top of the `.slurm` file, or
- Submit with `--export=ALL,CONDA_EXE=/full/path/to/conda,...` (e.g., `$HOME/miniconda3/bin/conda`).

## Summarize results

From the repo root:

```bash
python scripts/summarize_ogbench_csvs.py --roots exp logs --aggregate --out runs.csv --print
```

## Phase 1: critic-backbone variants (AntMaze large stitch, 3 seeds)

One environment (`antmaze-large-stitch-v0`), 3 seeds, 7 configs = 21 array tasks.

```bash
mkdir -p logs/phase_1
sbatch slurm/phase1_critics_antmaze_large_stitch_array.slurm
```

## Phase 1 (reduced): depth-6 untied vs tied (+ param-matched tied)

This is the recommended “fast” Phase 1 run: 3 seeds × 3 configs = 9 array tasks:

- `resnet` depth = 6
- `recur_tied` iters = 6
- `recur_tied` iters = 6 with `critic_backbone_hidden_dim` chosen to parameter-match `resnet:6`

```bash
mkdir -p logs/phase_1

# (1) Compute a good hidden-dim match on the login node.
OGBENCH_DATASET_DIR=.ogbench_data \
  python scripts/match_recur_hidden_dim.py --dataset antmaze-large-stitch-v0 --resnet-depth 6 --recur-iters 6

# (2) Submit with the printed RECUR_MATCH_HIDDEN_DIM.
sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=### slurm/phase1_min_antmaze_large_stitch_array.slurm
```

To test the tied model with more test-time compute (K_test), set `CRITIC_EVAL_NUM_ITERS` (only affects `recur_tied`):

```bash
sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=###,CRITIC_EVAL_NUM_ITERS=12 slurm/phase1_min_antmaze_large_stitch_array.slurm
```

## Phase 2: continue experiments (keep Phase 1 results intact)

Phase 2 scripts are identical to Phase 1 scripts, but default to:

- SLURM stdout/stderr: `logs/phase_2/`
- `--run_group=P2_*` (so `exp/OGBench/P2_*/...` doesn’t collide with Phase 1)

Full grid (21 tasks):

```bash
mkdir -p logs/phase_2
sbatch slurm/phase2_critics_antmaze_large_stitch_array.slurm
```

Reduced (9 tasks) + optional `K_test` knob:

```bash
mkdir -p logs/phase_2

OGBENCH_DATASET_DIR=.ogbench_data \
  python scripts/match_recur_hidden_dim.py --dataset antmaze-large-stitch-v0 --resnet-depth 6 --recur-iters 6

sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=### slurm/phase2_min_antmaze_large_stitch_array.slurm

# Optional: more test-time iterations (tied-only).
sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=###,CRITIC_EVAL_NUM_ITERS=12 slurm/phase2_min_antmaze_large_stitch_array.slurm
```

## If 5 hours isn’t enough: split Train vs Eval

On multi-task environments (like AntMaze stitch), evaluation can dominate wall-clock because it runs
`eval_tasks × eval_episodes` rollouts per eval interval. If your 5h GPU jobs don’t reach `train_steps`,
use this split:

- Train job: disable evaluation (`--eval_interval=0`) and save checkpoints frequently.
- Eval job: load a checkpoint and run full evaluation once.

Train-only (9 tasks, 5h default, saves every 200k steps):

```bash
mkdir -p logs/phase_2
sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=### slurm/phase2_train_min_antmaze_large_stitch_array.slurm
```

Eval-only (single job):

```bash
mkdir -p logs/phase_2
sbatch --export=ALL,RESTORE_PATH='exp/OGBench/P2_TrainOnly_Depth6/sd000_*',RESTORE_EPOCH=1000000 slurm/phase2_eval_only.slurm
```

If some eval jobs fail with CUDA initialization / “No visible GPU devices”, keep `EVAL_ON_CPU=1` (default).
This forces `JAX_PLATFORMS=cpu` in `slurm/phase2_eval_only.slurm` so JAX won’t touch CUDA.

### Resuming a train-only job

`main.py` now resumes cleanly based on the restored checkpoint’s internal step counter. To resume into the *same*
run directory (so you don’t create duplicate “seed 0” runs), set `EXP_NAME` and restore from that same directory.
In the Phase 2 SLURM scripts, `RESTORE_PATH` can be relative to `${SLURM_SUBMIT_DIR}` (it will be converted to an absolute path).

Example (resume seed 0 run in place):

```bash
sbatch --export=ALL,RECUR_MATCH_HIDDEN_DIM=###,RUN_GROUP=P2_TrainOnly_Depth6,EXP_NAME='sd000_s_123456.20260130_010203',RESTORE_PATH='exp/OGBench/P2_TrainOnly_Depth6/sd000_s_123456.20260130_010203',RESTORE_EPOCH=80000 slurm/phase2_train_min_antmaze_large_stitch_array.slurm
```

If a job ever runs on CPU by accident (slow), `JAX_PLATFORMS=cuda` makes it fail fast instead of burning hours.
