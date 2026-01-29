# SLURM

## CRL baseline repro (AntMaze stitch)

One run per array task: 4 envs × 5 seeds = 20 tasks.

```bash
# Create log directory for SLURM stdout/stderr.
mkdir -p logs/phase_0

# Ensure the upstream OGBench repo is present locally.
./scripts/bootstrap_ogbench.sh

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
