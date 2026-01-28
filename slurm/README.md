# SLURM

## CRL baseline repro (AntMaze stitch)

One run per array task: 4 envs × 5 seeds = 20 tasks.

```bash
# (Optional) prefetch datasets on login node (recommended if compute nodes have no internet)
conda run -n recurrent python scripts/prefetch_ogbench_datasets.py --dataset-dir .ogbench_data

sbatch slurm/repro_crl_antmaze_stitch_array.sbatch
```

Common overrides:

```bash
sbatch \
  --export=ALL,WANDB_MODE=online,CONDA_ENV=recurrent,DATASET_DIR=/scratch/$USER/ogbench_data,SAVE_DIR=/scratch/$USER/exp \
  slurm/repro_crl_antmaze_stitch_array.sbatch
```

