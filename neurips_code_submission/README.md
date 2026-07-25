# Recurrent Offline RL NeurIPS Code Artifact

This artifact contains the minimal source files needed to run the patched
OGBench implementation used for the paper experiments. It intentionally does
not include datasets, experiment logs, reference PDFs, paper drafts, or a full
third-party checkout.

## Contents

- `requirements.txt`: Python dependencies for local sanity checks and OGBench runs.
- `impls/`: self-contained training implementation source.
- `slurm/`: Slurm launchers for CRL, HIQL, and SAW experiments.
- `pbs/`: PBS wrappers for CRL, HIQL, and SAW on clusters that use PBS Pro instead of Slurm.
- `scripts/find_mlp_param_match.py`: parameter-matched MLP helper.
- `scripts/plot_ams_trajectory_bottleneck.py`: plotting helper for the AMS trajectory diagnostic.
- `OGBENCH_LICENSE`: license notice for the OGBench source files this artifact derives from.

## Setup

Create and activate a Python or Conda environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The patched implementation is included under:

```text
impls/
```

## Sanity Checks

Check that the expected source files exist:

```bash
test -f impls/main.py
test -f impls/agents/crl.py
test -f impls/agents/saw.py
```

## Recurrent Design

The artifact exposes one recurrent model design:

- `CRITIC_BACKBONE=recur` for CRL critics.
- `VALUE_BACKBONE=recur` for HIQL and SAW values.
- `M=<int>` controls the number of residual SwiGLU layers per recurrent iteration.
- `K=<int>` controls the number of recurrent iterations.
- Each recurrent step uses a learned step embedding `E_k` and a learned
  per-step LayerScale parameter `alpha_k` initialized by the layer-scale setting.

The older dense recurrent cell, stacked-vs-tied variants, input injection, and
unsupported algorithm families are intentionally removed from this submission layout.

Run a short local smoke test by overriding the training budget and disabling
online logging:

```bash
WANDB_MODE=offline \
DATASETS_OVERRIDE="antmaze-medium-stitch-v0" \
TRAIN_STEPS=1000 \
EVAL_INTERVAL=1000 \
LOG_INTERVAL=200 \
SAVE_INTERVAL=1000 \
EVAL_EPISODES=1 \
EVAL_TASKS=1 \
VIDEO_EPISODES=0 \
./slurm/train_crl.slurm
```

## Data

OGBench datasets are not included. By default, the launchers look for datasets
under `.ogbench_data/`. You can override the location with:

```bash
DATASET_DIR=/path/to/ogbench_data ./slurm/train_crl.slurm
```

## Notes

The source code is included directly in this artifact. External Python packages
and OGBench datasets are still installed or downloaded through their standard
public distribution channels.
