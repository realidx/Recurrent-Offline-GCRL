# OGBench (official) in this workspace

This repo vendors the official OGBench repo at `third_party/ogbench`.

- Upstream commit: `1d4140997f60c52c6fb0702ec100dc988b18c548`

## Install

From the workspace root:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
python -m pip install -r requirements.txt
```

Notes:

- OGBench environments require MuJoCo and `dm_control` (installed as dependencies of `ogbench`).
- OGBench datasets are downloaded on first use via `ogbench.make_env_and_datasets`, so the first run needs network access.

### CUDA server install

On your remote NVIDIA/CUDA machine:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
python -m pip install -r requirements-cuda12.txt
```

## Run the reference CRL implementation (example)

The reference implementations live in `third_party/ogbench/impls`.

```bash
cd third_party/ogbench/impls
python main.py --env_name=antmaze-large-stitch-v0 --eval_episodes=50 --agent=agents/crl.py
```

For the exact hyperparameters used by the OGBench authors for each task/algorithm, see:

- `third_party/ogbench/impls/hyperparameters.sh`

## Phase 0: reproduce all AntMaze stitching CRL baselines

From the workspace root:

```bash
# Quick CPU sanity check (short run; still downloads datasets on first use).
TRAIN_STEPS=20000 EVAL_INTERVAL=5000 LOG_INTERVAL=1000 SEEDS="0" ./scripts/repro_crl_antmaze_stitch.sh

# Full repro (typically on server).
SEEDS="0 1 2 3 4" WANDB_MODE=online ./scripts/repro_crl_antmaze_stitch.sh
```
