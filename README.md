# Coworker Setup

This repo does not depend on `third_party/ogbench` being pulled as a working submodule checkout.

Instead, the intended workflow is:

1. Clone this repo.
2. Activate your Python or Conda environment.
3. Install dependencies.
4. Run the bootstrap script once to reconstruct the patched OGBench tree locally.

## Quick Start

```bash
git clone <this-repo-url>
cd Recurrent-Offline-RL

# Activate your environment first, then install deps.
pip install -r requirements.txt

# Rebuild third_party/ogbench from upstream + our local patch.
./scripts/bootstrap_ogbench.sh CLEAN=1
```

After that, the expected OGBench code should exist under `third_party/ogbench/impls`, and the Slurm launchers should work on this machine.

## Why Bootstrap Is Required

The repo stores our OGBench modifications as a patch:

- `patches/ogbench_impls.patch`

The bootstrap script:

1. clones upstream OGBench
2. checks out the pinned upstream commit
3. resets the checkout to a clean state
4. applies `patches/ogbench_impls.patch`

So:

- `git pull` on this repo gives you the patch file and scripts
- `./scripts/bootstrap_ogbench.sh CLEAN=1` rebuilds the patched `third_party/ogbench` tree

This is why the workflow is not "just pull and run".

## When To Re-Run Bootstrap

Run bootstrap again when:

- you clone the repo on a new machine
- you pull changes that modify `patches/ogbench_impls.patch`
- you suspect `third_party/ogbench` has local drift
- a Slurm launcher says `Run ./scripts/bootstrap_ogbench.sh first`

Use:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```

`CLEAN=1` is the safe default because it resets the local OGBench checkout before applying the patch.

## Recommended Sharing Workflow

For coworkers, the simplest reliable setup is:

1. clone this repo normally
2. install dependencies
3. run `./scripts/bootstrap_ogbench.sh CLEAN=1`
4. run the desired Slurm launcher or Python command

Do not assume that a plain submodule update will reproduce the current patched OGBench tree.

## Sanity Checks

If bootstrap succeeds, you should have these files:

- `third_party/ogbench/impls/main.py`
- `third_party/ogbench/impls/agents/crl.py`
- `third_party/ogbench/impls/agents/saw.py`

You can also verify the bootstrap script parses correctly with:

```bash
bash -n scripts/bootstrap_ogbench.sh
```

## Maintainer Note

If you change code inside `third_party/ogbench`, refresh the patch before asking coworkers to pull:

```bash
git -C third_party/ogbench diff --binary HEAD > patches/ogbench_impls.patch
```

Then commit the updated patch file in the main repo.
