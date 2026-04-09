# Commit And Push Workflow

This repo uses a patched local checkout of OGBench under `third_party/ogbench`.
The important consequence is:

- `third_party/ogbench` is **not** the portable source of truth for our changes.
- `patches/ogbench_impls.patch` **is** the portable source of truth for OGBench-side edits.

If you only remember one rule, remember this:

> If you changed anything under `third_party/ogbench`, regenerate and commit `patches/ogbench_impls.patch`.


## Repo Structure

- Outer repo: this repo
- Nested OGBench checkout: `third_party/ogbench`
- Bootstrap script: `scripts/bootstrap_ogbench.sh`
- Patch file applied by bootstrap: `patches/ogbench_impls.patch`

`bootstrap_ogbench.sh` clones OGBench at a pinned commit, then applies `patches/ogbench_impls.patch`.
That means anyone pulling this repo can recreate the intended OGBench state by running:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```


## The Correct Mental Model

When you edit files under `third_party/ogbench`, do **not** treat that nested checkout as the thing to push.

Treat it as:

- a temporary working tree for editing upstream OGBench files
- from which you regenerate `patches/ogbench_impls.patch`

Then commit and push the **outer repo**.


## Safe Cases

### Case 1: You only changed outer-repo files

Examples:

- `slurm/*.slurm`
- `scripts/*.sh`
- `README.md`
- `paper/*`

Then commit and push normally:

```bash
git status --short
git add <outer repo files>
git commit -m "Your message"
git push origin phase-4
```


### Case 2: You changed anything inside `third_party/ogbench`

Examples:

- `third_party/ogbench/impls/agents/*.py`
- `third_party/ogbench/impls/utils/*.py`
- `third_party/ogbench/ogbench/*`

Then the workflow is:

1. Edit the files in `third_party/ogbench`
2. Regenerate `patches/ogbench_impls.patch`
3. Commit the patch file in the outer repo
4. Push the outer repo

Use:

```bash
git -C third_party/ogbench diff --binary HEAD > patches/ogbench_impls.patch
git add patches/ogbench_impls.patch
git commit -m "Describe the OGBench-side change"
git push origin phase-4
```


## Recommended Full Workflow

If you touched both outer files and OGBench files:

```bash
git status --short
git -C third_party/ogbench status --short

git -C third_party/ogbench diff --binary HEAD > patches/ogbench_impls.patch

git add patches/ogbench_impls.patch
git add <outer repo files you actually want to commit>

git commit -m "Describe the full change"
git push origin phase-4
```


## Important Verification Steps

Before committing:

```bash
git status --short
git -C third_party/ogbench status --short
wc -c patches/ogbench_impls.patch
rg -n "some_marker_you_expect" patches/ogbench_impls.patch
```

Good practice:

- make sure `patches/ogbench_impls.patch` is not empty
- make sure it contains the expected file names or markers
- avoid accidentally committing unrelated local edits


## What Not To Do

Do **not** do these as the standard workflow:

- do not rely on the dirty state of `third_party/ogbench` without regenerating the patch
- do not push only the nested repo and assume others can reproduce your changes
- do not commit the outer repo while forgetting to update `patches/ogbench_impls.patch`
- do not assume `third_party/ogbench` alone is enough on a fresh checkout


## After Pulling On Another Machine Or Cluster

After `git pull`, rebuild the patched OGBench checkout:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```

This is the expected way to sync the nested OGBench tree with the outer repo.


## When Would You Push `third_party/ogbench` Itself?

Only if you intentionally maintain your own fork/remote for OGBench.

That is **not** the standard workflow in this repo.

The standard workflow is:

- edit `third_party/ogbench`
- regenerate `patches/ogbench_impls.patch`
- commit/push the outer repo


## Quick Checklist

If you changed OGBench code:

- [ ] regenerate `patches/ogbench_impls.patch`
- [ ] verify patch is non-empty and contains the intended changes
- [ ] commit the patch in the outer repo
- [ ] push the outer repo branch
- [ ] after pulling elsewhere, run `./scripts/bootstrap_ogbench.sh CLEAN=1`

