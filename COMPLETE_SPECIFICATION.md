# COMPLETE SPECIFICATION (Current)
## Recurrent / Iterated Critic Backbones for OGBench CRL

Last updated: 2026-02-10

This document is the single source of truth for:
- the critic/actor architectures currently implemented in our OGBench fork,
- the key hyperparameters we actually use in SLURM runs,
- how “stack depth × iterations” maps to code flags.

---

## 1) Repo + Bootstrap Contract

Upstream benchmark: OGBench (official repo) pinned at commit:
- `1d4140997f60c52c6fb0702ec100dc988b18c548`

Local integration strategy:
- upstream code is vendored at `third_party/ogbench`
- all changes are applied by `./scripts/bootstrap_ogbench.sh` from `patches/ogbench_impls.patch`
- remote workflow: run `./scripts/bootstrap_ogbench.sh CLEAN=1` on the login node before submitting jobs

This guarantees compute nodes always run the patched implementation.

---

## 2) Tasks + Datasets

### 2.1 Stitch tasks (primary)
- `antmaze-medium-stitch-v0`
- `antmaze-large-stitch-v0` (primary)
- `antmaze-giant-stitch-v0` (paper default uses `discount=0.995`)
- `antmaze-teleport-stitch-v0`

### 2.2 Navigate tasks (follow-up)
- `antmaze-giant-navigate-v0` (paper default uses `discount=0.995`)

Dataset location:
- SLURM scripts set `OGBENCH_DATASET_DIR` (typically `${SLURM_SUBMIT_DIR}/.ogbench_data`)
- download helper: `scripts/prefetch_ogbench_datasets.py`

---

## 3) Algorithm (Fixed Across Comparisons)

We keep the dataset, CRL loss, actor architecture, training budget, and eval protocol fixed across comparisons.
Only the critic backbone is swapped.

### 3.1 Critic: CRL bilinear score

The critic is a contrastive bilinear score using two encoders:
- `phi(s,a) ∈ R^d`
- `psi(g) ∈ R^d`
- score/logit: `v(s,a,g) = <phi(s,a), psi(g)> / sqrt(d)`

Implementation: `third_party/ogbench/impls/utils/networks.py:GCBilinearValue`.

Ensemble:
- `ensemble=True` uses **2** critics implemented via `nn.vmap`
- the critic returns `(q1, q2)` and the actor uses `min(q1, q2)` (TD3-style conservatism)

### 3.2 Contrastive loss type

Default is backward-compatible sigmoid BCE:
- `--agent.contrastive_loss_type=bce`

Optional InfoNCE variant (ablation):
- `--agent.contrastive_loss_type=infonce`
- `--agent.infonce_temperature=0.1` (default)

### 3.3 Actor: DDPG+BC (fixed)

Actor is always a standard MLP policy trained with DDPG+BC:
- behavior cloning strength: `alpha` (default `0.1`)

Implementation: `third_party/ogbench/impls/agents/crl.py`.

---

## 4) Architectures (What We Swap)

All variants below are “backbones” used inside CRL’s `phi` and `psi`. We do not change the CRL loss.

Common default dimensions:
- `value_hidden_dims=(512, 512, 512)`
- `latent_dim=512`
- backbone width defaults to `value_hidden_dims[-1]` unless overridden

Optional width override (used for parameter-matching ablations):
- `--agent.critic_backbone_hidden_dim=<int>` (0 disables; when set, overrides backbone hidden width for `resnet` / `recur_tied` / `partial_tied`)

### 4.1 Baseline MLP backbone (`critic_backbone=mlp`)

Implementation: `MLP(hidden_dims=(*value_hidden_dims, latent_dim), activations=gelu, layer_norm=layer_norm)`.

This is **4 Dense layers total**: `512 → 512 → 512 → 512`, with:
- `GELU + LayerNorm` after layers 1–3
- no activation / no LN on the final latent layer

### 4.2 Feedforward ResNet backbone (`critic_backbone=resnet`)

Implementation: `DeepResNetBackbone(hidden_dim, out_dim=latent_dim, num_blocks=critic_resnet_depth)`.

ResNetBlock (per block):
1) (optional) `LayerNorm(h)`
2) `Dense(hidden_dim) → SiLU → Dense(hidden_dim)`
3) `LayerScale` (learned vector, init `layerscale_init=1e-2`)
4) residual add: `h ← h + u`

So `critic_resnet_depth=D` means **D residual MLP blocks** stacked, no iteration.

### 4.3 Tied iterated backbone (`critic_backbone=recur_tied`)

Implementation: `RecurTiedBackbone(hidden_dim, out_dim=latent_dim, num_iters=K, max_iters, ...)`.

Per-iteration update (K times):
1) (optional) LayerNorm on `h`
   - `--agent.critic_recur_tied_ln=1` uses a single LN module shared across iterations
2) step-conditioned FiLM:
   - step encoding is discrete lookup *or* sinusoidal
   - FiLM depends on `(step, h)`: produces `gamma, beta` and applies `h_film = (1+gamma)*LN(h) + beta`
3) tied 2-layer MLP: `Dense(hidden_dim) → SiLU → Dense(hidden_dim)`
4) gated residual: `h ← h + alpha[k] * u`, where `alpha[k]` is a learned scalar (init `layerscale_init`)

Step encoding modes:
- discrete lookup table (default): `--agent.critic_recur_sinusoidal=0` uses learned `step_embed[k]`
- sinusoidal (extrapolatable): `--agent.critic_recur_sinusoidal=1`, implemented in
  `third_party/ogbench/impls/utils/positional_encoding.py`

Max step table:
- `--agent.critic_recur_max_iters` (default 16; some sinusoidal runs used 24)

Checkpoint compatibility note:
- if you trained with old “per-iteration LN modules”, load with `--agent.critic_recur_tied_ln=0`
- current default is tied LN (`--agent.critic_recur_tied_ln=1`)

### 4.4 Stack-and-iterate backbone (`critic_backbone=partial_tied`)

This is the general “stack depth × outer iterations” model:
- stack depth `G = critic_partial_groups`
- outer iterations `T = critic_partial_iters_per_group`

Implementation: `PartiallyTiedBackbone(hidden_dim, out_dim=latent_dim, num_groups=G, iters_per_group=T, ...)`.

It uses:
- **group-specific** 2-layer residual MLP blocks (one block per group),
- the same FiLM mechanism as `recur_tied` (optionally per-group FiLM),
- the same LayerNorm + LayerScale/gating stabilizers as above.

Block schedule:
- `--agent.critic_partial_block_schedule=stack` (recommended; Universal-Transformer style)
- `--agent.critic_partial_block_schedule=grouped` (legacy “chunked” schedule)

Other partial flags (rarely changed):
- `--agent.critic_partial_cycle_step_params=1`: cycles per-step params (step encoding / alpha / optional per-step LN) so `T_test` doesn’t allocate untrained step parameters
- `--agent.critic_partial_per_group_film=0`: if set to 1, FiLM parameters are untied per group (more parameters; ablation-only)

Paper-friendly naming:
- we recommend describing models as `(G, T)` rather than “partial”
- `recur_tied` is the special case `(G=1, T=K)`
- `resnet` is the special case `(G=D, T=1)` but **without** FiLM/step conditioning

---

## 5) Hyperparameters We Actually Use

### 5.1 Defaults from `agents/crl.py`
- `lr=3e-4`
- `lr_min=1e-5`
- `lr_decay_steps=0` (when enabled in SLURM we set it to `train_steps`)
- `batch_size=1024`
- `alpha=0.1` (DDPG+BC)
- `discount=0.99` (paper default uses `0.995` on “giant” tasks)
- `actor_hidden_dims=(512,512,512)`
- `value_hidden_dims=(512,512,512)`
- `latent_dim=512`

### 5.2 Stitch runs (our standard overrides)
Used in most AntMaze-*stitch runs:
- `--agent.actor_p_randomgoal=0.5`
- `--agent.actor_p_trajgoal=0.5`
- `--agent.alpha=0.1`

### 5.3 Giant tasks (paper default)
For `*-giant-*` tasks we follow OGBench hyperparameters:
- `--agent.discount=0.995`

### 5.4 Typical training + eval budget in SLURM
- `train_steps=1_000_000`
- logging: `log_interval=20_000`, `validation_log_interval=100_000`
- evaluation during training: `eval_interval=200_000`, `eval_tasks=5`, `eval_episodes=20`
- final evaluation: `eval_tasks=5`, `eval_episodes=50`
- checkpoints: `save_interval=200_000`

---

## 6) Evaluation + Logging Notes (Debuggability)

We patched evaluation to make comparisons less ambiguous:
- deterministic per-episode seeding (reruns compare episode-by-episode),
- richer evaluation logging (critic ensemble disagreement, Q stats, action stats, policy-vs-behavior divergence).

Action refinement is evaluation-only:
- `eval_refine_steps`, `eval_refine_lr`, `eval_refine_l2`
Current results suggest refinement is not a reliable source of gains and should be treated as ablation-only.

---

## 7) Pointers

- CRL agent + config flags: `third_party/ogbench/impls/agents/crl.py`
- Backbones: `third_party/ogbench/impls/utils/networks.py`
- Sinusoidal step encoding: `third_party/ogbench/impls/utils/positional_encoding.py`
- Training driver + evaluation hooks: `third_party/ogbench/impls/main.py`
- SLURM scripts (canonical settings): `slurm/`
