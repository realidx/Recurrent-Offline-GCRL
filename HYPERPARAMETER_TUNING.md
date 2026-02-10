# Hyperparameter Tuning (Current)

Last updated: 2026-02-10

This doc records the *current* hyperparameters we use in experiments and a small set of tuning knobs we consider
scientifically acceptable (without drifting into “algorithm changes”).

For architecture definitions, see `COMPLETE_SPECIFICATION.md`.

---

## 1) Canonical Settings (What We Try Not to Change)

### 1.1 Training budget + logging
- `train_steps=1_000_000`
- `log_interval=20_000`
- `validation_log_interval=100_000`
- `eval_interval=200_000`
- `save_interval=200_000`
- eval during training: `eval_tasks=5`, `eval_episodes=20`
- final eval: `eval_tasks=5`, `eval_episodes=50`

### 1.2 Optimizer + batch
Defaults come from `third_party/ogbench/impls/agents/crl.py:get_config()` unless overridden in SLURM:
- `--agent.lr=3e-4`
- `--agent.lr_decay_steps=train_steps` (when enabled; otherwise 0 = constant LR)
- `--agent.lr_min=1e-5`
- `--agent.batch_size=1024`

### 1.3 Actor (fixed across critic comparisons)
- architecture: `actor_hidden_dims=(512,512,512)`
- loss: `actor_loss=ddpgbc`
- behavior cloning strength: `--agent.alpha=0.1` (canonical)

Goal sampling (stitch tasks; canonical in our runs):
- `--agent.actor_p_randomgoal=0.5`
- `--agent.actor_p_trajgoal=0.5`

### 1.4 Critic (fixed algorithm; tunable backbone only)
Shared dimensions (canonical):
- `value_hidden_dims=(512,512,512)`
- `latent_dim=512`
- `ensemble=2` critics (min used for actor update)
- `layer_norm=True`
- `--agent.critic_layerscale_init=1e-2`

Step-table / iteration limit:
- `--agent.critic_recur_max_iters=16` (canonical)
  - some sinusoidal runs used `24`; keep that explicit if you do it.

Optional width override (parameter-match ablations only):
- `--agent.critic_backbone_hidden_dim=<int>` (0 disables)

### 1.5 Discount (task-dependent; follow OGBench hyperparameters)
- default: `--agent.discount=0.99`
- “giant” tasks (paper default): `--agent.discount=0.995`

---

## 2) Architecture Knobs (Our Main “Scaling” Axes)

We use the paper-friendly `(G, T)` view:
- `G` = stack depth (number of distinct residual blocks)
- `T` = outer iterations (how many times to apply the stack)

Mapping to flags:
- feedforward ResNet baseline: `--agent.critic_backbone=resnet` + `--agent.critic_resnet_depth=G` (T=1, no FiLM)
- tied iteration: `--agent.critic_backbone=recur_tied` + `--agent.critic_recur_iters=T` (G=1)
- stack-and-iterate: `--agent.critic_backbone=partial_tied` + `--agent.critic_partial_groups=G` +
  `--agent.critic_partial_iters_per_group=T` +
  `--agent.critic_partial_block_schedule=stack` (recommended schedule)

Partial-only knobs (keep fixed unless explicitly ablated):
- `--agent.critic_partial_cycle_step_params=1`
- `--agent.critic_partial_per_group_film=0`

---

## 3) Step Conditioning (Discrete vs Sinusoidal)

Default (discrete lookup table):
- `--agent.critic_recur_sinusoidal=0`

Sinusoidal (extrapolatable step encoding; used in Phase 3 experiments):
- `--agent.critic_recur_sinusoidal=1`

Notes:
- sinusoidal removes dependence on a learned `step_embed[k]` table, but still uses FiLM and `alpha[k]`.
- for test-time `T` changes, discrete requires `max_iters` large enough; sinusoidal avoids “undefined step embeddings” but
  is not guaranteed to improve performance by itself.

---

## 4) LayerNorm Tying (Checkpoint Compatibility)

Current default:
- `--agent.critic_recur_tied_ln=1` (true tied LN module across iterations)

If you need to load an old checkpoint trained with per-iteration LN modules:
- set `--agent.critic_recur_tied_ln=0` at restore time.

---

## 5) Contrastive Loss Variant (Optional Ablation)

Default (canonical):
- `--agent.contrastive_loss_type=bce`

Optional:
- `--agent.contrastive_loss_type=infonce`
- `--agent.infonce_temperature=0.1`

This is an *algorithmic* change (loss change), so it should be treated as a separate ablation line, not part of the main
architecture-only comparison.

---

## 6) Evaluation-Only Knobs (Ablations)

### 6.1 Action refinement
Flags:
- `--eval_refine_steps`
- `--eval_refine_lr`
- `--eval_refine_l2`

Current finding:
- refinement has not produced consistent improvements across seeds in our runs; keep it off (`steps=0`) unless explicitly
  running a refinement ablation.

### 6.2 Video
Use `--video_episodes=2` (and keep `eval_episodes` > 0 unless you’re OK with missing summary stats).

---

## 7) Recommended Minimal Sweeps (Compute-Aware)

When moving to a new task (e.g., `antmaze-giant-navigate-v0`), keep everything canonical and only sweep:

1) Tied iterations:
- `recur_tied`: `T ∈ {4, 6, 8}` (with 3 seeds if possible)

2) Stack depth vs iterations (if you need the UT-style comparison):
- `(G, T) ∈ {(1,4), (2,2), (2,4), (4,1)}` with `partial_block_schedule=stack`

3) Step encoding:
- discrete vs sinusoidal (only if you need “test-time scaling” story)

Avoid adding new free knobs (width, batch size, alpha, etc.) unless you have a clear diagnostic reason.
