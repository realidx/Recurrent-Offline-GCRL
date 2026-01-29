# Project Plan: Recurrent Critic for Stitching in OGBench CRL

## Goals (what you’ll claim)
Demonstrate that a **tied, iterated critic** (a single critic “block” unrolled for *K* iterations with shared weights) can:

- **Type A:** Beat the baseline on `antmaze-*-stitch` (primary: `antmaze-large-stitch`).
- **Type B:** Match an **untied deep critic** with **fewer critic parameters**.
- **Type C:** Improve with **test-time compute** via **critic-guided action refinement** (evaluation-only).
- **Optional:** For tied critics only, performance improves as you increase **test-time iterations** `K_test` (a “knob” unavailable to untied models).

## Ground rules (so comparisons are defensible)
Hold constant across comparisons:

- Same dataset + same task split (train/eval) + same data loader.
- Same CRL algorithm and training code path.
- Same training budget: environment steps (or gradient updates), batch size, update ratio, eval frequency.
- Same actor architecture (standard MLP) and actor training procedure.
- Only swap **critic architecture** (and only what is strictly necessary to support it, e.g., depth/iterations).
- Refinement is **evaluation-only** and applied **identically** to all models.

## Metrics & evaluation protocol (define before coding)
For every run, log at minimum:

- **Primary:** success rate (and any “hard success” metric used by the benchmark).
- **Secondary:** eval return / reward.
- **Stability:** Q statistics and loss curves (details in “Instrumentation”).
- **Compute:** steps-per-second (SPS) and wall-clock to reach a target success.
- **Capacity:** exact critic parameter count.

Use fixed seeds (e.g., 5) and report mean ± std.

## Experiment matrix (what you’ll run)
Define 3 critic families:

1. **Baseline**: repo’s default MLP critic (untied, shallow).
2. **Deep untied**: residual-block backbone (untied), sweep depth `D ∈ {3, 6, 12}`.
3. **Tied iterated (“recurrent depth”)**: one block applied *K* times with shared parameters (sweep `K_train`).

Suggested comparisons:

- **A (baseline vs tied):**
  - Baseline critic vs tied critic at matched *effective depth* and similar optimizer settings.
- **B (deep untied vs tied):**
  - Deep untied with depth `D` vs tied with `K=D` (or nearest), but tuned to have *fewer* params.
- **C (refinement):**
  - Apply identical action-refinement to all critics and compare uplift.
  - For tied critics: test-time knob `K_test > K_train` and measure gains vs extra compute.

## Critic architecture spec (write this down precisely)
### Baseline critic
The exact existing critic in CRL (do not change).

### Deep untied critic
Same input/output as baseline; increased depth/width by stacking layers/blocks **without weight tying**.

### Tied iterated critic (the contribution)
Let `f_θ` be a critic “block” that maps a hidden state to a hidden state (or directly refines Q features).
Unroll:

`h_0 = embed(s, a)`
`h_{k+1} = f_θ(h_k, s, a)` for `k = 0..K-1`
`Q(s,a) = head(h_K)`

Key knobs:

- `K_train`: unroll steps during training
- `K_test`: unroll steps during evaluation (may be ≥ `K_train`)

Implementation notes (to keep results interpretable):

- Keep I/O identical to baseline Q interface.
- Use residual connections per step (recommended for stability).
- Consider per-step normalization (LayerNorm) if Q blows up.
- If using truncated BPTT or stop-gradient tricks, pre-register them as an ablation (don’t sneak them in).

### What “depth” means in this project (avoid ambiguity)
Because CRL uses a bilinear score `phi(s,a)^T psi(g)` rather than a scalar-Q TD critic, define depth in terms of the
**phi/psi backbone**:

- **Baseline MLP depth**: number of hidden layers in the MLP used for `phi` and `psi` (OGBench default is 3 hidden layers).
- **Untied ResNet depth `D`**: number of residual blocks applied in `phi` and in `psi` (each block is LN → 2-layer MLP → residual).
- **Tied iterated depth `K`**: number of iterations of the tied block applied in `phi` and in `psi` (shared weights across iterations).

### Critic-guided action refinement (evaluation-only)
At each env step, given state `s`:

1. Initialize `a_0 = π(s)` (actor output).
2. For `t=0..T-1`, update action to increase critic value:
   - `a_{t+1} = clip(a_t + η * normalize(∂Q(s,a_t)/∂a), action_low, action_high)`
3. Execute `a_T`.

Keep identical for all critics:

- same `T` (refinement steps), `η` (step size), and action projection
- same random seeds and evaluation episodes
- same compute measurement methodology

Guardrails:

- Always clip/project actions to env bounds.
- Log refinement success/failure stats (NaNs, grad norms, delta-a magnitude).

## Phase 0 — Reproduce official CRL on AntMaze stitching tasks
### P0.1 Tasks and compute budget
Reproduce **all 4 AntMaze stitching configs** (CRL, unchanged):

- `antmaze-medium-stitch-v0`
- `antmaze-large-stitch-v0`
- `antmaze-giant-stitch-v0`
- `antmaze-teleport-stitch-v0`

Define a single immutable “budget constant” for all runs:

- env steps (or gradient updates) exactly equal
- batch size equal
- eval frequency equal

Notes:

- For “official repro”, use the authors’ per-task hyperparameters (see `third_party/ogbench/impls/hyperparameters.sh`).
- For later architecture comparisons, keep the **training budget constant within each task**.

### P0.2 Repro checklist
Run official CRL unchanged and log:

- success rate (+ hard metric if available)
- eval reward/return
- training curves: critic loss, actor loss, temperature/alpha (if used)
- wall-clock throughput (SPS)

Pass condition:

- performance is “close enough” to reported numbers for the repo/benchmark
- training is stable across multiple seeds (stability > perfect match)

### P0.3 Add instrumentation early (before changing models)
Add logging for:

- `Q_mean`, `Q_std`, `Q_abs_max` (over a fixed-size batch every N updates)
- critic grad norm (pre-clip and post-clip if clipping exists)
- actor action norm stats (mean/std, abs max), and action saturation fraction
- target-Q stats (same summary stats as Q)

Also helpful:

- TD error stats (mean/std/abs max)
- fraction of NaNs/Infs in Q, targets, grads
- `K_train` / `K_test` and refinement settings (so runs are self-describing)

## Phase 1 — Critic-backbone variants inside CRL (architecture-only swap)
In this project, the “critic” is CRL’s differentiable **score function**:

- `score(s,a,g) = phi(s,a)^T psi(g) / sqrt(d)` (the same logit/energy used by CRL),

so Phase 1 swaps only the **phi/psi backbones** while keeping the CRL loss, ensemble semantics, and call sites unchanged.

1. Factor critic creation behind a single config switch (baseline / deep-untied / tied-iterated).
2. Ensure optimizer groups and target-network update logic still match the baseline.
3. Add a param-count logger for critic and actor separately.
4. Add a “sanity eval” that runs a tiny batch forward/backward and asserts finite tensors.

Exit criteria:

- swapping critic variants changes *only* critic params and compute, not the rest of the pipeline
- identical training loop and logging across variants

## Phase 2 — Type A and Type B experiments (train-time comparisons)
### A: Beat baseline on stitch
Grid (keep small):

- `K_train ∈ {3, 6, 12}` (tied)
- match baseline optimizer/lr; only adjust if clearly unstable

Report:

- best tied setting vs baseline under identical budget
- seed robustness (mean ± std)

### B: Match the best untied depth with fewer parameters
Do not assume “deeper is better” in offline RL. Instead, **sweep untied depth** and pick the best.

1. Sweep deep untied ResNet depth `D ∈ {3, 6, 12}` (same width as baseline).
2. Pick the best untied setting by mean success (with fixed budget).
3. Compare tied iterated using `K_train ∈ {3, 6, 12}` (matched-depth set vs the untied ResNet sweep) and tune width only if needed so:

- `params(tied) < params(deep_untied)`

Report:

- success vs param count
- success vs wall-clock (compute efficiency)

## Phase 3 — Type C experiments (test-time compute)
1. Implement refinement (evaluation-only).
2. Run refinement on baseline / deep untied / tied at the same `T, η`.
3. For tied critics only: sweep `K_test` (e.g., `K_test ∈ {K_train, 2K_train, 4K_train}`).

Report:

- uplift from refinement (absolute and relative)
- uplift from increasing `K_test` (compute knob)
- measured eval-time overhead (wall-clock per episode)

## Phase 4 — Analysis & reporting package (what makes it publishable)
Produce plots/tables:

- learning curves with confidence bands (success vs steps)
- bar chart: best success per model family with identical budget
- Pareto: success vs critic params; success vs wall-clock
- refinement ablation: `T` steps and `K_test` sweep
- stability diagnostics: Q stats and grad norms vs time

Write down:

- exact hyperparams and seeds
- compute budget definition and enforcement
- failure cases (if any) and how you prevented cherry-picking
