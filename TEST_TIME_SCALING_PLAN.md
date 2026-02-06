# Test-Time Compute Scaling Implementation Plan

**Created:** 2026-02-06
**Status:** Planning Phase
**Context:** Phase 3 K_test sweep (4→8→16) showed no improvement because iterations k≥4 were never trained

---

## 🎯 Two Distinct Approaches

### **Option 1: Anytime Prediction (Recommended First Step)**
- **What:** Train with random K ∈ [4, 16], test within that range
- **Generalization type:** Interpolation (test K within training distribution)
- **Implementation difficulty:** Low (5-10 lines of code)
- **Comparable to current model:** Yes (same architecture, just different training)
- **Research contribution:** "Non-hierarchical test-time compute scaling through anytime recurrent critics"

### **Option 2: True Test-Time Scaling (Advanced)**
- **What:** Train with random K ∈ [4, 16], test at K > 16 (e.g., 20, 24, 32)
- **Generalization type:** Extrapolation (test K beyond training distribution)
- **Implementation difficulty:** High (requires architectural redesign)
- **Comparable to current model:** No (different architecture = different experiment)
- **Research contribution:** "Extrapolative test-time compute scaling in offline RL"

---

## 📋 Option 1: Anytime Prediction Implementation

### **Core Idea**
Currently, `critic_recur_iters=4` is fixed during training → only step_embed[0:3] and alpha[0:3] get gradient updates. With random-K training, ALL iterations [4, 16] receive gradients, enabling the model to produce good Q-values at any K.

### **Code Changes**

#### **Change 1: Random-K Sampling in Critic Loss**

**File:** `third_party/ogbench/impls/agents/crl.py`
**Location:** Around line 145 (inside `contrastive_loss` method)

**Current code:**
```python
def contrastive_loss(self, batch, grad_params, net):
    """Contrastive loss for the critic."""
    # ... existing code ...

    # Critic forward pass (line ~60-80 depending on version)
    s_repr = self.network.select(net)(
        batch['observations'],
        batch['actions'],
        params=grad_params
    )
```

**Modified code:**
```python
def contrastive_loss(self, batch, grad_params, net, rng=None):
    """Contrastive loss for the critic."""
    # ... existing code ...

    # Sample random K for this batch (anytime prediction)
    if self.config.get('critic_recur_anytime', False):
        if rng is None:
            rng = self.rng
        rng, key = jax.random.split(rng)
        # Sample uniformly from [K_min, K_max]
        k_min = self.config.get('critic_recur_iters', 4)
        k_max = self.config.get('critic_recur_max_iters', 16)
        num_iters = jax.random.randint(key, shape=(), minval=k_min, maxval=k_max + 1)
    else:
        num_iters = None  # Use default from config

    # Critic forward pass with sampled K
    s_repr = self.network.select(net)(
        batch['observations'],
        batch['actions'],
        params=grad_params,
        num_iters=num_iters  # Pass to backbone
    )
```

**Key insight:** By sampling `num_iters` per batch, different gradient updates see different iteration counts → all step_embed[4:15] and alpha[4:15] receive training signal.

#### **Change 2: Add Config Flag**

**File:** `third_party/ogbench/impls/agents/crl.py`
**Location:** Around line 354 (config defaults)

```python
# Add to get_default_config()
critic_recur_anytime=False,  # Enable random-K anytime training
```

#### **Change 3: SLURM Script for Anytime Training**

**New file:** `slurm/phase3_train_anytime.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=p3_anytime
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --array=0-2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# ... (standard setup from phase3_train_recur_tied_ktrain.slurm) ...

AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
  "--agent.batch_size=1024"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"           # K_min
  "--agent.critic_recur_max_iters=16"      # K_max
  "--agent.critic_recur_anytime=1"         # NEW: Enable random-K training
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_layerscale_init=1e-2"
  "--agent.lr_decay_steps=${LR_DECAY_STEPS}"
  "--agent.lr_min=${LR_MIN}"
)

RUN_GROUP="P3_Anytime_K4to16"
EXP_NAME="sd$(printf %03d ${SEED})_anytime_k4to16_lrd1000000"
```

#### **Change 4: Evaluation Sweep**

**Modify:** `slurm/phase3_eval_ktest_sweep.slurm`

```bash
# Source checkpoints from anytime training run
TRAIN_RUN_GROUP="${TRAIN_RUN_GROUP:-P3_Anytime_K4to16}"
TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_anytime_k4to16_lrd1000000"

# Test at K ∈ {4, 8, 12, 16} with refinement
K_TESTS=(4 8 12 16)
REFINE_STEPS=(10)
```

### **Expected Results**

#### **Hypothesis:**
Success rate should increase monotonically with K_test:
- K_test=4: ~0.37 (baseline, matching current fixed-K=4 performance)
- K_test=8: ~0.42 (more iterations → better Q-estimates)
- K_test=12: ~0.45
- K_test=16: ~0.48 (best within trained range)

#### **Why this works:**
1. All iterations [4, 16] trained → step_embed and alpha learned for all k
2. More iterations = more refinement = better Q(s,a,g) estimates
3. Action refinement gradient ascent benefits from better Q-gradients

#### **Failure modes:**
- **Flat scaling curve:** If all K_test give same performance → critic not learning to refine progressively
- **U-shaped curve:** If middle K_test worse than extremes → instability in iteration dynamics
- **Worse than baseline:** If anytime training hurts K=4 performance → random-K is too noisy as a training signal

### **Timeline Estimate**
- **Code changes:** 30 minutes
- **Training (3 seeds):** ~9.5 hours per seed (same as current K=4 runs)
- **Evaluation sweep:** ~2 hours (4 K_test × 50 episodes × 5 tasks)
- **Total:** ~1.5 days (including queue time)

---

## 🔬 Option 2: True Test-Time Scaling (Extrapolation)

### **Why Current Architecture Cannot Extrapolate**

Looking at [networks.py:114-119](third_party/ogbench/impls/utils/networks.py#L114-L119):

```python
step_embed = self.param(
    'step_embed',
    nn.initializers.normal(stddev=0.02),
    (self.max_iters, self.hidden_dim),  # Fixed-size table
)
alpha = self.param('alpha', nn.initializers.constant(self.layerscale_init), (self.max_iters,))
```

**The problem:**
- `step_embed` is a **lookup table** of shape `(max_iters, hidden_dim)`
- At iteration k, we index: `step_embed[k]` → embeddings for k ≥ max_iters **do not exist**
- Even with anytime training, testing at K_test=32 requires `step_embed[32]` which is undefined

**Why this is fundamental:**
- Discrete step indices → discrete parameters → no extrapolation beyond max_iters
- Similar to positional embeddings in Transformers: you cannot process sequence length 2048 if you only have embeddings for positions [0, 1024)

### **Required Architectural Changes**

#### **Change A: Continuous Positional Encoding**

Replace discrete lookup table with a **learned function** `f: ℝ → ℝ^h` that maps normalized iteration progress to embeddings.

**Option A1: Sinusoidal Encoding (Transformer-style)**
```python
def sinusoidal_step_encoding(k, max_iters, hidden_dim):
    """
    Encode step k as sinusoidal position (extrapolatable).

    Args:
        k: Current iteration (can be > max_iters at test time)
        max_iters: Training max iterations (used for normalization)
        hidden_dim: Embedding dimension

    Returns:
        step_embed: (hidden_dim,) encoding of iteration k
    """
    # Normalize k to [0, 1] range (allows extrapolation beyond 1.0)
    t = k / max_iters

    # Create frequency bands (same as Transformer)
    i = jnp.arange(hidden_dim // 2)
    freq = 1.0 / (10000 ** (2 * i / hidden_dim))

    # Sinusoidal encoding
    angles = t * freq
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])
    return emb
```

**Pros:**
- Zero learnable parameters for step encoding
- Smooth interpolation and extrapolation
- Proven in Transformers (ALiBi, RoPE variants support extrapolation)

**Cons:**
- Fixed functional form (less flexible than learned embeddings)
- May not capture task-specific iteration dynamics

**Option A2: Learned MLP Encoding**
```python
class StepEncoder(nn.Module):
    """
    Learned function mapping normalized step → embedding.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, k, max_iters):
        # Normalize to [0, 1] (extrapolation means t > 1.0)
        t = k / max_iters

        # MLP: scalar → hidden_dim
        h = nn.Dense(64)(t[None])  # (1,) → (64,)
        h = nn.silu(h)
        h = nn.Dense(128)(h)
        h = nn.silu(h)
        emb = nn.Dense(self.hidden_dim)(h)  # → (hidden_dim,)
        return emb[0]
```

**Pros:**
- Learnable function → can adapt to data
- Smooth by construction (MLP is continuous)

**Cons:**
- Requires training signal at diverse k to learn extrapolation
- May overfit to training range [4, 16] if not regularized

#### **Change B: Iteration-Invariant Alpha**

Current: `alpha[k]` is per-iteration → needs alpha[20], alpha[24], etc. for extrapolation.

**Option B1: Learned Function of k**
```python
# Replace fixed alpha array with learned scalar function
alpha_mlp = nn.Dense(1)  # Input: step_embed, Output: scalar

# In loop:
alpha_k = alpha_mlp(step_embed_k)  # Compute alpha as function of step embedding
h = h + alpha_k * u
```

**Option B2: Fixed Schedule**
```python
# Simple geometric decay (no learnable params)
alpha_k = layerscale_init * (0.9 ** k)
```

### **Modified RecurTiedBackbone (Extrapolatable Version)**

```python
class RecurTiedBackboneExtrapolate(nn.Module):
    """
    Weight-tied recurrent backbone with EXTRAPOLATABLE positional encoding.
    """
    hidden_dim: int
    out_dim: int
    num_iters: int
    max_iters: int = 16  # Training max; can test beyond this
    layer_norm: bool = True
    tied_layer_norm: bool = True
    layerscale_init: float = 1e-2
    step_encoding: str = 'sinusoidal'  # or 'learned_mlp'

    @nn.compact
    def __call__(self, x, num_iters=None):
        iters = self.num_iters if num_iters is None else int(num_iters)
        # NOTE: No hard limit check — allow K > max_iters at test time

        h = nn.Dense(self.hidden_dim)(x)

        # Step encoder: continuous function (not lookup table)
        if self.step_encoding == 'sinusoidal':
            step_encoder = lambda k: sinusoidal_step_encoding(k, self.max_iters, self.hidden_dim)
        elif self.step_encoding == 'learned_mlp':
            step_encoder = StepEncoder(self.hidden_dim)

        # Alpha function (instead of fixed array)
        alpha_fn = nn.Dense(1, name='alpha_mlp')  # Input: step_embed → scalar

        # FiLM and tied layers (same as before)
        film_fc1_step = nn.Dense(self.hidden_dim, use_bias=False)
        film_fc1_h = nn.Dense(self.hidden_dim)
        film_fc2 = nn.Dense(2 * self.hidden_dim)
        tied_fc1 = nn.Dense(self.hidden_dim)
        tied_fc2 = nn.Dense(self.hidden_dim)
        ln = nn.LayerNorm() if self.layer_norm and self.tied_layer_norm else None

        for k in range(iters):
            # Compute step embedding on-the-fly (not precomputed)
            step_emb_k = step_encoder(k) if self.step_encoding == 'sinusoidal' else step_encoder(k, self.max_iters)

            h1 = ln(h) if ln is not None else h

            # FiLM conditioning
            fc1_out = film_fc1_step(step_emb_k) + film_fc1_h(h1)
            film = film_fc2(nn.silu(fc1_out))
            gamma, beta = jnp.split(film, 2, axis=-1)
            h_film = (1.0 + gamma) * h1 + beta

            # Tied MLP
            u = tied_fc2(nn.silu(tied_fc1(h_film)))

            # Learned alpha as function of step
            alpha_k = nn.sigmoid(alpha_fn(step_emb_k)) * 0.1  # Bound to [0, 0.1]
            h = h + alpha_k * u

        return nn.Dense(self.out_dim)(h)
```

### **Why This Is Not Comparable to Current Model**

#### **1. Different Inductive Bias**
- **Current (discrete):** Each iteration k has independent embedding → flexible but non-extrapolatable
- **New (continuous):** Iterations share functional structure → extrapolatable but constrained

**Analogy:** Discrete is like a lookup table; continuous is like a polynomial fit. The polynomial can extrapolate, but it's a fundamentally different model.

#### **2. Training Dynamics**
- **Current:** Gradients update independent `step_embed[k]` entries
- **New:** Gradients flow through shared `step_encoder` MLP → all iterations coupled

**Effect:** Harder optimization. The MLP must learn a single function that works for all k ∈ [4, 16], whereas the current model can specialize each k independently.

#### **3. Parameter Count Mismatch**
**Current:**
```
step_embed: max_iters × hidden_dim = 16 × 512 = 8,192 params
alpha: max_iters = 16 params
Total: 8,208 params
```

**New (sinusoidal):**
```
step_embed: 0 params (closed-form function)
alpha_mlp: hidden_dim → 1 = 513 params
Total: 513 params (16× fewer!)
```

**New (learned MLP encoder):**
```
step_encoder: 1→64 + 64→128 + 128→512 = 64 + 8,192 + 65,536 = 73,792 params (9× more!)
alpha_mlp: 513 params
Total: 74,305 params
```

**Implication:** Not a fair comparison. The sinusoidal version has far fewer params (undercapacity), and the learned MLP version has far more (overcapacity).

#### **4. Research Contribution Shifts**
- **Anytime (Option 1):** "Recurrent critics can scale test-time compute within their training range"
  - Claim: Architecture A with training protocol B achieves property C
  - Baseline: Same architecture A with training protocol B' (fixed-K)
  - **Fair comparison:** Only training differs

- **Extrapolation (Option 2):** "Continuous positional encodings enable extrapolative test-time scaling"
  - Claim: Architecture X with continuous encodings extrapolates better than architecture Y with discrete encodings
  - Baseline: Architecture Y (current discrete step_embed)
  - **Unfair comparison:** Both architecture AND training differ

### **If You Still Want to Pursue Option 2**

#### **Fair Comparison Protocol:**
1. **Baseline 1:** Current discrete RecurTied, trained with random-K ∈ [4, 16], tested at K=16 (anytime)
2. **Baseline 2:** Current discrete RecurTied, trained with random-K ∈ [4, 16], tested at K=20 (fails — out of bounds)
3. **Proposed:** Continuous RecurTied, trained with random-K ∈ [4, 16], tested at K ∈ {16, 20, 24, 32}

**Claim:** "Continuous encodings enable extrapolation beyond training range"

**Experiments:**
- Train both on same data, same hyperparams
- Show Baseline 1 matches Proposed at K=16 (within-distribution parity)
- Show Baseline 2 crashes at K=20 (out-of-bounds)
- Show Proposed maintains performance at K=20, 24, 32 (extrapolation)

#### **Expected Challenges:**
1. **Optimization instability:** Shared functional encoder may hurt convergence
2. **Extrapolation brittleness:** Model may learn "iteration 16 is the end" implicitly
3. **Diminishing returns:** K=20 may not help if the problem is solvable at K=16
4. **Engineering complexity:** Requires extensive hyperparameter tuning for the step_encoder MLP

---

## 🎯 Recommendation

### **Start with Option 1 (Anytime Prediction)**

**Why:**
1. ✅ **Simple:** 5-10 lines of code, existing architecture
2. ✅ **Fair:** Only training protocol differs from baseline
3. ✅ **Fast:** Results in ~1.5 days
4. ✅ **Interpretable:** Clear scaling curve K=4→8→12→16
5. ✅ **Publishable:** "Non-hierarchical test-time compute scaling" is a valid contribution

**Success criteria:**
- Monotonic improvement: success(K=16) > success(K=12) > success(K=8) > success(K=4)
- Absolute performance: success(K=16) ≥ 0.45 (current best is ~0.37 at K=4)

**If this works → strong paper section on test-time scaling within trained range**

### **Consider Option 2 only if:**
1. ✅ Option 1 shows promising scaling (validates the approach)
2. ✅ You have 2-3 weeks for architectural experiments
3. ✅ Extrapolation K > 16 is scientifically interesting for your domain (e.g., antmaze-ultra with 50-step horizons where K=16 is insufficient)

**Risk assessment:**
- 🔴 High complexity
- 🟡 Unfair comparison (different architecture)
- 🟡 Unclear reward (extrapolation may not help if K=16 is enough)

---

## 📊 Comparison Table

| Aspect | Current (Fixed-K=4) | Option 1 (Anytime) | Option 2 (Extrapolate) |
|--------|---------------------|-------------------|------------------------|
| **Architecture** | RecurTied (discrete step_embed) | Same | RecurTied (continuous encoder) |
| **Training K** | Fixed K=4 | Random K ∈ [4, 16] | Random K ∈ [4, 16] |
| **Test K range** | K=4 only | K ∈ [4, 16] | K ∈ [16, 32+] |
| **Generalization** | None | Interpolation | Extrapolation |
| **Code changes** | N/A (baseline) | ~10 lines | ~200 lines |
| **Training time** | 9.5 hrs | 9.5 hrs (same) | 10-12 hrs (MLP encoder overhead) |
| **Comparable?** | N/A | ✅ Yes (same arch) | ❌ No (different arch) |
| **Risk** | N/A | 🟢 Low | 🔴 High |
| **Research value** | Baseline | High (if works) | Very high (if extrapolates) |

---

## 🛠️ Implementation Checklist (Option 1)

### **Phase 1: Code Changes (Day 1, Morning)**
- [ ] Add `critic_recur_anytime` config flag to `crl.py`
- [ ] Modify `contrastive_loss` to sample random K per batch
- [ ] Ensure `num_iters` argument propagates to `RecurTiedBackbone.__call__`
- [ ] Create `slurm/phase3_train_anytime.slurm` script
- [ ] Test locally: `python main.py --agent.critic_recur_anytime=1 --train_steps=100` (quick smoke test)

### **Phase 2: Training (Day 1-2)**
- [ ] Submit 3-seed training job: `sbatch slurm/phase3_train_anytime.slurm`
- [ ] Monitor first 100k steps for training stability (check `critic/loss` doesn't spike)
- [ ] Wait for 1M step completion (~9.5 hrs × 3 seeds = ~30 hrs wall time with parallelism)

### **Phase 3: Evaluation (Day 2-3)**
- [ ] Modify `phase3_eval_ktest_sweep.slurm` to source anytime checkpoints
- [ ] Submit eval sweep: K_test ∈ {4, 8, 12, 16} × REFINE ∈ {0, 10} × 3 seeds
- [ ] Collect `eval.csv` results from all runs

### **Phase 4: Analysis (Day 3)**
- [ ] Plot scaling curves: success_rate vs K_test (separate lines for refine=0 vs refine=10)
- [ ] Compare to fixed-K=4 baseline (~0.37 mean success)
- [ ] Statistical significance test (bootstrap 95% CI)

---

**Next Action:** Await user decision on Option 1 vs Option 2 before proceeding with implementation.
