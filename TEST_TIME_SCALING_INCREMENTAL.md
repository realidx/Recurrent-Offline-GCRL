# Incremental Test-Time Scaling Validation

**Proposed Experiment:** Train sinusoidal encoding at K=4, test extrapolation to K=8
**Rationale:** Fast validation of continuous encoding concept before full commitment
**Timeline:** ~2 days (vs 2 weeks for full Option 2)

---

## 🎯 Experimental Design

### **The Key Insight**

Your idea is brilliant because it separates two questions:
1. **Does continuous encoding work at all?** (Can it extrapolate even 2×?)
2. **Is it worth the complexity?** (Performance vs discrete baseline)

**Testing at K=8 answers #1 with minimal investment.**

---

## 📊 Experiment Matrix

| Model | Train K | Test K | Extrapolation | Purpose |
|-------|---------|--------|---------------|---------|
| **Baseline (discrete)** | K=4 | K=4 | None | Current best (~0.37) |
| **Discrete (fail control)** | K=4 | K=8 | ❌ Crashes | Confirm discrete can't extrapolate |
| **Sinusoidal** | K=4 | K=4 | None | Parity check with baseline |
| **Sinusoidal** | K=4 | K=8 | 2× | **KEY TEST** — does it work? |

**Decision criteria:**
- ✅ **Success:** Sinusoidal @ K=8 ≥ 90% of Sinusoidal @ K=4
  - Proves smooth extrapolation works → pursue full Option 2
- ❌ **Failure:** Sinusoidal @ K=8 < 80% of Sinusoidal @ K=4
  - Extrapolation breaks down → stick with Option 1 (anytime)

---

## 🔧 Implementation Plan

### **Phase 1: Add Sinusoidal Encoding (2 hours)**

#### **Step 1: Create sinusoidal encoding function**

**New file:** `third_party/ogbench/impls/utils/positional_encoding.py`

```python
"""Continuous positional encodings for extrapolatable recurrent critics."""

import jax.numpy as jnp


def sinusoidal_step_encoding(k, max_iters, hidden_dim):
    """
    Sinusoidal positional encoding for iteration step k.

    Enables extrapolation beyond max_iters (unlike discrete lookup table).

    Args:
        k: Iteration index (int or float, can exceed max_iters)
        max_iters: Normalization constant from training (e.g., 16)
        hidden_dim: Output embedding dimension (must be even)

    Returns:
        embedding: (hidden_dim,) array of sinusoidal features

    Example:
        >>> emb_4 = sinusoidal_step_encoding(k=4, max_iters=16, hidden_dim=512)
        >>> emb_8 = sinusoidal_step_encoding(k=8, max_iters=16, hidden_dim=512)
        >>> # emb_8 is well-defined even if only trained with k ≤ 4
    """
    # Normalize to [0, 1] range (allows t > 1.0 at test time)
    t = k / max_iters

    # Frequency bands (Transformer-style)
    # freq[0] = 1.0 (high frequency), freq[-1] ≈ 1e-4 (low frequency)
    i = jnp.arange(hidden_dim // 2)
    freq = 1.0 / (10000.0 ** (2.0 * i / hidden_dim))

    # Compute angles and sinusoidal components
    angles = t * freq  # Shape: (hidden_dim // 2,)
    sin_component = jnp.sin(2.0 * jnp.pi * angles)
    cos_component = jnp.cos(2.0 * jnp.pi * angles)

    # Interleave [sin, cos, sin, cos, ...]
    embedding = jnp.concatenate([sin_component, cos_component])
    return embedding
```

#### **Step 2: Modify RecurTiedBackbone**

**File:** `third_party/ogbench/impls/utils/networks.py`
**Location:** Inside `RecurTiedBackbone` class (around line 97)

**Change 1: Add config parameter**
```python
class RecurTiedBackbone(nn.Module):
    hidden_dim: int
    out_dim: int
    num_iters: int
    max_iters: int = 16
    layer_norm: bool = True
    tied_layer_norm: bool = True
    layerscale_init: float = 1e-2
    use_sinusoidal_step_encoding: bool = False  # NEW: Enable continuous encoding
```

**Change 2: Replace step_embed with conditional logic**

Replace this (lines 114-119):
```python
step_embed = self.param(
    'step_embed',
    nn.initializers.normal(stddev=0.02),
    (self.max_iters, self.hidden_dim),
)
```

With this:
```python
if not self.use_sinusoidal_step_encoding:
    # Discrete lookup table (current approach)
    step_embed = self.param(
        'step_embed',
        nn.initializers.normal(stddev=0.02),
        (self.max_iters, self.hidden_dim),
    )
else:
    # Continuous sinusoidal encoding (extrapolatable)
    from utils.positional_encoding import sinusoidal_step_encoding
    step_embed = None  # Not used; computed on-the-fly
```

**Change 3: Update loop to use sinusoidal encoding when enabled**

Replace this (line 150):
```python
fc1_out = step_contributions[k] + film_fc1_h(h1)
```

With this:
```python
if not self.use_sinusoidal_step_encoding:
    # Discrete: use precomputed step_contributions
    step_contrib_k = step_contributions[k]
else:
    # Continuous: compute on-the-fly
    from utils.positional_encoding import sinusoidal_step_encoding
    step_emb_k = sinusoidal_step_encoding(k, self.max_iters, self.hidden_dim)
    step_contrib_k = film_fc1_step(step_emb_k)

fc1_out = step_contrib_k + film_fc1_h(h1)
```

**Note:** We still compute step_contributions for discrete mode (lines 139-140):
```python
if not self.use_sinusoidal_step_encoding:
    step_contributions = film_fc1_step(step_embed)
```

#### **Step 3: Add config flag to CRL agent**

**File:** `third_party/ogbench/impls/agents/crl.py`
**Location:** Line 354 (config defaults)

Add this flag:
```python
# Critic architecture
critic_backbone='mlp',
critic_backbone_hidden_dim=512,
critic_resnet_depth=3,
critic_recur_iters=4,
critic_recur_max_iters=16,
critic_recur_tied_ln=False,
critic_recur_sinusoidal=False,  # NEW: Use sinusoidal step encoding
```

**Propagate to network creation** (around line 200-250 where networks are built):

Find where RecurTiedBackbone is instantiated:
```python
if config['critic_backbone'] == 'recur_tied':
    backbone = RecurTiedBackbone(
        hidden_dim=config['critic_backbone_hidden_dim'],
        out_dim=config['latent_dim'],
        num_iters=config['critic_recur_iters'],
        max_iters=config['critic_recur_max_iters'],
        tied_layer_norm=config['critic_recur_tied_ln'],
        layerscale_init=config.get('critic_layerscale_init', 1e-2),
        use_sinusoidal_step_encoding=config['critic_recur_sinusoidal'],  # NEW
    )
```

---

### **Phase 2: Create Training Script (30 minutes)**

**New file:** `slurm/phase3_sinusoidal_k4.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=p3_sin_k4
#SBATCH --partition=gpu-long
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --array=0-2

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Same setup as phase3_train_recur_tied_ktrain.slurm...
# (copy lines 15-68 from that file)

SEEDS=(0 1 2)
idx="${SLURM_ARRAY_TASK_ID}"
SEED="${SEEDS[idx]}"

RUN_GROUP="P3_Sinusoidal_Incremental"
EXP_NAME="sd$(printf %03d ${SEED})_sin_k4_lrd1000000"

AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.1"
  "--agent.batch_size=1024"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=1"         # NEW: Enable sinusoidal encoding
  "--agent.critic_layerscale_init=1e-2"
  "--agent.lr_decay_steps=1000000"
  "--agent.lr_min=1e-5"
)

# ... rest same as phase3_train_recur_tied_ktrain.slurm
```

---

### **Phase 3: Create Evaluation Script (30 minutes)**

**New file:** `slurm/phase3_eval_sinusoidal_k4_to_k8.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=p3_sin_eval
#SBATCH --partition=gpu-long
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-5

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Array layout: seed (fastest) × ktest (slowest)
SEEDS=(0 1 2)
K_TESTS=(4 8)  # Test both within-range (4) and extrapolation (8)

idx="${SLURM_ARRAY_TASK_ID}"
seed_idx=$(( idx % ${#SEEDS[@]} ))
ktest_idx=$(( idx / ${#SEEDS[@]} ))

SEED="${SEEDS[seed_idx]}"
KTEST="${K_TESTS[ktest_idx]}"

# Source checkpoint from sinusoidal K=4 training
TRAIN_RUN_GROUP="P3_Sinusoidal_Incremental"
TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_sin_k4_lrd1000000"
RESTORE_PATH="${SLURM_SUBMIT_DIR}/exp/OGBench/${TRAIN_RUN_GROUP}/${TRAIN_BASE_NAME}"
RESTORE_EPOCH="1000000"

RUN_GROUP="P3_Sinusoidal_Eval_K4toK8"
EXP_NAME="sd$(printf %03d ${SEED})_sin_ktrain4_ktest${KTEST}_ref10"

# ... (same setup as phase3_eval_ktest_sweep.slurm) ...

cmd=(
  python main.py
  "--run_group=${RUN_GROUP}"
  "--save_dir=${SAVE_DIR}"
  "--env_name=${ENV_NAME}"
  "--seed=${SEED}"
  "--train_steps=0"
  "--eval_only=1"
  "--restore_path=${RESTORE_PATH}"
  "--restore_epoch=${RESTORE_EPOCH}"
  "--exp_name=${EXP_NAME}"
  "--eval_episodes=50"
  "--eval_refine_steps=10"
  "--agent=agents/crl.py"
  "--agent.critic_backbone=recur_tied"
  "--agent.critic_recur_iters=4"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_recur_tied_ln=1"
  "--agent.critic_recur_sinusoidal=1"         # Match training config
  "--agent.critic_layerscale_init=1e-2"
  "--agent.critic_eval_num_iters=${KTEST}"   # Override K at eval
  "--agent.alpha=0.1"
)

# ... rest same as phase3_eval_ktest_sweep.slurm
```

---

### **Phase 4: Control Experiment — Discrete K=4 → K=8 (Expect Crash)**

**Purpose:** Confirm that discrete embedding CANNOT extrapolate.

**File:** `slurm/phase3_eval_discrete_k4_to_k8_control.slurm`

```bash
#!/bin/bash
# Control: Try to eval discrete K=4 model at K=8 (should crash or use random embeddings)

# ... same structure as sinusoidal eval, but:
TRAIN_RUN_GROUP="P3_RecurTied_DynFiLM"  # Source from existing discrete baseline
TRAIN_BASE_NAME="sd$(printf %03d ${SEED})_ktrain4_a01_lrd1000000"

# Key difference: critic_recur_sinusoidal=0 (discrete)
"--agent.critic_recur_sinusoidal=0"
"--agent.critic_eval_num_iters=8"  # Try to use K=8 (out of trained range)
```

**Expected outcome:** Either crashes with IndexError, or uses `step_embed[4:7]` which are untrained (random) → poor performance.

---

## 📊 Expected Results

### **Scenario 1: Sinusoidal Works (Best Case)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete | 4 | 4 | 0.37 | Baseline |
| Discrete | 4 | 8 | **CRASH** or 0.10 | Confirms can't extrapolate |
| Sinusoidal | 4 | 4 | 0.35 | ~5% drop (acceptable) |
| Sinusoidal | 4 | 8 | **0.33** | Only 6% drop → smooth extrapolation! |

**Conclusion:** ✅ Continuous encoding works → proceed with full Option 2 (K=4→16, K=4→24, etc.)

---

### **Scenario 2: Sinusoidal Fails (Worst Case)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete | 4 | 4 | 0.37 | Baseline |
| Discrete | 4 | 8 | **CRASH** or 0.10 | Expected |
| Sinusoidal | 4 | 4 | 0.30 | ~20% drop (undercapacity) |
| Sinusoidal | 4 | 8 | **0.15** | 50% drop → extrapolation breaks |

**Conclusion:** ❌ Continuous encoding doesn't extrapolate well → stick with Option 1 (anytime with discrete)

---

### **Scenario 3: Mixed (Likely Case)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete | 4 | 4 | 0.37 | Baseline |
| Discrete | 4 | 8 | **CRASH** or 0.10 | Expected |
| Sinusoidal | 4 | 4 | 0.34 | ~8% drop (tolerable) |
| Sinusoidal | 4 | 8 | **0.28** | 18% drop → modest extrapolation |

**Conclusion:** 🟡 Extrapolation partially works → need to tune frequencies or try learned MLP

---

## 🎯 Decision Tree After Results

```
                    ┌─────────────────────────┐
                    │ Sinusoidal K=4 → K=8    │
                    │ result available        │
                    └───────────┬─────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Success ≥ 90%                   Success < 80%
                │                               │
                ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────────┐
    │ ✅ PROCEED          │       │ ❌ STOP                 │
    │ Full Option 2       │       │ Continuous encoding     │
    │                     │       │ doesn't work well       │
    │ Next: Train K=4,    │       │                         │
    │ test K∈{8,12,16,20} │       │ Fallback: Option 1      │
    │                     │       │ (anytime with discrete) │
    └─────────────────────┘       └─────────────────────────┘
```

---

## 💡 Why This Experiment Is Brilliant

### **1. Low Risk**
- Only 2× extrapolation (K=8) vs aggressive 4× (K=16→32)
- If it fails, we only lose 2 days, not 2 weeks

### **2. Fast Validation**
- Training: 9.5 hours (same as baseline)
- Eval: 2 hours (only 2 K_test values)
- Total: ~1.5 days

### **3. Clear Signal**
- If K=8 works → continuous encoding is viable
- If K=8 fails → saves weeks of wasted effort

### **4. Minimal Code Changes**
- Only ~100 lines (add sinusoidal function + conditional in RecurTiedBackbone)
- Easy to revert if it doesn't work

### **5. Fair Comparison**
- Compare Sinusoidal @ K=4 vs Discrete @ K=4 (parity check)
- Compare Sinusoidal @ K=8 vs Discrete @ K=8 (extrapolation test)

---

## 🔧 Implementation Checklist

### **Day 1 Morning (2 hours):**
- [ ] Create `utils/positional_encoding.py` with `sinusoidal_step_encoding`
- [ ] Modify `RecurTiedBackbone` to support `use_sinusoidal_step_encoding` flag
- [ ] Add `critic_recur_sinusoidal` config to `crl.py`
- [ ] Test locally: `python main.py --train_steps=100 --agent.critic_recur_sinusoidal=1`

### **Day 1 Afternoon (1 hour):**
- [ ] Create `slurm/phase3_sinusoidal_k4.slurm` training script
- [ ] Create `slurm/phase3_eval_sinusoidal_k4_to_k8.slurm` eval script
- [ ] Create `slurm/phase3_eval_discrete_k4_to_k8_control.slurm` control script
- [ ] Submit training: `sbatch slurm/phase3_sinusoidal_k4.slurm`

### **Day 2 Morning (wait for training):**
- [ ] Training completes (~9.5 hours)
- [ ] Submit eval: `sbatch slurm/phase3_eval_sinusoidal_k4_to_k8.slurm`
- [ ] Submit control: `sbatch slurm/phase3_eval_discrete_k4_to_k8_control.slurm`

### **Day 2 Afternoon (2 hours):**
- [ ] Evals complete
- [ ] Collect results from `eval.csv` files
- [ ] Plot: Success vs K_test (discrete vs sinusoidal)
- [ ] Decide: Proceed with full Option 2 or fallback to Option 1?

---

## 📈 Success Metrics

### **Parity Check (within training range):**
- Sinusoidal @ K=4 should be ≥ 90% of Discrete @ K=4
- If < 80% → sinusoidal encoding has too much inductive bias penalty

### **Extrapolation Check (2× beyond training):**
- Sinusoidal @ K=8 should be ≥ 90% of Sinusoidal @ K=4
- If < 80% → smooth extrapolation doesn't work for this architecture

### **Control Check:**
- Discrete @ K=8 should crash or perform terribly (< 0.15)
- Confirms that discrete can't extrapolate

---

## 🎬 What Happens Next?

### **If Successful:**
1. Write up K=4→K=8 result as proof-of-concept
2. Extend to K=4→K=16 (4× extrapolation)
3. Try aggressive extrapolation K=4→K=24, K=4→K=32
4. Compare anytime (Option 1) vs extrapolation (Option 2) in final paper

### **If Unsuccessful:**
1. Document why continuous encoding failed (undercapacity? bad frequencies?)
2. Fall back to Option 1 (anytime prediction with discrete embeddings)
3. Still publishable: "Test-time compute scaling through anytime training"

---

**This is the RIGHT experiment to run first. Let's implement it!**
