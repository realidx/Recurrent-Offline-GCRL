# Hyperparameter Tuning Guide

## Current Status: Phase 2 (Depth=4,8 with 3 seeds each)

**Last Updated:** 2026-02-03
**Current Config:** `alpha=0.1`, `lr=3e-4`, `actor_p_randomgoal=0.5`

---

## 🚨 Critical Issues Identified

### Issue 1: Actor Bottleneck
**Symptoms:**
- `actor_loss` stuck at ~1.7 (target: <0.8)
- `grad/global_norm` only 0.1-0.3 (should be 0.3-0.8)
- `action_norm/mean/saturation_frac` plateau after 100k steps
- Actor stops learning new behaviors

**Root Cause:** `alpha=0.1` too conservative → actor learns too slowly

**Diagnosis Date:** 2026-02-03 (initial), confirmed 2026-02-03 (validation analysis)

---

### Issue 2: Critic Overfitting (SEVERE) 🚨
**Symptoms (Training vs Validation Divergence):**
- `training/critic_loss` decreases ✅ BUT `validation/critic_loss` flat (d=4) or INCREASES (d=8) ❌
- `training/q_mean` increases ✅ BUT `validation/q_mean` DECREASES ❌
- `training/logits_pos` increases ✅ BUT `validation/logits_pos` DECREASES ❌
- `training/v_mean`: d=8 > d=4 BUT `validation/v_mean`: d=8 < d=4 ❌
- `training/score_margin` much higher than `validation/score_margin`
- d=8 overfits MORE than d=4 (explains why d=8 doesn't beat d=4 in evaluation)

**Root Cause:** `batch_size=1024` too small → critic memorizes training batch instead of learning general Q-value patterns

**Why This Matters:**
- Critic gives **wrong Q-values** on validation/evaluation data
- Actor gets **misleading gradients** from overfitted critic
- Task 1 forgetting: Actor learns from wrong Q-values, fails on actual task
- Task 2 failure: Q-values meaningless for exploration
- d=8 worse than d=4: More capacity = more overfitting

**Diagnosis Date:** 2026-02-03 (training/validation log comparison)

**CRITICAL:** These two issues are COUPLED. You must fix BOTH simultaneously.

---

## ✅ Recommended Changes (Priority Order)

### Priority 1: Fix Both Issues Simultaneously 🔴 CRITICAL

**You MUST apply BOTH changes together. Fixing only one will not work.**

#### Change 1A: Increase Actor Learning Weight
**Parameter:** `--agent.alpha`
**Current Value:** `0.1`
**Recommended Value:** `0.3`
**File to Edit:** [slurm/phase2_critics_antmaze_large_stitch_array.slurm:84](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L84)

**Change:**
```bash
# OLD:
"--agent.alpha=0.1"

# NEW:
"--agent.alpha=0.3"
```

**Why This Helps:**
- Actor learns 3x faster relative to critic
- Can better follow critic's Q-value gradients
- Reduces catastrophic forgetting

#### Change 1B: Increase Batch Size (Critic Regularization)
**Parameter:** `--agent.batch_size`
**Current Value:** `1024` (default)
**Recommended Value:** `2048`
**File to Edit:** [slurm/phase2_critics_antmaze_large_stitch_array.slurm:88](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L88)

**Change:**
```bash
AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.3"
  "--agent.batch_size=2048"  # ADD THIS LINE
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
  "--agent.critic_recur_max_iters=16"
  "--agent.critic_layerscale_init=1e-2"
)
```

**Why This Helps:**
- 2x more diverse state-action pairs per critic update
- Harder for critic to memorize specific training patterns
- Forces critic to learn general Q-value functions
- d=8 will generalize better (reduce overfitting)
- Validation metrics will track training metrics

**Trade-off:**
- Training 2x slower (more data per batch)
- But necessary to prevent critic overfitting

**Expected Impact:**
| Metric | Current | Target (alpha=0.3, batch=2048) | Improvement |
|--------|---------|-------------------------------|-------------|
| `actor_loss` | ~1.7 | <0.8 | 53% reduction |
| `grad/global_norm` | 0.1-0.3 | 0.3-0.8 | 2-3x increase |
| `validation/critic_loss` | Flat/increasing | Decreasing like training | Generalization ✅ |
| `validation/q_mean` | Decreasing ❌ | Increasing like training ✅ | Correct Q-values |
| Task 1 stability | Peak→drop | Peak→stable | No forgetting |
| Task 2 success | 0% | >5% | Exploration works |
| d=8 vs d=4 | Inconsistent | d=8 wins | Depth advantage |

**What to Monitor:**

**Training Metrics:**
- `training/actor_loss` - Should drop to <0.8 by 500k steps
- `training/grad/global_norm` - Should increase to 0.3-0.8 range
- `training/policy_behavior_mse` - Must stay <0.2 (safety check)

**Validation Metrics (CRITICAL - Check Every 100k Steps):**
- ✅ `validation/critic_loss` should DECREASE (not increase!)
- ✅ `validation/q_mean` should INCREASE (not decrease!)
- ✅ `validation/logits_pos` should INCREASE (not decrease!)
- ✅ `validation/score_margin` within 3 points of training
- ✅ `validation/critic_loss / training/critic_loss < 1.5` (max 50% higher)

**Evaluation Metrics:**
- `evaluation/task_1_success` - Should NOT drop after peaking
- `evaluation/task_2_success` - Should reach >5% by 1M steps
- d=8 should consistently beat d=4

**Risks:**
- If `policy_behavior_mse > 0.2`: Alpha too high → reduce to 0.2
- If `grad/global_norm > 2.0`: Instability → add gradient clipping
- If `validation/critic_loss` still increases for d=8: Try batch_size=4096 or reduce d=8 capacity

---

### Priority 2: Increase Learning Rate (If alpha=0.3 + batch=2048 insufficient) 🟡 CONDITIONAL

**Parameter:** `--agent.lr` (not currently in SLURM script, uses default 3e-4)
**Current Value:** `3e-4` (default from [impls/agents/crl.py:358](impls/agents/crl.py#L358))
**Recommended Value:** `5e-4`
**File to Edit:** [slurm/phase2_critics_antmaze_large_stitch_array.slurm:88](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L88)

**When to Apply:**
- ONLY if alpha=0.3 + batch_size=2048 still shows `actor_loss > 1.0` at 500k steps
- ONLY if validation metrics are healthy (not overfitting)
- Do NOT apply if critic still overfitting

**Change:**
```bash
AGENT_FLAGS=(
  "--agent.actor_p_randomgoal=0.5"
  "--agent.actor_p_trajgoal=0.5"
  "--agent.alpha=0.3"
  "--agent.batch_size=2048"
  "--agent.lr=5e-4"  # ADD THIS LINE
  "--agent.critic_backbone=${CRITIC_BACKBONE}"
  ...
)
```

**Expected Impact:**
- Faster convergence (reach low loss 200k steps earlier)
- May reduce stability (higher gradient variance)

**What to Monitor:**
- `training/critic_loss` - Should NOT spike above 0.01
- `validation/critic_loss` - Should still track training (not diverge)
- `training/ensemble_disagreement` - Should stay <1.0
- If training diverges OR validation overfits, revert to 3e-4

---

### Priority 3: Adjust Goal Sampling (For Task Stability) 🟢 OPTIONAL

**Parameter:** `--agent.actor_p_randomgoal`
**Current Value:** `0.5`
**Recommended Value:** `0.7`
**File to Edit:** [slurm/phase2_critics_antmaze_large_stitch_array.slurm:82](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L82)

**When to Apply:**
- If Task 1/5 still show high instability after alpha=0.3
- If per-task success has high variance across evaluation epochs

**Change:**
```bash
# OLD:
"--agent.actor_p_randomgoal=0.5"

# NEW:
"--agent.actor_p_randomgoal=0.7"
```

**Why This Helps:**
- More random goals → more exploration of state space
- Reduces overfitting to specific goal trajectories
- May improve generalization across tasks

**Trade-off:**
- Slightly slower convergence (noisier training signal)
- Better final performance on diverse tasks

---

## 📊 Monitoring Checklist

### Every 100k Steps (Training Metrics)

Check these in your WandB dashboard or CSV logs:

**Actor Health:**
- [ ] `training/actor_loss` - Decreasing trend? Target: <0.8 by 500k
- [ ] `training/policy_behavior_mse` - Staying below 0.2?
- [ ] `training/grad/actor_global_norm` - In range 0.3-0.8?
- [ ] `training/action_norm` - Changing over time? (Should NOT plateau at 100k)

**Critic Health (Training):**
- [ ] `training/critic_loss` - Converging to <0.005?
- [ ] `training/score_margin` - Staying above 10?
- [ ] `training/ensemble_disagreement` - Stable around 0.5?
- [ ] `training/q_mean` - Increasing trend?
- [ ] `training/logits_pos` - Increasing trend?

**Critic Health (Validation) 🚨 CRITICAL:**
- [ ] `validation/critic_loss` - DECREASING like training? (Not flat or increasing!)
- [ ] `validation/q_mean` - INCREASING like training? (Not decreasing!)
- [ ] `validation/logits_pos` - INCREASING like training? (Not decreasing!)
- [ ] `validation/score_margin` - Within 3 points of training score_margin?
- [ ] `validation/critic_loss / training/critic_loss` - Ratio < 1.5?

**Warning Signs:**
- 🚨 `policy_behavior_mse > 0.2` → Alpha too high, reduce to 0.2
- 🚨 `grad/global_norm > 2.0` → Add gradient clipping (see IMPLEMENTATION_NOTES.md)
- 🚨 `validation/critic_loss` increasing → Critic overfitting, increase batch_size further
- 🚨 `validation/q_mean` decreasing → Critic learning wrong Q-values, increase batch_size
- 🚨 `ensemble_disagreement > 1.5` → Training unstable
- 🚨 `action_norm` plateau before 500k → Actor stopped learning

---

### Every 200k Steps (Evaluation Metrics)

**Primary Target:**
- [ ] `evaluation/overall_success` - Increasing trend?

**Per-Task Analysis:**
- [ ] Task 1: Success stable after peak? (No catastrophic forgetting)
- [ ] Task 2: Success > 0%? (Previously all zeros)
- [ ] Task 3: High success maintained? (Easiest task, should stay high)
- [ ] Task 4: d=8 > d=4 consistently?
- [ ] Task 5: d=8 > d=4 consistently?

**Depth Comparison:**
- [ ] Depth=8 outperforming depth=4 on average?
- [ ] If not, actor still bottleneck → increase alpha further

---

## 🔬 Experimental Plan

### Phase 2a: Fix Actor Bottleneck + Critic Overfitting (Current Priority)

**Goal:** Fix both coupled issues simultaneously

**Runs:**
```bash
# Run 1: alpha=0.3, batch_size=2048, depth=4, seeds=[0,1,2]
# Run 2: alpha=0.3, batch_size=2048, depth=8, seeds=[0,1,2]
```

**Success Criteria:**

**Training Metrics:**
1. `actor_loss < 0.8` by 500k steps
2. `grad/global_norm` in range 0.3-0.8
3. `action_norm` continues changing (not plateau)

**Validation Metrics (CRITICAL):**
4. `validation/critic_loss` DECREASES (not increases!)
5. `validation/q_mean` INCREASES (not decreases!)
6. `validation/score_margin` within 3 points of training
7. `validation/critic_loss / training/critic_loss < 1.5`

**Evaluation Metrics:**
8. Task 1 success stable after peaking (no catastrophic forgetting)
9. Task 2 success > 5% (currently 0%)
10. Depth=8 consistently beats depth=4

**Timeline:** 2-3 days per depth (6 runs total, 2x slower due to batch_size increase)

---

### Phase 2b: LR Tuning (If Needed)

**Trigger:** alpha=0.3 results show `actor_loss > 1.0` at 500k steps

**Runs:**
```bash
# Run 3: alpha=0.3, lr=5e-4, depth=4, seeds=[0,1,2]
# Run 4: alpha=0.3, lr=5e-4, depth=8, seeds=[0,1,2]
```

**Success Criteria:**
1. `actor_loss < 0.8` by 300k steps (200k faster than baseline)
2. No training divergence (critic_loss stays <0.01)

---

### Phase 2c: Goal Sampling (If Needed)

**Trigger:** Task 1/5 still unstable after alpha tuning

**Runs:**
```bash
# Run 5: alpha=0.3, actor_p_randomgoal=0.7, depth=8, seeds=[0,1,2]
```

**Success Criteria:**
1. Lower variance in per-task success across evaluation epochs
2. Overall success ≥ alpha=0.3 baseline

---

## 📝 Results Log

### Run History

| Date | Alpha | Batch | LR | Goal | Depth | Seeds | Actor Loss | Val Healthy? | Task 1 | Task 2 | Notes |
|------|-------|-------|----|----|-------|-------|------------|--------------|--------|--------|-------|
| 2026-02-03 | 0.1 | 1024 | 3e-4 | 0.5 | 4 | 0,1,2 | ~1.7 | ❌ val_q↓ | ❌ Peak→drop | 0% | Actor bottleneck + critic overfitting |
| 2026-02-03 | 0.1 | 1024 | 3e-4 | 0.5 | 8 | 0,1,2 | ~1.7 | ❌ val_loss↑ | ❌ Peak→drop | 0% | d=8 overfits MORE than d=4 |
| TBD | 0.3 | 2048 | 3e-4 | 0.5 | 4 | 0,1,2 | ? | ? | ? | ? | **NEXT RUN** - Fix both issues |
| TBD | 0.3 | 2048 | 3e-4 | 0.5 | 8 | 0,1,2 | ? | ? | ? | ? | **NEXT RUN** - Check if d=8 generalizes |

**Legend:**
- **Val Healthy?**: ✅ = validation metrics track training, ❌ = validation diverges (overfitting)
- **Task 1**: ✅ = stable after peak, ❌ = catastrophic forgetting
- **Task 2**: Success percentage (currently 0% for all)

**Add new runs here as experiments complete.**

### Validation Analysis (2026-02-03)

**Key Finding:** Critic overfitting to training batch, especially for d=8

| Metric | Training Pattern | Validation Pattern | Healthy? |
|--------|------------------|-------------------|----------|
| critic_loss | Decreases ✅ | d=4: flat, d=8: increases ❌ | ❌ Overfitting |
| q_mean | Increases ✅ | Decreases ❌ | ❌ Wrong Q-values |
| logits_pos | Increases ✅ | Decreases ❌ | ❌ Memorization |
| score_margin | Increases ✅ | Much lower | ❌ No generalization |
| v_mean | d=8 > d=4 ✅ | d=8 < d=4 ❌ | ❌ d=8 capacity wasted |

**Conclusion:** batch_size=1024 too small for contrastive learning, especially for d=8

---

## 🎯 Decision Tree

```
Start: Critic overfitting + Actor bottleneck (coupled issues)
  │
  ├─→ Try alpha=0.3 + batch_size=2048 (MUST DO BOTH)
  │     │
  │     ├─→ Validation metrics healthy? (val_critic_loss↓, val_q_mean↑)
  │     │     ├─ YES → Check actor_loss < 0.8 by 500k
  │     │     │         ├─ YES → SUCCESS! Check evaluation
  │     │     │         │         ├─ Task 1 stable, Task 2 >5%, d=8>d=4? → Phase 3!
  │     │     │         │         └─ Still issues → Try actor_p_randomgoal=0.7
  │     │     │         │
  │     │     │         └─ NO (actor_loss still high) → Add lr=5e-4
  │     │     │               └─ Re-check actor_loss at 500k
  │     │     │
  │     │     └─ NO (validation still diverging)
  │     │           │
  │     │           ├─ Is d=4 OK? (validation tracks training for d=4)
  │     │           │   ├─ YES → Use d=4, skip d=8 (too much capacity)
  │     │           │   └─ NO → Increase batch_size to 4096
  │     │           │
  │     │           └─ d=8 overfits but d=4 doesn't?
  │     │                 └─ Reduce d=8 capacity: --agent.critic_backbone_hidden_dim=384
  │     │
  │     └─→ Training unstable? (loss spikes)
  │           └─ Reduce batch_size to 1536 (compromise)
  │
  └─→ Warning conditions:
        ├─ policy_behavior_mse > 0.2 → Reduce alpha to 0.2
        ├─ grad/global_norm > 2.0 → Add gradient clipping
        └─ validation diverging → Increase batch_size further
```

---

## 🧪 Additional Fixes (If Needed)

### If d=8 Still Overfits with batch_size=2048

**Option A: Increase Batch Size Further**
**Value:** `--agent.batch_size=4096`
**Trade-off:** 4x slower training, but may be necessary for d=8
**When:** If validation/critic_loss still increases for d=8 with batch=2048

**Option B: Reduce d=8 Hidden Dimension**
**Parameter:** `--agent.critic_backbone_hidden_dim=384`
**Current:** 512 (default for last layer of phi/psi)
**When:** If batch_size=4096 too slow
**Effect:** Reduces d=8 parameter count by ~25%, less capacity to overfit

**Option C: Use d=4 Only**
**Decision:** If d=8 consistently overfits and doesn't beat d=4, skip d=8
**Justification:** Phase 2 goal is finding best architecture; if d=8 doesn't help, use d=4

### Gradient Clipping
**When:** Only if `grad/global_norm > 100` or NaN losses
**See:** [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for implementation
**Value:** `critic_grad_clip=5.0`
**Note:** With batch_size=2048, this should NOT be needed

### Learning Rate Warmup
**Current:** None
**Alternative:** Linear warmup over 10k steps
**When:** If early training shows high loss variance with batch_size=2048
**Complexity:** Requires code changes, not worth it yet

---

## 📚 Related Documentation

- [COMPLETE_SPECIFICATION.md](COMPLETE_SPECIFICATION.md) - Full experimental design
- [LOGGING_AUDIT.md](LOGGING_AUDIT.md) - All available metrics
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - Gradient clipping implementation
- [CHANGES_REVIEW.md](CHANGES_REVIEW.md) - Code changes review

---

## 🔄 Update Instructions

**After each experimental run:**
1. Add results to "Results Log" table
2. Update "Current Status" section
3. Mark completed recommendations with ✅
4. Add new observations to "Decision Tree"

**When to create new sections:**
- If you discover new hyperparameter sensitivities
- If a recommended change fails (document why)
- If you find better values than recommended

---

**Next Actions:**
1. Change `alpha` from 0.1 to 0.3 in [slurm/phase2_critics_antmaze_large_stitch_array.slurm:84](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L84)
2. Add `--agent.batch_size=2048` in [slurm/phase2_critics_antmaze_large_stitch_array.slurm:88](slurm/phase2_critics_antmaze_large_stitch_array.slurm#L88)
3. Re-run depth=4,8 experiments with BOTH changes (they must be applied together)
