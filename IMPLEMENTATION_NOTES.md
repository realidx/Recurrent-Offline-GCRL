# Implementation Notes: Gradient Clipping & Reward Normalization

## Gradient Clipping (Optional Enhancement)

### When to Add:
- **Phase 3 ablations** (if instability observed)
- **Exploratory runs** with very deep networks (K > 12)
- **DO NOT add for Phase 1/2 baseline comparisons**

### How to Implement:

**In `third_party/ogbench/impls/agents/crl.py`:**

```python
# Add to config (line ~355):
critic_grad_clip=0.0,  # 0 disables, typical values: 1.0, 5.0, 10.0

# Modify network_tx (line ~324):
if config.get('critic_grad_clip', 0) > 0:
    network_tx = optax.chain(
        optax.clip_by_global_norm(config['critic_grad_clip']),
        optax.adam(learning_rate=config['lr']),
    )
else:
    network_tx = optax.adam(learning_rate=config['lr'])
```

**Typical values:**
- **Conservative:** `clip=10.0` (rarely triggers, safety net only)
- **Aggressive:** `clip=1.0` (constrains optimization, may slow learning)
- **Recommended for RecurTied:** `clip=5.0` (if you see grad norms > 100)

### How to Monitor:
Already implemented! Check your logs for:
- `training/grad/global_norm` - Should stay < 10 typically
- `training/grad/critic_global_norm` - Module-specific
- If these spike to 100+ or NaN, consider clipping

---

## Reward Normalization (NOT Recommended)

### Why NOT Needed:
1. **AntMaze rewards are binary** (0 or 1) - already normalized
2. **CRL doesn't use rewards** for learning - only for evaluation metrics
3. **Would complicate comparison** with baseline

### When it WOULD be useful:
- Dense reward tasks (e.g., Hopper, HalfCheetah with rewards in [-1000, +1000])
- Online RL where reward scale affects TD targets
- Non-contrastive methods (DQN, DDPG, SAC) that regress Q-values from rewards

### If You Really Want It (Not Recommended):
```python
# In dataset preprocessing (NOT recommended for this project):
rewards_mean = np.mean(dataset['rewards'])
rewards_std = np.std(dataset['rewards']) + 1e-8
dataset['rewards'] = (dataset['rewards'] - rewards_mean) / rewards_std
```

**But again:** This is **unnecessary and harmful** for your OGBench experiments.

---

## Summary Table

| Feature | Current Status | Should Add? | Priority | Reason |
|---------|---------------|-------------|----------|--------|
| **Gradient Clipping** | ❌ No | 🟡 Maybe | Low | Baseline doesn't use it; only if instability occurs |
| **Gradient Logging** | ✅ Yes | ✅ Done | N/A | Already implemented in your patches |
| **Reward Normalization** | ❌ No | ❌ No | N/A | Rewards are binary; CRL doesn't use them |
| **Reward Logging** | ✅ Yes | ✅ Done | N/A | Already in eval metrics |

---

## Recommendation for Your Project:

### ✅ **Keep Current Setup** (No Changes Needed)
- Your training is stable (no reported NaN/divergence)
- Matches baseline for fair comparison
- Gradient logging lets you monitor if issues arise

### ⚠️ **Only Add Gradient Clipping If:**
1. You see `grad/global_norm > 100` in logs
2. Training diverges (NaN losses, Q-values exploding)
3. RecurTied with K=12 becomes unstable

### 📊 **What to Monitor Instead:**
Check your current logs for warning signs:
```bash
# Look for gradient explosions:
grep "grad/global_norm" train.csv | awk -F, '{print $N}' | sort -n | tail -5

# Look for Q-value drift:
grep "q_abs_max" train.csv | awk -F, '{print $N}' | sort -n | tail -5
```

If `grad_norm > 50` or `q_abs_max > 1000`, consider adding clipping.

---

**Last Updated:** 2026-02-02
**Status:** Current implementation is appropriate for OGBench CRL experiments
