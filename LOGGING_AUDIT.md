# 📊 Logging Audit & Recommendations

## ✅ What You Currently Log

### **1. Training Metrics** (logged every 5,000 steps)

#### **Critic (Q-function) Statistics:**
- `training/critic_loss` - Contrastive loss value
- `training/v_mean`, `v_max`, `v_min` - Value function statistics
- `training/q_mean` - Mean Q-value (diagonal of logits matrix)
- `training/q_std` - Std dev of Q-values
- `training/q_abs_max` - Maximum absolute Q-value
- `training/logits_std` - Full logits matrix std dev
- `training/logits_abs_max` - Maximum absolute logit
- `training/binary_accuracy` - Binary classification accuracy
- `training/categorical_accuracy` - Categorical accuracy
- `training/logits_pos` - Mean positive logits
- `training/logits_neg` - Mean negative logits

#### **Actor (Policy) Statistics:**
- `training/actor_loss` - Policy loss (DDPG+BC)
- `training/bc_log_prob` - Behavior cloning log probability
- `training/mse` - MSE between policy output and data actions
- `training/std` - Policy standard deviation
- `training/action_mean` - Mean of policy actions
- `training/action_std` - Std dev of policy actions
- `training/action_norm` - L2 norm of actions
- `training/action_abs_max` - Maximum absolute action value
- `training/action_saturation_frac` - Fraction of actions near bounds (|a| > 0.99)

#### **Gradient Statistics:**
- `training/grad/max` - Maximum gradient value
- `training/grad/min` - Minimum gradient value
- `training/grad/norm` - Gradient norm (post-clipping if applied)
- `training/grad/global_norm` - Global gradient norm across all parameters
- `training/grad/critic_global_norm` - Critic-specific gradient norm
- `training/grad/actor_global_norm` - Actor-specific gradient norm
- `training/grad/value_global_norm` - Value network gradient norm (if AWR)

### **2. Validation Metrics** (logged every 100,000 steps)
- Same as training metrics, but prefixed with `validation/`
- Computed on validation split (if available)

### **3. Evaluation Metrics** (logged every 100,000 steps)

#### **Task Performance:**
- `evaluation/{task_name}_success` - Per-task success rate
- `evaluation/{task_name}_episode.return` - Per-task episode return
- `evaluation/overall_success` - **PRIMARY METRIC** - Average success across all tasks
- `evaluation/overall_episode.return` - Average return across all tasks

#### **Critic Quality on Fixed Batch:**
- `evaluation/Q_mean` - Mean critic score on fixed evaluation batch
- `evaluation/Q_std` - Std dev of critic scores
- `evaluation/Q_abs_max` - Maximum absolute critic score

#### **Action Refinement** (if enabled):
- `evaluation/overall_refine.q_pre` - Q-value before refinement
- `evaluation/overall_refine.q_post` - Q-value after refinement
- `evaluation/overall_refine.q_improve` - Improvement (q_post - q_pre)
- `evaluation/overall_refine.delta_a` - L2 distance from actor output
- `evaluation/overall_refine.grad_norm_mean` - Average gradient norm during refinement
- `evaluation/overall_refine.grad_norm_max` - Maximum gradient norm
- `evaluation/overall_refine.steps_taken_mean` - **IMPORTANT** - Actual iterations before early stop
- `evaluation/overall_refine.nonfinite_frac` - Fraction with NaN/Inf
- `evaluation/overall_refine.grad_vanished_frac` - Fraction stopped due to vanishing gradient
- `evaluation/overall_refine.q_plateau_frac` - Fraction stopped due to Q plateau
- `evaluation/overall_refine.max_steps_frac` - Fraction that used all T steps

### **4. Timing Metrics:**
- `time/epoch_time` - Time per log interval (seconds)
- `time/sps` - Steps per second (throughput)
- `time/total_time` - Total wall-clock time since start
- `time/eval_time` - Time spent on evaluation

---

## 🟡 What's USEFUL But Missing

### **Priority 1: Critical for Debugging** 🔴

#### **1. Parameter Statistics**
**Why:** Detect parameter drift, dead neurons, weight explosion
```python
# Add to CRL agent or main.py:
param_stats = jax.tree_util.tree_map(lambda p: {
    'mean': float(jnp.mean(jnp.abs(p))),
    'max': float(jnp.max(jnp.abs(p))),
}, agent.network.params)

# Log:
'params/critic_mean': ...
'params/critic_max': ...
'params/actor_mean': ...
'params/actor_max': ...
```
**When useful:** Detecting weight decay issues, initialization problems, or parameter drift

#### **2. Ensemble Disagreement** (CRL uses 2 critics)
**Why:** Monitor if ensemble members are learning different functions
```python
# In CRL critic loss:
v = agent.network.select('critic')(batch['observations'], batch['goals'], batch['actions'])
# v shape: (ensemble_size, batch_size)
ensemble_std = jnp.std(v, axis=0).mean()  # Std across ensemble members

# Log:
'critic/ensemble_disagreement': ensemble_std
```
**When useful:** High disagreement (>5) suggests instability; low disagreement (<0.1) suggests ensemble redundancy

#### **3. Batch Statistics**
**Why:** Detect data pipeline issues, distribution shift
```python
# In main.py training loop:
'batch/obs_mean': batch['observations'].mean()
'batch/obs_std': batch['observations'].std()
'batch/action_mean': batch['actions'].mean()
'batch/goal_dist_mean': jnp.linalg.norm(batch['observations'] - batch['goals'], axis=-1).mean()
```
**When useful:** If batch stats drift during training, suggests data pipeline bug

---

### **Priority 2: Useful for Analysis** 🟡

#### **4. Learning Rate** (if you add scheduling later)
```python
'optim/lr': current_learning_rate
```
**When useful:** If you add LR warmup/decay, essential for debugging

#### **5. Target Network Lag** (if using EMA target networks)
CRL may use target networks for stability. If so, log:
```python
target_lag = jax.tree_util.tree_map(
    lambda online, target: jnp.mean(jnp.abs(online - target)),
    online_params, target_params
)
'target/critic_lag': float(target_lag)
```
**When useful:** Target lag too small → unstable; too large → outdated targets

#### **6. Contrastive Loss Components**
**Why:** Decompose InfoNCE loss to see what's working
```python
# In CRL loss:
positive_scores = logits_diag  # Correct (s,a,g) pairs
negative_scores = logits[~I]   # Mismatched pairs

'critic/pos_score_mean': positive_scores.mean()
'critic/neg_score_mean': negative_scores.mean()
'critic/score_margin': positive_scores.mean() - negative_scores.mean()
```
**When useful:** Margin < 1 suggests critic isn't discriminating well

#### **7. Goal Distance at Episode End**
```python
# In evaluation.py, after episode ends:
final_goal_distance = jnp.linalg.norm(observation - goal)
'eval/final_goal_dist': float(final_goal_distance)
```
**When useful:** Correlates with success rate; helps debug "almost succeeded" cases

---

### **Priority 3: Nice to Have** 🟢

#### **8. Checkpoint Metadata**
```python
# When saving checkpoints:
'checkpoint/size_mb': os.path.getsize(checkpoint_path) / 1024**2
'checkpoint/params_count': sum([p.size for p in jax.tree_leaves(agent.network.params)])
```
**When useful:** Tracking storage costs, param count changes

#### **9. Episode Length Statistics**
```python
# In evaluation:
'eval/episode_length_mean': mean_episode_length
'eval/episode_length_std': std_episode_length
```
**When useful:** Detecting policies that give up early (short episodes)

#### **10. Action Diversity** (Entropy proxy)
```python
# In training:
action_entropy_proxy = -jnp.mean(jnp.log(action_std + 1e-8))
'actor/action_diversity': action_entropy_proxy
```
**When useful:** Offline RL policies can collapse to deterministic; this detects it

---

## ❌ What's NOT Useful (Don't Add)

### **Things to Skip:**

1. **Per-step metrics in evaluation** - You already aggregate per episode ✓
2. **Reward statistics** - Rewards are binary (0/1), no variance to analyze
3. **Replay buffer statistics** - You're offline, dataset is fixed
4. **Exploration metrics** (entropy, KL divergence) - Not relevant for offline RL
5. **Dense logging of every gradient** - Your current summary stats (mean/max/norm) are sufficient

---

## 📋 Recommended Additions (Prioritized)

### **Immediate (Add Now):**
1. ✅ **Ensemble disagreement** - Critical for CRL stability
2. ✅ **Parameter statistics** (mean/max of weights) - Detect drift

### **Phase 2 (If Issues Arise):**
3. 🟡 **Contrastive loss margin** - If training is unstable
4. 🟡 **Batch statistics** - If you suspect data pipeline issues

### **Phase 3 (Analysis):**
5. 🟢 **Goal distance at episode end** - For detailed ablations
6. 🟢 **Episode length stats** - Understand policy behavior

---

## 🛠️ Implementation Example

**Add to `third_party/ogbench/impls/agents/crl.py`:**

```python
# In critic loss function (line ~60):
def critic_loss(self, batch, grad_params):
    # ... existing code ...

    # Add ensemble disagreement:
    v = self.network.select('critic')(
        batch['observations'],
        batch['value_goals'],
        batch['actions']
    )  # Shape: (ensemble_size=2, batch_size)

    if v.ndim == 2:  # Ensemble output
        ensemble_disagreement = jnp.std(v, axis=0).mean()
    else:
        ensemble_disagreement = 0.0

    info.update({
        'critic/ensemble_disagreement': ensemble_disagreement,
        # Contrastive margin:
        'critic/pos_score_mean': logits_pos,
        'critic/neg_score_mean': logits_neg,
        'critic/score_margin': logits_pos - logits_neg,
    })

    return loss, info

# In create() method (after network initialization, line ~326):
@classmethod
def create(cls, seed, ex_observations, ex_actions, ex_goals, config):
    # ... existing code ...

    # Log parameter counts:
    param_count = sum([p.size for p in jax.tree_util.tree_leaves(network_params)])
    print(f"Total parameters: {param_count:,}")

    # Log parameter statistics (optional):
    critic_params = network_params['critic']
    critic_mean = float(jnp.mean(jnp.array([
        jnp.mean(jnp.abs(p)) for p in jax.tree_util.tree_leaves(critic_params)
    ])))
    print(f"Critic param mean (abs): {critic_mean:.6f}")

    return cls(rng, network=network, config=flax.core.FrozenDict(**config))
```

---

## 📊 Logging Best Practices (You're Already Following)

✅ **What you're doing right:**
1. Separate training/validation/evaluation namespaces
2. Per-task AND overall metrics for evaluation
3. Timing metrics (SPS, epoch time)
4. Comprehensive gradient statistics
5. Refinement metrics with early-stop breakdown

✅ **Keep doing:**
- Log every 5k steps (not too frequent)
- Eval every 100k steps (reasonable for offline RL)
- Use CSV + WandB (redundancy is good)

---

## 🎯 Final Recommendation

**Minimal additions for maximum value:**

```python
# Add these 3 metrics to CRL agent:
'critic/ensemble_disagreement': ensemble_std
'critic/score_margin': logits_pos_mean - logits_neg_mean
'params/critic_abs_mean': critic_param_abs_mean
```

**That's it!** These three additions cover the biggest gaps:
1. **Ensemble disagreement** - Stability monitoring
2. **Score margin** - Contrastive learning quality
3. **Param stats** - Weight drift detection

Everything else is nice-to-have but not critical for your experiments.

---

**Last Updated:** 2026-02-02
**Status:** Your current logging is already excellent. Add the 3 recommended metrics for completeness.
