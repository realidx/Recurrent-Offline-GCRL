# 📚 COMPLETE PROJECT SPECIFICATION
## Recurrent Offline RL for OGBench CRL - All Components & Setup

---

## 1️⃣ BENCHMARK & ENVIRONMENT

### **Benchmark Framework**
- **Name:** OGBench (Offline Goal-Conditioned Benchmark)
- **Repository:** https://github.com/seohongpark/ogbench
- **Commit:** `1d4140997f60c52c6fb0702ec100dc988b18c548`
- **Purpose:** Goal-conditioned offline reinforcement learning evaluation

### **Environments/Tasks**
**Primary Focus:** AntMaze Stitching Tasks
- `antmaze-medium-stitch-v0`
- `antmaze-large-stitch-v0` (primary benchmark)
- `antmaze-giant-stitch-v0`
- `antmaze-teleport-stitch-v0`

**Environment Properties:**
- **Observation Space:** Continuous (state vectors)
  - AntMaze-large: 29-dimensional (ant position, velocity, goal)
- **Action Space:** Continuous, bounded [-1, 1]
  - AntMaze: 8-dimensional (ant joint torques)
- **Episode Length:** Variable (terminated on goal reach or timeout)
- **Success Metric:** Binary success (reach goal within threshold)
- **Task Type:** Stitching (requires connecting suboptimal trajectories)

### **Dataset**
- **Type:** Offline (fixed) trajectories
- **Format:** `.npz` files with keys: `observations`, `actions`, `rewards`, `terminals`, `goals`
- **Storage:** Downloaded to `~/.ogbench/data` or `OGBENCH_DATASET_DIR`
- **Dataset Class:** `GCDataset` (Goal-Conditioned Dataset)
- **Compact Format:** Uses memory-efficient storage (compact_dataset=True)

---

## 2️⃣ REINFORCEMENT LEARNING ALGORITHM

### **Base Algorithm: CRL (Contrastive RL)**

**Core Concept:**
- **Q-value:** Expected cumulative discounted reward from state-action pair
  - Mathematical definition: `Q(s,a,g) = E[Σ γ^t r_t | s_0=s, a_0=a, goal=g]`
  - In CRL: Represented as bilinear score `φ(s,a)ᵀ ψ(g) / √d`

**Key Components:**

1. **Contrastive Loss (InfoNCE)**
   - **Purpose:** Train critic to distinguish positive (s,a,g) triplets from negatives
   - **Formula:** `sigmoid_binary_cross_entropy(logits=φᵀψ/√d, labels=I)`
   - **Positive pairs:** Actual (state, action, achieved goal)
   - **Negative pairs:** Mismatched goals within batch
   - **Batch size effect:** Larger batch = more negatives = better contrastive signal

2. **Bilinear Critic Architecture**
   - **φ (phi):** State-action encoder `φ: (s,a) → R^d`
   - **ψ (psi):** Goal encoder `ψ: g → R^d`
   - **Score:** `Q(s,a,g) = φ(s,a)ᵀ ψ(g) / √d`
   - **Scaling factor:** `1/√d` normalizes by latent dimension
   - **Ensemble:** Uses 2 critics, takes minimum for robustness

3. **Actor Training**
   - **Method:** DDPG+BC (Deep Deterministic Policy Gradient + Behavior Cloning)
   - **Actor loss:** `L = α * MSE(π(s,g), a_data) - Q(s, π(s,g), g)`
   - **BC coefficient (α):** 0.1 (balances imitation vs Q-maximization)
   - **Policy:** Deterministic `π: (s,g) → a`
   - **Output:** Actions clipped to [-1, 1]

---

## 3️⃣ NEURAL NETWORK ARCHITECTURES

### **A. Baseline MLP Critic**
**Type:** Standard Multi-Layer Perceptron (untied weights)

**Architecture:**
```
φ(s,a): [s;a] → Dense(512) → Dense(512) → Dense(512) → Dense(512) [latent]
ψ(g):   g     → Dense(512) → Dense(512) → Dense(512) → Dense(512) [latent]
```

**Components:**
- **Hidden layers:** 3 layers of 512 units each
- **Activation:** Implicitly uses default (likely ReLU or SiLU)
- **Layer Normalization:** Enabled (applied after each layer)
- **Output dimension:** 512 (latent_dim)
- **Parameters:** ~1.3M per backbone (φ or ψ)

### **B. Deep ResNet Critic (Untied)**
**Type:** Residual Network with stacked blocks

**Architecture:**
```
φ(s,a): [s;a] → Dense(hidden_dim) → [ResBlock × D] → Dense(latent_dim)
```

**ResNet Block:**
```python
ResNetBlock(h):
  h1 = LayerNorm(h)               # Pre-activation normalization
  u = Dense(hidden_dim)(h1)       # First dense layer
  u = SiLU(u)                     # Swish activation
  u = Dense(hidden_dim)(u)        # Second dense layer
  u = u * layerscale              # Per-channel learned scaling
  return h + u                    # Residual skip connection
```

**Hyperparameters:**
- **Depth (D):** 3, 6, or 12 blocks (swept in experiments)
- **Hidden dimension:** 512 (default) or matched to recurrent
- **Latent dimension:** 512
- **LayerScale init:** 1e-2
- **Layer normalization:** Enabled

### **C. Recurrent Tied Critic (Proposed)**
**Type:** Weight-tied iterative refinement with FiLM conditioning

**Architecture:**
```python
φ(s,a): [s;a] → Dense(hidden_dim) → [TiedBlock × K] → Dense(latent_dim)
```

**Tied Iteration Block:**
```python
for k in 0..K-1:
  # Step conditioning via FiLM
  e_k = step_embed[k]                    # Learnable per-step embedding
  film = Dense(2*h)(SiLU(Dense(h)(e_k))) # FiLM parameters
  gamma, beta = split(film)

  # Apply FiLM + tied transformation
  h1 = LayerNorm(h)                      # ⚠️ BUG: should be tied!
  h_film = (1 + gamma) * h1 + beta       # Affine conditioning
  u = Dense(h)(SiLU(Dense(h)(h_film)))   # Tied dense layers (shared weights)
  h = h + alpha[k] * u                   # Per-step learned scaling
```

**Hyperparameters:**
- **Training iterations (K_train):** 3, 4, 6, or 12
- **Test iterations (K_test):** Can override at eval (e.g., 12 when K_train=6)
- **Max iterations:** 32 (for parameter pre-allocation)
- **Hidden dimension:** 512 or matched via script
- **Step embedding stddev:** 0.02
- **Alpha (per-step scale) init:** 1e-2

**Novel Features:**
- **FiLM (Feature-wise Linear Modulation):** Conditions each iteration with learned `(γ, β)`
- **Per-iteration scaling:** Learned `α[k]` weights each iteration's contribution
- **Test-time compute knob:** Can increase K at inference for better performance

### **D. Actor Network**
**Type:** Gaussian policy (for continuous actions)

**Architecture:**
```
π(s,g): [s;g] → Dense(512) → Dense(512) → Dense(512) → Dense(action_dim)
```

**Output:**
- **Mean:** Tanh-scaled to [-1, 1]
- **Std:** Constant (not state-dependent) when `const_std=True`
- **Sampling:** Diagonal Gaussian `N(μ, σI)`

---

## 4️⃣ TRAINING HYPERPARAMETERS

### **Optimization**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Optimizer** | Adam | Adaptive learning rate optimizer |
| **Learning rate** | 3e-4 | Fixed throughout training |
| **Batch size** | 1024 | Samples per gradient update |
| **Discount factor (γ)** | 0.99 (0.995 for giant) | Future reward decay |
| **Gradient clipping** | Enabled | Via `optax.global_norm` (logged) |

### **Training Schedule**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Total train steps** | 1,000,000 | Total gradient updates |
| **Log interval** | 5,000 | Steps between training logs |
| **Validation log interval** | 100,000 | Steps between validation logs |
| **Eval interval** | 100,000 | Steps between full evaluations |
| **Save interval** | 1,000,000 | Steps between checkpoints |
| **Estimated wall-clock** | ~5 hours on GPU | For 1M steps |

### **CRL-Specific Hyperparameters**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Alpha (BC coefficient)** | 0.1 | Weight of behavior cloning in actor loss |
| **Actor p_randomgoal** | 0.5 | Prob of random goal for actor training |
| **Actor p_trajgoal** | 0.5 | Prob of trajectory goal for actor |
| **Value p_trajgoal** | 1.0 | Always use trajectory goals for critic |
| **Value geometric sampling** | True | Sample future goals geometrically |
| **Ensemble size** | 2 | Number of critic networks |

### **Architecture-Specific Settings**
| Setting | MLP | ResNet | RecurTied |
|---------|-----|--------|-----------|
| **Backbone type** | 'mlp' | 'resnet' | 'recur_tied' |
| **Depth/Iters** | 3 layers | 3/6/12 blocks | 3/4/6/12 iters |
| **Hidden dim** | 512 | 512 or matched | 512 or matched |
| **LayerScale init** | N/A | 1e-2 | 1e-2 |
| **Max iters** | N/A | N/A | 32 |

---

## 5️⃣ EVALUATION SETUP

### **Evaluation Protocol**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Eval episodes** | 50 | Rollouts per task |
| **Eval tasks** | All (varies by env) | AntMaze-large has multiple task IDs |
| **Video episodes** | 0 (can set to 1+) | Rendered episodes for visualization |
| **Video frame skip** | 3 | Subsample frames for video |
| **Eval on CPU** | True | Transfer to CPU for eval (saves GPU memory) |
| **Temperature** | 0.0 | Deterministic policy (no exploration) |
| **Gaussian noise** | None | No action noise at eval |

### **Primary Metrics**
1. **Success rate:** Fraction of episodes reaching goal
2. **Episode return:** Cumulative reward per episode
3. **Q-value statistics:** Mean/std/max of critic scores (logged on fixed batch)

### **Action Refinement (Phase 3)**
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Refine steps (T)** | 0 (default), swept in Phase 3 | Max gradient ascent iterations |
| **Refine learning rate (η)** | 0.05 | Step size for action updates |
| **Refine L2 penalty (λ)** | 0.0 (default) | Regularization weight |
| **Early stop: grad_eps** | 1e-6 | Stop if gradient norm below this |
| **Early stop: q_eps** | 1e-5 | Stop if Q improvement below this |

**Refinement Metrics Logged:**
- `q_pre`, `q_post` - Q-value before/after refinement
- `q_improve` - Improvement (q_post - q_pre)
- `delta_a` - L2 distance from actor output
- `steps_taken_mean` - Actual iterations before early stop
- `grad_norm_mean`, `grad_norm_max` - Gradient statistics
- `nonfinite_frac` - Fraction with NaN/Inf
- `grad_vanished_frac` - Fraction stopped due to small gradient
- `q_plateau_frac` - Fraction stopped due to Q plateau
- `max_steps_frac` - Fraction that used all T steps

**Refinement Algorithm with Early Stopping:**
```python
a_0 = π(s, g)                           # Initial action from actor
for t in 0..T-1:
  grad = ∇_a [Q(s,a,g) - λ||a-a_0||²]  # Compute gradient w.r.t. action

  # Early stopping checks:
  if isnan(grad) or isinf(grad):       # NaN/Inf detected
    break  # Stop with reason="nonfinite"
  if ||grad|| < 1e-6:                  # Gradient vanished
    break  # Stop with reason="grad_vanished"

  a = a + η * (grad / ||grad||)        # Normalized gradient ascent
  a = clip(a, -1, 1)                   # Enforce action bounds

  if Q(s,a,g) - Q(s,a_prev,g) < 1e-5:  # Negligible improvement
    break  # Stop with reason="q_plateau"

execute a_t                             # Final action (t ≤ T)
```

**Benefits of Early Stopping:**
1. Saves computation when refinement converges early
2. Prevents wasted iterations on plateaued actions
3. Catches numerical instabilities (NaN/Inf)
4. Reveals which models benefit from refinement (`steps_taken_mean` metric)

---

## 6️⃣ MATHEMATICAL CONCEPTS & COMPONENTS

### **Core RL Concepts**

**Q-value (Action-Value Function):**
- **Definition:** Expected return starting from (s,a) and following policy π
- **Bellman equation:** `Q(s,a,g) = r(s,a,g) + γ 𝔼[Q(s',π(s',g),g)]`
- **In CRL:** Learned via contrastive loss, not Bellman updates

**Policy (π):**
- **Definition:** Mapping from states and goals to actions
- **Type:** Deterministic Gaussian (constant variance)
- **Training:** Maximizes Q(s,π(s,g),g) with BC regularization

**Discount Factor (γ):**
- **Purpose:** Weights future rewards vs immediate rewards
- **Value:** 0.99 (exponential decay: reward 100 steps away weighted 0.99^100 ≈ 0.366)

### **Neural Network Components**

**Layer Normalization:**
- **Formula:** `LN(x) = γ * (x - μ)/σ + β`
- **Purpose:** Stabilizes training by normalizing activations
- **Parameters:** Learnable scale (γ) and bias (β) per feature

**LayerScale:**
- **Formula:** `x_out = x_in + α * f(x_in)`
- **Purpose:** Scales residual branch to stabilize deep networks
- **Init:** 1e-2 (small initial contribution from residual)

**FiLM (Feature-wise Linear Modulation):**
- **Formula:** `FiLM(x, γ, β) = (1 + γ) ⊙ x + β`
- **Purpose:** Conditionally modulates features based on context (iteration step)
- **Use case:** Differentiates recurrent iterations in tied architecture

**SiLU (Swish) Activation:**
- **Formula:** `SiLU(x) = x * sigmoid(x)`
- **Properties:** Smooth, non-monotonic, better gradients than ReLU

**Residual Connection:**
- **Formula:** `h_out = h_in + f(h_in)`
- **Purpose:** Enables gradient flow in deep networks
- **Variants:** Used in ResNet (untied) and RecurTied (tied)

### **Loss Functions**

**Contrastive Loss (InfoNCE):**
```python
logits[i,j] = φ(s_i, a_i)ᵀ ψ(g_j) / √d  # All pairs in batch
I[i,j] = 1 if i==j else 0                # Identity matrix (positives)
loss = mean(sigmoid_binary_cross_entropy(logits, I))
```

**DDPG+BC Actor Loss:**
```python
L_actor = α * ||π(s,g) - a_behavior||² - Q(s, π(s,g), g)
```

**Gradient Normalization:**
- **Purpose:** Makes updates invariant to gradient magnitude
- **Formula:** `a ← a + η * (∇Q / ||∇Q||)`

**Ensemble (Min):**
- **Purpose:** Reduces Q-value overestimation
- **Formula:** `Q_ensemble(s,a,g) = min(Q_1(s,a,g), Q_2(s,a,g))`

---

## 7️⃣ IMPLEMENTATION DETAILS

### **Software Stack**
- **Language:** Python 3.8+
- **ML Framework:** JAX + Flax (functional neural networks)
- **Optimizer:** Optax (JAX optimization library)
- **Environment:** MuJoCo + dm_control + OGBench
- **Logging:** WandB + CSV files
- **Compute:** SLURM job scheduler (GPU clusters)

### **JAX/Flax Specifics**
- **@nn.compact:** Inline module definition (layers defined in forward pass)
- **PyTreeNode:** Immutable dataclass for agent state
- **vmap:** Vectorized mapping over batches/ensembles
- **jit:** Just-in-time compilation for speed
- **grad:** Automatic differentiation

### **Reproducibility**
- **Random seeds:** 0, 1, 2, 3, 4 (5 seeds per experiment)
- **Deterministic policy:** No randomness at eval (temp=0)
- **Fixed datasets:** Offline data doesn't change
- **Checkpointing:** Save model every 1M steps

### **Parameter Counting**
**Closed-form formulas** (from `match_recur_hidden_dim.py`):

**Dense layer:** `params = in_dim × out_dim + out_dim` (weights + bias)

**LayerNorm:** `params = 2 × dim` (scale + bias)

**ResNet backbone:**
```
params = Dense(in→h) + D × [LN(h) + 2×Dense(h→h) + h] + Dense(h→latent)
       = in×h + h + D×(2h + 2(h² + h) + h) + h×latent + latent
```

**RecurTied backbone:**
```
params = Dense(in→h) + max_iters×h + max_iters +  # step_embed + alpha
         2×Dense(h→h) + Dense(h→2h) +              # FiLM network
         2×Dense(h→h) +                            # tied layers
         K×2h +                                    # K LayerNorms (⚠️BUG)
         Dense(h→latent)
```

---

## 8️⃣ EXPERIMENT PHASES

### **Phase 0: Baseline Reproduction**
- **Goal:** Validate OGBench CRL implementation
- **Tasks:** All 4 AntMaze stitching environments
- **Seeds:** 5 (0-4)
- **Config:** Official OGBench hyperparameters
- **Total runs:** 4 envs × 5 seeds = 20

### **Phase 1: Architecture Variants**
- **Goal:** Compare MLP, ResNet, RecurTied on antmaze-large-stitch
- **Configurations:**
  - MLP (baseline)
  - ResNet depth 3/6/12
  - RecurTied iters 3/6/12
  - RecurTied param-matched to ResNet
- **Seeds:** 3 (0-2)
- **Total runs:** 7-9 configs × 3 seeds = 21-27

### **Phase 2: Final Experiments**
- **Goal:** Validate claims A/B (beat baseline, match params)
- **Same as Phase 1** but with different run_group name

### **Phase 3: Test-Time Compute**
- **Goal:** Validate claim C (K_test scaling for tied critics)
- **Method:** Evaluate trained models with K_test > K_train
- **Example:** Train with K=6, eval with K=12/18/24
- **Includes:** Action refinement sweeps (T=0/5/10/20)

---

## 9️⃣ INFRASTRUCTURE

### **File Organization**
```
/exp/                           # Experiment results
  OGBench/
    {run_group}/
      sd{seed}_{timestamp}/
        flags.json              # Hyperparameters
        train.csv               # Training metrics
        eval.csv                # Evaluation metrics
        checkpoints/            # Model weights
```

### **SLURM Job Arrays**
- **Job scheduler:** SLURM (for HPC clusters)
- **Array indexing:** Maps to (seed, config) pairs
- **Resources per job:** 1 GPU, 8 CPUs, 24GB RAM, 5 hours
- **Parallelism:** 20+ jobs simultaneously

### **Logging & Monitoring**
- **Training logs:** Every 5,000 steps
- **Validation logs:** Every 100,000 steps
- **Evaluation:** Every 100,000 steps (full rollouts)
- **WandB:** Online experiment tracking (optional)
- **CSV:** Local persistent logs (always enabled)

### **Utility Scripts**
- `bootstrap_ogbench.sh`: Clone and patch OGBench
- `repro_crl_antmaze_stitch.sh`: Run baseline reproduction
- `match_recur_hidden_dim.py`: Compute param-matched hidden dims
- `summarize_ogbench_csvs.py`: Aggregate results across runs
- `inspect_ogbench_run.py`: Quick run directory inspection

---

## 🔟 KEY INNOVATIONS (This Project)

1. **Tied Recurrent Critic:** Weight-shared iterative refinement for critics
2. **FiLM Conditioning:** Step-wise modulation to differentiate tied iterations
3. **Test-Time Compute Scaling:** Increase K at inference (recur_tied only)
4. **Action Refinement:** Gradient-based action optimization at eval
5. **Parameter Matching:** Fair comparison by matching model capacity

---

## 📊 EXPECTED OUTCOMES

### **Claim A (Type A):**
Tied critic with K=6 beats MLP baseline on antmaze-large-stitch

### **Claim B (Type B):**
Tied critic with K=6, fewer params ≈ ResNet depth-6 performance

### **Claim C (Type C):**
Tied critic performance improves with K_test > K_train (test-time compute)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-02
**Author:** Comprehensive specification generated from codebase analysis
