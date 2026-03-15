# Current Model Structure Reference

This file describes the model structures currently used in this repository.

Source of truth:
- local runtime defaults in `third_party/ogbench/impls/...`
- patch-applied model code in `patches/ogbench_impls.patch`

For remote runs, these structures only exist after:

```bash
./scripts/bootstrap_ogbench.sh CLEAN=1
```

## Shared High-Level Layout

CRL uses:
- an **actor**
- a **critic/value backbone** for `phi` and `psi`
- optional visual/state encoders before the actor/critic

The actor and critic backbones are now independently configurable.

---

## Actor Backbones

### 1. `actor_backbone=mlp`

Default actor.

Structure:

```text
input
-> MLP(hidden_dims, activate_final=True)
-> output head
```

Continuous actor output head:
- `mean_net`
- optional `log_std_net` or constant std path

Discrete actor output head:
- `logit_net`

Notes:
- `actor_hidden_dims` defines hidden width/depth
- default is `(512, 512, 512)`

### 2. `actor_backbone=residual_dense`

Residual MLP actor for deep-actor experiments.

Structure:

```text
input
-> Dense(input_proj)
-> repeat total Dense layers = actor_num_dense_layers:
     Dense(hidden_dim)
     LayerNorm
     SiLU
     residual add every actor_residual_span layers (or at final layer)
-> output head
```

Properties:
- LayerNorm is mandatory inside the residual actor backbone
- activation is `SiLU`
- residual span is configurable via `actor_residual_span`
- final action head remains unchanged

Key actor flags:
- `actor_backbone`
- `actor_hidden_dims`
- `actor_backbone_hidden_dim`
- `actor_num_dense_layers`
- `actor_residual_span`

---

## Critic Backbones

Critic backbone is used inside `GCBilinearValue` / `GCDiscreteBilinearCritic` for both:
- `phi`
- `psi`

### 1. `critic_backbone=mlp`

Plain baseline critic.

Structure:

```text
input
-> MLP((*value_hidden_dims, latent_dim), activate_final=False, layer_norm=layer_norm)
```

Notes:
- this is the cleanest baseline
- default hidden dims are `(512, 512, 512)`

### 2. `critic_backbone=resnet`

Untied residual critic backbone.

Structure:

```text
input
-> Dense(hidden_dim)
-> repeat critic_resnet_depth times:
     ResNetBlock
-> Dense(out_dim)
```

Each `ResNetBlock` does:

```text
h
-> optional LayerNorm
-> Dense(hidden_dim)
-> SiLU
-> Dense(hidden_dim)
-> LayerScale(alpha)
-> residual add
```

Key critic flags:
- `critic_resnet_depth`
- `critic_layerscale_init`
- `critic_backbone_hidden_dim`

### 3. `critic_backbone=recur_tied`

Main recurrent/tied critic.

User-facing notation:
- `N x K`
- `N` = number of Dense layers inside one tied recurrent residual update
- `K` = recurrent iterations during training

Structure:

```text
input
-> Dense(hidden_dim)                         # hidden-state initialization
-> repeat K times:
     step encoding
     optional FiLM conditioning
     tied residual update with N Dense layers
     residual add across loops
-> Dense(out_dim)
```

More explicitly:

```text
h0 = Dense(x)
for k in {1..K}:
    step_k = discrete lookup or sinusoidal encoding

    if use_film:
        u = FiLM(step_k, hk)
    else:
        u = hk + step_proj(step_k)

    for i in {1..N}:
        u = Dense_i(u)
        if i < N:
            optional LayerNorm_i
            SiLU

    hk+1 = hk + alpha_k * u

out = Dense(hK)
```

Current design details:
- residual connection is **between recurrent loops**
- `alpha_k` is LayerScale-style residual scaling per iteration
- FiLM is optional via `critic_recur_use_film`
- step encoding is optional sinusoidal via `critic_recur_sinusoidal`
- LayerNorm options:
  - `critic_recur_tied_ln=True`:
    - tied per-layer LayerNorm inside the recurrent branch
    - one LN for each hidden Dense layer except the last
  - `critic_recur_tied_ln=False`:
    - legacy untied per-iteration LN on the loop state

Important:
- current tied design uses **shared FiLM**, **shared Dense layers**, and **shared per-layer LN modules** across loop iterations when `critic_recur_tied_ln=True`

Key critic flags:
- `critic_recur_iters`
- `critic_recur_num_dense_layers`
- `critic_recur_max_iters`
- `critic_recur_tied_ln`
- `critic_recur_sinusoidal`
- `critic_recur_use_film`
- `critic_layerscale_init`
- `critic_backbone_hidden_dim`

### 4. `critic_backbone=partial_tied`

Partially tied recurrent critic.

Structure:

```text
input
-> Dense(hidden_dim)
-> repeat according to partial schedule:
     choose group/block
     apply group-specific 2-layer residual update
     residual add
-> Dense(out_dim)
```

Properties:
- blocks are untied across groups
- blocks are reused within a group
- supports:
  - `grouped` schedule
  - `stack` schedule
- supports optional per-group FiLM

Key critic flags:
- `critic_partial_groups`
- `critic_partial_iters_per_group`
- `critic_partial_per_group_film`
- `critic_partial_cycle_step_params`
- `critic_partial_block_schedule`

---

## Optional Critic Components

### FiLM

Used by `recur_tied` and optionally grouped in `partial_tied`.

Role:
- modulates hidden state using step information and current state

`recur_tied`:
- enabled by `critic_recur_use_film=True`
- disabled cleanly for no-FiLM ablation with `False`

### Step Encoding

Two modes for recurrent critics:

1. discrete step lookup
2. sinusoidal step encoding

Flag:
- `critic_recur_sinusoidal`

### LayerNorm

Actor:
- residual actor always uses LN after every hidden Dense layer

Critic:
- MLP uses the baseline `MLP(..., layer_norm=layer_norm)` implementation
- ResNet uses optional LN inside each residual block
- recurrent tied critic now prefers tied per-layer LN for hidden layers

### LayerScale

Used in:
- `resnet`
- `recur_tied`
- `partial_tied`

Flag:
- `critic_layerscale_init`

Interpretation:
- scales residual updates
- set near zero for conservative residual start

---

## Practical Mapping Examples

### `critic_backbone=mlp`

```text
3 hidden layer MLP critic + output projection to latent_dim
```

### `critic_backbone=recur_tied`, `critic_recur_num_dense_layers=3`, `critic_recur_iters=4`

```text
3x4 recurrent tied critic
= one tied 3-Dense residual update repeated 4 times
```

### `actor_backbone=residual_dense`, `actor_num_dense_layers=16`, `actor_residual_span=4`

```text
16 hidden Dense layers total
with residual add every 4 Dense layers
```

---

## Current Ablation-Relevant Knobs

Most important current architecture ablations:

- `critic_backbone`
- `critic_recur_num_dense_layers`
- `critic_recur_iters`
- `critic_recur_use_film`
- `critic_recur_sinusoidal`
- `critic_recur_tied_ln`
- `critic_layerscale_init`
- `actor_backbone`
- `actor_num_dense_layers`
- `actor_residual_span`

---

## Scope Note

This file describes the **current codebase design**, not an idealized or original-paper design.
If the patch changes, this file should be updated together with:

- `patches/ogbench_impls.patch`
- `slurm/phase4_train_crl_generic_array.slurm`
- `slurm/phase4_train_recur_tied_giant_navigate.slurm`
