# Recur-Tied Model Reference

This note reflects the current repo after the architecture cleanup.

- Actor networks no longer have recurrent variants. Actors now use the baseline MLP path only.
- On the value side, only `mlp` and `recur_tied` remain.
- This file documents only the reachable `recur_tied` surface that is still wired through agents and launchers.

Primary code references:

- `third_party/ogbench/impls/utils/networks.py`
- `third_party/ogbench/impls/agents/crl.py`
- `third_party/ogbench/impls/agents/qrl.py`
- `third_party/ogbench/impls/agents/hiql.py`
- `third_party/ogbench/impls/agents/saw.py`
- `third_party/ogbench/impls/agents/cgivl.py`
- `slurm/train_crl.slurm`
- `slurm/train_hiql.slurm`
- `slurm/train_saw.slurm`
- `slurm/train_cgivl.slurm`

## Core Backbone

`RecurTiedBackbone` is defined in `third_party/ogbench/impls/utils/networks.py`.

Current arguments:

| Arg | Default | Meaning |
| --- | --- | --- |
| `hidden_dim` | required | Internal recurrent width |
| `out_dim` | required | Output width |
| `num_iters` | required | Train-time recurrent steps |
| `num_dense_layers` | `2` | Number of tied Dense layers inside each recurrent residual update |
| `block_type` | `'dense'` | Inner recurrent cell: `dense` or `swiglu` |
| `max_iters` | `16` | Maximum number of supported recurrent steps when per-step parameters are used |
| `layer_norm` | `True` | Whether LayerNorm is enabled |
| `ln_mode` | `'per_layer_final'` | One of `pre_loop`, `per_layer`, `per_layer_final`, `pre_loop_per_layer` |
| `layerscale_init` | `1e-2` | Initial value for LayerScale residual weights |
| `use_step_info` | `False` | Whether loop-step information is injected |
| `use_sinusoidal_step_encoding` | `False` | Use sinusoidal step encoding instead of a learned step table |
| `use_film` | `True` | Enable FiLM modulation |
| `film_mode` | `'hidden'` | One of `hidden`, `context`, `hidden_context` |
| `use_layerscale` | `True` | Enable learned residual scaling |
| `use_act` | `False` | Enable ACT halting |
| `act_epsilon` | `0.01` | ACT halting slack |
| `act_min_iters` | `1` | Minimum recurrent steps before halting is allowed |

High-level forward structure:

1. Project inputs to `hidden_dim`.
2. Run `num_iters` recurrent refinement steps.
3. Each step applies a tied residual block selected by `block_type`.
4. `dense` keeps the existing tied Dense stack; `swiglu` uses a fixed pre-norm `W_a/W_b/W_o` SwiGLU cell.
5. Optional step information is added either from a learned step table or a sinusoidal encoding.
6. Optional FiLM modulation is applied before the inner recurrent cell.
7. Optional LayerScale multiplies the residual update by a learned per-step `alpha_k`.
8. Optional ACT forms a weighted final hidden state.
9. Final projection maps to `out_dim`.

## Simplified Surface

The following older knobs are gone:

- width override via `*_backbone_hidden_dim`
- untied LayerNorm via `*_recur_tied_ln`
- separate context-FiLM toggle via `*_recur_use_film_context`
- shared-alpha LayerScale via `*_recur_shared_alpha`

The current FiLM interface is:

- `*_recur_use_film`: on or off
- `*_recur_film_mode`: `hidden`, `context`, or `hidden_context`

So the current usage is:

```text
use_film = False
```

or

```text
use_film = True
film_mode = hidden | context | hidden_context
```

There is no extra context-FiLM switch anymore.

LayerNorm is always the tied/shared version when it is enabled. The remaining control is only placement via `ln_mode`.

LayerScale now always uses non-shared per-step `alpha_k` when enabled. The parameter shape is always `(max_iters,)`.

## Block Types

`recur_tied` now supports two inner block variants:

- `block_type='dense'`: the original tied Dense stack controlled by `num_dense_layers` and `ln_mode`
- `block_type='swiglu'`: a fixed pre-norm recurrent cell

```text
z_k = LN(h_{k-1})
z~_k = (1 + gamma(c, e_k)) ⊙ z_k + beta(c, e_k)
a_k = W_a z~_k
b_k = W_b z~_k
g_k = SiLU(a_k) ⊙ b_k
Δ_k = W_o g_k
h_k = h_{k-1} + alpha_k Δ_k
```

Notes:

- `swiglu` keeps the outer recurrent shell, FiLM, LayerScale, and ACT logic unchanged.
- `swiglu` uses a fixed 2-layer gated cell, so `num_dense_layers` must stay at `2`.
- `swiglu` uses pre-norm semantics internally even if the old `ln_mode` default remains `per_layer_final`.

## Iteration Bound

`recur_tied` only allows `num_iters > max_iters` when there are no per-step learned parameters.

In the current implementation, the bound is required when either of these is true:

- `use_step_info=True` with learned discrete step embeddings
- `use_layerscale=True`

Because `use_layerscale=True` is the default everywhere, test-time iteration overrides usually still need `num_iters <= max_iters` unless LayerScale is explicitly disabled.

## Conditioning Design

Step information has three regimes:

1. No step info:
   - `use_step_info=False`
2. Learned discrete step embeddings:
   - `use_step_info=True`
   - `use_sinusoidal_step_encoding=False`
3. Sinusoidal step encodings:
   - `use_step_info=True`
   - `use_sinusoidal_step_encoding=True`

FiLM modes mean:

- `hidden`: condition from the current hidden state only
- `context`: condition from the external `context` tensor only
- `hidden_context`: add hidden-state and context projections together

If `film_mode` uses `context`, the caller must provide a context tensor.

## Where Recur-Tied Is Still Used

### CRL

CRL uses `recur_tied` on the critic/value side only, through the bilinear value modules in `third_party/ogbench/impls/utils/networks.py`.

Agent defaults in `third_party/ogbench/impls/agents/crl.py`:

| Config | Default |
| --- | --- |
| `critic_backbone` | `mlp` |
| `critic_recur_iters` | `4` |
| `critic_recur_num_dense_layers` | `2` |
| `critic_recur_max_iters` | `16` |
| `critic_recur_ln_mode` | `per_layer_final` |
| `critic_recur_use_step_info` | `False` |
| `critic_recur_sinusoidal` | `False` |
| `critic_recur_use_film` | `True` |
| `critic_recur_film_mode` | `hidden` |
| `critic_recur_use_layerscale` | `True` |
| `critic_recur_use_act` | `False` |
| `critic_recur_act_epsilon` | `0.01` |
| `critic_recur_act_min_iters` | `1` |
| `critic_recur_act_ponder_weight` | `0.0` |
| `critic_layerscale_init` | `1e-2` |

`critic_recur_num_blocks` still exists only as a deprecated fallback that maps to `2 * num_blocks` dense layers if `critic_recur_num_dense_layers` is not set.

### QRL

QRL uses `recur_tied` on the critic side for `GCMRNValue` or `GCIQEValue`.

Agent defaults in `third_party/ogbench/impls/agents/qrl.py`:

| Config | Default |
| --- | --- |
| `critic_backbone` | `mlp` |
| `critic_recur_iters` | `4` |
| `critic_recur_num_dense_layers` | `2` |
| `critic_recur_max_iters` | `16` |
| `critic_recur_ln_mode` | `per_layer_final` |
| `critic_recur_use_step_info` | `False` |
| `critic_recur_sinusoidal` | `False` |
| `critic_recur_use_film` | `True` |
| `critic_recur_film_mode` | `hidden` |
| `critic_recur_use_layerscale` | `True` |
| `critic_layerscale_init` | `1e-2` |

Important limitation:

- QRL does not currently thread the ACT arguments into its recurrent value constructors, so `critic_recur_use_act` style knobs are not active there.

### HIQL

HIQL uses `recur_tied` for the scalar value network only.

Agent defaults in `third_party/ogbench/impls/agents/hiql.py`:

| Config | Default |
| --- | --- |
| `value_backbone` | `mlp` |
| `value_recur_iters` | `4` |
| `value_recur_num_dense_layers` | `2` |
| `value_recur_max_iters` | `4` |
| `value_recur_ln_mode` | `per_layer_final` |
| `value_recur_use_step_info` | `False` |
| `value_recur_sinusoidal` | `False` |
| `value_recur_use_film` | `True` |
| `value_recur_film_mode` | `hidden` |
| `value_recur_use_layerscale` | `True` |
| `value_recur_use_act` | `False` |
| `value_recur_act_epsilon` | `0.01` |
| `value_recur_act_min_iters` | `1` |
| `value_recur_act_ponder_weight` | `0.0` |
| `value_layerscale_init` | `1e-2` |

### SAW

After cleanup, SAW uses `recur_tied` for the value network only. Actor and low-actor recurrent variants were removed.

Agent defaults in `third_party/ogbench/impls/agents/saw.py`:

| Config | Default |
| --- | --- |
| `value_backbone` | `mlp` |
| `value_recur_iters` | `4` |
| `value_recur_num_dense_layers` | `2` |
| `value_recur_max_iters` | `16` |
| `value_recur_ln_mode` | `per_layer_final` |
| `value_recur_use_step_info` | `False` |
| `value_recur_sinusoidal` | `False` |
| `value_recur_use_film` | `True` |
| `value_recur_film_mode` | `hidden` |
| `value_recur_use_layerscale` | `True` |
| `value_recur_use_act` | `False` |
| `value_recur_act_epsilon` | `0.01` |
| `value_recur_act_min_iters` | `1` |
| `value_recur_act_ponder_weight` | `0.0` |
| `value_layerscale_init` | `1e-2` |

### CGIVL

CGIVL uses `recur_tied` for the value network only.

Agent defaults in `third_party/ogbench/impls/agents/cgivl.py`:

| Config | Default |
| --- | --- |
| `value_backbone` | `mlp` |
| `value_recur_iters` | `4` |
| `value_recur_num_dense_layers` | `2` |
| `value_recur_max_iters` | `4` |
| `value_recur_ln_mode` | `per_layer_final` |
| `value_recur_use_step_info` | `False` |
| `value_recur_sinusoidal` | `False` |
| `value_recur_use_film` | `True` |
| `value_recur_film_mode` | `hidden` |
| `value_recur_use_layerscale` | `True` |
| `value_recur_use_act` | `False` |
| `value_recur_act_epsilon` | `0.01` |
| `value_recur_act_min_iters` | `1` |
| `value_layerscale_init` | `1e-2` |

## Launcher Defaults That Matter

The launchers are now the practical source of truth for many experiments.

### `slurm/train_crl.slurm`

Defaults:

| Env var | Default |
| --- | --- |
| `CRITIC_BACKBONE` | `mlp` |
| `KTRAIN` | `4` |
| `RECUR_NUM_DENSE_LAYERS` | unset |
| `RECUR_NUM_BLOCKS` | `1` |
| `RECUR_MAX_ITERS` | `8` |
| `CRITIC_RECUR_LN_MODE` | `per_layer_final` |
| `RECUR_USE_STEP_INFO` | `0` |
| `RECUR_SINUSOIDAL` | `0` |
| `CRITIC_RECUR_FILM_MODE` | `hidden` |
| `RECUR_USE_ACT` | `0` |
| `RECUR_ACT_EPSILON` | `0.01` |
| `RECUR_ACT_MIN_ITERS` | `1` |
| `RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `LAYERSCALE_INIT` | `1e-2` |

Notes:

- If `RECUR_NUM_DENSE_LAYERS` is left unset, the agent-side fallback based on `RECUR_NUM_BLOCKS` still applies.
- The launcher does not expose `critic_recur_use_film` or `critic_recur_use_layerscale`, so those remain at the agent defaults (`True`).

### `slurm/train_hiql.slurm`

Defaults:

| Env var | Default |
| --- | --- |
| `VALUE_BACKBONE` | `recur_tied` |
| `VALUE_RECUR_ITERS` | `6` |
| `VALUE_RECUR_NUM_DENSE_LAYERS` | `4` |
| `VALUE_RECUR_MAX_ITERS` | `6` |
| `VALUE_RECUR_LN_MODE` | `per_layer_final` |
| `VALUE_RECUR_USE_STEP_INFO` | `0` |
| `VALUE_RECUR_FILM_MODE` | `hidden` |
| `VALUE_RECUR_USE_ACT` | `0` |
| `VALUE_RECUR_ACT_EPSILON` | `0.01` |
| `VALUE_RECUR_ACT_MIN_ITERS` | `1` |
| `VALUE_RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `VALUE_LAYERSCALE_INIT` | `1e-2` |

Notes:

- The launcher does not expose `value_recur_sinusoidal`, `value_recur_use_film`, or `value_recur_use_layerscale`, so those stay at the agent defaults (`False`, `True`, `True`).

### `slurm/train_saw.slurm`

Defaults:

| Env var | Default |
| --- | --- |
| `VALUE_BACKBONE` | `mlp` |
| `VALUE_RECUR_ITERS` | `4` |
| `VALUE_RECUR_NUM_DENSE_LAYERS` | `2` |
| `VALUE_RECUR_MAX_ITERS` | `16` |
| `VALUE_RECUR_LN_MODE` | `per_layer_final` |
| `VALUE_RECUR_USE_STEP_INFO` | `0` |
| `VALUE_RECUR_FILM_MODE` | `hidden` |
| `VALUE_RECUR_USE_ACT` | `0` |
| `VALUE_RECUR_ACT_EPSILON` | `0.01` |
| `VALUE_RECUR_ACT_MIN_ITERS` | `1` |
| `VALUE_RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `VALUE_LAYERSCALE_INIT` | `1e-2` |

Notes:

- The launcher does not expose `value_recur_sinusoidal`, `value_recur_use_film`, or `value_recur_use_layerscale`, so those stay at the agent defaults (`False`, `True`, `True`).

### `slurm/train_cgivl.slurm`

Defaults:

| Env var | Default |
| --- | --- |
| `VALUE_BACKBONE` | `mlp` |
| `VALUE_RECUR_ITERS` | `4` |
| `VALUE_RECUR_NUM_DENSE_LAYERS` | `2` |
| `VALUE_RECUR_MAX_ITERS` | `4` |
| `VALUE_RECUR_LN_MODE` | `per_layer_final` |
| `VALUE_RECUR_USE_STEP_INFO` | `0` |
| `VALUE_RECUR_SINUSOIDAL` | `0` |
| `VALUE_RECUR_USE_FILM` | `1` |
| `VALUE_RECUR_FILM_MODE` | `hidden` |
| `VALUE_RECUR_USE_LAYERSCALE` | `1` |
| `VALUE_RECUR_USE_ACT` | `0` |
| `VALUE_RECUR_ACT_EPSILON` | `0.01` |
| `VALUE_RECUR_ACT_MIN_ITERS` | `1` |
| `VALUE_LAYERSCALE_INIT` | `1e-2` |

There is no dedicated `qrl` Slurm launcher in the current repo.

## Practical Summary

If you ignore the MLP baseline and look only at the remaining recurrent design, the repo now mostly uses one simplified family:

- tied residual refinement with `num_dense_layers` Dense layers per step
- tied LayerNorm with configurable placement
- optional step information
- optional FiLM with a single explicit mode switch
- optional per-step LayerScale
- optional ACT in CRL, HIQL, SAW, and CGIVL

The biggest remaining degrees of freedom are now:

- `num_iters`
- `num_dense_layers`
- `max_iters`
- `ln_mode`
- step information on/off
- sinusoidal vs learned step encoding
- FiLM on/off and `film_mode`
- LayerScale on/off
- ACT on/off

Everything else that previously made the recurrent surface harder to reason about has been removed.
