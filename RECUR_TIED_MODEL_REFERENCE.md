# Recurrent Model Reference

This note reflects the current repo after removing the retired recurrent variants.

Only core fixed-depth recurrent refinement remains. Earlier modulation, halting, auxiliary-loss, recurrent-readout, and actor-gating experiments are no longer part of the reachable model surface.

Primary code references:

- `third_party/ogbench/impls/utils/networks.py`
- `third_party/ogbench/impls/agents/crl.py`
- `third_party/ogbench/impls/agents/hiql.py`
- `third_party/ogbench/impls/agents/saw.py`
- `third_party/ogbench/impls/agents/qrl.py`
- `third_party/ogbench/impls/agents/gciql.py`
- `third_party/ogbench/impls/agents/cgivl.py`
- `slurm/train_crl.slurm`
- `slurm/train_hiql.slurm`
- `slurm/train_saw.slurm`
- `slurm/train_qrl.slurm`
- `slurm/train_gciql.slurm`
- `slurm/train_cgivl.slurm`

## Current Backbones

The value/critic recurrent path supports:

- `mlp`: baseline non-recurrent path
- `recur_tied`: tied recurrent residual refinement
- `recur_stacked_swiglu`: stacked SwiGLU recurrent refinement

Actors use the baseline MLP path. HIQL and SAW actor losses read the value network at its configured depth; there is no separate actor-facing recurrent-depth override.

## Shared Behavior

Step information is now learned and discrete only:

- `use_step_info=False`: no step signal
- `use_step_info=True`: add a learned step table entry for each refinement step

Because the step table and LayerScale are indexed by step, runs with these enabled must keep `num_iters <= max_iters`.

The recurrent auxiliaries that remain are diagnostic only:

- hidden norm
- hidden drift

There is no learned early-exit distribution, multi-step readout distribution, or recurrent auxiliary loss.

## `recur_tied`

`RecurTiedBackbone` is defined in `third_party/ogbench/impls/utils/networks.py`.

Current arguments:

| Arg | Default | Meaning |
| --- | --- | --- |
| `hidden_dim` | required | Internal recurrent width |
| `out_dim` | required | Output width |
| `num_iters` | required | Train-time recurrent steps |
| `num_dense_layers` | `2` | Number of tied Dense layers inside each residual update |
| `block_type` | `dense` | Inner cell: `dense` or `swiglu` |
| `max_iters` | `16` | Maximum supported recurrent steps for step-indexed parameters |
| `layer_norm` | `True` | Whether LayerNorm is enabled |
| `ln_mode` | `per_layer_final` | One of `pre_loop`, `per_layer`, `per_layer_final`, `pre_loop_per_layer` |
| `use_step_info` | `False` | Whether to add learned step embeddings |
| `use_layerscale` | `True` | Whether to use per-step residual scaling |
| `layerscale_init` | `1e-2` | Initial residual scale |
| `swiglu_pre_ln` | `True` | Whether the SwiGLU cell applies pre-normalization |

High-level forward structure:

1. Project inputs to `hidden_dim`.
2. Run exactly `num_iters` recurrent refinement steps.
3. Optionally add the learned step embedding for the current step.
4. Apply the tied inner block selected by `block_type`.
5. Optionally scale the residual update with per-step LayerScale.
6. Project the final hidden state to `out_dim`.

`block_type='dense'` uses the tied Dense stack controlled by `num_dense_layers` and `ln_mode`.
`block_type='swiglu'` uses a gated SwiGLU update and requires `num_dense_layers=2`.

## `recur_stacked_swiglu`

`RecurStackedSwiGLUBackbone` is also defined in `third_party/ogbench/impls/utils/networks.py`.

Current arguments:

| Arg | Default | Meaning |
| --- | --- | --- |
| `hidden_dim` | required | Internal recurrent width |
| `out_dim` | required | Output width |
| `num_iters` | required | Train-time recurrent steps |
| `num_dense_layers` | `2` | Number of stacked SwiGLU blocks |
| `max_iters` | `16` | Maximum supported recurrent steps for step-indexed parameters |
| `layer_norm` | `True` | Whether LayerNorm is enabled |
| `use_step_info` | `False` | Whether to add learned step embeddings |
| `step_info_inner_mode` | `add` | How step embeddings enter inner blocks |
| `use_layerscale` | `True` | Whether to use per-step residual scaling |
| `use_inner_residual` | `False` | Whether inner blocks use residual connections |
| `layerscale_init` | `1e-2` | Initial residual scale |

## Agent Usage

CRL wires recurrent controls through the critic/value modules:

- `critic_backbone`
- `critic_recur_iters`
- `critic_recur_num_dense_layers`
- `critic_recur_block_type`
- `critic_recur_max_iters`
- `critic_recur_ln_mode`
- `critic_recur_use_step_info`
- `critic_recur_use_layerscale`
- `critic_recur_swiglu_pre_ln`
- `critic_layerscale_init`

HIQL and SAW wire recurrent controls through the scalar value module:

- `value_backbone`
- `value_recur_iters`
- `value_recur_num_dense_layers`
- `value_recur_block_type`
- `value_recur_max_iters`
- `value_recur_ln_mode`
- `value_recur_use_step_info`
- `value_recur_use_layerscale`
- `value_recur_swiglu_pre_ln`
- `value_recur_use_inner_residual`
- `value_layerscale_init`

QRL, GCIQL, and CGIVL use the same cleaned recurrent constructor surface for their value/critic modules. Their launchers expose only the remaining recurrent knobs.

## Launchers

The SLURM launchers have been cleaned so removed flags are no longer exported. The active recurrent launcher knobs are:

- `RECUR_NUM_DENSE_LAYERS`
- `RECUR_MAX_ITERS`
- `RECUR_LN_MODE`
- `RECUR_USE_STEP_INFO`
- `RECUR_USE_LAYERSCALE`
- `CRITIC_RECUR_BLOCK_TYPE`
- `CRITIC_RECUR_SWIGLU_PRE_LN`
- `CRITIC_LAYERSCALE_INIT`
- `VALUE_RECUR_BLOCK_TYPE`
- `VALUE_RECUR_SWIGLU_PRE_LN`
- `VALUE_RECUR_USE_INNER_RESIDUAL`
- `VALUE_LAYERSCALE_INIT`

Algorithm-specific launchers may prefix these with `CRITIC_` or `VALUE_` depending on whether the recurrent module is a critic/value network.
