# Recur-Tied Model Reference

This note summarizes the `recur_tied` design as it is actually implemented in this repo, including:

- the core backbone architecture
- parameterization and initialization details
- agent-level config surfaces
- launcher defaults that override agent defaults in practice
- a few repo-specific caveats

Primary implementation references:

- `third_party/ogbench/impls/utils/networks.py:214`
- `third_party/ogbench/impls/utils/networks.py:1075`
- `third_party/ogbench/impls/utils/networks.py:1150`
- `third_party/ogbench/impls/agents/crl.py:467`
- `third_party/ogbench/impls/agents/qrl.py:366`
- `third_party/ogbench/impls/agents/hiql.py:462`
- `third_party/ogbench/impls/agents/saw.py:440`

## 1. Core `RecurTiedBackbone` design

Core class:

- `third_party/ogbench/impls/utils/networks.py:214`

Declared backbone arguments:

| Arg | Default | Meaning |
| --- | --- | --- |
| `hidden_dim` | required | Internal feature width of the recurrent state |
| `out_dim` | required | Output width after the recurrent stack |
| `num_iters` | required | Number of recurrent refinement steps used at train/eval unless overridden |
| `num_dense_layers` | `2` | Number of tied Dense layers inside each recurrent update |
| `max_iters` | `16` | Maximum length of learned per-step parameter tables |
| `layer_norm` | `True` | Whether LayerNorm is used at all |
| `tied_layer_norm` | `True` | Whether LN modules are shared across recurrent steps |
| `ln_mode` | `'per_layer'` | LN placement: `'pre_loop'`, `'per_layer'`, `'per_layer_final'` |
| `layerscale_init` | `1e-2` | Initial value of residual scaling parameter `alpha` |
| `use_step_info` | `True` | Whether step-dependent information is injected |
| `use_sinusoidal_step_encoding` | `False` | If `True`, use sinusoidal encoding instead of a learned step table |
| `use_film` | `True` | Whether step conditioning is applied through FiLM |
| `use_layerscale` | `True` | Whether the residual branch is multiplied by learned `alpha` |
| `shared_alpha` | `False` | Whether one `alpha` is shared across all steps |
| `use_act` | `False` | Whether ACT halting is enabled |
| `act_epsilon` | `0.01` | ACT halting slack |
| `act_min_iters` | `1` | Minimum recurrent steps before halting is allowed |

### 1.1 Forward structure

The forward pass in `third_party/ogbench/impls/utils/networks.py:233` is:

1. Input projection: `x -> Dense(hidden_dim)` at `:258`.
2. Optional step parameterization:
   - Learned step table `step_embed[max_iters, hidden_dim]` if `use_step_info=True` and `use_sinusoidal_step_encoding=False` at `:261-267`.
   - Otherwise no learned table, and sinusoidal encodings are produced on the fly if sinusoidal mode is enabled at `:333-338`.
3. Optional LayerScale parameter `alpha`:
   - shape `(max_iters,)` if `shared_alpha=False`
   - shape `(1,)` if `shared_alpha=True`
   - declared at `:272-275`
4. Optional conditioning layers:
   - `step_proj = Dense(hidden_dim, use_bias=False)` at `:277`
   - `film_fc1_h = Dense(hidden_dim)` at `:278`
   - `film_fc2 = Dense(2 * hidden_dim)` at `:279`
   - `act_halting = Dense(1)` when ACT is enabled at `:280`
5. Tied residual branch:
   - `tied_fc1 ... tied_fcN`, each `Dense(hidden_dim)`, created once and reused at every recurrent step at `:282-284`
6. LayerNorm placement:
   - pre-loop LN if `ln_mode='pre_loop'` at `:289-295`, `:327-331`
   - per-layer LN if `ln_mode in ('per_layer', 'per_layer_final')` at `:296-310`, `:350-363`
7. Recurrent update:
   - FiLM path: `u = (1 + gamma) * h1 + beta` after `film_fc2(silu(step + hidden))` at `:342-346`
   - non-FiLM path: `u = h1 + step_contrib` or `u = h1` at `:347-348`
   - pass through tied Dense stack with SiLU between internal layers at `:350-363`
   - residual update:
     - `h_next = h + alpha_k * u` if LayerScale is enabled at `:365-367`
     - `h_next = h + u` otherwise at `:368`
8. Optional ACT halting and weighted final state accumulation at `:370-387`.
9. Final projection: `Dense(out_dim)` at `:389`.

### 1.2 Iteration-bound logic

The backbone allows test-time iteration overrides, but only safely when it is not indexing into untrained per-step parameters.

Bound enforcement is at `third_party/ogbench/impls/utils/networks.py:247-256`.

The strict `iters <= max_iters` constraint is required if any of these are true:

- learned discrete step embeddings are used
- per-iteration `alpha_k` is used
- untied per-iteration LayerNorm is used

If all step-dependent parameters are fully shared, the code explicitly allows extrapolation beyond `max_iters`.

## 2. Explicit initialization details

Only a few initializations are explicitly set in this repo for `recur_tied`.

| Component | Where | Init |
| --- | --- | --- |
| learned discrete `step_embed` | `networks.py:263-267` | `normal(stddev=0.02)` |
| residual `alpha` / LayerScale | `networks.py:272-275` | constant `layerscale_init` |
| actor final mean head | `networks.py:778` | `default_init(1e-2)` by default |
| discrete actor final logit head | `networks.py:968` | `default_init(1e-2)` by default |

Important detail:

- The recurrent Dense layers inside `RecurTiedBackbone` do not override `kernel_init`, so they use Flax `nn.Dense` defaults. The repo does not specify a custom numeric initializer for `input_proj`, `tied_fc*`, `step_proj`, `film_fc1_h`, `film_fc2`, `act_halting`, or `output_proj`.

LayerScale example:

- The repo-wide default `layerscale_init` is almost always `1e-2`.
- That means the residual update starts as `h_next = h + 0.01 * u` when `shared_alpha=True`, or `h_next = h + alpha_k * u` with every `alpha_k` initialized to `0.01` when `shared_alpha=False`.

## 3. Conditioning design

### 3.1 Step information

There are three step-conditioning regimes:

1. No step information:
   - `use_step_info=False`
   - no `step_proj`
   - FiLM can still be applied from hidden state alone

2. Learned discrete step embeddings:
   - `use_step_info=True`
   - `use_sinusoidal_step_encoding=False`
   - learned table of shape `(max_iters, hidden_dim)`

3. Sinusoidal step encodings:
   - `use_step_info=True`
   - `use_sinusoidal_step_encoding=True`
   - computed by `sinusoidal_step_encoding` in `third_party/ogbench/impls/utils/positional_encoding.py:4`

Sinusoidal helper details:

- frequencies follow the standard transformer-style log spacing at `positional_encoding.py:13-18`
- `max_steps` is currently ignored in the helper at `positional_encoding.py:6`

### 3.2 FiLM path

FiLM is implemented at `third_party/ogbench/impls/utils/networks.py:342-346`.

The exact transformation is:

```text
fc1_out = film_fc1_h(h1)              if no step info
fc1_out = step_contrib_k + film_fc1_h(h1) otherwise
film = film_fc2(silu(fc1_out))
gamma, beta = split(film)
u = (1 + gamma) * h1 + beta
```

So FiLM is not just step-only conditioning; it always includes a hidden-state projection, and step information is additive on top of that.

## 4. Where `recur_tied` is used

### 4.1 CRL critic / value

CRL wires `recur_tied` into `GCBilinearValue` or `GCDiscreteBilinearCritic`, which build bilinear `phi` and `psi` towers on top of `RecurTiedBackbone`.

References:

- wiring: `third_party/ogbench/impls/agents/crl.py:280-360`
- default config: `third_party/ogbench/impls/agents/crl.py:488-509`
- bilinear recurrent backbone wrapper: `third_party/ogbench/impls/utils/networks.py:1150-1302`

Key CRL defaults for `recur_tied`:

| Config | Default |
| --- | --- |
| `critic_recur_iters` | `4` |
| `critic_recur_num_dense_layers` | `2` |
| `critic_recur_max_iters` | `16` |
| `critic_recur_tied_ln` | `True` |
| `critic_recur_ln_mode` | `'per_layer'` |
| `critic_recur_use_step_info` | `True` |
| `critic_recur_sinusoidal` | `False` |
| `critic_recur_use_film` | `True` |
| `critic_recur_use_layerscale` | `True` |
| `critic_recur_use_film_context` | `False` |

This ordering matches the current hypothesis that critic geometry matters most for CRL, then HIQL, and only weakly for SAW.
| `critic_recur_shared_alpha` | `False` |
| `critic_recur_use_act` | `False` |
| `critic_recur_act_epsilon` | `0.01` |
| `critic_recur_act_min_iters` | `1` |
| `critic_recur_act_ponder_weight` | `0.0` |
| `critic_layerscale_init` | `1e-2` |
| `critic_backbone_hidden_dim` | `0` meaning “use `value_hidden_dims[-1]`” |
| `value_hidden_dims` | `(512, 512, 512)` |
| `latent_dim` | `512` |

Design consequence:

- each of `phi` and `psi` is its own recurrent tower
- each outputs a `latent_dim`-dimensional embedding
- the final score is `sum(phi * psi / sqrt(latent_dim))` at `networks.py:1304-1347`
- if `ensemble=True`, the recurrent towers are vmapped into a 2-head ensemble at `networks.py:1298-1301`

### 4.2 QRL critic

QRL also allows `critic_backbone='recur_tied'`, but it uses `GCMRNValue` or `GCIQEValue` rather than the CRL bilinear value.

References:

- wiring: `third_party/ogbench/impls/agents/qrl.py:230-299`
- defaults: `third_party/ogbench/impls/agents/qrl.py:389-405`

Key QRL defaults for `recur_tied`:

| Config | Default |
| --- | --- |
| `critic_recur_iters` | `4` |
| `critic_recur_num_dense_layers` | `2` |
| `critic_recur_max_iters` | `16` |
| `critic_recur_tied_ln` | `True` |
| `critic_recur_ln_mode` | `'per_layer'` |
| `critic_recur_use_step_info` | `True` |
| `critic_recur_sinusoidal` | `False` |
| `critic_recur_use_film` | `True` |
| `critic_recur_use_layerscale` | `True` |
| `critic_recur_shared_alpha` | `False` |
| `critic_layerscale_init` | `1e-2` |
| `value_hidden_dims` | `(512, 512, 512)` |
| `latent_dim` | `512` |
| `quasimetric_type` | `'iqe'` by default |

Important caveat:

- Unlike CRL, QRL does not thread ACT arguments through its `GCMRNValue` / `GCIQEValue` constructors. So CRL launcher flags like `critic_recur_use_act` are meaningful for `crl`, but not for `qrl`.

### 4.3 HIQL recurrent value

HIQL uses a scalar `GCRecurrentValue`, not a bilinear critic.

References:

- recurrent value wrapper: `third_party/ogbench/impls/utils/networks.py:1075-1137`
- HIQL wiring: `third_party/ogbench/impls/agents/hiql.py:299-322`
- HIQL defaults: `third_party/ogbench/impls/agents/hiql.py:462-489`

Key HIQL defaults:

| Config | Default |
| --- | --- |
| `value_recur_iters` | `4` |
| `value_recur_num_dense_layers` | `2` |
| `value_recur_max_iters` | `4` |
| `value_recur_tied_ln` | `True` |
| `value_recur_ln_mode` | `'pre_loop'` |
| `value_recur_use_step_info` | `False` |
| `value_recur_sinusoidal` | `False` |
| `value_recur_use_film` | `True` |
| `value_recur_use_layerscale` | `True` |
| `value_recur_shared_alpha` | `True` |
| `value_recur_use_act` | `False` |
| `value_recur_act_epsilon` | `0.01` |
| `value_recur_act_min_iters` | `1` |
| `value_recur_act_ponder_weight` | `0.0` |
| `value_layerscale_init` | `1e-2` |
| `value_backbone_hidden_dim` | `0` meaning “use `value_hidden_dims[-1]`” |
| `value_hidden_dims` | `(512, 512, 512)` |

Design consequence:

- HIQL’s default recurrent value is more conservative than CRL/QRL:
  - fewer supported steps by default (`max_iters=4`)
  - no step information by default
  - `pre_loop` LN by default
  - shared single `alpha` by default

### 4.4 SAW recurrent value, actor, and low actor

SAW uses the recurrent backbone in more places than the other agents.

References:

- value wiring: `third_party/ogbench/impls/agents/saw.py:282-304`
- actor and low-actor wiring: `third_party/ogbench/impls/agents/saw.py:327-342`, `:365-380`
- defaults: `third_party/ogbench/impls/agents/saw.py:442-508`

SAW value defaults:

| Config | Default |
| --- | --- |
| `value_recur_iters` | `4` |
| `value_recur_num_dense_layers` | `2` |
| `value_recur_max_iters` | `16` |
| `value_recur_tied_ln` | `True` |
| `value_recur_ln_mode` | `'per_layer'` |
| `value_recur_use_step_info` | `True` |
| `value_recur_sinusoidal` | `False` |
| `value_recur_use_film` | `True` |
| `value_recur_use_layerscale` | `True` |
| `value_recur_shared_alpha` | `False` |
| `value_recur_use_act` | `False` |
| `value_recur_act_epsilon` | `0.01` |
| `value_recur_act_min_iters` | `1` |
| `value_recur_act_ponder_weight` | `0.0` |
| `value_layerscale_init` | `1e-2` |

SAW actor defaults if `actor_backbone='recur_tied'`:

| Config | Default |
| --- | --- |
| `actor_recur_iters` | `4` |
| `actor_recur_num_dense_layers` | `2` |
| `actor_recur_max_iters` | `16` |
| `actor_recur_tied_ln` | `True` |
| `actor_recur_ln_mode` | `'per_layer'` |
| `actor_recur_use_step_info` | `True` |
| `actor_recur_sinusoidal` | `False` |
| `actor_recur_use_film` | `True` |
| `actor_recur_use_layerscale` | `True` |
| `actor_recur_shared_alpha` | `False` |
| `actor_layerscale_init` | `1e-2` |

SAW low-actor defaults if `low_actor_backbone='recur_tied'`:

| Config | Default |
| --- | --- |
| `low_actor_recur_iters` | `4` |
| `low_actor_recur_num_dense_layers` | `2` |
| `low_actor_recur_max_iters` | `16` |
| `low_actor_recur_tied_ln` | `True` |
| `low_actor_recur_ln_mode` | `'per_layer'` |
| `low_actor_recur_use_step_info` | `True` |
| `low_actor_recur_sinusoidal` | `False` |
| `low_actor_recur_use_film` | `True` |
| `low_actor_recur_use_layerscale` | `True` |
| `low_actor_recur_shared_alpha` | `False` |
| `low_actor_layerscale_init` | `1e-2` |

Extra actor-head detail:

- if SAW uses `GCActor` with a recurrent backbone, the recurrent module outputs `hidden_dim`, then the actor mean head uses `default_init(1e-2)` at `networks.py:778`

## 5. Launcher defaults that matter in practice

The raw agent defaults are not always what runs. Several Slurm launchers override them.

### 5.1 `slurm/train_crl_generic_array.slurm`

References:

- recurrent env defaults: `slurm/train_crl_generic_array.slurm:115-138`
- flag mapping: `slurm/train_crl_generic_array.slurm:364-380`

Default recurrent launcher values:

| Env var | Launcher default |
| --- | --- |
| `KTRAIN` | `4` |
| `RECUR_NUM_BLOCKS` | `1` |
| `RECUR_NUM_DENSE_LAYERS` | unset, then derived as `2 * RECUR_NUM_BLOCKS` if not provided |
| `RECUR_MAX_ITERS` | `24` |
| `SINUSOIDAL` | `1` |
| `TIED_LN` | `1` |
| `RECUR_LN_MODE` | `per_layer` |
| `RECUR_USE_STEP_INFO` | `1` |
| `RECUR_USE_FILM` | `1` |
| `RECUR_USE_LAYERSCALE` | `1` |
| `RECUR_SHARED_ALPHA` | `0` |
| `RECUR_USE_ACT` | `0` |
| `RECUR_ACT_EPSILON` | `0.01` |
| `RECUR_ACT_MIN_ITERS` | `1` |
| `RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `LAYERSCALE_INIT` | `1e-2` |

Important difference versus CRL agent defaults:

- the launcher defaults to sinusoidal step encoding and `max_iters=24`
- the agent default config itself uses learned discrete step embeddings and `max_iters=16`

Named presets in comments:

- `Recur3x4`: `KTRAIN=4`, `RECUR_NUM_DENSE_LAYERS=3`, `RECUR_MAX_ITERS=24`, `SINUSOIDAL=1`, `TIED_LN=1` at `slurm/train_crl_generic_array.slurm:19-24`
- `Recur6x4`: `KTRAIN=4`, `RECUR_NUM_DENSE_LAYERS=6`, `RECUR_MAX_ITERS=24`, `SINUSOIDAL=1`, `TIED_LN=1` at `slurm/train_crl_generic_array.slurm:23-24`

### 5.2 `slurm/train_hiql_recur_value_array.slurm`

References:

- recurrent value defaults: `slurm/train_hiql_recur_value_array.slurm:83-102`
- flag mapping: `slurm/train_hiql_recur_value_array.slurm:316-332`

Launcher defaults:

| Env var | Launcher default |
| --- | --- |
| `VALUE_BACKBONE` | `recur_tied` |
| `VALUE_RECUR_ITERS` | `6` |
| `VALUE_RECUR_NUM_DENSE_LAYERS` | `4` |
| `VALUE_RECUR_MAX_ITERS` | `6` |
| `VALUE_RECUR_USE_STEP_INFO` | `0` |
| `VALUE_RECUR_SINUSOIDAL` | `0` |
| `VALUE_RECUR_TIED_LN` | `1` |
| `VALUE_RECUR_LN_MODE` | `per_layer` |
| `VALUE_RECUR_USE_FILM` | `1` |
| `VALUE_RECUR_USE_LAYERSCALE` | `1` |
| `VALUE_RECUR_SHARED_ALPHA` | `0` |
| `VALUE_RECUR_USE_ACT` | `0` |
| `VALUE_RECUR_ACT_EPSILON` | `0.01` |
| `VALUE_RECUR_ACT_MIN_ITERS` | `1` |
| `VALUE_RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `VALUE_LAYERSCALE_INIT` | `1e-2` |

Important difference versus HIQL agent defaults:

- launcher runs a deeper recurrent value by default: `6` iterations and `4` dense layers per update
- launcher switches `ln_mode` from HIQL default `pre_loop` to `per_layer`
- launcher switches `shared_alpha` from HIQL default `True` to `False`

### 5.3 `slurm/train_saw_array.slurm`

References:

- launcher defaults: `slurm/train_saw_array.slurm:80-143`
- value flag mapping: `slurm/train_saw_array.slurm:287-302`
- actor flag mapping: `slurm/train_saw_array.slurm:332-343`
- low-actor flag mapping: `slurm/train_saw_array.slurm:385-396`

Value-side launcher defaults:

| Env var | Launcher default |
| --- | --- |
| `VALUE_RECUR_ITERS` | `4` |
| `VALUE_RECUR_NUM_DENSE_LAYERS` | `2` |
| `VALUE_RECUR_MAX_ITERS` | `16` |
| `VALUE_RECUR_LN_MODE` | `per_layer` |
| `VALUE_RECUR_USE_STEP_INFO` | `0` |
| `VALUE_RECUR_SINUSOIDAL` | `0` |
| `VALUE_RECUR_USE_FILM` | `1` |
| `VALUE_RECUR_USE_LAYERSCALE` | `1` |
| `VALUE_RECUR_SHARED_ALPHA` | `0` |
| `VALUE_RECUR_USE_ACT` | `0` |
| `VALUE_RECUR_ACT_EPSILON` | `0.01` |
| `VALUE_RECUR_ACT_MIN_ITERS` | `1` |
| `VALUE_RECUR_ACT_PONDER_WEIGHT` | `0.0` |
| `VALUE_LAYERSCALE_INIT` | `1e-2` |

Actor-side launcher defaults if you choose `actor_backbone='recur_tied'`:

| Env var | Launcher default |
| --- | --- |
| `ACTOR_RECUR_ITERS` | `4` |
| `ACTOR_RECUR_NUM_DENSE_LAYERS` | `2` |
| `ACTOR_RECUR_MAX_ITERS` | `16` |
| `ACTOR_RECUR_LN_MODE` | `per_layer` |
| `ACTOR_RECUR_USE_STEP_INFO` | `0` |
| `ACTOR_RECUR_SINUSOIDAL` | `0` |
| `ACTOR_RECUR_USE_FILM` | `1` |
| `ACTOR_RECUR_USE_LAYERSCALE` | `1` |
| `ACTOR_RECUR_SHARED_ALPHA` | `0` |
| `ACTOR_LAYERSCALE_INIT` | `1e-2` |

Low-actor launcher defaults if you choose `low_actor_backbone='recur_tied'`:

| Env var | Launcher default |
| --- | --- |
| `LOW_ACTOR_RECUR_ITERS` | `4` |
| `LOW_ACTOR_RECUR_NUM_DENSE_LAYERS` | `2` |
| `LOW_ACTOR_RECUR_MAX_ITERS` | `16` |
| `LOW_ACTOR_RECUR_LN_MODE` | `per_layer` |
| `LOW_ACTOR_RECUR_USE_STEP_INFO` | `0` |
| `LOW_ACTOR_RECUR_SINUSOIDAL` | `0` |
| `LOW_ACTOR_RECUR_USE_FILM` | `1` |
| `LOW_ACTOR_RECUR_USE_LAYERSCALE` | `1` |
| `LOW_ACTOR_RECUR_SHARED_ALPHA` | `0` |
| `LOW_ACTOR_LAYERSCALE_INIT` | `1e-2` |

Important launcher caveat:

- `train_saw_array.slurm` does not expose `*_recur_tied_ln`; the agent defaults therefore remain in force, which means tied LayerNorm stays `True` unless the code/config is changed elsewhere.

## 6. Practical reading of the repo’s recur-tied “designs”

If you ignore MLP/resnet baselines and focus only on `recur_tied`, the repo currently uses a few recurring design patterns:

### Design A: CRL/QRL-style recurrent bilinear critic

- hidden width usually resolves to `512`
- latent output width `512`
- two recurrent towers `phi` and `psi`
- `K_train=4`
- per-layer LayerNorm
- FiLM enabled
- LayerScale enabled with `init=1e-2`
- usually untied `alpha_k`
- often sinusoidal in launcher defaults, even though the raw agent config default is learned step embeddings

### Design B: HIQL recurrent scalar value

- scalar output via `GCRecurrentValue`
- default raw agent config is conservative:
  - `iters=4`
  - `dense_layers=2`
  - `max_iters=4`
  - `pre_loop` LN
  - no step info
  - shared `alpha`
- the dedicated launcher uses a stronger setting:
  - `iters=6`
  - `dense_layers=4`
  - `max_iters=6`
  - `per_layer` LN
  - no step info
  - unshared `alpha`

### Design C: SAW recurrent value / actor / low actor

- the same recurrent backbone can appear in value, actor, and low actor
- raw SAW defaults use tied LN, per-layer LN mode, step info on, FiLM on, LayerScale on, unshared alpha
- the SAW launcher turns step info off by default for value, actor, and low actor

## 7. Caveats and gotchas

1. `critic_recur_num_blocks` is deprecated surface area.
   - In CRL/QRL it is only a fallback.
   - Effective dense-layer count becomes `critic_recur_num_dense_layers` if provided, otherwise `2 * critic_recur_num_blocks`.
   - See `crl.py:294-299`, `qrl.py:244-249`.

2. QRL does not currently expose ACT through its value constructors.
   - CRL does.
   - HIQL and SAW do for recurrent scalar values.

3. Most recurrent Dense initializers are not numerically specified in this repo.
   - The only explicit recurrent init values here are:
     - `step_embed` stddev `0.02`
     - `LayerScale alpha` init `1e-2` unless overridden

4. The term “2x4” or “6x4” in comments is repo shorthand, not a formal class name.
   - In practice it means:
     - number of Dense layers per recurrent residual update
     - times number of recurrent iterations

## 8. Short answer: what is the default LayerScale init?

Across the core backbone, CRL, QRL, HIQL, SAW, and the Slurm launchers, the explicit LayerScale initialization is almost always:

```text
layerscale_init = 1e-2
```

That value is used to initialize the recurrent residual scaling parameter `alpha`.

## 9. External References

Run commands for the FiLM-context ablation now live in [FILM_CONTEXT_ABLATION_RUNS.md](/Users/bruce/Recurrent-Offline-RL/FILM_CONTEXT_ABLATION_RUNS.md).

Task- and algorithm-level non-model hyperparameters now live in [ALGORITHM_HYPERPARAMETERS_REFERENCE.md](/Users/bruce/Recurrent-Offline-RL/ALGORITHM_HYPERPARAMETERS_REFERENCE.md).
