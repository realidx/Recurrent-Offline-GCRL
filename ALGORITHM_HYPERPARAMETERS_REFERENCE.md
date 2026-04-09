# Algorithm Hyperparameters Reference

This file summarizes non-model hyperparameters only.

Included:

- optimization / RL knobs such as `discount`, `tau`, `expectile`, `alpha`
- goal-sampling probabilities such as `actor_p_*` and `value_p_*`
- algorithm behavior flags such as `gc_negative`, `value_geom_sample`, `actor_geom_sample`

Excluded:

- architecture and model-shape knobs such as `KTRAIN`, recurrent depth, hidden dims, backbone choice, and layer counts

Source policy:

- `CRL` and `HIQL` antmaze task rows are derived from OGBench third-party benchmark entries in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh), plus the corresponding agent defaults in [third_party/ogbench/impls/agents/crl.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/crl.py) and [third_party/ogbench/impls/agents/hiql.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/hiql.py).
- `SAW` is not part of the OGBench third-party benchmark table in this repo. Its values below are local worktree defaults from [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm) and [third_party/ogbench/impls/agents/saw.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/saw.py).

## CRL

Shared non-model defaults from [crl.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/crl.py#L468):

- `lr=3e-4`
- `batch_size=1024`
- `actor_loss=ddpgbc`
- `const_std=True`
- `value_p_curgoal=0.0`
- `value_p_trajgoal=1.0`
- `value_p_randomgoal=0.0`
- `value_geom_sample=True`
- `actor_geom_sample=False`
- `gc_negative=False`
- `p_aug=0.0`

Effective antmaze locomotion hyperparameters:

| Task | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `antmaze-medium-navigate-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-large-navigate-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-giant-navigate-v0` | `0.1` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-teleport-navigate-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-medium-stitch-v0` | `0.1` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |
| `antmaze-large-stitch-v0` | `0.1` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |
| `antmaze-giant-stitch-v0` | `0.1` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |
| `antmaze-teleport-stitch-v0` | `0.1` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |

Pattern from OGBench:

- `navigate`: actor goal mix is `1.0 / 0.0`
- `stitch`: actor goal mix is `0.5 / 0.5`
- `giant`: `discount=0.995`
- other listed antmaze locomotion tasks: `discount=0.99`

## HIQL

Shared non-model defaults from [hiql.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/hiql.py#L468):

- `lr=3e-4`
- `batch_size=1024`
- `tau=0.005`
- `expectile=0.7`
- `subgoal_steps=25`
- `rep_dim=10`
- `value_p_curgoal=0.2`
- `value_p_trajgoal=0.5`
- `value_p_randomgoal=0.3`
- `value_geom_sample=True`
- `actor_p_curgoal=0.0`
- `actor_geom_sample=False`
- `gc_negative=True`
- `p_aug=0.0`

Effective antmaze locomotion hyperparameters:

| Task | low_alpha | high_alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `antmaze-medium-navigate-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `antmaze-large-navigate-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `antmaze-giant-navigate-v0` | `3.0` | `3.0` | `0.995` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `antmaze-teleport-navigate-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `antmaze-medium-stitch-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| `antmaze-large-stitch-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| `antmaze-giant-stitch-v0` | `3.0` | `3.0` | `0.995` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| `antmaze-teleport-stitch-v0` | `3.0` | `3.0` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |

Pattern from OGBench:

- `navigate`: actor goal mix is `1.0 / 0.0`
- `stitch`: actor goal mix is `0.5 / 0.5`
- `giant`: `discount=0.995`
- other listed antmaze locomotion tasks: `discount=0.99`

## SAW

`SAW` does not appear in the OGBench third-party benchmark table in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh), so there is no repo-provided task-specific benchmark grid analogous to `CRL` or `HIQL`.

Current local non-model defaults used by [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm#L46) and [saw.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/saw.py#L509):

| Hyperparameter | Value |
| --- | --- |
| `lr` | `3e-4` |
| `batch_size` | `1024` |
| `discount` | `0.99` |
| `tau` | `0.005` |
| `expectile` | `0.7` |
| `low_alpha` | `3.0` |
| `awr_alpha` | `3.0` |
| `kl_alpha` | `3.0` |
| `subgoal_steps` | `25` |
| `rep_dim` | `10` |
| `share_goal_rep` | `False` |
| `gc_negative` | `True` |
| `value_geom_sample` | `True` |
| `actor_geom_sample` | `False` |
| `actor_geom_discount` | `0.99` |
| `value_p_curgoal` | `0.2` |
| `value_p_trajgoal` | `0.5` |
| `value_p_randomgoal` | `0.3` |
| `actor_p_curgoal` | `0.0` |
| `actor_p_trajgoal` | `1.0` |
| `actor_p_randomgoal` | `0.0` |
| `p_aug` | `0.0` |

Practical note:

- if you run `SAW` through [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm) without overrides, these are the effective algorithm hyperparameters regardless of antmaze task
- if you want task-specific `SAW` settings analogous to OGBench `CRL` / `HIQL`, they need to be defined explicitly by local run commands or a new task table
