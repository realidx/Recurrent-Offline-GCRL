# Algorithm Hyperparameters Reference

This file summarizes non-model hyperparameters only.

Included:

- optimization / RL knobs such as `discount`, `tau`, `expectile`, `alpha`
- goal-sampling probabilities such as `actor_p_*` and `value_p_*`
- algorithm behavior flags such as `gc_negative`, `value_geom_sample`, `actor_geom_sample`

Excluded:

- architecture and model-shape knobs such as `KTRAIN`, recurrent depth, hidden dims, backbone choice, and layer counts

Source policy:

- `CRL`, `QRL`, `GCIQL`, and `HIQL` task rows are derived from OGBench third-party benchmark entries in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh), plus the corresponding agent defaults in [third_party/ogbench/impls/agents/crl.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/crl.py), [third_party/ogbench/impls/agents/qrl.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/qrl.py), [third_party/ogbench/impls/agents/gciql.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/gciql.py), and [third_party/ogbench/impls/agents/hiql.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/hiql.py).
- `CGIVL` / `CGCIVL` values below are taken from the paper appendix in [references/CGCIVL.pdf](/Users/bruce/Recurrent-Offline-RL/references/CGCIVL.pdf), the authors' README example in [third_party/CGCIVL/readme.md](/Users/bruce/Recurrent-Offline-RL/third_party/CGCIVL/readme.md), and the authors' agent defaults in [third_party/CGCIVL/impls/agents/cgivl.py](/Users/bruce/Recurrent-Offline-RL/third_party/CGCIVL/impls/agents/cgivl.py). Where the paper gives only task-family rules rather than per-task commands, those rows are marked as default-backed or inferred from the published rule.
- `SAW` is not part of the OGBench third-party benchmark table in this repo. Its values below are taken from the SAW paper appendix in [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), plus local worktree defaults from [slurm/train_saw.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw.slurm) and [third_party/ogbench/impls/agents/saw.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/saw.py).

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
| `antmaze-medium-explore-v0` | `0.003` | `0.99` | `0.0` | `0.0` | `1.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-large-explore-v0` | `0.003` | `0.99` | `0.0` | `0.0` | `1.0` | `0.0` | `1.0` | `0.0` |
| `antmaze-teleport-explore-v0` | `0.003` | `0.99` | `0.0` | `0.0` | `1.0` | `0.0` | `1.0` | `0.0` |

Pattern from OGBench:

- `navigate`: actor goal mix is `1.0 / 0.0`
- `stitch`: actor goal mix is `0.5 / 0.5`
- `explore`: actor goal mix is `0.0 / 1.0`, and antmaze `alpha` drops to `0.003`
- `giant`: `discount=0.995`
- other listed antmaze locomotion tasks: `discount=0.99`

Effective humanoidmaze locomotion hyperparameters:

| Task | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `humanoidmaze-medium-navigate-v0` | `0.1` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-large-navigate-v0` | `0.1` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-giant-navigate-v0` | `0.1` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-medium-stitch-v0` | `0.1` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-large-stitch-v0` | `0.1` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-giant-stitch-v0` | `0.1` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `1.0` | `0.0` |

Pattern from OGBench:

- all listed humanoidmaze CRL tasks use `discount=0.995`
- `navigate`: actor goal mix is `1.0 / 0.0`
- `stitch`: actor goal mix is `0.5 / 0.5`

Effective state-manipulation hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cube` play | `cube-single-play-v0`, `cube-double-play-v0`, `cube-triple-play-v0`, `cube-quadruple-play-v0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `cube` noisy | `cube-single-noisy-v0`, `cube-double-noisy-v0`, `cube-triple-noisy-v0`, `cube-quadruple-noisy-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `scene` play | `scene-play-v0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `scene` noisy | `scene-noisy-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `puzzle` play | `puzzle-3x3-play-v0`, `puzzle-4x4-play-v0`, `puzzle-4x5-play-v0`, `puzzle-4x6-play-v0` | `3.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |
| `puzzle` noisy | `puzzle-3x3-noisy-v0`, `puzzle-4x4-noisy-v0`, `puzzle-4x5-noisy-v0`, `puzzle-4x6-noisy-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` | `0.0` |

Pattern from OGBench:

- `play` state-manipulation CRL tasks use `alpha=3.0`
- `noisy` state-manipulation CRL tasks use `alpha=0.1`
- those rows do not override `discount` or goal-sampling probabilities, so they inherit the shared CRL defaults

## QRL

Shared non-model defaults from [qrl.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/qrl.py#L340):

- `lr=3e-4`
- `batch_size=1024`
- `eps=0.05`
- `actor_loss=ddpgbc`
- `const_std=True`
- `value_p_curgoal=0.0`
- `value_p_trajgoal=0.0`
- `value_p_randomgoal=1.0`
- `value_geom_sample=True`
- `actor_geom_sample=False`
- `gc_negative=False`
- `p_aug=0.0`

The local launcher [slurm/train_qrl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_qrl.slurm) carries the QRL training path in this repo. Its recurrent critic/value switches such as `CRITIC_BACKBONE`, `KTRAIN`, `RECUR_NUM_DENSE_LAYERS`, and `CRITIC_EVAL_NUM_ITERS` are architecture knobs only, so they are intentionally excluded from the task grid below.

Effective antmaze locomotion hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | `antmaze-medium-navigate-v0`, `antmaze-large-navigate-v0`, `antmaze-teleport-navigate-v0` | `0.003` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| giant navigate | `antmaze-giant-navigate-v0` | `0.003` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| stitch | `antmaze-medium-stitch-v0`, `antmaze-large-stitch-v0`, `antmaze-teleport-stitch-v0` | `0.003` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |
| giant stitch | `antmaze-giant-stitch-v0` | `0.003` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |
| explore | `antmaze-medium-explore-v0`, `antmaze-large-explore-v0`, `antmaze-teleport-explore-v0` | `0.001` | `0.99` | `0.0` | `0.0` | `1.0` | `0.0` | `0.0` | `1.0` |

Effective pointmaze locomotion hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | `pointmaze-medium-navigate-v0`, `pointmaze-large-navigate-v0`, `pointmaze-teleport-navigate-v0` | `0.0003` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| giant navigate | `pointmaze-giant-navigate-v0` | `0.0003` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| stitch | `pointmaze-medium-stitch-v0`, `pointmaze-large-stitch-v0`, `pointmaze-teleport-stitch-v0` | `0.0003` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |
| giant stitch | `pointmaze-giant-stitch-v0` | `0.0003` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |

Effective humanoidmaze and antsoccer hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| humanoidmaze navigate | `humanoidmaze-medium-navigate-v0`, `humanoidmaze-large-navigate-v0`, `humanoidmaze-giant-navigate-v0` | `0.001` | `0.995` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| humanoidmaze stitch | `humanoidmaze-medium-stitch-v0`, `humanoidmaze-large-stitch-v0`, `humanoidmaze-giant-stitch-v0` | `0.001` | `0.995` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |
| antsoccer navigate | `antsoccer-arena-navigate-v0`, `antsoccer-medium-navigate-v0` | `0.003` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| antsoccer stitch | `antsoccer-arena-stitch-v0`, `antsoccer-medium-stitch-v0` | `0.003` | `0.99` | `0.0` | `0.5` | `0.5` | `0.0` | `0.0` | `1.0` |

Effective state-manipulation hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cube` play | `cube-single-play-v0`, `cube-double-play-v0`, `cube-triple-play-v0`, `cube-quadruple-play-v0` | `0.3` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| `cube` noisy | `cube-single-noisy-v0`, `cube-double-noisy-v0`, `cube-triple-noisy-v0`, `cube-quadruple-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| `scene` play | `scene-play-v0` | `0.3` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| `scene` noisy | `scene-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| `puzzle` play | `puzzle-3x3-play-v0`, `puzzle-4x4-play-v0`, `puzzle-4x5-play-v0`, `puzzle-4x6-play-v0` | `0.3` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |
| `puzzle` noisy | `puzzle-3x3-noisy-v0`, `puzzle-4x4-noisy-v0`, `puzzle-4x5-noisy-v0`, `puzzle-4x6-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.0` | `0.0` | `1.0` |

Visual-task overrides from OGBench:

- visual locomotion and manipulation tasks use `train_steps=500000`, `batch_size=256`, and `encoder=impala_small`
- visual antmaze keeps the corresponding state antmaze `alpha`, `discount`, and actor goal mix
- visual humanoidmaze keeps `alpha=0.001`, `discount=0.995`, and the corresponding navigate/stitch actor goal mix
- visual cube/scene/puzzle tasks use `p_aug=0.5`
- powderworld tasks use `train_steps=500000`, `batch_size=256`, `encoder=impala_small`, `discrete=True`, `actor_loss=awr`, `alpha=3.0`, and `eval_temperature=0.3`

Patterns from OGBench:

- `QRL` always keeps the value-goal mix at `0.0 / 0.0 / 1.0`, unlike `CRL`, `GCIQL`, and `HIQL`
- locomotion `navigate`: actor goal mix is `1.0 / 0.0`
- locomotion `stitch`: actor goal mix is `0.5 / 0.5`
- locomotion `explore`: actor goal mix is `0.0 / 1.0`
- `giant` antmaze and all listed humanoidmaze tasks use `discount=0.995`
- locomotion `alpha` is family-specific: `pointmaze=0.0003`, `antmaze=0.003`, `humanoidmaze=0.001`, `antsoccer=0.003`
- state-manipulation `play` tasks use `alpha=0.3`, while `noisy` tasks use `alpha=0.03`

## GCIQL

Shared non-model defaults from [gciql.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/gciql.py#L292):

- `lr=3e-4`
- `batch_size=1024`
- `tau=0.005`
- `expectile=0.9`
- `actor_loss=ddpgbc`
- `const_std=True`
- `value_p_curgoal=0.2`
- `value_p_trajgoal=0.5`
- `value_p_randomgoal=0.3`
- `value_geom_sample=True`
- `actor_p_curgoal=0.0`
- `actor_geom_sample=False`
- `gc_negative=True`
- `p_aug=0.0`

The local launcher [slurm/train_gciql.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_gciql.slurm) follows the OGBench command table below by default. It also exposes optional `VALUE_BACKBONE` and `CRITIC_BACKBONE` switches for local recurrent-value experiments; these are architecture knobs and are not part of the published OGBench non-model hyperparameter grid.

Effective antmaze locomotion hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | `antmaze-medium-navigate-v0`, `antmaze-large-navigate-v0`, `antmaze-teleport-navigate-v0` | `0.3` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| giant navigate | `antmaze-giant-navigate-v0` | `0.3` | `0.995` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| stitch | `antmaze-medium-stitch-v0`, `antmaze-large-stitch-v0`, `antmaze-teleport-stitch-v0` | `0.3` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| giant stitch | `antmaze-giant-stitch-v0` | `0.3` | `0.995` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| explore | `antmaze-medium-explore-v0`, `antmaze-large-explore-v0`, `antmaze-teleport-explore-v0` | `0.01` | `0.99` | `0.0` | `0.0` | `1.0` | `0.2` | `0.5` | `0.3` |

Effective pointmaze locomotion hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| navigate | `pointmaze-medium-navigate-v0`, `pointmaze-large-navigate-v0`, `pointmaze-teleport-navigate-v0` | `0.003` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| giant navigate | `pointmaze-giant-navigate-v0` | `0.003` | `0.995` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| stitch | `pointmaze-medium-stitch-v0`, `pointmaze-large-stitch-v0`, `pointmaze-teleport-stitch-v0` | `0.003` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| giant stitch | `pointmaze-giant-stitch-v0` | `0.003` | `0.995` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |

Effective humanoidmaze and antsoccer hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| humanoidmaze navigate | `humanoidmaze-medium-navigate-v0`, `humanoidmaze-large-navigate-v0`, `humanoidmaze-giant-navigate-v0` | `0.1` | `0.995` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| humanoidmaze stitch | `humanoidmaze-medium-stitch-v0`, `humanoidmaze-large-stitch-v0`, `humanoidmaze-giant-stitch-v0` | `0.1` | `0.995` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| antsoccer navigate | `antsoccer-arena-navigate-v0`, `antsoccer-medium-navigate-v0` | `0.1` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| antsoccer stitch | `antsoccer-arena-stitch-v0`, `antsoccer-medium-stitch-v0` | `0.1` | `0.99` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |

Effective state-manipulation hyperparameters:

| Task family | Tasks covered | alpha | discount | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cube` play | `cube-single-play-v0`, `cube-double-play-v0`, `cube-triple-play-v0`, `cube-quadruple-play-v0` | `1.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `cube` noisy | `cube-single-noisy-v0`, `cube-double-noisy-v0`, `cube-triple-noisy-v0`, `cube-quadruple-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `scene` play | `scene-play-v0` | `1.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `scene` noisy | `scene-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `puzzle` play | `puzzle-3x3-play-v0`, `puzzle-4x4-play-v0`, `puzzle-4x5-play-v0`, `puzzle-4x6-play-v0` | `1.0` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `puzzle` noisy | `puzzle-3x3-noisy-v0`, `puzzle-4x4-noisy-v0`, `puzzle-4x5-noisy-v0`, `puzzle-4x6-noisy-v0` | `0.03` | `0.99` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |

Visual-task overrides from OGBench:

- visual locomotion and manipulation tasks use `train_steps=500000`, `batch_size=256`, and `encoder=impala_small`
- visual antmaze keeps the corresponding state antmaze `alpha`, `discount`, and actor goal mix
- visual humanoidmaze keeps `alpha=0.1`, `discount=0.995`, and the corresponding navigate/stitch actor goal mix
- visual cube/scene/puzzle tasks use `p_aug=0.5`
- powderworld tasks use `train_steps=500000`, `batch_size=256`, `encoder=impala_small`, `discrete=True`, `actor_loss=awr`, `alpha=3.0`, and `eval_temperature=0.3`

## CGIVL / CGCIVL

Published shared non-model settings for `CGIVL` / `CGCIVL`:

- `lr=3e-4`
- `batch_size=1024`
- `tau=0.005`
- `expectile=0.7`
- `value_p_curgoal=0.2`
- `value_p_trajgoal=0.5`
- `value_p_randomgoal=0.3`
- `gd_p_curgoal=0.0`
- `gd_p_trajgoal=0.8`
- `gd_p_randomgoal=0.2`
- `value_geom_sample=True`
- `gd_geom_sample=True`
- `actor_p_curgoal=0.0`
- `gc_negative=True`
- `p_aug=0.0`
- `neg_eps=0.01`

Published task-family sampling rules from Appendix C.3:

- `navigate` tasks use actor-sampling `alpha=0.9`, which maps most naturally to `actor_p_trajgoal=0.9`, `actor_p_randomgoal=0.1`
- `stitch` tasks use actor-sampling `alpha=0.5`, which maps to `actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`
- quasimetric distillation uses `alpha_d=0.8`, matching `gd_p_trajgoal=0.8`, `gd_p_randomgoal=0.2`

Published task-specific training-step overrides:

| Task | train_steps | actor_p_trajgoal | actor_p_randomgoal | gd_p_trajgoal | gd_p_randomgoal | Source status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pointmaze-medium-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `pointmaze-large-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `pointmaze-giant-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `pointmaze-medium-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `pointmaze-large-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `pointmaze-giant-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `antmaze-medium-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `antmaze-large-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `antmaze-giant-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `antmaze-medium-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `antmaze-large-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `antmaze-giant-stitch-v0` | `2000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix explicit override |
| `humanoidmaze-medium-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `humanoidmaze-large-navigate-v0` | `1000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix rule |
| `humanoidmaze-giant-navigate-v0` | `3000000` | `0.9` | `0.1` | `0.8` | `0.2` | appendix explicit override |
| `humanoidmaze-medium-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `humanoidmaze-large-stitch-v0` | `1000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix rule |
| `humanoidmaze-giant-stitch-v0` | `3000000` | `0.5` | `0.5` | `0.8` | `0.2` | appendix explicit override |

State manipulation tasks reported in the paper do not get separate appendix sampling overrides. The safest reference is therefore the published shared settings plus the agent defaults:

| Task | train_steps | actor_p_trajgoal | actor_p_randomgoal | gd_p_trajgoal | gd_p_randomgoal | Source status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cube-single-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |
| `cube-double-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |
| `cube-triple-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |
| `scene-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |
| `puzzle-4x4-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |
| `puzzle-4x5-play-v0` | `1000000` | `1.0` | `0.0` | `0.8` | `0.2` | default-backed |

Important caveats:

- the paper appendix gives a task-family rule for actor sampling and explicit training-step overrides, but it does not publish a complete per-task command grid for `CGIVL`
- the authors' README provides one exact command for [pointmaze-large-stitch-v0](/Users/bruce/Recurrent-Offline-RL/third_party/CGCIVL/readme.md#L29): `discount=0.995`, `low_alpha=10.0`, `actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`, `goaldistance_latent_dim=512`, `neg_eps=0.01`, `init_goal_rep=False`
- that same README command also passes `--agent.alpha=0.003`, but `alpha` is not materially used by the current `cgivl.py` loss path in either the authors' code or the local port, so it is not treated here as a meaningful task hyperparameter
- unlike `HIQL`, there is no published `CGIVL` discount table in the repo; the only exact non-default discount visible locally is the README example above
- `low_alpha=3.0` and `high_alpha=3.0` are agent defaults, but no appendix table maps those temperatures across every task

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

Effective humanoidmaze locomotion hyperparameters:

| Task | low_alpha | high_alpha | discount | subgoal_steps | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal | value_p_curgoal | value_p_trajgoal | value_p_randomgoal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `humanoidmaze-medium-navigate-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `humanoidmaze-large-navigate-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `humanoidmaze-giant-navigate-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `1.0` | `0.0` | `0.2` | `0.5` | `0.3` |
| `humanoidmaze-medium-stitch-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| `humanoidmaze-large-stitch-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |
| `humanoidmaze-giant-stitch-v0` | `3.0` | `3.0` | `0.995` | `100` | `0.0` | `0.5` | `0.5` | `0.2` | `0.5` | `0.3` |

Additional OGBench pattern for humanoidmaze:

- all listed humanoidmaze HIQL tasks use `discount=0.995`
- all listed humanoidmaze HIQL tasks use `subgoal_steps=100`
- `navigate`: actor goal mix is `1.0 / 0.0`
- `stitch`: actor goal mix is `0.5 / 0.5`

## SAW

`SAW` does not appear in the OGBench third-party benchmark table in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh), so there is no repo-provided task-specific benchmark grid analogous to `CRL` or `HIQL`.

Published SAW task-specific settings come from Table 2 in [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf). The paper states that:

- common hyperparameters follow OGBench / HIQL settings from Park et al. (2024a)
- state, subgoal, and goal-sampling distributions are identical to `HIQL`
- Table 2 only overrides `expectile`, `AWR alpha`, `KLD beta`, and `subgoal steps`

Notation mapping from the paper to the local SAW code:

- paper `AWR alpha` maps to both `low_alpha` and `awr_alpha`
- paper `KLD beta` maps to `kl_alpha`

Effective published state-based SAW hyperparameters:

| Task | discount | expectile | low_alpha | awr_alpha | kl_alpha | subgoal_steps | value_p_curgoal | value_p_trajgoal | value_p_randomgoal | actor_p_curgoal | actor_p_trajgoal | actor_p_randomgoal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pointmaze-medium-navigate-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `pointmaze-large-navigate-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `pointmaze-giant-navigate-v0` | `0.995` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `antmaze-medium-navigate-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `antmaze-large-navigate-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `antmaze-giant-navigate-v0` | `0.995` | `0.7` | `3.0` | `3.0` | `3.0` | `25` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-medium-navigate-v0` | `0.995` | `0.7` | `3.0` | `3.0` | `3.0` | `100` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-large-navigate-v0` | `0.995` | `0.7` | `3.0` | `3.0` | `3.0` | `100` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `humanoidmaze-giant-navigate-v0` | `0.995` | `0.7` | `3.0` | `3.0` | `3.0` | `100` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `cube-single-play-v0` | `0.99` | `0.9` | `3.0` | `3.0` | `0.3` | `10` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `cube-double-play-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `1.0` | `10` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `cube-triple-play-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `1.0` | `10` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |
| `scene-play-v0` | `0.99` | `0.7` | `3.0` | `3.0` | `1.0` | `10` | `0.2` | `0.5` | `0.3` | `0.0` | `1.0` | `0.0` |

Patterns from the SAW paper:

- `antmaze-giant-navigate-v0` uses `discount=0.995`, `expectile=0.7`, `low_alpha=awr_alpha=3.0`, `kl_alpha=3.0`, `subgoal_steps=25`
- `humanoidmaze-giant-navigate-v0` uses the same temperatures as antmaze giant, but increases `subgoal_steps` to `100`
- `cube-double-play-v0` and `scene-play-v0` both use `expectile=0.7`, `low_alpha=awr_alpha=3.0`, `kl_alpha=1.0`, `subgoal_steps=10`
- `cube-single-play-v0` is the main manipulation outlier with `expectile=0.9` and `kl_alpha=0.3`

Current local non-model defaults used by [slurm/train_saw.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw.slurm) and [saw.py](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/agents/saw.py):

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

- if you run `SAW` through [slurm/train_saw.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw.slurm) without overrides, these local defaults are used regardless of task
- the local launcher does not automatically encode the paper's task-specific SAW overrides such as `discount=0.995` for giant mazes, `subgoal_steps=100` for humanoidmaze navigation, or `kl_alpha=1.0` / `subgoal_steps=10` for `cube-double-play-v0` and `scene-play-v0`
- if you want paper-matched `SAW` settings, those task-specific overrides need to be passed explicitly in local run commands
