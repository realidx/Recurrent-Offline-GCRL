# FiLM-Context Ablation Runs

This file collects runnable `sbatch` commands for the current FiLM-context ablation.

Launcher sources:

- [slurm/train_crl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_crl.slurm)
- [slurm/train_hiql_recur_value_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_hiql_recur_value_array.slurm)
- [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm)

The new launcher env vars exposed for this ablation are:

- `RECUR_USE_FILM_CONTEXT`, `RECUR_USE_STEP_INFO` for `CRL` / `QRL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO` for `HIQL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO`, `ACTOR_RECUR_USE_FILM_CONTEXT`, `ACTOR_RECUR_USE_STEP_INFO`, `LOW_ACTOR_RECUR_USE_FILM_CONTEXT`, `LOW_ACTOR_RECUR_USE_STEP_INFO` for `SAW`

## CRL on `antmaze-giant-navigate-v0`

These `CRL` commands are aligned to the OGBench third-party benchmark entry in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh#L140), which for `antmaze-giant-navigate-v0` specifies `--agent.alpha=0.1 --agent.discount=0.995` and does not override actor goal-mix defaults.

Baseline recurrent critic, hidden-only FiLM:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=Baseline,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=0,RECUR_USE_FILM_CONTEXT=0,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

Hybrid hidden + context FiLM critic:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=FiLMCtx,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=0,RECUR_USE_FILM_CONTEXT=1,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

Hybrid hidden + context + step FiLM critic:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=FiLMCtxStep,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=1,RECUR_USE_FILM_CONTEXT=1,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

## HIQL on `antmaze-large-stitch-v0`

These `HIQL` commands are aligned to the OGBench third-party benchmark entry in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh#L181), which for `antmaze-large-stitch-v0` uses `--agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.actor_p_trajgoal=0.5 --agent.actor_p_randomgoal=0.5`. The launcher now also exposes `VALUE_RECUR_LN_MODE`, so you can test `pre_loop` directly.

Five-way single-seed FiLM mode ladder for the recurrent value:

`hidden`, no step:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=Hidden_NoStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=0,VALUE_RECUR_FILM_MODE=hidden,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm
```

`context`, no step:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=Context_NoStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm
```

`hidden_context`, no step:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=HiddenContext_NoStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=hidden_context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm
```

`context`, with step:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=Context_WithStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm
```

`hidden_context`, with step:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=HiddenContext_WithStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=hidden_context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm
```

## SAW on `antmaze-giant-navigate-v0`

`SAW` is not part of the OGBench third-party benchmark table, so these are local ablation commands rather than benchmark-matching references.

Value-only baseline:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_ValueRecur_Baseline,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=16,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=0,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

Value-only hidden + context FiLM:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_ValueRecur_FiLMCtx,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=16,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

Optional all-recurrent SAW ablation:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_AllRecur_FiLMCtx,VALUE_BACKBONE=recur_tied,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,ACTOR_BACKBONE=recur_tied,ACTOR_RECUR_USE_STEP_INFO=0,ACTOR_RECUR_USE_FILM_CONTEXT=1,LOW_ACTOR_BACKBONE=recur_tied,LOW_ACTOR_RECUR_USE_STEP_INFO=0,LOW_ACTOR_RECUR_USE_FILM_CONTEXT=1,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

## Single-seed Smoke Test

Before launching 3 seeds, use a one-seed smoke test:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=Smoke_CRL_FiLMCtx,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=24,RECUR_USE_STEP_INFO=0,RECUR_USE_FILM_CONTEXT=1,ALPHA=0.1,DISCOUNT=0.995,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_crl_generic_array.slurm
```

## Suggested Comparison Order

1. `CRL`: baseline vs `RECUR_USE_FILM_CONTEXT=1`
2. `CRL`: `RECUR_USE_FILM_CONTEXT=1` vs `RECUR_USE_FILM_CONTEXT=1, RECUR_USE_STEP_INFO=1`
3. `HIQL`: baseline vs `VALUE_RECUR_USE_FILM_CONTEXT=1`


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=sd000_antmaze_large_stitch_hiql_recur4x6_mlpactor,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_Repro,EXP_NAME=sd000_antmaze_large_stitch_crl_recur2x4_preloop_sinusoidal,ALPHA=0.1,DISCOUNT=0.99,BATCH_SIZE=1024,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,CRITIC_BACKBONE=recur_tied,LAYERSCALE_INIT=1e-2,LR_DECAY_STEPS=1000000,LR_MIN=1e-5,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=24,RECUR_SINUSOIDAL=1,CRITIC_RECUR_LN_MODE=pre_loop,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
