# FiLM-Context Ablation Runs

This file collects runnable `sbatch` commands for the current FiLM-context ablation.

Launcher sources:

- [slurm/train_crl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_crl.slurm)
- [slurm/train_hiql_recur_value_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_hiql_recur_value_array.slurm)
- [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm)

The new launcher env vars exposed for this ablation are:

- `CRITIC_RECUR_USE_FILM`, `CRITIC_RECUR_FILM_MODE`, `RECUR_USE_STEP_INFO` for `CRL` / `QRL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO` for `HIQL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO`, `ACTOR_RECUR_USE_FILM_CONTEXT`, `ACTOR_RECUR_USE_STEP_INFO`, `LOW_ACTOR_RECUR_USE_FILM_CONTEXT`, `LOW_ACTOR_RECUR_USE_STEP_INFO` for `SAW`

## CRL on `antmaze-giant-navigate-v0`

These `CRL` commands are aligned to the OGBench third-party benchmark entry in [third_party/ogbench/impls/hyperparameters.sh](/Users/bruce/Recurrent-Offline-RL/third_party/ogbench/impls/hyperparameters.sh#L140), which for `antmaze-giant-navigate-v0` specifies `--agent.alpha=0.1 --agent.discount=0.995` and does not override actor goal-mix defaults.

Baseline recurrent critic, hidden-only FiLM:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=Baseline,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

Hybrid hidden + context FiLM critic:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=FiLMCtx,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden_context,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

Hybrid hidden + context + step FiLM critic:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate_FiLM_Ablation,EXP_NAME=FiLMCtxStep,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=8,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden_context,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
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
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=HiddenContext_NoStep,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=3,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_LN_MODE=pre_loop_per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_FILM_MODE=hidden_context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm

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
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_ValueRecur_Baseline,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=16,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=0,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

Value-only hidden + context FiLM:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_ValueRecur_FiLMCtx,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=16,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

Optional all-recurrent SAW ablation:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=P4_SAW_AntGiantNavigate_AllRecur_FiLMCtx,VALUE_BACKBONE=recur_tied,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,ACTOR_BACKBONE=recur_tied,ACTOR_RECUR_USE_STEP_INFO=0,ACTOR_RECUR_USE_FILM_CONTEXT=1,LOW_ACTOR_BACKBONE=recur_tied,LOW_ACTOR_RECUR_USE_STEP_INFO=0,LOW_ACTOR_RECUR_USE_FILM_CONTEXT=1,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw_array.slurm
```

## CGIVL on `antmaze-large-stitch-v0`

These commands use the local [train_cgivl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_cgivl.slurm) launcher plus the published CGIVL stitch-task defaults summarized in [ALGORITHM_HYPERPARAMETERS_REFERENCE.md](/Users/bruce/Recurrent-Offline-RL/ALGORITHM_HYPERPARAMETERS_REFERENCE.md): `actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`, `discount=0.99`, and `neg_eps=0.01`.

Default task hyperparameters, seeds `0 1 2`:

```bash
sbatch --array=0-2 --export="ALL,SEEDS=0 1 2,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CGIVL_AntLargeStitch_Default,DISCOUNT=0.99,NEG_EPS=0.01,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_cgivl.slurm
```

3x4 tied value network, `pre_loop_per_layer` LN, context-only FiLM, seed `0`:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CGIVL_AntLargeStitch,EXP_NAME=3x4_HiddenContextFiLM_sd000,DISCOUNT=0.99,NEG_EPS=0.01,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=3,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_LN_MODE=pre_loop_per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM=1,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=hidden_context,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_cgivl.slurm
```

Baseline value network, seed `0`:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CGIVL_AntLargeStitch,EXP_NAME=BaselineMLP_sd000,DISCOUNT=0.99,NEG_EPS=0.01,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,VALUE_BACKBONE=mlp,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_cgivl.slurm
```

## Single-seed Smoke Test

Before launching 3 seeds, use a one-seed smoke test:

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=Smoke_CRL_FiLMCtx,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden_context,ALPHA=0.1,DISCOUNT=0.995,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_crl.slurm
```

## CRL SwiGLU on `antmaze-medium-stitch-v0`

This command uses the new `recur_tied` inner block variant with `CRITIC_RECUR_BLOCK_TYPE=swiglu`, while keeping the benchmark-aligned CRL task defaults for `antmaze-medium-stitch-v0`: `alpha=0.1`, `discount=0.99`, `actor_p_trajgoal=0.5`, and `actor_p_randomgoal=0.5`.

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi_hidden,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi_context,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=context,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi_hidden_context,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=0,CRITIC_RECUR_USE_FILM=1,CRITIC_RECUR_FILM_MODE=hidden_context,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

## Suggested Comparison Order

1. `CRL`: hidden-only FiLM vs `CRITIC_RECUR_FILM_MODE=hidden_context`
2. `CRL`: `CRITIC_RECUR_FILM_MODE=hidden_context` without step info vs with `RECUR_USE_STEP_INFO=1`
3. `HIQL`: baseline vs `VALUE_RECUR_USE_FILM_CONTEXT=1`


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=sd000_antmaze_large_stitch_hiql_recur4x6_mlpactor,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_Repro,EXP_NAME=sd000_antmaze_large_stitch_crl_recur2x4_preloop_sinusoidal,ALPHA=0.1,DISCOUNT=0.99,BATCH_SIZE=1024,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,CRITIC_BACKBONE=recur_tied,LAYERSCALE_INIT=1e-2,LR_DECAY_STEPS=1000000,LR_MIN=1e-5,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=24,RECUR_SINUSOIDAL=1,CRITIC_RECUR_LN_MODE=pre_loop,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_FiLM_Ablation,EXP_NAME=Context_NoStep_perlayerLN,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=context,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_hiql.slurm


## SAW paper-aligned references

From [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), Supplementary Section G / Table 2:
- `antmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=25`
- `humanoidmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=100`

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNavigate,EXP_NAME=Baseline_Repro,VALUE_BACKBONE=mlp,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_saw.slurm




sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNavigate,EXP_NAME=sd0_SAW_MLP4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,VALUE_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512,EVAL_ON_CPU=1,WANDB_MODE=online" slurm/train_saw.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=humanoidmaze-giant-navigate-v0,RUN_GROUP=SAW_HumanoidMazeGiantNavigate,EXP_NAME=sd0_3x4_HumanoidMazeGiantNavigate,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=100,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=3,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_LN_MODE=pre_loop_per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=context,VALUE_RECUR_USE_ACT=0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_saw.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CGIVL_AntLargeStitch,EXP_NAME=ValueRecur3x4_HiddenContextStep_sd000,DISCOUNT=0.99,NEG_EPS=0.01,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=3,VALUE_RECUR_NUM_DENSE_LAYERS=4,VALUE_RECUR_MAX_ITERS=3,VALUE_RECUR_LN_MODE=pre_loop_per_layer,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM=1,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=hidden_context,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_cgivl.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNavigate,EXP_NAME=sd000_SAW_3x4_HiddenContext,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=3,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_LN_MODE=pre_loop_per_layer,VALUE_RECUR_USE_STEP_INFO=0,VALUE_RECUR_USE_FILM_CONTEXT=1,VALUE_RECUR_FILM_MODE=hidden_context,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_saw.slurm


sbatch --job-name=hiql_geom_eval --partition=gpu-long --gpus=h100-47 --time=05:00:00 --cpus-per-task=8 --mem=24G --output=logs/phase_4/slurm-%x-%j.out --error=logs/phase_4/slurm-%x-%j.err --wrap='cd "$SLURM_SUBMIT_DIR/third_party/ogbench/impls" && export OGBENCH_DATASET_DIR="$SLURM_SUBMIT_DIR/.ogbench_data" && conda run -n recurrent python analyze_hiql_value_geometry.py --run_dir "$SLURM_SUBMIT_DIR/exp/OGBench/HIQL_AntLargeStitch_4x6/sd000_antmaze_large_stitch_hiql_recur4x6_mlpactor" --epochs 800000 --task_ids 1,2,3,4,5 --policy_rollout_steps 25 --reset_eval_episodes 50 --train_adv_batches 8 --train_adv_batch_size 1024 --output_dir "$SLURM_SUBMIT_DIR/exp/value_geometry_hiql_antlarge_4x6_sd000_ckpt800000"'


sbatch --job-name=hiql_geom_eval --partition=gpu-long --gpus=h100-47 --time=05:00:00 --cpus-per-task=8 --mem=24G --output=logs/phase_4/slurm-%x-%j.out --error=logs/phase_4/slurm-%x-%j.err --wrap='cd "$SLURM_SUBMIT_DIR/third_party/ogbench/impls" && export OGBENCH_DATASET_DIR="$SLURM_SUBMIT_DIR/.ogbench_data" && conda run -n recurrent python analyze_hiql_value_geometry.py --run_dir "$SLURM_SUBMIT_DIR/exp/OGBench/HIQL_AntLargeStitch_Baseline/sd000_antmaze_large_stitch_hiql_mlp_baseline" --epochs 1000000 --task_ids 1,2,3,4,5 --policy_rollout_steps 25 --reset_eval_episodes 50 --train_adv_batches 8 --train_adv_batch_size 1024 --output_dir "$SLURM_SUBMIT_DIR/exp/value_geometry_hiql_baseline_sd000_ckpt1000000"'
