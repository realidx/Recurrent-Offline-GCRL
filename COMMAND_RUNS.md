# FiLM-Context Ablation Runs

This file collects runnable `sbatch` commands for the current FiLM-context ablation.

Launcher sources:

- [slurm/train_crl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_crl.slurm)
- [slurm/train_qrl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_qrl.slurm)
- [slurm/train_hiql_recur_value_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_hiql_recur_value_array.slurm)
- [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm)

The new launcher env vars exposed for this ablation are:

- `CRITIC_RECUR_USE_FILM`, `CRITIC_RECUR_FILM_MODE`, `RECUR_USE_STEP_INFO` for `CRL` / `QRL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO` for `HIQL`
- `VALUE_RECUR_USE_FILM_CONTEXT`, `VALUE_RECUR_USE_STEP_INFO`, `ACTOR_RECUR_USE_FILM_CONTEXT`, `ACTOR_RECUR_USE_STEP_INFO`, `LOW_ACTOR_RECUR_USE_FILM_CONTEXT`, `LOW_ACTOR_RECUR_USE_STEP_INFO` for `SAW`

## QRL Full Training

`QRL` now has its own launcher in [slurm/train_qrl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_qrl.slurm). It exposes the same recurrent quasimetric-value controls as the old shared path, including the optional evaluation-time iteration override `CRITIC_EVAL_NUM_ITERS`.

```bash
# QRL baseline on antmaze-medium-stitch-v0 (OGBench task defaults).
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=QRL_AntMediumStitch,EXP_NAME=sd000_ALS_QRL_mlp,CRITIC_BACKBONE=mlp,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm

# QRL recurrent critic/value on the same task, mirroring the CRL recur_tied setup.
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=QRL_AntMediumStitch,EXP_NAME=sd000_AMS_QRL,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm

sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=QRL_AntLargetitch,EXP_NAME=sd000_ALS_QRL,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm


```

## CRL SwiGLU on `antmaze-medium-stitch-v0`

This command uses the new `recur_tied` inner block variant with `CRITIC_RECUR_BLOCK_TYPE=swiglu`, while keeping the benchmark-aligned CRL task defaults for `antmaze-medium-stitch-v0`: `alpha=0.1`, `discount=0.99`, `actor_p_trajgoal=0.5`, and `actor_p_randomgoal=0.5`.

```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi_noFiLM,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noFiLM_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_SWIGLU_PRE_LN=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_Step_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_SWIGLU_PRE_LN=0,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noLRdecay,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,LR_DECAY_STEPS=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-large-stitch-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch,EXP_NAME=sd000_ALS_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_USE_INPUT_INJECTION=1,CRITIC_RECUR_USE_SOFT_MIXTURE=0,RECUR_USE_ACT=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-giant-navigate-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate,EXP_NAME=sd000_AGN_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_USE_INPUT_INJECTION=1,CRITIC_RECUR_USE_SOFT_MIXTURE=0,RECUR_USE_ACT=0,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-medium-explore-v0
# OGBench uses different non-model hyperparameters here than antmaze navigate/stitch:
# alpha=0.003, discount=0.99, actor goal mix = (cur=0.0, traj=0.0, random=1.0).
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-explore-v0,RUN_GROUP=CRL_AntMediumExplore,EXP_NAME=sd000_AME_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_USE_INPUT_INJECTION=1,CRITIC_RECUR_USE_SOFT_MIXTURE=0,RECUR_USE_ACT=0,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.0,ACTOR_P_RANDOMGOAL=1.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch,EXP_NAME=sd000_ALS_baseline,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate,EXP_NAME=sd000_AGN_baseline,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0 pbs/train_crl.pbs

```

## CRL actor-scaling interaction grid on `antmaze-large-stitch-v0` (`seed=0`)

These four `CRL` runs are the minimal interaction test for the hypothesis:
does actor scaling become useful only when the critic is iterative?

The grid is:
- baseline actor + baseline critic
- baseline actor + iterative critic
- scaled actor + baseline critic
- scaled actor + iterative critic

All four runs use the standard `antmaze-large-stitch-v0` CRL task defaults:
- `alpha=0.1`
- `discount=0.99`
- `actor_p_curgoal=0.0`
- `actor_p_trajgoal=0.5`
- `actor_p_randomgoal=0.5`

The scaled actor here uses the new `residual_mlp` backbone, while the iterative critic uses `recur_tied` with the SwiGLU cell.

```bash
# 1. Baseline: MLP actor + MLP critic
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_ActorScaleInteraction,EXP_NAME=sd000_ALS_actorint_mlp_mlp,CRITIC_BACKBONE=mlp,ACTOR_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_crl.slurm

# 2. Critic-only scaling: MLP actor + iterative critic
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_ActorScaleInteraction,EXP_NAME=sd000_ALS_actorint_mlp_recurcritic,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ACTOR_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_crl.slurm

# 3. Actor-only scaling: residual actor + MLP critic
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_ActorScaleInteraction,EXP_NAME=sd000_ALS_actor8_resactor_mlpcritic,CRITIC_BACKBONE=mlp,ACTOR_BACKBONE=residual_mlp,ACTOR_NUM_LAYERS=8,ACTOR_LAYER_WIDTH=512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online pbs/train_crl.pbs

# 4. Full interaction: residual actor + iterative critic
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_ActorScaleInteraction,EXP_NAME=sd000_ALS_actor8_resactor_recurcritic,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ACTOR_BACKBONE=residual_mlp,ACTOR_NUM_LAYERS=8,ACTOR_LAYER_WIDTH=512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online pbs/train_crl.pbs

```

## CRL InfoNCE comparison on `antmaze-large-stitch-v0` (`seed=0`)

These runs are the cleanest first comparison against your existing `BCE` CRL results on ALS:
- keep the same task, seed, batch size, actor loss, LR, LR decay, and critic architecture
- change only `CONTRASTIVE_LOSS_TYPE` from `bce` to `infonce`
- start with `INFONCE_TEMPERATURE=0.1`

The most important matched comparison is the recurrent critic, since that is the core mechanism story. If `temp=0.1` is promising but clearly under-tuned, the next sweep should be temperature only, while keeping `BATCH_SIZE=1024` fixed.

```bash
# 1. Matched ALS baseline: MLP critic + InfoNCE
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_InfoNCE,EXP_NAME=sd000_ALS_mlp_infonce_t01,CRITIC_BACKBONE=mlp,CONTRASTIVE_LOSS_TYPE=infonce,INFONCE_TEMPERATURE=0.1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,BATCH_SIZE=1024,WANDB_MODE=online" slurm/train_crl.slurm

# 2. Matched ALS recurrent critic: SwiGLU recur_tied + InfoNCE
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_InfoNCE,EXP_NAME=sd000_ALS_recur_infonce_t01,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CONTRASTIVE_LOSS_TYPE=infonce,INFONCE_TEMPERATURE=0.1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,BATCH_SIZE=1024,WANDB_MODE=online" slurm/train_crl.slurm

# 3. Recurrent critic temperature sweep: lower temperature
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_InfoNCE,EXP_NAME=sd000_ALS_recur_infonce_t003,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CONTRASTIVE_LOSS_TYPE=infonce,INFONCE_TEMPERATURE=0.03,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,BATCH_SIZE=1024,WANDB_MODE=online" slurm/train_crl.slurm

# 4. Recurrent critic temperature sweep: higher temperature
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_InfoNCE,EXP_NAME=sd000_ALS_recur_infonce_t03,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CONTRASTIVE_LOSS_TYPE=infonce,INFONCE_TEMPERATURE=0.3,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,BATCH_SIZE=1024,WANDB_MODE=online" slurm/train_crl.slurm
```

## CRL parameter-matched MLP comparisons on `antmaze-medium-stitch-v0`

These compare the `hidden_context` SwiGLU recurrent critic against two MLP critics matched by `params/critic_count` for the same `antmaze-medium-stitch-v0` setup. The target recurrent critic has `params/critic_count=7,497,744`. The closest matches found with [find_mlp_param_match.py](/Users/bruce/Recurrent-Offline-RL/scripts/find_mlp_param_match.py) are:


```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_depthmatch_5x512,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=3,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd003_AMS_MLP_depthmatch_5x512,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_widthmatch_3x688,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=688x688x688,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=3,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd003_AMS_MLP_widthmatch_3x688,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=688x688x688,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

```

## CRL recurrent critic on `scene-play-v0`

This is the state-based `scene-play-v0` `CRL` run with the recurrent SwiGLU critic. It follows the OGBench CRL task defaults for scene-play:
- `alpha=3.0`
- `discount=0.99`
- `actor_p_curgoal=0.0`
- `actor_p_trajgoal=1.0`
- `actor_p_randomgoal=0.0`

```bash

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay,EXP_NAME=sd000_SP_recur_actormlp,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay,EXP_NAME=sd000_SP_recur_resactor8,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ACTOR_BACKBONE=residual_mlp,ACTOR_NUM_LAYERS=8,ACTOR_LAYER_WIDTH=512,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay,EXP_NAME=sd000_SP_recur_resactor16,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ACTOR_BACKBONE=residual_mlp,ACTOR_NUM_LAYERS=16,ACTOR_LAYER_WIDTH=512,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

```


## HIQL confidence-gate comparison on `antmaze-large-stitch-v0` (`seed=4`)

These four runs compare:
- MLP HIQL baseline
- MLP HIQL + confidence-scaled AWR
- recurrent-value HIQL
- recurrent-value HIQL + confidence-scaled AWR

They use the current [slurm/train_hiql.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_hiql.slurm) launcher, `seed=4`, and the standard `antmaze-large-stitch-v0` HIQL task defaults: `discount=0.99`, `low_alpha=3.0`, `high_alpha=3.0`, `subgoal_steps=25`, `actor_p_trajgoal=0.5`, `actor_p_randomgoal=0.5`.

```bash
# 1. HIQL baseline (MLP value)
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_mlp,VALUE_BACKBONE=mlp,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 2. HIQL baseline + confidence-scaled AWR (MLP value)
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_mlp_conf,VALUE_BACKBONE=mlp,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=1,ACTOR_CONFIDENCE_NUM_NEGATIVES=4,ACTOR_CONFIDENCE_TAU=0.0,ACTOR_CONFIDENCE_TEMPERATURE=1.0,ACTOR_CONFIDENCE_MIN=0.1,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 3. HIQL recurrent value
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_recur,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 4. HIQL recurrent value + confidence-scaled AWR
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_recur_conf,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=1,ACTOR_CONFIDENCE_NUM_NEGATIVES=4,ACTOR_CONFIDENCE_TAU=0.0,ACTOR_CONFIDENCE_TEMPERATURE=1.0,ACTOR_CONFIDENCE_MIN=0.1,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm
```


## SAW paper-aligned references

From [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), Supplementary Section G / Table 2:
- `antmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=25`,`discount=0.995`
- `humanoidmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=100`,`discount=0.995`

```bash
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=3,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd003_AGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=3,ENV_NAME=humanoidmaze-giant-navigate-v0,RUN_GROUP=SAW_HumanoidGiantNav_SwiGLU,EXP_NAME=sd003_HGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=100,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=3,ENV_NAME=cube-double-play-v0,RUN_GROUP=SAW_CubeDoublePlay_SwiGLU,EXP_NAME=sd003_CDP_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=2,ENV_NAME=scene-play-v0,RUN_GROUP=SAW_ScenePlay_SwiGLU,EXP_NAME=sd002_SP_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

```


## SAW input/output Ablation
```bash
# 1. Control: recurrent SwiGLU, no input group, no output group
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_ctrl,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

# 2. Input Group Only: separate input anchor/state + per-step learned lambda
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_inj,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=1,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

# 3. Output Group Only: shared per-step head + soft mixture over steps
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_mix,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=1,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

# 4. Both Groups: input anchor injection + soft mixture
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_inj_mix,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=1,VALUE_RECUR_USE_SOFT_MIXTURE=1,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm
```

### NSCC PBS equivalents

On NSCC ASPIRE2A with the repo defaults in `pbs/` and `slurm/`, the same four runs become:

```bash
# 1. Control
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_ctrl,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 pbs/train_saw.pbs

# 2. Input group only
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_inj,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=1,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 pbs/train_saw.pbs

# 3. Output group only
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_mix,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=1,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 pbs/train_saw.pbs

# 4. Both groups
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_NewDesignAbl,EXP_NAME=sd000_AGN_SAW_Swi_inj_mix,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_INPUT_INJECTION=1,VALUE_RECUR_USE_SOFT_MIXTURE=1,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 pbs/train_saw.pbs


```

```bash
# 1. Recurrent value only
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=GCIQL_AntMediumStitch,EXP_NAME=sd000_AMS_GCIQL_v_recur,VALUE_BACKBONE=recur_tied,CRITIC_BACKBONE=mlp,KTRAIN=4,RECUR_MAX_ITERS=4,RECUR_BLOCK_TYPE=swiglu,RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,ALPHA=0.3,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_gciql.slurm

# 2. Recurrent critic only
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=GCIQL_AntMediumStitch,EXP_NAME=sd000_AMS_GCIQL_q_recur,VALUE_BACKBONE=mlp,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_MAX_ITERS=4,RECUR_BLOCK_TYPE=swiglu,RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,ALPHA=0.3,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_gciql.slurm

# 3. Recurrent value + recurrent critic
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=GCIQL_AntMediumStitch,EXP_NAME=sd000_AMS_GCIQL_vq_recur,VALUE_BACKBONE=recur_tied,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_MAX_ITERS=4,RECUR_BLOCK_TYPE=swiglu,RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,ALPHA=0.3,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_gciql.slurm

sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=GCIQL_AntMediumStitch,EXP_NAME=sd000_AMS_GCIQL_baseline,VALUE_BACKBONE=mlp,CRITIC_BACKBONE=mlp,DISCOUNT=0.99,ALPHA=0.3,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_gciql.slurm
```

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-medium-explore-v0,RUN_GROUP=CRL_AntMediumExplore,EXP_NAME=sd000_AME_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_ACT=0,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.0,ACTOR_P_RANDOMGOAL=1.0,EVAL_ON_CPU=0 pbs/train_crl.pbs
