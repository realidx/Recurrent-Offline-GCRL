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

## CRL SwiGLU on `antmaze-medium-stitch-v0`

This command uses the new `recur_tied` inner block variant with `CRITIC_RECUR_BLOCK_TYPE=swiglu`, while keeping the benchmark-aligned CRL task defaults for `antmaze-medium-stitch-v0`: `alpha=0.1`, `discount=0.99`, `actor_p_trajgoal=0.5`, and `actor_p_randomgoal=0.5`.

```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi_noFiLM,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noFiLM_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_SWIGLU_PRE_LN=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_Step_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_SWIGLU_PRE_LN=0,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noLRdecay,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,LR_DECAY_STEPS=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```

## CRL parameter-matched MLP comparisons on `antmaze-medium-stitch-v0`

These compare the `hidden_context` SwiGLU recurrent critic against two MLP critics matched by `params/critic_count` for the same `antmaze-medium-stitch-v0` setup. The target recurrent critic has `params/critic_count=7,497,744`. The closest matches found with [find_mlp_param_match.py](/Users/bruce/Recurrent-Offline-RL/scripts/find_mlp_param_match.py) are:

- depth-matched MLP: `VALUE_HIDDEN_DIMS=512x512x512x512x512x512x512` with `params/critic_count=7,452,672`
- width-matched MLP: `VALUE_HIDDEN_DIMS=832x832x832` with `params/critic_count=7,383,552`

```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_depthmatch_7x512,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_widthmatch_3x832,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=832x832x832,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_baseline_noln,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512,LAYER_NORM=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```


## SAW paper-aligned references

From [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), Supplementary Section G / Table 2:
- `antmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=25`,`discount=0.995`
- `humanoidmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=100`,`discount=0.995`

```bash
sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd000_AGN_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=humanoidmaze-giant-navigate-v0,RUN_GROUP=SAW_HumanoidGiantNav_SwiGLU,EXP_NAME=sd000_HGN_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=100,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=cube-double-play-v0,RUN_GROUP=SAW_CubeDoublePlay_SwiGLU,EXP_NAME=sd000_CDP_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm


sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=SAW_ScenePlay_SwiGLU,EXP_NAME=sd000_SP_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

```

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ActorRecur,EXP_NAME=sd000_ALS_actor_recur,ACTOR_BACKBONE=recur_tied,ACTOR_RECUR_ITERS=4,ACTOR_RECUR_MAX_ITERS=4,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ActorRecur,EXP_NAME=sd000_ALS_actor_critic_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ACTOR_BACKBONE=recur_tied,ACTOR_RECUR_ITERS=4,ACTOR_RECUR_MAX_ITERS=4,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_crl.slurm


