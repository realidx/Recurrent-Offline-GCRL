# Recurrent Value Runs

This file collects runnable `sbatch` commands for the current recurrent-value experiments.

Launcher sources:

- [slurm/train_crl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_crl.slurm)
- [slurm/train_qrl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_qrl.slurm)
- [slurm/train_hiql_recur_value_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_hiql_recur_value_array.slurm)
- [slurm/train_saw_array.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_saw_array.slurm)

Current launcher env vars for recurrent runs include:

- `CRITIC_BACKBONE`, `KTRAIN`, `RECUR_NUM_DENSE_LAYERS`, `CRITIC_RECUR_BLOCK_TYPE`, `RECUR_MAX_ITERS`, `RECUR_USE_STEP_INFO` for `CRL` / `QRL`
- `VALUE_BACKBONE`, `VALUE_RECUR_ITERS`, `VALUE_RECUR_NUM_DENSE_LAYERS`, `VALUE_RECUR_BLOCK_TYPE`, `VALUE_RECUR_MAX_ITERS`, `VALUE_RECUR_USE_STEP_INFO` for `HIQL` / `SAW`

## QRL Full Training

`QRL` now has its own launcher in [slurm/train_qrl.slurm](/Users/bruce/Recurrent-Offline-RL/slurm/train_qrl.slurm). It exposes the same recurrent quasimetric-value controls as the old shared path, including the optional evaluation-time iteration override `CRITIC_EVAL_NUM_ITERS`.

```bash
# QRL baseline on antmaze-medium-stitch-v0 (OGBench task defaults).
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=QRL_AntMediumStitch,EXP_NAME=sd000_ALS_QRL_mlp,CRITIC_BACKBONE=mlp,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm

# QRL recurrent critic/value on the same task, mirroring the CRL recur_tied setup.
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=QRL_AntMediumStitch,EXP_NAME=sd000_AMS_QRL,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm

sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=QRL_AntLargetitch,EXP_NAME=sd000_ALS_QRL,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_qrl.slurm


```

## HIQL Value-Refinement Mechanism Probe on `antmaze-large-stitch-v0`

HIQL reproduction note: use `LR_DECAY_STEPS=0` for all HIQL baseline and recurrent comparisons. Earlier HIQL runs that omitted this env var used the old launcher default cosine decay over 1M steps; treat those as exploratory and rerun before using them in tables.

This run uses the new HIQL-specific diagnostics:

- `evaluation/probe/hiql_actor/low_actor_value/adv_*`
- `evaluation/probe/hiql_actor/low_actor_awr_weight_*`
- `evaluation/probe/hiql_actor/high_actor_value/adv_*`
- `evaluation/probe/hiql_actor/high_actor_awr_weight_*`
- `evaluation/refine/value_step_{k}_hiql_actor/...` for per-refinement-step low/high actor-facing value signals

```bash
# SLURM seed 4, MLP HIQL value baseline. This logs the fixed-probe HIQL actor-facing metrics.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_mlp_constlr,VALUE_BACKBONE=mlp,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# SLURM seed 4, recurrent HIQL value, explicit probe dumps at start/mid/final.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_recur2_constlr,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=2,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=2,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,REFINE_DUMP_STEPS=1:500000:1000000,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm



# NSCC PBS seed 4 MLP baseline equivalent.
qsub -J 0 -P personal -q normal -v SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_mlp_mech_constlr,VALUE_BACKBONE=mlp,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,EVAL_ON_CPU=0 pbs/train_hiql.pbs

# NSCC PBS seed 4 equivalent. The -J 0 override is needed because this is a single-seed SEEDS list.
qsub -J 0 -P personal -q normal -v SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_recur_mech_constlr,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,REFINE_DUMP_STEPS=1:500000:1000000,EVAL_ON_CPU=0 pbs/train_hiql.pbs
```

## CRL SwiGLU on `antmaze-medium-stitch-v0`

This command uses the new `recur_tied` inner block variant with `CRITIC_RECUR_BLOCK_TYPE=swiglu`, while keeping the benchmark-aligned CRL task defaults for `antmaze-medium-stitch-v0`: `alpha=0.1`, `discount=0.99`, `actor_p_trajgoal=0.5`, and `actor_p_randomgoal=0.5`.

```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_SwiGLU,EXP_NAME=sd000_AMS_Swi,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_SWIGLU_PRE_LN=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_Step_noLN,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,CRITIC_RECUR_SWIGLU_PRE_LN=0,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch_SwiGLU,EXP_NAME=sd000_ALS_Swi_noLRdecay,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,LR_DECAY_STEPS=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-large-stitch-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch,EXP_NAME=sd000_ALS_Swi_step,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-giant-navigate-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate,EXP_NAME=sd000_AGN_Swi_step,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-medium-explore-v0
# OGBench uses different non-model hyperparameters here than antmaze navigate/stitch:
# alpha=0.003, discount=0.99, actor goal mix = (cur=0.0, traj=0.0, random=1.0).
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-explore-v0,RUN_GROUP=CRL_AntMediumExplore,EXP_NAME=sd000_AME_Swi_step,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=0.003,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.0,ACTOR_P_RANDOMGOAL=1.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch,EXP_NAME=sd000_ALS_baseline,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate,EXP_NAME=sd000_AGN_baseline,CRITIC_BACKBONE=mlp,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0 pbs/train_crl.pbs

```


## Stacked-SwiGLU recurrent backbone

This is the new recurrent design where each iteration contains a stacked residual SwiGLU cell, and the outer recurrent update applies a learned per-step acceptance `alpha_k`. The backbone name is `recur_stacked_swiglu`.

`RECUR_NUM_DENSE_LAYERS` / `VALUE_RECUR_NUM_DENSE_LAYERS` is the inner per-iteration stack depth `m`, not the old `recur_tied` meaning.

```bash
# CRL on antmaze-giant-navigate-v0 with the new stacked-SwiGLU recurrent critic.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=AGN_StackedSwiGLU,EXP_NAME=sd000_AGN_crl_stacked_swi_m2k4,CRITIC_BACKBONE=recur_stacked_swiglu,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,BATCH_SIZE=1024,WANDB_MODE=online" slurm/train_crl.slurm

# HIQL on antmaze-giant-navigate-v0 with the new stacked-SwiGLU recurrent value backbone.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=AGN_StackedSwiGLU,EXP_NAME=sd000_AGN_hiql_stacked_swi_m2k4_constlr,VALUE_BACKBONE=recur_stacked_swiglu,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.995,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,WANDB_MODE=online" slurm/train_hiql.slurm

# HIQL old AGN
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=HIQL_AGN_RecurTied,EXP_NAME=sd000_AGN_hiql_recur_tied_k4_constlr,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.995,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# HIQL ALS seed 0, old recur_tied backbone, constant LR
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_ALS_ConstLR,EXP_NAME=sd000_ALS_hiql_recur_tied_k4_constlr,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# HIQL ALS seed 0, new stacked SwiGLU backbone, constant LR
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_ALS_ConstLR,EXP_NAME=sd000_ALS_hiql_stacked_swi_m2k4_constlr,VALUE_BACKBONE=recur_stacked_swiglu,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_STEP_INFO_INNER_MODE=all,VALUE_RECUR_USE_INNER_RESIDUAL=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm


# HIQL old scene
sbatch --array=0 --gpus=a100-40 --export="ALL,SEEDS=1,ENV_NAME=scene-play-v0,RUN_GROUP=HIQL_ScenePlay_RecurTied,EXP_NAME=sd001_SP_hiql_recur_tied_k4_constlr,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=10,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# HIQL new scene 
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=1,ENV_NAME=scene-play-v0,RUN_GROUP=HIQL_ScenePlay_StackedSwiGLU,EXP_NAME=sd001_SP_hiql_stacked_swi_m2k4_constlr,VALUE_BACKBONE=recur_stacked_swiglu,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_STEP_INFO_INNER_MODE=all,VALUE_RECUR_USE_INNER_RESIDUAL=1,LR_DECAY_STEPS=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=10,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# CRL old scene
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay_RecurTied,EXP_NAME=sd004_SP_crl_recur_tied_k4,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL new scene
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=1,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay_StackedSwiGLU,EXP_NAME=sd001_SP_crl_stacked_swi_m2k4,CRITIC_BACKBONE=recur_stacked_swiglu,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_LAYERSCALE=1,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm


# SAW on antmaze-giant-navigate-v0 with the new stacked-SwiGLU recurrent value backbone.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=AGN_StackedSwiGLU,EXP_NAME=sd004_AGN_saw_stacked_swi_m2k4,VALUE_BACKBONE=recur_stacked_swiglu,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm


sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=5,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_ALS_Repro,EXP_NAME=ds005_hiql_ALS_repro,VALUE_BACKBONE=mlp,LR_DECAY_STEPS=0,EVAL_ON_CPU=0,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,WANDB_MODE=online" slurm/train_hiql.slurm

# Swiglu ablation
sbatch --array=0-1 --gpus=h100-47 --export="ALL,SEEDS='0 1',ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AMS_BlockAblation_Dense,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=dense,RECUR_MAX_ITERS=4,CRITIC_RECUR_LN_MODE=per_layer_final,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_LAYERSCALE=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0-1 --gpus=h100-47 --export="ALL,SEEDS='0 1',ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_ALS_BlockAblation_Dense,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=dense,RECUR_MAX_ITERS=4,CRITIC_RECUR_LN_MODE=per_layer_final,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_LAYERSCALE=1,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

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

qsub -P personal -q normal -v SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay,EXP_NAME=sd000_SP_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs


qsub -P personal -q normal -v SEEDS=0,ENV_NAME=cube-single-play-v0,RUN_GROUP=CRL_CubeSinglePlay,EXP_NAME=sd000_CSP_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=1,ENV_NAME=cube-single-play-v0,RUN_GROUP=CRL_CubeSinglePlay,EXP_NAME=sd001_CSP_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=2,ENV_NAME=cube-single-play-v0,RUN_GROUP=CRL_CubeSinglePlay,EXP_NAME=sd002_CSP_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

qsub -P personal -q normal -v SEEDS=3,ENV_NAME=cube-single-play-v0,RUN_GROUP=CRL_CubeSinglePlay,EXP_NAME=sd003_CSP_recur,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online pbs/train_crl.pbs

```

## CRL stacked-SwiGLU critic on `scene-play-v0` and `cube-single-play-v0`

These use the new `recur_stacked_swiglu` critic backbone with the same OGBench CRL task defaults as the recurrent commands above:
- `alpha=3.0`
- `discount=0.99`
- `actor_p_curgoal=0.0`
- `actor_p_trajgoal=1.0`
- `actor_p_randomgoal=0.0`

They use the more comparable `m=2, K=4` setting:
- `RECUR_NUM_DENSE_LAYERS=2`
- `KTRAIN=4`
- `RECUR_MAX_ITERS=4`

```bash
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=scene-play-v0,RUN_GROUP=CRL_ScenePlay_StackedSwiGLU,EXP_NAME=sd000_SP_stacked_swi_m2k4,CRITIC_BACKBONE=recur_stacked_swiglu,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=cube-single-play-v0,RUN_GROUP=CRL_CubeSinglePlay_StackedSwiGLU,EXP_NAME=sd000_CSP_stacked_swi_m2k4,CRITIC_BACKBONE=recur_stacked_swiglu,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,ALPHA=3.0,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm
```



## SAW paper-aligned references

From [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), Supplementary Section G / Table 2:
- `antmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=25`,`discount=0.995`
- `humanoidmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=100`,`discount=0.995`

```bash
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd004_AGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=humanoidmaze-giant-navigate-v0,RUN_GROUP=SAW_HumanoidGiantNav_SwiGLU,EXP_NAME=sd004_HGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=100,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=cube-double-play-v0,RUN_GROUP=SAW_CubeDoublePlay_SwiGLU,EXP_NAME=sd004_CDP_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=2,ENV_NAME=scene-play-v0,RUN_GROUP=SAW_ScenePlay_SwiGLU,EXP_NAME=sd002_SP_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

```


### NSCC PBS equivalents

On NSCC ASPIRE2A with the repo defaults in `pbs/` and `slurm`, the primary recurrent SAW run becomes:

```bash
qsub -P personal -q normal -v SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd000_AGN_SAW_Swi,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25 pbs/train_saw.pbs


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
