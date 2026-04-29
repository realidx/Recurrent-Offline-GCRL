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

## HIQL Value-Refinement Mechanism Probe on `antmaze-large-stitch-v0`

This run uses the new HIQL-specific diagnostics:

- `evaluation/probe/hiql_actor/low_actor_value/adv_*`
- `evaluation/probe/hiql_actor/low_actor_awr_weight_*`
- `evaluation/probe/hiql_actor/high_actor_value/adv_*`
- `evaluation/probe/hiql_actor/high_actor_awr_weight_*`
- `evaluation/refine/value_step_{k}_hiql_actor/...` for per-refinement-step low/high actor-facing value signals

```bash
# SLURM seed 4, recurrent HIQL value, explicit probe dumps at start/mid/final.
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_recur_mech,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=2,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=2,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM=0,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,REFINE_DUMP_STEPS=1:500000:1000000,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_recur_mech,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM=0,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,REFINE_DUMP_STEPS=1:500000:1000000,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm



# NSCC PBS seed 4 equivalent. The -J 0 override is needed because this is a single-seed SEEDS list.
qsub -J 0 -P personal -q normal -v SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_Mechanism,EXP_NAME=sd004_ALS_hiql_recur_mech,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=6,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=6,VALUE_RECUR_LN_MODE=pre_loop,VALUE_RECUR_USE_STEP_INFO=1,VALUE_RECUR_USE_FILM=0,VALUE_RECUR_USE_INPUT_INJECTION=0,VALUE_RECUR_USE_SOFT_MIXTURE=0,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,REFINE_PROBE_SIZE=512,REFINE_DUMP_STEPS=1:500000:1000000,EVAL_ON_CPU=0 pbs/train_hiql.pbs
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
