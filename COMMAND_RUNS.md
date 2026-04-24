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

# CRL full model on antmaze-large-stitch-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=CRL_AntLargeStitch,EXP_NAME=sd000_ALS_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_USE_INPUT_INJECTION=1,CRITIC_RECUR_USE_SOFT_MIXTURE=0,RECUR_USE_ACT=0,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

# CRL full model on antmaze-giant-navigate-v0
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=0,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=CRL_AntGiantNavigate,EXP_NAME=sd000_AGN_Swi_in,CRITIC_BACKBONE=recur_tied,KTRAIN=4,RECUR_NUM_DENSE_LAYERS=2,CRITIC_RECUR_BLOCK_TYPE=swiglu,RECUR_MAX_ITERS=4,RECUR_USE_STEP_INFO=1,CRITIC_RECUR_USE_FILM=0,CRITIC_RECUR_USE_INPUT_INJECTION=1,CRITIC_RECUR_USE_SOFT_MIXTURE=0,RECUR_USE_ACT=0,ALPHA=0.1,DISCOUNT=0.995,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=1.0,ACTOR_P_RANDOMGOAL=0.0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

```

## CRL parameter-matched MLP comparisons on `antmaze-medium-stitch-v0`

These compare the `hidden_context` SwiGLU recurrent critic against two MLP critics matched by `params/critic_count` for the same `antmaze-medium-stitch-v0` setup. The target recurrent critic has `params/critic_count=7,497,744`. The closest matches found with [find_mlp_param_match.py](/Users/bruce/Recurrent-Offline-RL/scripts/find_mlp_param_match.py) are:


```bash

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_depthmatch_5x512,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=3,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd003_AMS_MLP_depthmatch_5x512,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=512x512x512x512x512,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

sbatch --array=0 --export="ALL,SEEDS=0,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd000_AMS_MLP_widthmatch_3x688,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=688x688x688,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_crl.slurm

qsub -P personal -q normal -v SEEDS=3,ENV_NAME=antmaze-medium-stitch-v0,RUN_GROUP=CRL_AntMediumStitch_ParamMatch,EXP_NAME=sd003_AMS_MLP_widthmatch_3x688,CRITIC_BACKBONE=mlp,VALUE_HIDDEN_DIMS=688x688x688,ALPHA=0.1,DISCOUNT=0.99,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,EVAL_ON_CPU=0 pbs/train_crl.pbs

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
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_mlp,VALUE_BACKBONE=mlp,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 2. HIQL baseline + confidence-scaled AWR (MLP value)
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_mlp_conf,VALUE_BACKBONE=mlp,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=1,ACTOR_CONFIDENCE_NUM_NEGATIVES=4,ACTOR_CONFIDENCE_TAU=0.0,ACTOR_CONFIDENCE_TEMPERATURE=1.0,ACTOR_CONFIDENCE_MIN=0.1,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 3. HIQL recurrent value
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_recur,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=0,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm

# 4. HIQL recurrent value + confidence-scaled AWR
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=4,ENV_NAME=antmaze-large-stitch-v0,RUN_GROUP=HIQL_AntLargeStitch_ConfGate,EXP_NAME=sd004_ALS_HIQL_recur_conf,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_MAX_ITERS=4,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_NUM_DENSE_LAYERS=2,DISCOUNT=0.99,LOW_ALPHA=3.0,HIGH_ALPHA=3.0,SUBGOAL_STEPS=25,ACTOR_P_CURGOAL=0.0,ACTOR_P_TRAJGOAL=0.5,ACTOR_P_RANDOMGOAL=0.5,ACTOR_CONFIDENCE_GATE=1,ACTOR_CONFIDENCE_NUM_NEGATIVES=4,ACTOR_CONFIDENCE_TAU=0.0,ACTOR_CONFIDENCE_TEMPERATURE=1.0,ACTOR_CONFIDENCE_MIN=0.1,EVAL_ON_CPU=0,WANDB_MODE=online" slurm/train_hiql.slurm
```


## SAW paper-aligned references

From [references/SAW.pdf](/Users/bruce/Recurrent-Offline-RL/references/SAW.pdf), Supplementary Section G / Table 2:
- `antmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=25`,`discount=0.995`
- `humanoidmaze-giant-navigate-v0`: `EXPECTILE=0.7`, `LOW_ALPHA=3.0`, `AWR_ALPHA=3.0`, `KL_ALPHA=3.0`, `SUBGOAL_STEPS=100`,`discount=0.995`

```bash
sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=2,ENV_NAME=antmaze-giant-navigate-v0,RUN_GROUP=SAW_AntGiantNav_SwiGLU,EXP_NAME=sd002_AGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=25,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=2,ENV_NAME=humanoidmaze-giant-navigate-v0,RUN_GROUP=SAW_HumanoidGiantNav_SwiGLU,EXP_NAME=sd002_HGN_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.995,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=3.0,SUBGOAL_STEPS=100,WANDB_MODE=online" slurm/train_saw.slurm

sbatch --array=0 --gpus=h100-47 --export="ALL,SEEDS=2,ENV_NAME=cube-double-play-v0,RUN_GROUP=SAW_CubeDoublePlay_SwiGLU,EXP_NAME=sd002_CDP_SAW,VALUE_BACKBONE=recur_tied,VALUE_RECUR_ITERS=4,VALUE_RECUR_NUM_DENSE_LAYERS=2,VALUE_RECUR_BLOCK_TYPE=swiglu,VALUE_RECUR_MAX_ITERS=4,DISCOUNT=0.99,EXPECTILE=0.7,LOW_ALPHA=3.0,AWR_ALPHA=3.0,KL_ALPHA=1.0,SUBGOAL_STEPS=10,WANDB_MODE=online" slurm/train_saw.slurm

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
