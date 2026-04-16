# Research Plan

## Working Thesis

This project should be framed as a **critic-architecture paper for long-horizon offline goal-conditioned RL**.

The central claim is:

> Long-horizon offline GCRL is bottlenecked in part by value learning. Prior work mainly attacks this through algorithmic changes such as multi-step objectives, hierarchy, or policy extraction changes. We study a complementary axis: critic architecture. Standard MLP critics can produce diffuse and low-margin value signals on hard long-horizon state-goal pairs, while an iterative SwiGLU critic with FiLM, LayerScale, and pre-LN can produce sharper and more useful value signals at similar parameter or compute budgets.

This is already enough for a full paper. The plan should stay centered on this claim.

## Scope Control

There are three possible questions:

1. Does critic architecture matter for long-horizon offline GCRL?
2. Does a sharper critic improve downstream actor learning?
3. Is the method general across more than one value-based offline GCRL algorithm?

These should not be treated as equally central.

Recommended priority:

- Primary claim: critic architecture matters
- Secondary consequence: a sharper critic improves actor learning
- Secondary breadth: the method is plug-and-play across more than one algorithm

The paper should not try to equally prove all three at full strength.

## Final Intended Framing

The strongest concise framing is:

> Long-horizon offline GCRL is hard. Existing work mostly attacks this through algorithmic changes. We study the critic architecture instead. Standard MLP critics become diffuse on hard state-goal pairs. Iterative refinement with SwiGLU, FiLM, LayerScale, and pre-LN produces sharper value signals at better efficiency, which improves downstream actor learning and transfers across value-based offline GCRL methods.

This keeps the paper centered on **critic architecture as a neglected axis** rather than claiming to solve every bottleneck in offline GCRL.

## Narrative Structure

### 1. Motivation

Long-horizon offline GCRL is difficult. Prior work identifies issues in value learning, horizon length, and policy extraction. Most solutions change the algorithm.

### 2. Gap

The critic architecture itself is underexplored. Standard MLP critics may be too weak or too inefficient to extract a sharp signal from `(s, g)` on hard long-horizon tasks.

### 3. Observation

On hard tasks, a plain MLP critic produces a diffuse, low-margin, or ambiguous signal. Scaling width or depth helps somewhat, but is inefficient in parameters or compute.

### 4. Method

Use iterative refinement over the same `(s, g)` pair with a SwiGLU update block and FiLM conditioning, stabilized by LayerScale and pre-LN. The point is not to change the TD rule. The point is to iteratively sharpen the critic prediction.

### 5. Main Result

At matched parameter count or matched compute, the iterative critic outperforms plain MLP critics.

### 6. Mechanism

Refinement produces a sharper critic signal: larger margin, better ranking, better separation, and more useful supervision for actor learning.

### 7. Secondary Result

A sharper critic improves downstream actor extraction in a strong offline GCRL algorithm.

### 8. Breadth

Because this is a model-side intervention, it can plug into more than one value-based offline GCRL method.

## What To Avoid

Do not frame the paper as:

- "there are many bottlenecks and we address one of them and maybe partially the others"
- "we solve the long-horizon problem"
- "we improve all offline GCRL algorithms"
- "we reduce horizon" unless the method explicitly does that
- "recurrence is always better than feedforward models"

Prefer safer claims:

- "we study a model-side manifestation of the long-horizon value-learning bottleneck"
- "we improve critic sharpness and downstream actor learning"
- "the method is complementary to explicit horizon-reduction techniques"
- "the method is plug-and-play across multiple value-based offline GCRL methods"

## Algorithm Roles

### Main paper algorithm: HIQL

Use `HIQL` as the main benchmark because:

- it is a strong offline GCRL baseline
- it has a clear value-learning component and a clear actor-learning component
- it supports the story "a sharper critic teaches the actor better"
- it is already competitive on long-horizon stitching-style tasks

### Analysis / isolation algorithm: CRL

Use `CRL` for mechanistic and critic-side diagnosis because:

- it isolates the critic phenomenon more cleanly
- the interpretation is less entangled with actor complexity
- it is well-suited for demonstrating critic sharpness directly

### What not to include as central algorithms

Avoid making the paper center depend on too many algorithms such as:

- `IQL`
- `SAC+BC`
- `QRL`
- `SAW`
- `CGCIVL`
- `SHARSA`
- `flow BC`
- `n-step DQN`

These are only useful if they serve a very specific secondary role.

## Core Model Definition

Assume the proposed model is the full iterative critic:

- iterative refinement
- SwiGLU update block
- FiLM conditioning
- LayerScale
- pre-LN

This full model is the reference point for all ablations.

## Claim-Aligned Experiment Plan

The experimental slate should be organized by claim, not by algorithm count.

### Claim 1. Critic architecture matters

Minimal evidence needed:

- one algorithmic setting that isolates critic behavior well
- one easy task, one medium task, one hard long-horizon task
- MLP baseline, deeper MLP, wider MLP, iterative critic
- matched-parameter or matched-compute comparison

Recommended setting:

- use `CRL` for the cleanest isolation

Goal:

> Show that critic architecture matters, especially as horizon and task difficulty increase.

### Claim 2. Iterative refinement sharpens critic signal

Minimal evidence needed:

- horizon-sensitive analysis
- value margin and ranking metrics
- refinement-step analysis

Recommended setting:

- use `CRL`, or use the `HIQL` value network if the implementation is cleaner

Goal:

> Show that MLP critics become diffuse on hard state-goal pairs, while iterative refinement degrades more slowly and sharpens the signal step by step.

### Claim 3. Sharper critics help actor learning

Minimal evidence needed:

- one strong offline GCRL algorithm
- actor held fixed
- standard critic vs refined critic

Recommended setting:

- use `HIQL`

Goal:

> Show that the refined critic is not merely numerically nicer; it improves downstream policy learning.

### Claim 4. The method is general

Minimal evidence needed:

- one additional algorithm only
- one compact compositionality result

Goal:

> Show the method is not tied to one value objective or one actor design.

## Must-Have Experiments

These are the experiments that should exist before the paper is considered complete.

### A. Main comparison: baseline vs proposed model

Purpose:

- establish the core result
- provide the main table or figure for the paper

Compare:

- plain MLP baseline
- deeper MLP
- wider MLP
- iterative SwiGLU + FiLM + LayerScale + pre-LN critic

Prefer:

- `CRL` for critic-isolation results
- `HIQL` for final task performance and actor-transfer results

Tasks:

- one easy short-horizon task
- one medium task
- one hard stitching or long-horizon task

Required outputs:

- final success or return
- parameter count
- compute budget
- uncertainty across seeds

### B. Mechanism analysis

Purpose:

- explain why iterative refinement helps
- connect the performance gain to critic signal quality

Required measurements:

- value margin
- ranking quality
- positive vs negative future-state separation
- one horizon-sensitive analysis bucketed by distance or goal difficulty
- refinement-step curves showing how the signal changes across iterations

This section must directly answer:

> What critic pathology does the MLP have, and what does iterative refinement fix?

### C. Component ablation

Purpose:

- show that the final design is not arbitrarily assembled

Use the full model as the reference and test:

- remove FiLM
- remove LayerScale
- remove pre-LN
- replace iterative refinement with a single step

Important:

- do not only run leave-one-out ablations
- also include a small build-up sequence if budget permits

Recommended build-up:

- MLP baseline
- SwiGLU block
- + pre-LN
- + FiLM
- + LayerScale
- + iteration

This is stronger than only saying each component is "essential."

### D. Iteration-count study

Purpose:

- support the claim that iterative refinement behaves like a controllable sharpening process

Compare:

- 1 iteration
- 2 iterations
- 4 iterations
- 6 or more iterations if stable and affordable

Desired pattern:

- simple tasks peak earlier
- harder tasks benefit from more iterations
- beyond the useful range, performance saturates or degrades

This should include both:

- final task performance
- signal-quality-by-step analysis

### E. Matched-parameter comparison

Purpose:

- show the gain is not just from having more parameters

Compare:

- proposed iterative critic
- deeper MLP matched by parameter count
- wider MLP matched by parameter count

Important:

- use at least one matched point that is very close
- ideally use a small 3-point scaling curve instead of only one point

### F. Matched-compute comparison

Purpose:

- show the gain is not just from spending more computation

Compare:

- proposed iterative critic
- deeper MLP
- wider MLP

Do not rely on wall-clock alone. Use wall-clock as a secondary metric.

Primary fairness metrics should be one or more of:

- parameter count
- total parameter updates
- parameter-sample products
- estimated critic FLOPs if available

## Critical Missing Controls

These controls are easy to overlook, but they materially strengthen the paper.

### 1. Recurrence vs stronger block family

You need at least one control showing the gain is from iterative refinement rather than only from using a better feedforward block.

Recommended controls:

- 1-step version of the same SwiGLU + FiLM + LayerScale + pre-LN block
- if feasible, a small untied or unrolled feedforward control

This helps isolate:

- iterative refinement itself
- weight sharing / repeated application
- block expressivity

### 2. Seed protocol

Every main comparison needs a clear seed protocol.

Minimum expectation:

- enough seeds to report mean and spread for main results
- the same seeds across compared methods where possible

Without this, matched-budget claims will be weak.

### 3. Actor-benefit control

When claiming the critic helps actor extraction, keep the actor design fixed.

Strong optional controls:

- critic frozen before actor extraction
- compare actor behavior when trained from matched-quality or matched-time critic checkpoints

The point is to make the actor-benefit claim causal rather than merely correlated.

### 4. Batch-composition sanity check

If goal sampling or difficulty composition drifts across runs, the mechanism story becomes fragile.

Track batch statistics so you can verify that differences are not caused by data-composition changes.

## Nice-To-Have Experiments

These help if the core paper is already strong, but they are not required for acceptance of the main claim.

### A. Small build-up ablation sequence

This is useful if the leave-one-out ablation is noisy or hard to interpret.

### B. Small scaling curve

Instead of a single matched-parameter point, show small, medium, and large budgets.

This makes the efficiency story more robust and less cherry-picked.

### C. One extra generality result

Use only one additional algorithm beyond the core story.

Good use:

- one compact result showing the method also helps another value-based offline GCRL algorithm

Bad use:

- turning the paper into a multi-algorithm leaderboard

## Do Not Spend Time On These Until The Core Paper Is Finished

- iterative refinement on the actor side
- online GCRL extension
- many additional algorithms
- many environment families beyond what is needed to show easy, medium, hard difficulty
- speculative theory sections that are not supported by the diagnostics

These are good future-work directions, not core-paper requirements.

## Recommended Minimal Experimental Slate

If budget is tight, the minimum convincing paper is:

1. Main result with `HIQL` on easy, medium, and hard OGBench tasks
2. Critic diagnosis and mechanism with `CRL`
3. Matched-parameter and matched-compute comparison against deeper and wider MLPs
4. Component ablation on the full iterative critic
5. Iteration-count analysis
6. One compact compositionality result on one additional algorithm

Anything beyond this should be justified carefully.

## Metrics And Logging Plan

Logs should support paper claims directly rather than functioning as generic debugging metrics.

### Main outcome metrics

Use:

- `evaluation/return`
- `evaluation/success`

Purpose:

- report task performance
- define the main result figures and tables

### Fairness and efficiency metrics

Use:

- `params/critic_count`
- `params/value_count`
- `params/actor_count`
- `time/hours_elapsed`
- `time/samples_per_second`
- `compute/total_param_updates`
- `compute/critic_param_updates`
- `compute/value_param_updates`
- `compute/total_param_samples`
- `compute/critic_param_samples`
- `compute/value_param_samples`

Purpose:

- make matched-parameter and matched-compute comparisons explicit
- prevent vague fairness claims

### Critic signal metrics

Use:

- `evaluation/critic_signal/*` or `evaluation/value_signal/*`
- core metrics such as positive mean, negative mean, `margin_mean`, `rank_acc`, `separation_z`

Purpose:

- show critic sharpness directly
- connect performance to signal quality

### Geometry and horizon-sensitive metrics

Use when available:

- `evaluation/critic_geometry/*` or `evaluation/value_geometry/*`
- `graph_corr`
- `graph_spearman`
- `hard_graph_corr`
- `hard_graph_spearman`
- source buckets: `current`, `traj`, `random`
- horizon buckets: `traj_short`, `traj_medium`, `traj_long`
- maze-aware buckets:
- `maze_xy_short`, `maze_xy_medium`, `maze_xy_long`
- `maze_path_short`, `maze_path_medium`, `maze_path_long`

Purpose:

- show that MLP critics degrade more sharply with difficulty
- show that refinement preserves ranking and separation on hard pairs

### Refinement-step metrics

Use:

- `evaluation/refine/step_k_*`
- `evaluation/refine/step_k_signal/*`

Purpose:

- show how signal quality changes with each refinement step
- support the iteration-count story

### Actor-transfer metrics

Use:

- CRL-style actor diagnostics: `evaluation/actor_critic/*`, `evaluation/actor_behavior_*`
- HIQL-style actor diagnostics: `evaluation/low_actor_*`, `evaluation/high_actor_*`

Purpose:

- show critic improvements transfer to policy extraction quality

### Batch-composition sanity checks

Use:

- `batch/value_goal_source_*_frac`
- `batch/value_goal_horizon_mean`
- `batch/value_goal_horizon_std`
- `batch/value_goal_maze_xy_distance_*`
- `batch/value_goal_maze_path_*`

Purpose:

- verify stable goal sampling and horizon composition
- rule out batch-composition drift as an explanation

## Decision Rules For Paper Scope

If the following are true, the paper is already strong:

- the iterative critic beats MLP baselines on hard tasks
- the gain survives matched-parameter and matched-compute controls
- critic signal metrics improve in a horizon-sensitive way
- actor learning improves in `HIQL`

If those four are not yet true, do not expand the scope.

## Immediate Next Steps

1. Lock the paper claim as a critic-architecture paper.
2. Finalize the minimal task set: one easy, one medium, one hard.
3. Finalize the main baselines: MLP, deeper MLP, wider MLP, full iterative critic.
4. Add the missing recurrence-vs-block-family control.
5. Define the seed protocol for all main experiments.
6. Define the fairness protocol for matched-parameter and matched-compute comparisons.
7. Implement at least one horizon-sensitive signal metric and one refinement-step metric.
8. Keep generality to one compact additional algorithm only.
