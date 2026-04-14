# Research Plan

## Working Thesis

This project should be framed primarily as a **critic-architecture intervention for long-horizon offline goal-conditioned RL**.

The main claim is:

> Long-horizon offline GCRL is bottlenecked in part by value learning. Prior work mostly attacks this at the algorithmic level through value objectives, multi-step returns, hierarchy, or policy extraction changes. This project studies a complementary axis: critic architecture. Standard MLP critics can produce diffuse and low-margin value signals on hard long-horizon state-goal pairs, while iterative refinement can produce sharper and more useful value signals at similar parameter or compute budgets.

This is already enough for a full paper.

## Scope Control

There are at least three possible questions:

1. Does critic architecture matter for long-horizon offline GCRL?
2. Does a better critic improve actor extraction and generalization?
3. Is the method general and composable with other algorithms?

These should not be treated as equally central.

Recommended prioritization:

- Primary claim: critic architecture matters
- Secondary consequence: a sharper critic improves downstream actor learning
- Secondary breadth: the method is plug-and-play across more than one algorithm

The paper should not try to equally prove all three at full strength.

## Recommended Narrative

### 1. Motivation

Long-horizon offline GCRL is difficult. Prior work identifies horizon-related issues in value learning and policy learning. Existing solutions mainly modify the **algorithm** through n-step returns, hierarchy, special value objectives, or policy extraction changes.

### 2. Gap

The **critic architecture** itself is underexplored. Standard MLP critics may be too weak or too inefficient at extracting a sharp long-horizon signal from `(s, g)`.

### 3. Observation

On hard tasks, a plain MLP critic produces a diffuse, low-margin, or ambiguous signal. Increasing width or depth helps somewhat, but is inefficient in parameter count or compute.

### 4. Method

Use iterative refinement with FiLM-conditioned updates over the same `(s, g)` pair. The point is not to change the TD rule, but to iteratively sharpen the critic’s prediction.

### 5. Main Result

At matched parameter count or matched compute, iterative refinement outperforms plain MLP critics.

### 6. Mechanism

Refinement produces a sharper critic signal: larger margin, better ranking, and more confident supervision for actor learning.

### 7. Secondary Result

A sharper critic improves downstream actor extraction and may help generalization.

### 8. Breadth

Because this is a model-side intervention, it can be plugged into more than one value-based offline GCRL algorithm.

## What To Avoid

Do not frame the paper as:

- “there are many bottlenecks and we address one of them and maybe partially the others”
- “we solve the long-horizon problem”
- “we improve all offline GCRL algorithms”
- “we reduce horizon” unless the method explicitly does that

Prefer safer claims:

- “we study a model-side manifestation of the long-horizon value-learning bottleneck”
- “we improve critic sharpness and downstream actor learning”
- “the method is complementary to explicit horizon-reduction techniques”
- “the method is plug-and-play across multiple value-based offline GCRL methods”

## Algorithm Roles

### Main paper algorithm: HIQL

Recommended as the primary benchmark algorithm because:

- it is a strong offline GCRL baseline
- it has a clear value-learning component and a clear actor-learning component
- it naturally supports the story “a sharper value signal teaches the actor better”
- it is already competitive on stitching-style tasks

### Analysis / isolation algorithm: CRL

Use CRL for mechanistic or diagnostic experiments because:

- it isolates the critic phenomenon more cleanly
- the interpretation is less entangled with actor complexity
- it is well-suited to demonstrating the critic-side effect itself

### What not to include as central algorithms

Avoid making the paper center depend on too many algorithms such as:

- IQL
- SAC+BC
- QRL
- SAW
- CGCIVL
- SHARSA
- flow BC
- n-step DQN

These should only appear if they serve a specific secondary role.

## Recommended Experimental Slate

### Core algorithms

- `HIQL` as the main benchmark algorithm
- `CRL` as the critic-isolation / diagnosis algorithm

### Core architectures

- MLP baseline
- deeper MLP
- wider MLP
- iterative refinement critic

### Core sections

1. Main result on long-horizon OGBench tasks with HIQL
2. Critic diagnosis and mechanism with CRL or HIQL
3. Downstream actor benefit in HIQL
4. One small compositionality experiment on one additional method

## Minimal Claim-Aligned Experiment Design

### Part I. Does a critic-model problem exist?

Use CRL or the HIQL value network with actor effects minimized.

Compare:

- plain MLP baseline
- wider MLP
- deeper MLP
- iterative refinement model

Match either:

- parameter count
- training FLOPs or wall-clock
- ideally both

Tasks:

- one easy short-horizon task
- one medium task
- one hard stitching or long-horizon task

Measurements:

- return or success
- critic margin metric
- value ranking quality
- possibly variance or entropy over candidate actions or subgoals

Goal of this section:

> Show that critic architecture matters, especially as horizon grows.

### Part II. What phenomenon is being fixed?

This is the analysis section.

Possible measurements:

- value gap between better and worse candidate actions
- AUC or pairwise ranking accuracy on action comparisons
- separation of positive vs negative future states
- signal quality as a function of goal distance
- calibration-like curves if possible
- actor target sharpness

At least one metric should be explicitly horizon-sensitive.

Recommended ways to bucket:

- by goal distance
- by trajectory-to-goal length
- by stitching difficulty

Goal of this section:

> Show that MLP critics become diffuse as horizon increases, while iterative refinement degrades more slowly.

### Part III. Does this help actor extraction?

Use HIQL.

Compare:

- standard HIQL critic
- refined HIQL critic

Keep the actor design fixed.

Measurements:

- final success or return
- actor agreement with high-value actions
- policy loss or extraction quality
- optionally performance with the actor fixed or critic frozen

Goal of this section:

> Show that the refined critic is not just numerically nicer; it improves downstream policy learning.

### Part IV. Is it general or composable?

Keep this small.

Use only one additional algorithmic context beyond the main story.

Recommended options:

- refinement inside `HIQL`
- refinement inside `CRL`

and stop there, or add one compact experiment showing complementarity with an explicit horizon-reduction method.

Goal of this section:

> Show the method is not tied to one value objective or one actor design.

## What Each Algorithm Is Good For

- `CRL`: best for isolating the critic phenomenon
- `HIQL`: best for showing critic-to-actor improvement in offline GCRL
- `IQL` / `SAC+BC`: more standard offline RL, but less aligned with the paper center
- `DDPG+BC` / `AWR` / `SfBC`: only useful if the paper becomes mainly about policy extraction, which it should not
- `flow BC`: useful as a BC floor or hierarchical component, not a central baseline
- `SHARSA` / `SAW` / `CGCIVL`: useful only for a compositionality section
- `n-step DQN`: useful as conceptual inspiration, not as a benchmark center

## Final Recommended Claim

The strongest concise framing is:

> Long-horizon offline GCRL is hard. Prior work mostly attacks this through algorithmic changes. We study the critic architecture instead. Standard MLP critics become diffuse on hard state-goal pairs; iterative refinement produces sharper value signals at better efficiency. This improves downstream actor learning and transfers across value-based offline GCRL methods.

That keeps the paper centered on **critic architecture as a neglected axis**, rather than trying to explain every bottleneck in offline GCRL at once.

## Experimental Budget Heuristic

Plan experiments by **claim**, not by algorithm count.

For each claim, ask for the minimum evidence needed:

- Claim 1: critic architecture matters
  - one algorithm
  - matched-compute or matched-parameter comparison

- Claim 2: iterative refinement sharpens critic signal
  - one analysis section
  - margin, ranking, and distance-bucket plots

- Claim 3: sharper critic helps actor learning
  - one main algorithm
  - fixed actor design

- Claim 4: the method is general
  - one additional algorithm only

This should sharply reduce the experimental surface area.

## Immediate Next Steps

1. Lock the main paper claim as a critic-architecture paper, not a general bottleneck paper.
2. Use `HIQL` as the main benchmark algorithm.
3. Use `CRL` as the critic-isolation and mechanism section.
4. Define a matched-parameter or matched-compute comparison between MLP and iterative refinement.
5. Choose a small horizon-stratified task set: easy, medium, hard.
6. Implement at least one horizon-sensitive analysis metric.
7. Keep compositionality to one small section only.
