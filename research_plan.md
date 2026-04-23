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
- Step embedding
- LayerScale
- pre-LN

This full model is the reference point for all ablations.

## Claim-Aligned Experiment Plan

The experimental slate should be organized by claim, not by algorithm count.

### Claim 1. Critic architecture matters

Minimal evidence needed:

- one algorithmic setting that isolates critic behavior well
- one easy task, one medium task, one hard long-horizon task: antmaze-medium-stitch, antmaze-giant-navigate, antmaze-large-stitch
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
- compare MLP baseline and iterative critic

Recommended setting:

- use `CRL`

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

Fixed proposed model:

- iterative SwiGLU + Step embedding + LayerScale + pre-LN critic

Compare:

- plain MLP critic baseline
- fixed proposed iterative critic

Important:

- this section is not for model search
- this section is not for matched-parameter or matched-compute claims
- this section should only compare the frozen proposed model against the standard MLP critic

Algorithms:

- `CRL` is required because it gives the clearest critic-side comparison
- `HIQL` is strongly recommended because it strengthens relevance for offline goal-conditioned RL
- `IQL` and `SAC+BC` should be included if budget permits, as supporting breadth rather than as the center of the paper

Tasks:

- include at least two long-horizon AntMaze tasks
- include one easier long-horizon setting such as `antmaze-medium-stitch-v0`
- include one harder long-horizon locomotion setting such as `antmaze-large-stitch-v0` or `antmaze-giant-navigate-v0`
- include one non-AntMaze OGBench manipulation task if budget permits, such as a `scene`, `cube`, or `puzzle` task

Recommended minimum task slate:

- `antmaze-medium-stitch-v0`
- `antmaze-large-stitch-v0`
- `antmaze-giant-navigate-v0`

Optional coverage upgrade:

- add one manipulation task once the locomotion story is stable

Protocol:

- use the same seed set for every compared method whenever possible
- keep actor architecture, training horizon, and task hyperparameters fixed within each algorithm-task comparison
- do not mix ablation variants such as no-Step, no-LayerScale, or no-pre-LN into this section
- do not present this section as a fairness-controlled parameter or compute comparison yet; that belongs later

Required outputs:

- final task performance using the standard benchmark metric for each environment
- mean and spread across seeds
- per-seed values saved so the same runs can later support mechanism analysis
- one benchmark-facing main table
- one compact figure summarizing the main comparison

Primary question answered by this section:

> Does the frozen proposed critic architecture outperform the standard MLP critic on meaningful long-horizon offline control benchmarks, across more than one algorithm family?

### B. Mechanism analysis

Purpose:

- explain why iterative refinement helps
- connect the performance gain to actor-usable critic/value signal quality
- make the mechanism story reviewer-facing rather than a loose collection of diagnostics
- define the exact mechanism-only experiment slate before running new jobs

Mechanism-only thesis:

> Recurrent refinement helps when the offline dataset contains weak but usable stitching signal. It improves the local preference signal used by the actor, especially on hard state-goal pairs. It does not necessarily learn a uniformly better global geometry, and it helps less when the data is too sparse or underdetermined.

This section must directly answer:

> What critic pathology does the MLP have, and what does iterative refinement fix?

The answer should be:

> The MLP critic/value network can produce weak or ambiguous local preferences on hard goal-conditioned pairs. Recurrent refinement makes this signal more actor-usable through repeated correction, not simply by learning a better global maze-distance metric or by adding more parameters.

Mechanism claims to support:

- the baseline critic/value network can learn coarse structure while still producing one-shot judgments that are too blunt or brittle for actor extraction
- iterative refinement progressively disambiguates hard state-goal pairs rather than merely acting as a deeper one-shot predictor
- the first major gain is not only sharper final margins, but a signal that changes actor extraction in the right direction
- the gain is selective to stitchable hard pairs rather than uniform across all cases
- iterative refinement makes the signal more operationally useful for control, not necessarily more globally metric-like

What not to claim:

- do not claim recurrence is always better than feedforward models
- do not claim the model solves sparse long-horizon exploration or all offline GCRL bottlenecks
- do not claim the mechanism is better global maze geometry unless graph-distance metrics support it
- do not reduce the story to "margin goes up"
- do not make input injection or output soft mixture part of the final mechanism because current evidence does not show benefit

Final recurrent model for the mechanism story:

- tied iterative refinement over a critic/value hidden state
- SwiGLU update block
- step embedding or step conditioning
- FiLM modulation when enabled by the chosen config
- LayerScale stabilization
- pre-LN recurrent update
- fixed iteration count, with `iter4` as the current default final model

The final model does not include:

- repeated input injection into every recurrent step
- output soft mixture over all recurrent steps

Those variants can remain in appendix or design-search notes, but they should not be central mechanism figures.

Algorithm roles for mechanism:

- `CRL` should be the main mechanism microscope because it exposes a clean contrastive state-action-goal critic.
- `IQL` / `GCIQL` should be the value-network validation algorithm because actor extraction is mediated by learned value/advantage estimates.
- `HIQL` should not be central for this mechanism section because its high-level and low-level actors make the causal value-network story less clean.
- `SAW` and other methods can be future work or appendix breadth after the CRL/IQL mechanism is stable.

Be precise about IQL:

- do not say IQL uses only a value network
- say that IQL/GCIQL represents a value-learning and advantage-weighted actor-extraction setting
- say that IQL/GCIQL checks whether the same recurrent-refinement mechanism appears outside the contrastive CRL critic

Mechanism environment slate:

| Role | Environment | Reason |
| --- | --- | --- |
| Clean positive case | `antmaze-medium-stitch-v0` | Best for showing iteration-depth and early actor-usability effects clearly. |
| Main hard positive case | `antmaze-large-stitch-v0` | Stronger long-horizon stitching task for capacity controls and headline mechanism plots. |
| Boundary case | `antmaze-giant-navigate-v0` | Tests whether the method still helps when stitching signal is sparse or underdetermined. |

A manipulation task is optional for the mechanism section. Add it only if the paper later needs evidence that the mechanism is not maze-specific.

Seed protocol:

- minimum acceptable mechanism evidence: 3 seeds
- preferred headline mechanism evidence: 5 seeds
- qualitative plots such as critic fields or per-step trajectory visualizations: 1 representative seed is fine, but must be backed by multi-seed quantitative plots
- recommended seed set: start with `0,1,2`; add `3,4` for final figures

Required CRL mechanism experiments:

| Environment | Variants | Purpose |
| --- | --- | --- |
| `antmaze-medium-stitch-v0` | MLP, iter1, iter2, iter4 | Clean iteration-depth mechanism and early learning. |
| `antmaze-large-stitch-v0` | MLP, wider MLP, deeper MLP, iter1, iter2, iter4 | Main hard case and capacity control. |
| `antmaze-giant-navigate-v0` | MLP, iter4, optionally iter2 | Boundary case where recurrence may help less. |

Required IQL/GCIQL mechanism experiments:

| Environment | Variants | Purpose |
| --- | --- | --- |
| `antmaze-large-stitch-v0` | MLP value, iter4 value, optionally iter2 | Validate that recurrent value refinement improves actor extraction. |
| `antmaze-giant-navigate-v0` | MLP value, iter4 value | Check whether the value mechanism weakens in sparse navigate setting. |

If compute is tight, skip IQL on `medium-stitch`; CRL already covers the clean mechanism case.

Feedforward controls:

- standard MLP baseline
- parameter-matched or near-parameter-matched wider MLP
- parameter-matched or near-parameter-matched deeper MLP
- one-step recurrent/SwiGLU model if available
- iterative recurrent model at `iter2` and `iter4`

The key comparison is not only `MLP` versus `iter4`.
The key comparison is:

> Does multi-step tied refinement outperform comparable one-shot feedforward capacity?

Core empirical story from current CRL results on `antmaze-medium-stitch-v0`:

- early performance gains are much larger than early margin gains
- at `100k`, baseline and recurrent critics have similar overall margin, but recurrent has much higher success and much better actor-facing diagnostics
- iteration depth shows a threshold effect: `iter1` helps only modestly, while `iter2+` gives the major jump
- actor-extraction metrics improve almost monotonically with refinement depth
- medium-horizon and path-sensitive buckets improve more than easy buckets
- global maze-geometry correlation can worsen as refinement depth increases, so the correct claim is not "better shortest-path regression everywhere"

Recommended section structure:

1. Set up the failure mode:
   long-horizon offline goal-conditioned RL requires compositional reachability judgments under dataset support constraints, so a one-shot MLP can produce value estimates that are too coarse or brittle on hard pairs even if its global geometry statistics are reasonable.
2. Show the early-stage effect:
   the recurrent critic improves actor usability before it becomes dramatically sharper in aggregate margin.
3. Show that refinement itself matters:
   `iter1` is not enough; the main gain appears once the critic can revise its estimate multiple times.
4. Show selectivity:
   the improvement is strongest on stitchable hard pairs, medium-horizon/path-sensitive buckets, and the hardest stitch subtasks.
5. State the geometry caveat:
   the recurrent critic is more useful for control even when it is not a uniformly better global maze-distance surrogate.
6. State the boundary:
   the mechanism should help less on sparse navigate settings where the offline data lacks enough stitchable signal.

Recommended figures:

1. Baseline-pathology figure.
   Compare MLP and recurrent models on success plus fixed-probe hard-pair signal metrics.
   Goal:
   show that the baseline is not merely lower reward; its learned signal is less useful on the pairs that require stitching.

2. Iteration-depth mechanism figure.
   Plot fixed-probe signal metrics for `iter1`, `iter2`, and `iter4`, plus per-recurrent-step metrics from the same trained model.
   Goal:
   show that later refinement steps do real corrective work and that the effect concentrates on hard or stitchable buckets.
   This figure is the main evidence that iterative refinement itself matters.

3. Actor-extraction figure.
   For CRL, plot actor-action score minus behavior-action score and the fraction where the actor action scores higher.
   For IQL/GCIQL, plot advantage distribution, positive-advantage fraction, AWR weight mean/max, and AWR effective sample size.
   Goal:
   show that the critic/value signal becomes more usable for actor extraction.

4. Recurrence-versus-capacity figure.
   Compare MLP, wider MLP, deeper MLP, iter1, iter2, and iter4.
   Goal:
   show that the effect is not explained only by parameter count or a stronger feedforward block.

5. Regime-boundary figure.
   Compare `antmaze-large-stitch-v0` and `antmaze-giant-navigate-v0`.
   Goal:
   show that recurrence helps when the dataset has stitchable signal and helps less when information is sparse or underdetermined.

How the figures and narrative should connect:

- Figure 1 answers:
  what mistake does the MLP critic make?
- Figure 2 answers:
  what do extra refinement steps actually change?
- Figure 3 answers:
  why does that matter for offline policy extraction?
- Figure 4 answers:
  why is this not just parameter count?
- Figure 5 answers:
  where does the mechanism stop helping?

The mechanism subsection should not read as three disjoint diagnostics.
It should read as one argument:

- the baseline critic makes the wrong kind of approximation
- refinement progressively corrects it
- this correction first appears as improved actor usability
- and later appears as stronger medium-horizon discrimination and better stitching success
- the effect weakens when the dataset does not contain enough stitchable signal

Most valuable quantitative points to emphasize:

- at `100k`, recurrent and baseline margins can be similar while success differs drastically
- `q_delta_mean` and policy-behavior drift improve sharply and early
- the `iter1` to `iter2` jump is more important than the `iter3` to `iter4` gap
- `task3` and `task4` or similarly hard stitch subtasks carry a disproportionate share of the gain
- medium-horizon buckets are more diagnostic than global averages alone
- CRL actor-usability metrics should move in the same direction as success
- IQL/GCIQL advantage and AWR-weight diagnostics should show the same qualitative trend on at least one hard stitch task

Mechanism logging to track during training and evaluation:

- outcome metrics: `evaluation/overall_success`, `evaluation/best_so_far_success`, per-task success, and return as secondary
- fairness metrics: critic/value parameter count, actor parameter count, wall-clock hours, updates per second, number of gradient updates, and recurrent iteration count
- fixed-probe CRL metrics: `evaluation/probe/critic_signal/margin_mean`, `margin_std`, `positive_score_mean`, `negative_score_mean`, `rank_accuracy`, and `hard_margin_mean`
- fixed-probe IQL/GCIQL metrics: `evaluation/probe/value_signal/margin_mean`, `margin_std`, `rank_accuracy`, and `hard_margin_mean`
- CRL actor-usability metrics: `evaluation/probe/actor_critic/q_actor_mean`, `q_behavior_mean`, `q_delta_mean`, and `prefer_actor_frac`
- IQL/GCIQL actor-usability metrics: `evaluation/probe/actor_value/adv_mean`, `adv_std`, `positive_adv_frac`, `awr_weight_mean`, `awr_weight_max`, and `awr_weight_ess_frac`
- policy-behavior metrics when available: `evaluation/probe/policy_behavior/mse_mean`, `l2_mean`, and `evaluation/probe/actor_behavior_log_prob`
- recurrent-step metrics: `evaluation/refine/critic_step_k_signal/*`, `evaluation/refine/value_step_k_signal/*`, final-minus-initial deltas, hidden-state norm, and hidden-state drift
- geometry controls: graph-distance correlation and hard-pair graph-distance correlation
- training health metrics: critic/value loss, actor loss, critic/value gradient norm, actor gradient norm, CRL contrastive score margin, and IQL/GCIQL advantage/AWR-weight statistics
- dataset sanity metrics: goal-source composition, horizon or temporal-distance buckets, easy/medium/hard probe-pair fractions, and task ID distribution

Metrics to demote:

- input-injection ratios
- soft-mixture weights or entropy
- absolute Q/value magnitude without paired ranking or margin interpretation
- duplicate percentiles for every metric
- per-ensemble min/max debug metrics unless diagnosing instability
- wall-clock-only comparisons without parameter and update counts

Decision rules:

- strong enough: `iter4` beats MLP and matched feedforward controls on a stitch task
- strong enough: `iter2` or `iter4` improves fixed-probe hard-pair signal over MLP
- strong enough: recurrent-step metrics show signal improvement from early to final refinement steps
- strong enough: actor-usability metrics improve in the same direction as success
- strong enough: IQL/GCIQL shows the same qualitative value/actor-extraction trend on at least one hard stitch task
- strong enough: the navigate boundary case shows smaller or less consistent gains
- needs revision: success improves but actor-usability metrics do not move
- needs revision: fixed-probe signal improves only on easy pairs
- needs revision: wider or deeper MLP matches recurrence at similar compute
- needs revision: IQL/GCIQL shows the opposite trend from CRL on the same task

What not to claim:

- do not claim that iterative refinement simply learns a uniformly better maze metric
- do not claim that every geometry statistic must improve
- do not reduce the mechanism story to "margin goes up"
- do not claim that input injection or output soft mixture are part of the final mechanism

Preferred conclusion of this section:

> Recurrent refinement helps because long-horizon offline goal-conditioned value estimation contains a sequential disambiguation problem. The recurrent critic/value network repeatedly corrects an initially coarse judgment and makes the resulting signal more useful for actor extraction. This improvement appears most clearly on stitchable hard pairs and does not require the model to become a uniformly better global maze-distance predictor.

Immediate mechanism experiment order:

1. Smoke-test the logging on one short CRL run on `antmaze-medium-stitch-v0`.
2. Run CRL `MLP`, `iter1`, `iter2`, and `iter4` on `antmaze-medium-stitch-v0` for 3 seeds.
3. Run CRL capacity controls on `antmaze-large-stitch-v0` for 3 seeds.
4. Run CRL `MLP` and `iter4` on `antmaze-giant-navigate-v0` for the boundary case.

### C. Component ablation

Purpose:

- show that the frozen proposed critic is not arbitrarily assembled
- identify which parts of the final design are responsible for the gain

Reference model:

- full proposed critic from Section A:
  iterative SwiGLU + Step embedding + LayerScale + pre-LN

Required ablations:

- remove Step embedding
- remove LayerScale
- remove pre-LN
- replace iterative refinement with a 1-step version of the same SwiGLU critic

Important interpretation:

- the 1-step control is not the MLP baseline
- it should keep the same block family and differ mainly in the absence of iterative refinement
- this is the key control for separating block expressivity from iterative refinement

Recommended scope:

- run this section on one primary algorithm only, preferably `CRL`
- use one medium and one hard task rather than the full benchmark slate
- good defaults are `antmaze-large-stitch-v0` and `antmaze-giant-navigate-v0`
- add more algorithms or tasks only if the main ablation pattern is already clear

Important:

- do not let this section explode into a full benchmark matrix
- do not mix matched-parameter or matched-compute claims into this section
- do not mix iteration-count sweeps into this section; those belong in Part D
- where possible, pair performance changes with at least one signal-quality metric from Part B

Required outputs:

- final task performance
- delta relative to the full proposed model
- same-seed comparison across ablations
- at least one mechanism-side measurement when possible

Optional build-up sequence:

- MLP baseline
- 1-step SwiGLU block
- + pre-LN
- + LayerScale
- + Step embedding
- + iteration

This optional build-up is useful if the leave-one-out ablations are noisy or if a more constructive design story is needed.

### D. Iteration-count study

Purpose:

- support the claim that iterative refinement behaves like a controllable sharpening process, and probably show that there is no single sweetspot for all kind of environment using all kind of algorithm.

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

Core idea:

- hold the algorithm, task, actor, batch size, and train-step budget fixed
- change only the critic-side architecture
- compare the frozen proposed critic against MLP baselines that match the changed module parameter count as closely as possible

Compare:

- full proposed critic
- deeper MLP matched by parameter count
- wider MLP matched by parameter count
- optional standard MLP baseline as an anchor row

Matching rule:

- match the changed module rather than only total model size
- for `CRL`, match `params/critic_count`
- for `HIQL`, match `params/value_count`
- use the real parameter count from the instantiated model, not a hand-derived formula

Recommended scope:

- start with one primary algorithm, preferably `CRL`
- start with one representative hard task, preferably `antmaze-large-stitch-v0`
- then add `antmaze-giant-navigate-v0`
- extend to `HIQL` only after the protocol is stable

Implementation note:

- use the existing parameter-count search utility to find close width-matched and depth-matched MLP baselines
- if input dimensions differ across task families, recompute the match rather than assuming one match transfers automatically

Important:

- use at least one MLP match that is very close to the target module count
- report the exact matched counts in the paper table
- use the same seeds and train-step budget across the matched models
- ideally include both one depth-matched and one width-matched MLP
- a small scaling curve is a nice upgrade, but one clean matched point is acceptable for the first pass

Required outputs:

- final task performance across seeds
- exact module parameter counts for every compared model
- delta relative to the full proposed model
- one concise table showing fairness and outcome together

### F. Matched-compute comparison

Purpose:

- show the gain is not just from spending more computation

Core idea:

- tied recurrence reuses the same parameters multiple times per update, so parameter count alone is not enough
- this section should compare the proposed critic against MLP baselines with similar empirical training cost per update

Compare:

- full proposed critic
- deeper MLP matched by empirical training cost
- wider MLP matched by empirical training cost

Recommended fairness protocol:

- run short pilot jobs on the same hardware, with the same algorithm, task, batch size, and logging cadence
- ignore warmup and use a stable post-JIT window
- measure empirical per-update cost using `time/step_time`
- choose MLP baselines whose median per-update cost is close to the proposed model
- after the pilot matching is fixed, run the full training comparison at the same train-step budget

Important:

- do not use wall-clock alone as the primary fairness metric
- do not rely only on parameter-update proxies for tied recurrent models
- current `compute/critic_param_updates` and related metrics are useful bookkeeping, but they do not fully capture repeated application of tied parameters inside one update
- use wall-clock as a secondary practical-efficiency result, not as the only fairness story

Recommended scope:

- start with `CRL` on `antmaze-large-stitch-v0`
- then add `antmaze-giant-navigate-v0`
- only expand to additional algorithms after the pilot protocol is stable

Primary fairness measurements should include:

- empirical per-update runtime from the training loop
- same number of updates
- same hardware type and batch size

Secondary measurements may include:

- wall-clock to threshold
- wall-clock to final score
- parameter count
- parameter-update and parameter-sample proxies
- estimated critic FLOPs if they can later be measured reliably

Required outputs:

- pilot calibration table showing the matched compute baselines
- final task performance across seeds
- practical-efficiency curves or table using wall-clock as a secondary view

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
