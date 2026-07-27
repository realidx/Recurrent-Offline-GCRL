# Meta Review of Submission13890 by Area Chair 31Hq
Metareview:
This paper proposes a new architecture (iterative latent refinement) for the value function in goal-conditioned RL, which iteratively refines a latent representation using recurrent updates. Experiments show that the architecture can be used as a drop-in replacement for three base RL algorithms (CRL, HIQL, SAW).

Reviewers found the results strong and surprising [wpus], praising for the paper for detailed baselines [Ym8U] (e.g., with equal number of parameters), analysis (e.g., what exactly is the iterative architecture doing) [wpus], and for clearly reporting the computational overhead [Ym8U].

The main reviewer feedback focused on comparisons with alternative critic architectures [fQ1u, wpus]. For example, reviewer [wpus] specifically mentioned BroNet and SimbaV2, two strong architectures that do not use iterative computation. There is also work on iterative computation architectures in RL, which I (AC) didn't see discussed. I feel the paper could be strengthened by discussing works such as the following:

Guez, Arthur, et al. "An investigation of model-free planning." ICML 2019
Tamar, Aviv, et al. "Value iteration networks." NeurIPS 2016.
Bush, Thomas, et al. "Interpreting emergent planning in model-free reinforcement learning."ICLR 2025.
Ghugare, Raj, et al. "On Computation and Reinforcement Learning." ICML 2026.
Reviewers also raised a questions about statistical significance, additional analysis, and hyperparameters. While also important to address, I think the main focus of the reviewers should be on additional baselines. If the proposed method compares favorably to these additional baselines, I expect the reviewers will reconsider their scores.

# Official Review of Submission13890 by Reviewer Ym8U
Summary:
The paper proposes replacing feedforward MLP value/critic backbones in offline goal-conditioned RL with an iterative latent-refinement module. The module applies shared recurrent update weights over K outer refinement steps, each containing m inner SwiGLU update blocks with step embeddings, LayerNorm, and LayerScale. It is used as a drop-in replacement in CRL, HIQL, and SAW while keeping losses, data, actors, goal sampling, optimizer, and evaluation protocol fixed. Across 21 reported algorithm-dataset rows, the paper reports a mean improvement of 10.7 percentage points, with the largest gains on stitching and bottleneck maze tasks. The paper also provides CRL-focused diagnostics, depth/capacity sweeps, matched-parameter feedforward controls, component ablations, and compute-overhead measurements.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

The matched-parameter feedforward control is a strong experiment. The authors compare the recurrent CRL critic against wider and deeper feedforward critics with nearly matched parameter counts, and neither feedforward alternative closes the gap. This directly addresses the obvious “it just has more parameters” explanation.

The method is tested across three algorithms with different value-to-policy interfaces: CRL’s contrastive score, HIQL’s hierarchical AWR-style value, and SAW’s subgoal-weighted value. This is more convincing than a single-algorithm architecture study.

The paper includes useful engineering analysis: depth sweeps show non-monotonic saturation, per-step-capacity sweeps show algorithm-dependent behavior, component ablations identify important architectural pieces, and compute overhead is transparently reported.

The paper is also candid that graph-distance alignment is not a complete explanation for the gains, since graph-distance correlations do not improve uniformly even when control improves.

Weaknesses

The largest concern is benchmark selection. The paper reports 21 algorithm-dataset rows from a subset of OGBench, but does not clearly state how this subset was selected or whether it was fixed in advance. The limitations section acknowledges that the study does not exhaustively cover the full OGBench suite, all algorithms, or all hyperparameter regimes. Without a selection protocol, the 10.7pp average is difficult to interpret as representative rather than potentially curated.

The proposed mechanism is not established across algorithms. The paper argues that recurrent refinement improves the actor-facing value/critic signal, but the detailed diagnostics are run only for CRL and only on three AntMaze tasks. HIQL and SAW, despite being part of the headline empirical claim, are not given comparable diagnostics.

Moreover, the capacity sweep suggests that CRL and HIQL respond differently to the same architectural change: increasing m helps HIQL but hurts or does not help CRL. The paper acknowledges that the cause is not isolated, but this undercuts the unified mechanism story.

The bottleneck-navigation explanation is suggestive but not systematic. Figure 7 provides an illustrative trajectory example, and the authors explicitly state that a single trajectory example is only illustrative. This should not be used as strong evidence for a general bottleneck mechanism.

The paper reports means and standard deviations, but no formal paired or matched-seed significance tests. This matters because the aggregate mixes very large effects with marginal or flat rows, such as cube-style settings where improvements are small relative to variance.

Finally, the manipulation evidence is mixed. Scene shows meaningful gains for some algorithms, but Cube gains are small or flat. This weakens any broad framing that recurrent refinement reliably improves offline GCRL beyond the maze/stitching settings.

Quality: 2: not good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
What was the selection protocol for the 21 reported algorithm-dataset rows? Were any evaluated rows excluded from Table 1?

Can the authors provide the same CRL-style diagnostics for HIQL and SAW?

Given that increasing m helps HIQL but hurts or does not help CRL, what is the evidence for a single shared mechanism?

Can the authors report paired or matched-seed significance tests for marginal rows?

Do matched-parameter and matched-wall-clock feedforward controls still fail outside CRL AntMaze, for example on HIQL, SAW, Cube, or Scene?

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
none

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

## Rebuttal



# Official Review of Submission13890 by Reviewer wpus
Summary:
The paper proposes an alternative architecture for the critic in GCRL: iterative latent refinement. The paper shows that several algorithms can increase performance by solely changing to the proposed iterative recurrent architecture.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:

The topic the paper studies is very interesting, and the results are surprising: just by using a recurrent and iterative architecture, several RL algorithms can directly improve in performance. This is interesting and highly relevant for RL algorithms research.
To my knowledge, such iterative architecture is novel.
The authors also provide a nice analysis on what the iterative architecture is doing.
Weakness:

[Main concern] The main contribution of this paper is a novel architecture for value/critic learning. However, there are no comparisons against other recent critic architectures that aims to improve critic learning, such as BroNet [1] and SimbaV2 [2]. It is known in the community that the critic architecture can impact performance [3], and so this finding alone is not novel, and the authors should compare against existing architecture approaches that aim to improve critic learning.

Table 1 mentions that baselines numbers are taken directly from reference papers. Is there any difficulties in running those baselines in the same exact setting? In general I am skeptical about directly taking numbers from papers since there can be very small technical details that matters a lot in terms of performance. These experiments should be quick and cheap to run.

Ablation experiments (other than Table 1) are only ran on simple antmaze experiments, and it’s not quite convincing enough. It would be nice to also add manipulation environments from OGBench.

[1] Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control
[2] Hyperspherical Normalization for Scalable Deep Reinforcement Learning
[3] Compute-Optimal Scaling for Value-Based Deep RL

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
Is your findings on the recurrent architecture specific to offline GCRL specifically, or is it more applicable to critic learning in RL in general (e.g. single task RL, online RL)? See my other questions in the weakness section.

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
n/a

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

## Rebuttal
We thank the reviewer for the constructive feedback and for highlighting the importance of critic architecture. We agree both that the original submission did not compare against enough modern critic backbones. We are therefore adding a controlled CRL architecture comparison in which every row uses the same local implementation, dataset, contrastive objective, actor, goal sampling, optimizer, number of updates, and evaluation protocol. We want to clarify again that our method is simple and can be apply directly to many of existing algorithm, because we simply replace the critic/value backbone with a iterative one. Our method can be complement to many of others.

> Comparison with BroNet and SimBaV2

For BroNet, we transplant the normalized residual critic backbone into both \(\phi\) and \(\psi\). We intentionally do not add the other components of the full BRO algorithm—such as parameter resets, distributional critics, weight decay, increased replay ratio, or optimistic exploration because these would change the optimization or online-exploration procedure rather than only the critic architecture.

For SimBa, we use its pre-LayerNorm residual feedforward blocks in both CRL representation networks while leaving CRL’s inputs and optimizer unchanged. We label this row “SimBa backbone.” We do not present it as a full SimBaV2 implementation. Full SimBaV2 combines hyperspherical weight and feature normalization with distributional value prediction and reward scaling. Applying the full SimBaV2 output and value-learning design would therefore change score geometry and objective, rather than constitute only a backbone replacement. We use SimBa as the clean residual architecture control and will state this distinction and limitation explicitly.

We additionally include IRU-4 from the recent work on computation in RL. IRU is the closest alternative recurrent architecture: it repeatedly applies a shared interpolation recurrent unit. Our adaptation preserves its recurrent normalization, forget-gate initialization, zero initial state, interpolation update, residual connection, and four shared computation steps. [TODO] (need to discuss On the Role of Computation in a deeper way)

| Critic backbone                          | AMS | ALS | Scene | Params | Time/update |
| ---------------------------------------- | --: | --: | ----: | -----: | ----------: |
| Local MLP                                |     |     |       |        |             |
| BroNet backbone                          |     |     |       |        |             |
| SimBa backbone                           |     |     |       |        |             |
| IRU-4                                    |     |     |       |        |             |
| Ours (\(K=4,m=1\), tied)                 |     |     |       |        |             |

The architectures have different parameter counts and computational costs, so we report both rather than implying an exact capacity match. Several alternatives have more parameters than our tied model; if the smaller tied model performs better, this is a conservative comparison with respect to parameter count.

> Why were the original baseline numbers taken from prior papers?

Table 1 followed the benchmark convention of using official OGBench/SAW reference results for broad coverage. We agree that architecture comparisons require a same-code anchor, so all newly added backbone comparisons include a locally rerun MLP with matched seeds. We will clearly separate the broad published-reference table from the locally controlled architecture table.

> Ablations on manipulation tasks

We agree that the original diagnostic and ablation evidence was too concentrated on AntMaze. In addition to including Scene in the architecture table above, we are adding the following CRL ablation on `scene-play-v0`:

| CRL critic on Scene                       | Purpose                              | Success | Params | Time/update |
| ----------------------------------------- | ------------------------------------ | ------: | -----: | ----------: |
| Local MLP                                 | Local anchor                         |         |        |             |
| Tied recurrent, \(K=1,m=1\)              | New block without iterative depth    |         |        |             |
| Untied SwiGLU ResNet, \(K=4,m=1\)        | Sequential depth without sharing     |         |        |             |
| Tied recurrent, \(K=4,m=1\) (full model) | Iterative refinement with sharing    |         |        |             |

Together, these rows test manipulation-domain transfer, the effect of iterative depth beyond the \(K=1\) block, and the contribution of weight sharing at \(K=4\).

> Is this specific to offline GCRL, or applicable to critic learning generally?

Our empirical claim is currently specific to offline GCRL. The module is architecturally generic and could replace value or critic backbones in single-task or online RL, but we have not tested those settings and therefore do not claim broad generality.

**We appreciate the reviewer's acknowledgement of our contributions. We hope that our answers address all the reviewer's concerns, and we are happy to continue the discussion for any future concerns.**



# Official Review of Submission13890 by Reviewer fQ1u
Summary:
The paper is involved with offline goal-conditioned RL. In this variant of RL, the agent cannot interact with the environment to improve its performance. The objective of the goal state is, given an arbitrary goal state as input, to move to this goal state as quickly as possible. The paper considers actor-critic-based RL. They propose a new architecture of the critic, which, as they call it, uses iterative latent refinement. The idea is to process the input by multiple blocks placed after each other, which have the same structure and share the same weights. The authors have implemented this approach and applied it on a set of case studies from the standard case study set for this research area. Here, it performs quite well.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
The overall idea is interesting. Overall, the method seems well and the approach performs reasonably well in practice.

One problem is however that they only compare against results of the original paper introducing the research area, OGBench, not against better numbers from successor papers. For example, I think that for antmaze-giant-navigate-v0 better numbers exist in "Test-Time Graph Search for Goal-Conditioned Reinforcement Learning (Opryshko et al., 2025)." and also for the other case studies, the numbers they always but sometimes better than SOTA. Other papers they do not compare the numbers against are

García, C. V., Cazorla, M., & Pomares, J. (2026). Negative energy as reward: Optimizing beyond demonstrations in offline goal-conditioned control. ICRA Workshop on Reinforcement Learning in the Era of Imitation Learning.
Wang, Z., Li, D., Chen, Y., Shi, Y., Bai, L., Yu, T., & Fu, Y. (2026). One-step generative policies with Q-learning: A reformulation of MeanFlow. Proceedings of the AAAI Conference on Artificial Intelligence, 40(1), 1-9.
Xia, Y., & Sun, F. (2026). Behavior regularization with flow latent policy for offline reinforcement learning. Proceedings of the AAAI Conference on Artificial Intelligence, 40(32), 27028–27036.

While the relevant background literature is provided, the paper is hard to read for researchers outside the quite specific fields the authors work in. The discussion of what they target at doing is, while being formalised appropriately, hard to understand. Certain terms (compatibility score, stitching, not discussed whether GCRL is the same as CRL, ...) they use are not introduced appropriately.

The architecture for the critic which they introduce works quite well in practive, but its use is not very well motivated. See also the questions later on.

Quality: 3: good
Clarity: 2: not good
Significance: 3: good
Originality: 3: good
Questions:
Is CRL the same as GCRL? What is GCIVL?
Why is it better to use enforce using the same weights in all of the blocks?
What is the motivation for the iterative method? I see it being described, and it seems to work fine in the experiments, but what was the initial motivation, why do you think it works well?
Limitations:
yes

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
no

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes

## Rebuttal
We thank the reviewer for the constructive feedback and positive assessment. We agree that the relationship to recent OGBench results and several background concepts were not explained clearly enough. We will clarify the scope of our comparisons, define the terminology at first use, and strengthen the motivation and controls for the iterative architecture.

> Why not compare against the other recent methods?

The cited works make important but different changes to the learning or inference pipeline. Our main experiment isolates an architecture question: we replace only the value/critic backbone while holding the offline GCRL algorithm, data, actor, losses, goal sampling, optimizer, and evaluation protocol fixed.

TTGS adds graph search at test time; NEaR performs trajectory-level energy optimization; MeanFlowQL and Flow Latent Policy change the policy class and behavior regularization. These methods are relevant to absolute OGBench performance but do not isolate critic-backbone architecture under an otherwise fixed GCRL algorithm. We will cite them as complementary methods and clarify that Table 1 is not a global OGBench leaderboard. To strengthen the architecture claim, we are also locally rerunning the CRL MLP and architecture baselines under the same pipeline.

> Please explain the background and terminology. Is CRL the same as GCRL? What is GCIVL?

We apologize that these distinctions were not introduced clearly enough. We will define these terms, as well as “compatibility score” and “trajectory stitching,” at first use and add an intuitive overview before the formal objectives.

Goal-conditioned reinforcement learning (GCRL) is the problem setting: it learns a policy conditioned on a desired goal. Contrastive RL (CRL) is one particular algorithm for this setting. CRL learns a contrastive state-action-goal compatibility score and trains its policy to select actions with high compatibility. Our method replaces the network producing this learned signal with an iterative-refinement backbone; it does not change CRL’s objective, actor, or data. A compatibility score \(f(s,a,g)\) is CRL’s learned scalar indicating how compatible taking action \(a\) at state \(s\) is with subsequently reaching goal \(g\). It is learned contrastively and is not necessarily a calibrated Q-value.

GCIVL denotes goal-conditioned implicit value learning. It is an offline GCRL method that learns an action-free goal-conditioned value function and uses value-derived advantages for policy extraction. In our paper, it appears in the background and provides the value-learning foundation used by related methods such as SAW.

Trajectory stitching means combining useful segments from different offline trajectories to reach a goal even when the dataset contains no complete demonstrated trajectory from the initial state to that goal. OGBench includes dedicated stitching datasets to test this ability.

> Why enforce the same weights in all blocks? What was the initial motivation for the iterative method?

| Critic backbone                      | AMS | ALS | Scene | Params | Time/update |
| ------------------------------------ | --: | --: | ----: | -----: | ----------: |
| Local MLP                            |     |     |       |        |             |
| Untied SwiGLU ResNet (\(K=4,m=1\))   |     |     |       |        |             |
| Ours (\(K=4,m=1\), tied)             |     |     |       |        |             |

Our initial motivation comes from the representational burden placed on an offline goal-conditioned critic. From fixed, goal-agnostic trajectories, it must learn a reachability or compatibility signal spanning long horizons and disconnected trajectory fragments, and that signal must then support behavior-constrained policy extraction. Prior offline CRL results found that conventional feedforward depth scaling did not reliably improve OGBench performance.

We therefore considered a different allocation of computation. Starting from a projected latent \(h^{(0)}\), the architecture repeatedly applies a learned correction:

\[
h^{(k+1)} = h^{(k)} + \alpha_k \Delta_\theta(h^{(k)}, x, e_k),
\]

where \(\Delta_\theta\) is shared across refinement steps, \(e_k\) is a learned step embedding, and \(\alpha_k\) is a LayerScale parameter. This allows the representation to be revised several times before producing the final critic output. We interpret this as learned latent refinement.

We do not assume that weight sharing is universally better. We chose it because it allows \(K\) to increase sequential computation without adding an independent transformation at every step, and because it represents refinement as repeated application of a learned correction operator. Whether tying itself helps is an empirical question: our \(K=1\) comparison tests whether the modern update block alone is sufficient, while the new untied \(K=4,m=1\) control preserves the input/output projections, step embeddings, LayerNorm, LayerScale, and update topology but gives every step independent weights. This control tests whether the gain comes from tying or simply from sequential depth. We are evaluating these controls on AntMaze stitching and Scene manipulation tasks and will report parameter counts and time per update.

**We appreciate the reviewer's acknowledgement of our contributions. We hope that our answers address all the reviewer's concerns, and we are happy to continue the discussion for any future concerns.**
