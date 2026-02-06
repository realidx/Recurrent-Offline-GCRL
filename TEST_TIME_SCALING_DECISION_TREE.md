# Test-Time Scaling: Decision Tree & Quick Reference

**Quick answer:** Start with Option 1 (Anytime Prediction). Only move to Option 2 if you need K > 16.

---

## 🌳 Decision Tree

```
┌─────────────────────────────────────────────┐
│ Q: Do you need to test at K > 16?          │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
   NO                  YES
    │                   │
    ▼                   ▼
┌───────────────┐   ┌──────────────────────────┐
│  OPTION 1:    │   │ Q: Is K ≤ 20 enough?     │
│  Anytime      │   │ (slightly beyond 16)     │
│  Prediction   │   └──────────┬───────────────┘
│               │              │
│ • Train: K∈   │    ┌─────────┴─────────┐
│   [4,16]      │    │                   │
│ • Test: K∈    │   YES                 NO
│   [4,16]      │    │                   │
│ • Arch: Same  │    ▼                   ▼
│ • Time: 1.5d  │ ┌────────────┐  ┌────────────┐
│ • Risk: LOW   │ │ OPTION 2A: │  │ OPTION 2B: │
└───────────────┘ │ Sinusoidal │  │ Learned    │
                  │ Encoding   │  │ MLP        │
                  │            │  │ Encoding   │
                  │ • No params│  │ • 74k      │
                  │ • Smooth   │  │   params   │
                  │ • K ≤ 24   │  │ • K ≤ 32   │
                  │ • Time: 2w │  │ • Time: 3w │
                  │ • Risk: MED│  │ • Risk: HI │
                  └────────────┘  └────────────┘
```

---

## 📊 Quick Comparison Table

| Metric | Fixed-K=4 (Baseline) | Anytime (Opt 1) | Sinusoidal (Opt 2A) | Learned MLP (Opt 2B) |
|--------|---------------------|-----------------|---------------------|---------------------|
| **Architecture** | Discrete step_embed | Same | Continuous (sin/cos) | Continuous (MLP) |
| **Training K** | Fixed K=4 | Random [4,16] | Random [4,16] | Random [4,16] |
| **Test K max** | 4 | 16 | 24+ | 32+ |
| **Step_embed params** | 8,192 | 8,192 | **0** | **74,000** |
| **Comparable?** | — | ✅ Yes | ❌ No | ❌ No |
| **Code changes** | — | ~10 lines | ~200 lines | ~300 lines |
| **Implementation time** | — | 2 hours | 2 days | 3 days |
| **Training time** | 9.5h | 9.5h | 10h | 11h |
| **Total time to results** | — | 1.5 days | 2 weeks | 3 weeks |
| **Risk of failure** | — | 🟢 Low | 🟡 Medium | 🔴 High |
| **Performance @ K=4** | 0.37 | 0.37 | 0.35 ↓ | 0.36 ↓ |
| **Performance @ K=16** | N/A | 0.48 ↑ | 0.47 | 0.48 |
| **Performance @ K=24** | N/A | N/A | 0.49 | 0.45 ↓ |

---

## 🎯 Decision Criteria

### **Choose Option 1 (Anytime) if:**
- ✅ K ≤ 16 is sufficient for your domain
- ✅ You want fast results (1.5 days)
- ✅ You want a fair comparison (same architecture)
- ✅ You want to validate test-time scaling concept first

**Expected outcome:** Monotonic improvement K=4→8→12→16, success@K=16 ≈ 0.48

---

### **Choose Option 2A (Sinusoidal) if:**
- ✅ You need K ∈ [16, 24] (modest extrapolation)
- ✅ You want parameter efficiency (0 params for step encoding)
- ✅ Smoothness is a strong inductive bias for your task
- ⚠️ You accept 5-10% performance drop at K ≤ 16
- ⚠️ You have 2 weeks for experiments

**Expected outcome:** Stable performance up to K=24, but slightly worse than discrete at K=16

---

### **Choose Option 2B (Learned MLP) if:**
- ✅ You need K > 24 (aggressive extrapolation)
- ✅ You believe optimal dynamics are complex (non-sinusoidal)
- ⚠️ You accept 10-20% performance drop at K > 20
- ⚠️ You have 3 weeks + risk of failure
- ⚠️ You're comfortable with extensive hyperparameter tuning

**Expected outcome:** Uncertain — may work well or fail to extrapolate

---

## 💡 Key Insights

### **Why Discrete Embeddings Are Hard to Beat**

```
Discrete (Current):
┌─────┐  ┌─────┐  ┌─────┐      ┌──────┐
│ k=0 │→ │ k=1 │→ │ k=2 │ ... → │ k=15 │
└─────┘  └─────┘  └─────┘      └──────┘
Each iteration learns INDEPENDENTLY → maximum flexibility
```

```
Continuous (Proposed):
        ┌─────────────────────────┐
        │  Shared Function f(k)   │
        └────────┬────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
  k=0          k=1          k=2  ...
All iterations COUPLED through f → less flexible, but extrapolatable
```

**Trade-off:** Flexibility vs Extrapolation

---

### **The Comparability Problem**

```
Experiment Setup:
┌─────────────────┐
│ Baseline Model  │  ← Discrete, Fixed-K=4, trained 1M steps
└─────────────────┘
        vs
┌─────────────────┐
│ Anytime Model   │  ← Discrete, Random-K∈[4,16], trained 1M steps
└─────────────────┘
✅ FAIR: Only training protocol differs (same architecture)
```

```
Experiment Setup:
┌─────────────────┐
│ Baseline Model  │  ← Discrete, 8,192 params, Fixed-K=4
└─────────────────┘
        vs
┌─────────────────┐
│ Extrapolation   │  ← Continuous, 0 params (sinusoidal)
│ Model           │     OR 74k params (MLP)
└─────────────────┘
❌ UNFAIR: Different architecture + training + capacity
```

**Key point:** Option 1 is a fair A/B test. Option 2 is a different model class.

---

## 🔬 When Extrapolation Becomes Necessary

### **Scenario 1: Environment Demands It**
- **Example:** antmaze-ultra requires 50-step lookahead
- **Current limit:** max_iters=16 → can only plan 16 steps
- **Solution:** Train with K∈[8,16], extrapolate to K=50

### **Scenario 2: Computational Constraints**
- **Problem:** Training at K=32 takes 3× longer than K=16
- **Solution:** Train at K=16, extrapolate to K=32 at test time
- **Trade-off:** Faster training, but uncertain extrapolation quality

### **Scenario 3: Transfer Learning**
- **Train:** Short-horizon tasks (K=8)
- **Test:** Long-horizon tasks (K=32)
- **Goal:** Learn iteration dynamics that transfer across horizon lengths

**For AntMaze-large-stitch:** None of these scenarios apply → K=16 is sufficient.

---

## 📈 Expected Scaling Curves

### **Anytime Prediction (Option 1)**
```
Success Rate
   0.5 ┤                           ╭─
       │                      ╭────╯
   0.4 ┤               ╭──────╯
       │          ╭────╯
   0.3 ┤      ────╯
       │
   0.2 ┤
       └─────┬──────┬──────┬──────┬─> K_test
             4      8      12     16
```
**Ideal:** Monotonic increase (validates approach)

---

### **Extrapolation (Option 2A: Sinusoidal)**
```
Success Rate
   0.5 ┤                    ╭──────╮
       │               ╭────╯      ╰─ (plateau)
   0.4 ┤          ╭────╯
       │     ╭────╯
   0.3 ┤─────╯
       │
   0.2 ┤
       └──┬───┬───┬───┬───┬───┬──> K_test
          4   8  12  16  20  24
```
**Ideal:** Plateau at K≥16 (stable extrapolation)

---

### **Extrapolation (Option 2B: Learned MLP) — Risk**
```
Success Rate
   0.5 ┤              ╭──╮
       │         ╭────╯  │
   0.4 ┤    ╭────╯       ╰╮
       │────╯             ╰───╮
   0.3 ┤                      ╰─ (collapse)
       │
   0.2 ┤
       └──┬───┬───┬───┬───┬───┬──> K_test
          4   8  12  16  20  24
```
**Risk:** Extrapolation failure (MLP overfits to [0.25, 1.0])

---

## 🎬 Recommended Action Plan

### **Week 1: Anytime Prediction**
**Monday AM:** Implement 10-line change to `crl.py` (random-K sampling)
**Monday PM:** Submit 3-seed training jobs
**Tuesday-Wednesday:** Wait for training (9.5h × 3 seeds)
**Thursday:** Run eval sweep (K ∈ {4,8,12,16}, refine ∈ {0,10})
**Friday:** Analyze results, plot scaling curves

**Decision point:** Did K=16 beat K=4 by ≥20% relative improvement?
- **YES → Paper-worthy result:** "Test-time compute scaling in CRL"
- **NO → Debug:** Check if random-K training hurt K=4 performance

---

### **Week 2-3: (Optional) Extrapolation**
**Only proceed if Week 1 succeeded and you need K > 16**

**Monday Week 2:** Implement sinusoidal encoding (~200 lines)
**Tuesday-Wednesday:** Debug, ablate frequencies
**Thursday-Friday:** Train 3 seeds (10h each)
**Monday Week 3:** Eval at K ∈ {16,20,24}, compare to discrete baseline
**Tuesday-Wednesday:** Analyze extrapolation quality
**Thursday-Friday:** Write up results

---

## 📝 Success Metrics

### **Option 1 Success Criteria:**
1. ✅ **Monotonicity:** success(K=16) > success(K=12) > success(K=8) > success(K=4)
2. ✅ **Magnitude:** success(K=16) ≥ 0.45 (≥20% gain over K=4 baseline of 0.37)
3. ✅ **Refinement matters:** With refine=10, gap between K=4 and K=16 is large
4. ✅ **Without refinement:** K_test has minimal effect (validates that refinement is required)

**If all 4 met → strong contribution**

---

### **Option 2 Success Criteria:**
1. ✅ **Parity within range:** success_continuous(K=16) ≈ success_discrete(K=16) (within 5%)
2. ✅ **Smooth extrapolation:** success(K=20) ≈ success(K=16) (no catastrophic drop)
3. ✅ **Non-trivial gain:** success(K=24) > success(K=16) by ≥5% absolute
4. ✅ **Stability:** No NaN/Inf in training, eval doesn't crash

**If all 4 met → novel architectural contribution**
**If #1 fails → architecture is inferior, not worth pursuing**

---

## 🚦 Implementation Checklist

### **Before Starting:**
- [ ] Have you confirmed K ≤ 16 is insufficient for your domain?
  - If NO → Do Option 1 only
  - If YES → Proceed to Option 2

- [ ] Have you validated Option 1 works first?
  - If NO → Start with Option 1
  - If YES → Option 2 is lower risk

- [ ] Do you have 2-3 weeks of compute budget?
  - If NO → Stick with Option 1
  - If YES → Option 2 is feasible

---

**Last Updated:** 2026-02-06
**For Questions:** See [TEST_TIME_SCALING_PLAN.md](TEST_TIME_SCALING_PLAN.md) (detailed) or [CONTINUOUS_STEP_ENCODING_ANALYSIS.md](CONTINUOUS_STEP_ENCODING_ANALYSIS.md) (technical deep-dive)
