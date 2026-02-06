# Sinusoidal Step Encoding Implementation Summary

**Date:** 2026-02-06
**Status:** ✅ Implementation Complete, Ready for Testing

---

## 🎯 What Was Implemented

Incremental test of continuous positional encoding for test-time scaling:
- **Train:** K=4 with sinusoidal encoding (instead of discrete step_embed)
- **Test:** K=4 (parity check) and K=8 (2× extrapolation)
- **Goal:** Validate if continuous encoding can extrapolate smoothly

---

## 📝 Code Changes

### **1. New File: Sinusoidal Encoding Function**
**File:** `third_party/ogbench/impls/utils/positional_encoding.py`
- Implements Transformer-style sinusoidal positional encoding
- Frequency range: [1.0, 1e-4] for smooth extrapolation
- 0 learnable parameters (closed-form function)

### **2. Modified: RecurTiedBackbone**
**File:** `third_party/ogbench/impls/utils/networks.py`
- Added `use_sinusoidal_step_encoding` parameter
- Conditional logic: discrete lookup table OR sinusoidal encoding
- Backward compatible (default=False, uses discrete)

**Lines changed:**
- `RecurTiedBackbone` class: Added parameter (line ~104)
- `__call__` method: Conditional step_embed creation (lines ~114-125)
- Loop: Compute step contribution per iteration (lines ~152-160)

### **3. Modified: GCBilinearValue Network**
**File:** `third_party/ogbench/impls/utils/networks.py`
- Added `recur_sinusoidal` field (line ~474)
- Pass to RecurTiedBackbone constructor (lines ~509, ~519)

### **4. Modified: CRL Agent Config**
**File:** `third_party/ogbench/impls/agents/crl.py`
- Added `critic_recur_sinusoidal` config flag (line ~359)
- Pass to GCBilinearValue at 3 instantiation sites (lines ~243, ~262, ~284)

### **5. New: Training SLURM Script**
**File:** `slurm/phase3_train_sinusoidal_k4.slurm`
- 3 seeds (0-2), K_train=4, sinusoidal encoding
- Run group: `P3_Sinusoidal_Incremental`
- Exp names: `sd{000,001,002}_sin_k4_lrd1000000`

### **6. New: Evaluation SLURM Script**
**File:** `slurm/phase3_eval_sinusoidal_k4_to_k8.slurm`
- Test at K ∈ {4, 8} with refine_steps=10
- Sources checkpoints from training
- Run group: `P3_Sinusoidal_Eval_K4toK8`

### **7. New: Smoke Test Script**
**File:** `test_sinusoidal_encoding.py`
- Validates sinusoidal encoding function
- Tests RecurTiedBackbone in both modes
- Compares parameter counts
- **Run this before submitting to SLURM!**

---

## 🚀 Usage Instructions

### **Step 1: Run Smoke Tests (2 minutes)**

```bash
cd /Users/bruce/Recurrent-Offline-RL
python test_sinusoidal_encoding.py
```

**Expected output:**
```
✅ Sinusoidal encoding test PASSED
✅ Discrete mode test PASSED
✅ Sinusoidal mode test PASSED
✅ Parameter count test PASSED
✅ ALL TESTS PASSED!
```

**If tests fail:** Debug before submitting to SLURM.

---

### **Step 2: Submit Training (9.5 hours)**

```bash
sbatch slurm/phase3_train_sinusoidal_k4.slurm
```

**What happens:**
- Trains 3 seeds in parallel (array job 0-2)
- Each seed: K_train=4, sinusoidal encoding, LR decay
- Output: `exp/OGBench/P3_Sinusoidal_Incremental/sd{000,001,002}_sin_k4_lrd1000000/`
- Checkpoints saved every 200k steps

**Monitor progress:**
```bash
# Check SLURM logs
tail -f logs/phase_3/slurm-p3_sin_k4-*.out

# Check training metrics (if WandB offline)
cat exp/OGBench/P3_Sinusoidal_Incremental/sd000_sin_k4_lrd1000000/train.csv | tail -20
```

---

### **Step 3: Submit Evaluation (2 hours)**

**Wait for training to reach 1M steps, then:**

```bash
sbatch slurm/phase3_eval_sinusoidal_k4_to_k8.slurm
```

**What happens:**
- Tests 3 seeds × 2 K_test = 6 jobs
- K_test=4: Parity check (should match discrete baseline ~0.37)
- K_test=8: Extrapolation test (should be ≥90% of K=4 if it works)
- Refine_steps=10 (action refinement enabled)

**Collect results:**
```bash
# Extract mean success rates
for seed in 0 1 2; do
  for ktest in 4 8; do
    csv="exp/OGBench/P3_Sinusoidal_Eval_K4toK8/sd$(printf %03d ${seed})_sin_ktrain4_ktest${ktest}_ref10/eval.csv"
    if [[ -f "$csv" ]]; then
      success=$(grep "success" "$csv" | tail -1 | cut -d',' -f2)
      echo "Seed ${seed}, K_test=${ktest}: success=${success}"
    fi
  done
done
```

---

## 📊 Expected Results

### **Scenario 1: Success (Continuous Encoding Works)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete (baseline) | 4 | 4 | 0.37 | Reference |
| **Sinusoidal** | 4 | 4 | **0.35** | ~5% drop (acceptable) |
| **Sinusoidal** | 4 | 8 | **0.33** | ~6% drop → smooth extrapolation! |

**Conclusion:** ✅ Continuous encoding enables extrapolation → proceed with full Option 2

---

### **Scenario 2: Failure (Extrapolation Breaks)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete (baseline) | 4 | 4 | 0.37 | Reference |
| **Sinusoidal** | 4 | 4 | **0.30** | ~20% drop (undercapacity) |
| **Sinusoidal** | 4 | 8 | **0.15** | 50% drop → extrapolation fails |

**Conclusion:** ❌ Continuous encoding doesn't work → stick with Option 1 (anytime with discrete)

---

### **Scenario 3: Mixed (Needs Tuning)**

| Model | Train K | Test K | Mean Success | Interpretation |
|-------|---------|--------|--------------|----------------|
| Discrete (baseline) | 4 | 4 | 0.37 | Reference |
| **Sinusoidal** | 4 | 4 | **0.34** | ~8% drop (tolerable) |
| **Sinusoidal** | 4 | 8 | **0.28** | ~18% drop → partial extrapolation |

**Conclusion:** 🟡 Extrapolation works but needs improvement → tune frequencies or try learned MLP

---

## 🎯 Decision Criteria

### **Proceed with Full Option 2 if:**
- ✅ Sinusoidal @ K=4 ≥ 90% of Discrete @ K=4 (parity check)
- ✅ Sinusoidal @ K=8 ≥ 90% of Sinusoidal @ K=4 (extrapolation)
- ✅ Action refinement matters (gap between refine=0 and refine=10)

### **Fallback to Option 1 if:**
- ❌ Sinusoidal @ K=4 < 80% of Discrete @ K=4 (too much capacity loss)
- ❌ Sinusoidal @ K=8 < 80% of Sinusoidal @ K=4 (extrapolation fails)

---

## 🔧 Debugging Tips

### **If training crashes:**

**Error:** `TypeError: sinusoidal_step_encoding() takes 3 positional arguments but 4 were given`
- **Fix:** Check import statement in `networks.py` line ~157
- Make sure it's: `from utils.positional_encoding import sinusoidal_step_encoding`

**Error:** `ModuleNotFoundError: No module named 'utils.positional_encoding'`
- **Fix:** File was created in wrong location
- Should be: `third_party/ogbench/impls/utils/positional_encoding.py`

**Error:** `ValueError: hidden_dim must be even for sinusoidal encoding`
- **Fix:** Ensure `hidden_dim=512` (even number)
- Check config: `--agent.critic_backbone_hidden_dim`

---

### **If evaluation crashes:**

**Error:** `ValueError: num_iters=8 exceeds max_iters=16`
- **Fix:** This should NOT happen (K=8 < 16)
- Check that `critic_recur_max_iters=16` is set

**Error:** `IndexError: index 8 is out of bounds for axis 0 with size 16`
- **Fix:** Means discrete mode is being used instead of sinusoidal
- Verify `--agent.critic_recur_sinusoidal=1` is passed to eval command

---

### **If results are bad:**

**Symptom:** Sinusoidal @ K=4 << Discrete @ K=4 (>30% gap)
- **Diagnosis:** Undercapacity (0 params vs 8,192 params for step_embed)
- **Fix:** Try learned MLP encoder (Option 2B) instead of sinusoidal

**Symptom:** Sinusoidal @ K=8 crashes with NaN/Inf
- **Diagnosis:** Numerical instability in gradient ascent
- **Fix:** Check `refine_lr`, `refine_l2` in evaluation.py

**Symptom:** K_test has no effect (K=4 ≈ K=8)
- **Diagnosis:** Action refinement not enabled
- **Fix:** Verify `--eval_refine_steps=10` is passed

---

## 📚 Related Files

### **Documentation:**
- [TEST_TIME_SCALING_INCREMENTAL.md](TEST_TIME_SCALING_INCREMENTAL.md) — Full implementation plan
- [TEST_TIME_SCALING_PLAN.md](TEST_TIME_SCALING_PLAN.md) — Original plan (Option 1 & 2)
- [CONTINUOUS_STEP_ENCODING_ANALYSIS.md](CONTINUOUS_STEP_ENCODING_ANALYSIS.md) — Technical deep-dive

### **Code:**
- `utils/positional_encoding.py` — Sinusoidal encoding function
- `utils/networks.py` — RecurTiedBackbone with sinusoidal support
- `agents/crl.py` — Config flag plumbing

### **Scripts:**
- `slurm/phase3_train_sinusoidal_k4.slurm` — Training
- `slurm/phase3_eval_sinusoidal_k4_to_k8.slurm` — Evaluation
- `test_sinusoidal_encoding.py` — Smoke tests

---

## ✅ Pre-Submission Checklist

- [ ] Smoke tests pass: `python test_sinusoidal_encoding.py`
- [ ] Bootstrap completed: `./scripts/bootstrap_ogbench.sh`
- [ ] Conda env activated: `conda activate recurrent`
- [ ] Dataset downloaded: `ls .ogbench_data/antmaze*`
- [ ] Logs directory exists: `mkdir -p logs/phase_3`
- [ ] SLURM queue ready: `squeue -u $USER`

**If all checks pass → Ready to submit!**

```bash
sbatch slurm/phase3_train_sinusoidal_k4.slurm
```

---

**Questions or issues?** See [TEST_TIME_SCALING_INCREMENTAL.md](TEST_TIME_SCALING_INCREMENTAL.md) for detailed troubleshooting.
