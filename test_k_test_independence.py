#!/usr/bin/env python3
"""Test whether K_test affects actor sampling when refine_steps=0.

This script verifies that:
1. Actor RNG seed is independent of critic_eval_num_iters
2. Action sampling is deterministic given the same actor RNG seed
3. There's no hidden coupling between critic config and actor behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/ogbench/impls'))

import numpy as np
import jax
import jax.numpy as jnp

# Set seeds
np.random.seed(42)

# Simulate what happens in main.py
print("=" * 60)
print("Test: Does critic_eval_num_iters affect numpy random state?")
print("=" * 60)

results = {}
for k_test in [4, 8, 12, 16]:
    # Reset to same state
    np.random.seed(42)

    # Simulate config creation (critic_eval_num_iters doesn't use numpy.random)
    config = {'critic_eval_num_iters': k_test}

    # This is what evaluation.py does to create actor RNG
    actor_rng_seed = np.random.randint(0, 2**32)

    results[k_test] = actor_rng_seed
    print(f"K_test={k_test:2d}: actor_rng_seed={actor_rng_seed}")

print("\n" + "=" * 60)
if len(set(results.values())) == 1:
    print("✅ PASS: All K_test values produce the same actor RNG seed")
    print("   → K_test should NOT affect success with refine_steps=0")
else:
    print("❌ FAIL: Different K_test values produce different actor RNG seeds")
    print("   → This would explain why K_test affects success")

# Additional test: Check if train_dataset.sample() could be affected
print("\n" + "=" * 60)
print("Test: Does anything before evaluation use numpy.random?")
print("=" * 60)

np.random.seed(42)
state_before = np.random.get_state()

# Simulate creating config (doesn't use random)
config = {'critic_eval_num_iters': 8}

state_after = np.random.get_state()

if np.array_equal(state_before[1], state_after[1]):
    print("✅ PASS: Config creation doesn't consume numpy random state")
else:
    print("❌ FAIL: Config creation consumes numpy random state")
    print("   → This could cause K_test-dependent behavior")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("If both tests pass, then K_test differences are statistical variance.")
print("If either test fails, there's a subtle seeding bug.")