#!/usr/bin/env python3
"""Test partial tying with cycling support.

This is a quick smoke-test to catch common implementation mistakes:
- K_test > K_train should run (cycling works)
- K_test > max_iters should error
- Sinusoidal step encoding should run for K_test > K_train without introducing new parameters
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/ogbench/impls'))

import jax
import jax.numpy as jnp
from utils.networks import PartiallyTiedBackbone

def test_partial_cycling():
    """Test that partial tying can cycle beyond total_iters."""
    print("=" * 60)
    print("Test: Partial Tying with Cycling")
    print("=" * 60)

    batch_size = 2
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    max_iters = 16

    # Create 2×4 partial tying model (discrete step embeddings, cycled step params).
    model = PartiallyTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_groups=2,
        iters_per_group=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=False,
        cycle_step_params=True,
    )

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, input_dim))
    params = model.init(key, x)

    print(f"\nModel configuration:")
    print(f"  Groups: 2")
    print(f"  Iters per group: 4")
    print(f"  Total iters (training): 2 × 4 = 8")
    print(f"  Max iters: {max_iters}")

    # Test K=8 (within total_iters)
    print(f"\n✓ Test K=8 (no cycling):")
    out_8 = model.apply(params, x, num_iters=8)
    print(f"  Output shape: {out_8.shape}")
    print(f"  Output finite: {jnp.all(jnp.isfinite(out_8))}")

    # Test K=12 (requires cycling: 8 base + 4 extra)
    print(f"\n✓ Test K=12 (cycling!):")
    try:
        out_12 = model.apply(params, x, num_iters=12)
        print(f"  Output shape: {out_12.shape}")
        print(f"  Output finite: {jnp.all(jnp.isfinite(out_12))}")
        print(f"  ✅ CYCLING WORKS!")
    except ValueError as e:
        print(f"  ❌ FAILED: {e}")
        return False

    # Test K=16 (requires cycling: 8 base + 8 extra, full cycle)
    print(f"\n✓ Test K=16 (full double cycle!):")
    try:
        out_16 = model.apply(params, x, num_iters=16)
        print(f"  Output shape: {out_16.shape}")
        print(f"  Output finite: {jnp.all(jnp.isfinite(out_16))}")
        print(f"  ✅ FULL CYCLING WORKS!")
    except ValueError as e:
        print(f"  ❌ FAILED: {e}")
        return False

    # Test beyond max_iters (should fail)
    print(f"\n✓ Test K=20 (exceeds max_iters=16, should fail):")
    try:
        out_20 = model.apply(params, x, num_iters=20)
        print(f"  ❌ Should have failed but didn't!")
        return False
    except ValueError as e:
        print(f"  ✅ Correctly rejected: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nCycling pattern for 2×4 with K=12:")
    print("  K=0-3:   Group 0 (first pass)")
    print("  K=4-7:   Group 1 (first pass)")
    print("  K=8-11:  Group 0 (cycle back) ← Uses same group weights as K=0-3")
    print("\nNote: in discrete+cycled mode, per-step params are cycled,")
    print("so step conditioning repeats every K_train steps. Extra iterations")
    print("still change outputs because h changes over time.")

    # Also test sinusoidal step encoding (extrapolatable, no step_embed table).
    print(f"\n✓ Test sinusoidal encoding (K=12):")
    sin_model = PartiallyTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_groups=2,
        iters_per_group=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=True,
        cycle_step_params=True,
    )
    sin_params = sin_model.init(key, x)
    out_12_sin = sin_model.apply(sin_params, x, num_iters=12)
    print(f"  Output shape: {out_12_sin.shape}")
    print(f"  Output finite: {jnp.all(jnp.isfinite(out_12_sin))}")
    return True


if __name__ == '__main__':
    try:
        success = test_partial_cycling()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
