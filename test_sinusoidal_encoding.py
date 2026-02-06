#!/usr/bin/env python3
"""
Quick smoke test for sinusoidal step encoding.

Usage:
    python test_sinusoidal_encoding.py

Verifies:
1. Sinusoidal encoding function works
2. RecurTiedBackbone can use sinusoidal encoding
3. Extrapolation to K=8 doesn't crash (even if trained at K=4)
"""

import sys
import os

# Add ogbench impls to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/ogbench/impls'))

import jax
import jax.numpy as jnp
import flax.linen as nn
from utils.positional_encoding import sinusoidal_step_encoding
from utils.networks import RecurTiedBackbone


def test_sinusoidal_encoding():
    """Test basic sinusoidal encoding function."""
    print("=" * 60)
    print("Test 1: Sinusoidal Encoding Function")
    print("=" * 60)

    hidden_dim = 512
    max_iters = 16

    # Compute embeddings for k=4 and k=8
    emb_4 = sinusoidal_step_encoding(4, max_iters, hidden_dim)
    emb_8 = sinusoidal_step_encoding(8, max_iters, hidden_dim)

    print(f"✓ Embedding for k=4: shape={emb_4.shape}, norm={jnp.linalg.norm(emb_4):.4f}")
    print(f"✓ Embedding for k=8: shape={emb_8.shape}, norm={jnp.linalg.norm(emb_8):.4f}")

    # Check smoothness: embeddings for nearby k should be similar
    emb_7 = sinusoidal_step_encoding(7, max_iters, hidden_dim)
    cosine_sim = jnp.dot(emb_7, emb_8) / (jnp.linalg.norm(emb_7) * jnp.linalg.norm(emb_8))
    print(f"✓ Cosine similarity(k=7, k=8): {cosine_sim:.4f} (should be > 0.9 for smoothness)")

    # Check extrapolation beyond max_iters
    emb_20 = sinusoidal_step_encoding(20, max_iters, hidden_dim)
    print(f"✓ Extrapolation k=20 (beyond max_iters=16): shape={emb_20.shape}, norm={jnp.linalg.norm(emb_20):.4f}")

    assert emb_4.shape == (hidden_dim,), f"Wrong shape: {emb_4.shape}"
    assert jnp.all(jnp.isfinite(emb_4)), "Non-finite values in embedding"
    assert cosine_sim > 0.9, f"Poor smoothness: {cosine_sim:.4f}"

    print("✅ Sinusoidal encoding test PASSED\n")


def test_recur_tied_discrete():
    """Test RecurTiedBackbone with discrete embeddings (baseline)."""
    print("=" * 60)
    print("Test 2: RecurTiedBackbone (Discrete Mode)")
    print("=" * 60)

    batch_size = 2
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    max_iters = 16

    # Create model with discrete embeddings
    model = RecurTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_iters=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=False,  # Discrete
    )

    # Initialize
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, input_dim))
    params = model.init(key, x)

    # Forward pass at K=4 (training)
    out_4 = model.apply(params, x, num_iters=4)
    print(f"✓ Forward pass K=4: output shape={out_4.shape}")

    # Try K=8 (should work, uses step_embed[4:7])
    out_8 = model.apply(params, x, num_iters=8)
    print(f"✓ Forward pass K=8: output shape={out_8.shape}")

    # But step_embed[4:7] are untrained (random), so outputs differ a lot
    diff = jnp.linalg.norm(out_8 - out_4) / jnp.linalg.norm(out_4)
    print(f"✓ Relative difference K=8 vs K=4: {diff:.4f} (high → untrained iterations)")

    assert out_4.shape == (batch_size, output_dim), f"Wrong output shape: {out_4.shape}"
    assert jnp.all(jnp.isfinite(out_4)), "Non-finite outputs"

    print("✅ Discrete mode test PASSED\n")


def test_recur_tied_sinusoidal():
    """Test RecurTiedBackbone with sinusoidal encoding."""
    print("=" * 60)
    print("Test 3: RecurTiedBackbone (Sinusoidal Mode)")
    print("=" * 60)

    batch_size = 2
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    max_iters = 16

    # Create model with sinusoidal encoding
    model = RecurTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_iters=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=True,  # Sinusoidal
    )

    # Initialize
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, input_dim))
    params = model.init(key, x)

    # Forward pass at K=4 (training)
    out_4 = model.apply(params, x, num_iters=4)
    print(f"✓ Forward pass K=4: output shape={out_4.shape}")

    # Extrapolate to K=8 (should work smoothly)
    out_8 = model.apply(params, x, num_iters=8)
    print(f"✓ Forward pass K=8 (EXTRAPOLATION): output shape={out_8.shape}")

    # With sinusoidal encoding, K=8 should be smoother extension of K=4
    diff = jnp.linalg.norm(out_8 - out_4) / jnp.linalg.norm(out_4)
    print(f"✓ Relative difference K=8 vs K=4: {diff:.4f} (should be smooth extrapolation)")

    # Test aggressive extrapolation K=16
    out_16 = model.apply(params, x, num_iters=16)
    print(f"✓ Forward pass K=16 (2× training): output shape={out_16.shape}")

    assert out_4.shape == (batch_size, output_dim), f"Wrong output shape: {out_4.shape}"
    assert out_8.shape == (batch_size, output_dim), f"Wrong output shape: {out_8.shape}"
    assert jnp.all(jnp.isfinite(out_8)), "Non-finite outputs at K=8"
    assert jnp.all(jnp.isfinite(out_16)), "Non-finite outputs at K=16"

    print("✅ Sinusoidal mode test PASSED\n")


def test_parameter_count():
    """Compare parameter counts: discrete vs sinusoidal."""
    print("=" * 60)
    print("Test 4: Parameter Count Comparison")
    print("=" * 60)

    input_dim = 64
    hidden_dim = 512
    output_dim = 256
    max_iters = 16

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, input_dim))

    # Discrete model
    model_discrete = RecurTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_iters=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=False,
    )
    params_discrete = model_discrete.init(key, x)

    # Sinusoidal model
    model_sinusoidal = RecurTiedBackbone(
        hidden_dim=hidden_dim,
        out_dim=output_dim,
        num_iters=4,
        max_iters=max_iters,
        use_sinusoidal_step_encoding=True,
    )
    params_sinusoidal = model_sinusoidal.init(key, x)

    # Count parameters
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    n_discrete = count_params(params_discrete)
    n_sinusoidal = count_params(params_sinusoidal)

    # Expected difference: step_embed is (max_iters, hidden_dim) = 16 × 512 = 8,192 params
    expected_diff = max_iters * hidden_dim
    actual_diff = n_discrete - n_sinusoidal

    print(f"✓ Discrete params:    {n_discrete:,}")
    print(f"✓ Sinusoidal params:  {n_sinusoidal:,}")
    print(f"✓ Difference:         {actual_diff:,} (expected: {expected_diff:,})")
    print(f"✓ Sinusoidal uses {100 * n_sinusoidal / n_discrete:.1f}% of discrete params")

    assert actual_diff == expected_diff, f"Unexpected param diff: {actual_diff} vs {expected_diff}"

    print("✅ Parameter count test PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("SINUSOIDAL STEP ENCODING SMOKE TESTS")
    print("=" * 60 + "\n")

    try:
        test_sinusoidal_encoding()
        test_recur_tied_discrete()
        test_recur_tied_sinusoidal()
        test_parameter_count()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Submit training: sbatch slurm/phase3_train_sinusoidal_k4.slurm")
        print("2. Wait ~9.5 hours for training to complete")
        print("3. Submit eval: sbatch slurm/phase3_eval_sinusoidal_k4_to_k8.slurm")
        print("4. Check results: compare K=4 vs K=8 success rates")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
