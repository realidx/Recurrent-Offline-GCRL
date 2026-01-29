#!/usr/bin/env python3
"""
Find a recurrent-tied critic hidden dim that matches the parameter count of an untied ResNet critic.

This uses the OGBench impls `GCBilinearValue` module directly (no MuJoCo needed) and reads
observation/action dims from the downloaded dataset `.npz`.

Example:
  OGBENCH_DATASET_DIR=/path/to/data \
    python scripts/match_recur_hidden_dim.py --dataset antmaze-large-stitch-v0 --resnet-depth 6 --recur-iters 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def count_params(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(np.prod(x.shape) for x in leaves))


def load_dims(dataset_dir: Path, dataset: str) -> Tuple[int, int, bool]:
    train_path = dataset_dir / f"{dataset}.npz"
    with np.load(train_path) as f:
        obs = f["observations"]
        acts = f["actions"]
    obs_dim = int(obs.shape[-1])
    # Actions may be int for discrete datasets.
    discrete = np.issubdtype(acts.dtype, np.integer) and acts.ndim == 1
    act_dim = int(acts.max() + 1) if discrete else int(acts.shape[-1])
    return obs_dim, act_dim, bool(discrete)


def build_critic_params(
    *,
    obs_dim: int,
    act_dim: int,
    hidden_dims=(512, 512, 512),
    latent_dim=512,
    layer_norm=True,
    ensemble=True,
    backbone: str,
    resnet_num_blocks: int = 6,
    recur_num_iters: int = 6,
    recur_max_iters: int = 32,
    layerscale_init: float = 1e-2,
    backbone_hidden_dim: int | None = None,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    impl_dir = repo_root / "third_party" / "ogbench" / "impls"
    os.environ.setdefault("PYTHONPATH", "")
    # Make sure `utils.networks` resolves like it does when running from impl_dir.
    import sys

    if str(impl_dir) not in sys.path:
        sys.path.insert(0, str(impl_dir))

    from utils.networks import GCBilinearValue  # noqa: E402

    critic = GCBilinearValue(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        layer_norm=layer_norm,
        ensemble=ensemble,
        value_exp=False,
        state_encoder=None,
        goal_encoder=None,
        backbone=backbone,
        resnet_num_blocks=resnet_num_blocks,
        recur_num_iters=recur_num_iters,
        recur_max_iters=recur_max_iters,
        layerscale_init=layerscale_init,
        backbone_hidden_dim=backbone_hidden_dim,
    )

    key = jax.random.PRNGKey(0)
    obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    goal = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    variables = critic.init(key, obs, goal, act)
    return variables["params"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", default=os.environ.get("OGBENCH_DATASET_DIR", "~/.ogbench/data"))
    parser.add_argument("--resnet-depth", type=int, default=6)
    parser.add_argument("--recur-iters", type=int, default=6)
    parser.add_argument("--recur-max-iters", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512, help="Base hidden dim (value_hidden_dims[-1]).")
    parser.add_argument("--search-min", type=int, default=256)
    parser.add_argument("--search-max", type=int, default=2048)
    parser.add_argument("--step", type=int, default=32, help="Search granularity for hidden dim.")
    args = parser.parse_args()

    dataset_dir = Path(os.path.expanduser(args.dataset_dir))
    obs_dim, act_dim, discrete = load_dims(dataset_dir, args.dataset)
    if discrete:
        raise SystemExit("This helper currently expects continuous actions (antmaze is continuous).")

    resnet_params = build_critic_params(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=(args.hidden_dim, args.hidden_dim, args.hidden_dim),
        latent_dim=args.latent_dim,
        backbone="resnet",
        resnet_num_blocks=args.resnet_depth,
    )
    target = count_params(resnet_params)

    lo = int(args.search_min // args.step) * args.step
    hi = int(args.search_max // args.step) * args.step

    best_h = None
    best_diff = None
    best_count = None
    for h in range(lo, hi + 1, int(args.step)):
        recur_params = build_critic_params(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=(args.hidden_dim, args.hidden_dim, args.hidden_dim),
            latent_dim=args.latent_dim,
            backbone="recur_tied",
            recur_num_iters=args.recur_iters,
            recur_max_iters=args.recur_max_iters,
            backbone_hidden_dim=h,
        )
        c = count_params(recur_params)
        diff = abs(c - target)
        if best_diff is None or diff < best_diff:
            best_h, best_diff, best_count = h, diff, c

    print(f"dataset={args.dataset} obs_dim={obs_dim} act_dim={act_dim}")
    print(f"target: resnet depth={args.resnet_depth} hidden_dim={args.hidden_dim} params={target}")
    print(
        f"match: recur iters={args.recur_iters} backbone_hidden_dim={best_h} params={best_count} diff={best_diff}"
    )
    print(f"export RECUR_MATCH_HIDDEN_DIM={best_h}")


if __name__ == "__main__":
    main()
