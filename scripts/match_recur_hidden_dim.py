#!/usr/bin/env python3
"""
Find a recurrent-tied critic hidden dim that matches the parameter count of an untied ResNet critic.

This uses a closed-form parameter count (no JAX needed) and reads observation/action dims from
the downloaded dataset `.npz`. This avoids failures on login nodes whose CPUs do not support
AVX (some prebuilt jaxlib wheels require AVX).

Example:
  OGBENCH_DATASET_DIR=/path/to/data \
    python scripts/match_recur_hidden_dim.py --dataset antmaze-large-stitch-v0 --resnet-depth 6 --recur-iters 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def dense_params(in_dim: int, out_dim: int, *, bias: bool = True) -> int:
    return int(in_dim) * int(out_dim) + (int(out_dim) if bias else 0)


def layer_norm_params(dim: int) -> int:
    # scale + bias
    return 2 * int(dim)


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


def resnet_backbone_params(*, in_dim: int, hidden_dim: int, latent_dim: int, depth: int, layer_norm: bool = True) -> int:
    """
    Match utils.networks.DeepResNetBackbone + ResNetBlock parameterization.
    """
    h = int(hidden_dim)
    d = int(latent_dim)
    depth = int(depth)

    total = 0
    total += dense_params(in_dim, h, bias=True)
    for _ in range(depth):
        if layer_norm:
            total += layer_norm_params(h)
        total += dense_params(h, h, bias=True)
        total += dense_params(h, h, bias=True)
        total += h  # layerscale vector
    total += dense_params(h, d, bias=True)
    return int(total)


def recur_tied_backbone_params(
    *,
    in_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_iters: int,
    max_iters: int = 16,
    layer_norm: bool = True,
) -> int:
    """
    Match utils.networks.RecurTiedBackbone parameterization.

    Note: LayerNorm is tied across iterations (single module instance), so LN params do not grow with K.
    """
    h = int(hidden_dim)
    d = int(latent_dim)
    k = int(num_iters)
    m = int(max_iters)

    total = 0
    total += dense_params(in_dim, h, bias=True)
    total += m * h  # step_embed
    total += m  # alpha (per-step scalar)
    total += dense_params(h, h, bias=True)  # film_fc1
    total += dense_params(h, 2 * h, bias=True)  # film_fc2
    total += dense_params(h, h, bias=True)  # tied_fc1
    total += dense_params(h, h, bias=True)  # tied_fc2
    if layer_norm:
        total += layer_norm_params(h)
    total += dense_params(h, d, bias=True)
    return int(total)


def bilinear_critic_params(
    *,
    obs_dim: int,
    act_dim: int,
    latent_dim: int,
    backbone: str,
    hidden_dim: int,
    resnet_depth: int = 6,
    recur_iters: int = 6,
    recur_max_iters: int = 16,
    layer_norm: bool = True,
    ensemble_size: int = 2,
) -> int:
    """
    Match utils.networks.GCBilinearValue when ensemble=True (ensemblize(..., 2)).
    Counts critic params only (phi + psi backbones), including ensemble duplication.
    """
    in_phi = int(obs_dim) + int(act_dim)
    in_psi = int(obs_dim)

    if backbone == "resnet":
        phi = resnet_backbone_params(
            in_dim=in_phi, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=resnet_depth, layer_norm=layer_norm
        )
        psi = resnet_backbone_params(
            in_dim=in_psi, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=resnet_depth, layer_norm=layer_norm
        )
    elif backbone == "recur_tied":
        phi = recur_tied_backbone_params(
            in_dim=in_phi,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_iters=recur_iters,
            max_iters=recur_max_iters,
            layer_norm=layer_norm,
        )
        psi = recur_tied_backbone_params(
            in_dim=in_psi,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_iters=recur_iters,
            max_iters=recur_max_iters,
            layer_norm=layer_norm,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return int(ensemble_size) * int(phi + psi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="antmaze-large-stitch-v0")
    parser.add_argument("--dataset-dir", default=os.environ.get("OGBENCH_DATASET_DIR", "~/.ogbench/data"))
    parser.add_argument("--resnet-depth", type=int, default=6)
    parser.add_argument("--recur-iters", type=int, default=6)
    parser.add_argument("--recur-max-iters", type=int, default=16)
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

    target = bilinear_critic_params(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=args.latent_dim,
        backbone="resnet",
        hidden_dim=args.hidden_dim,
        resnet_depth=args.resnet_depth,
        recur_iters=args.recur_iters,
        recur_max_iters=args.recur_max_iters,
    )

    recur_base = bilinear_critic_params(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=args.latent_dim,
        backbone="recur_tied",
        hidden_dim=args.hidden_dim,
        resnet_depth=args.resnet_depth,
        recur_iters=args.recur_iters,
        recur_max_iters=args.recur_max_iters,
    )

    lo = int(args.search_min // args.step) * args.step
    hi = int(args.search_max // args.step) * args.step

    best_h = None
    best_diff = None
    best_count = None
    for h in range(lo, hi + 1, int(args.step)):
        c = bilinear_critic_params(
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_dim=args.latent_dim,
            backbone="recur_tied",
            hidden_dim=h,
            resnet_depth=args.resnet_depth,
            recur_iters=args.recur_iters,
            recur_max_iters=args.recur_max_iters,
        )
        diff = abs(c - target)
        if best_diff is None or diff < best_diff:
            best_h, best_diff, best_count = h, diff, c

    print(f"dataset={args.dataset} obs_dim={obs_dim} act_dim={act_dim}")
    print(f"target: resnet depth={args.resnet_depth} hidden_dim={args.hidden_dim} params={target}")
    print(f"base: recur iters={args.recur_iters} hidden_dim={args.hidden_dim} params={recur_base} diff={abs(recur_base - target)}")
    print(
        f"match: recur iters={args.recur_iters} backbone_hidden_dim={best_h} params={best_count} diff={best_diff}"
    )
    print(f"export RECUR_MATCH_HIDDEN_DIM={best_h}")


if __name__ == "__main__":
    main()
