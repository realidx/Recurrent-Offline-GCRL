#!/usr/bin/env python3
"""Find width- and depth-matched MLP baselines for a recurrent target config.

This script instantiates the real agent implementation and reads the real module
parameter count, rather than relying on a hand-written formula. It is intended
to help build fair MLP baselines against recurrent critics / value networks.

Example:
  conda run -n recurrent python scripts/find_mlp_param_match.py \
    --agent crl \
    --env-name antmaze-large-stitch-v0 \
    --set critic_backbone=recur_tied \
    --set critic_recur_iters=4 \
    --set critic_recur_num_dense_layers=2 \
    --set critic_recur_block_type=swiglu
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMPL_DIR = ROOT / "third_party" / "ogbench" / "impls"
OGBENCH_DIR = ROOT / "third_party" / "ogbench"

for path in (str(IMPL_DIR), str(OGBENCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

AGENT_SPECS = {
    "crl": {
        "backbone_key": "critic_backbone",
        "hidden_dims_key": "value_hidden_dims",
        "primary_module": "critic",
        "slurm_prefix": "CRITIC",
    },
    "qrl": {
        "backbone_key": "critic_backbone",
        "hidden_dims_key": "value_hidden_dims",
        "primary_module": "value",
        "slurm_prefix": "CRITIC",
    },
    "hiql": {
        "backbone_key": "value_backbone",
        "hidden_dims_key": "value_hidden_dims",
        "primary_module": "value",
        "slurm_prefix": "VALUE",
    },
}

INTERESTING_MODULES = (
    "critic",
    "value",
    "target_value",
    "actor",
    "low_actor",
    "high_actor",
    "goal_rep",
)


@dataclass
class Candidate:
    kind: str
    hidden_dims: tuple[int, ...]
    module_count: int
    diff: int


def _parse_scalar_or_dims(raw: str):
    raw = raw.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "x" in raw and all(tok.isdigit() for tok in raw.split("x")):
        return tuple(int(tok) for tok in raw.split("x"))
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_override(spec: str):
    if "=" not in spec:
        raise ValueError(f"Override must use key=value format, got: {spec!r}")
    key, value = spec.split("=", 1)
    return key.strip(), _parse_scalar_or_dims(value)


def _count_params(param_tree) -> int:
    import jax

    if param_tree is None:
        return 0
    total = 0
    for leaf in jax.tree_util.tree_leaves(param_tree):
        total += int(np.asarray(leaf).size)
    return total


def _extract_module_params(params, module_name: str):
    if not isinstance(params, dict):
        return None
    if module_name in params:
        return params[module_name]
    for key in (f"modules_{module_name}", f"module_{module_name}", module_name):
        if key in params:
            return params[key]
    for key, value in params.items():
        if module_name in str(key):
            return value
    return None


def _format_dims_x(hidden_dims: tuple[int, ...]) -> str:
    return "x".join(str(x) for x in hidden_dims)


def _format_dims_py(hidden_dims: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in hidden_dims) + ")"


def _load_agent_config(agent_name: str) -> dict:
    module = importlib.import_module(f"agents.{agent_name}")
    config = dict(module.get_config())
    return config


def _apply_overrides(config: dict, overrides: list[tuple[str, object]]) -> dict:
    updated = dict(config)
    for key, value in overrides:
        updated[key] = value
    return updated


def _prepare_example_batch(env_name: str, config: dict, seed: int):
    from utils.env_utils import make_env_and_datasets
    from utils.datasets import GCDataset, HGCDataset, SHGCDataset

    env, train_raw, _ = make_env_and_datasets(
        env_name,
        frame_stack=config.get("frame_stack"),
        seed=seed,
    )
    dataset_classes = {
        "GCDataset": GCDataset,
        "HGCDataset": HGCDataset,
        "SHGCDataset": SHGCDataset,
    }
    dataset_class = dataset_classes[config["dataset_class"]]
    train_dataset = dataset_class(train_raw, config)
    example_batch = train_dataset.sample(1, evaluation=True)
    if config.get("discrete", False):
        example_batch["actions"] = np.full_like(example_batch["actions"], env.action_space.n - 1)
    return env, example_batch


def _build_agent(agent_name: str, config: dict, example_batch, seed: int):
    from agents import agents as agent_registry

    agent_class = agent_registry[agent_name]
    return agent_class.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        config,
    )


def _module_counts(params) -> dict[str, int]:
    return {name: _count_params(_extract_module_params(params, name)) for name in INTERESTING_MODULES}


def _evaluate_config(agent_name: str, config: dict, example_batch, seed: int) -> dict[str, int]:
    agent = _build_agent(agent_name, config, example_batch, seed)
    return _module_counts(agent.network.params)


def _search_depth_matches(
    agent_name: str,
    baseline_config: dict,
    example_batch,
    seed: int,
    hidden_dims_key: str,
    primary_module: str,
    target_count: int,
    *,
    width: int,
    min_depth: int,
    max_depth: int,
) -> list[Candidate]:
    results = []
    for depth in range(min_depth, max_depth + 1):
        candidate_dims = tuple([int(width)] * int(depth))
        cur_config = dict(baseline_config)
        cur_config[hidden_dims_key] = candidate_dims
        counts = _evaluate_config(agent_name, cur_config, example_batch, seed)
        module_count = int(counts[primary_module])
        results.append(
            Candidate(
                kind="depth",
                hidden_dims=candidate_dims,
                module_count=module_count,
                diff=abs(module_count - target_count),
            )
        )
    results.sort(key=lambda item: (item.diff, item.module_count))
    return results


def _search_width_matches(
    agent_name: str,
    baseline_config: dict,
    example_batch,
    seed: int,
    hidden_dims_key: str,
    primary_module: str,
    target_count: int,
    *,
    depth: int,
    min_width: int,
    max_width: int,
    width_step: int,
) -> list[Candidate]:
    results = []
    for width in range(min_width, max_width + 1, width_step):
        candidate_dims = tuple([int(width)] * int(depth))
        cur_config = dict(baseline_config)
        cur_config[hidden_dims_key] = candidate_dims
        counts = _evaluate_config(agent_name, cur_config, example_batch, seed)
        module_count = int(counts[primary_module])
        results.append(
            Candidate(
                kind="width",
                hidden_dims=candidate_dims,
                module_count=module_count,
                diff=abs(module_count - target_count),
            )
        )
    results.sort(key=lambda item: (item.diff, item.module_count))
    return results


def _default_width(depth_dims: tuple[int, ...]) -> int:
    if not depth_dims:
        return 512
    return int(depth_dims[0])


def _default_depth(depth_dims: tuple[int, ...]) -> int:
    if not depth_dims:
        return 3
    return int(len(depth_dims))


def _slurm_hint(agent_name: str, hidden_dims: tuple[int, ...]) -> str:
    spec = AGENT_SPECS[agent_name]
    dims_x = _format_dims_x(hidden_dims)
    if spec["slurm_prefix"] == "CRITIC":
        return (
            f"CRITIC_BACKBONE=mlp,"
            f"VALUE_HIDDEN_DIMS={dims_x},"
            f"VALUE_NUM_LAYERS={len(hidden_dims)},"
            f"VALUE_LAYER_WIDTH={hidden_dims[0]}"
        )
    return (
        f"VALUE_BACKBONE=mlp,"
        f"VALUE_HIDDEN_DIMS={dims_x},"
        f"VALUE_NUM_LAYERS={len(hidden_dims)},"
        f"VALUE_LAYER_WIDTH={hidden_dims[0]}"
    )


def _boundary_warning(kind: str, best: Candidate, *, min_depth: int, max_depth: int, min_width: int, max_width: int):
    if kind == "depth":
        depth = len(best.hidden_dims)
        if depth == min_depth or depth == max_depth:
            return f"warning: best depth match is on the search boundary ({depth}); expand --min-depth/--max-depth."
        return None
    width = int(best.hidden_dims[0])
    if width == min_width or width == max_width:
        return f"warning: best width match is on the search boundary ({width}); expand --min-width/--max-width."
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(AGENT_SPECS), required=True, help="Agent family to inspect.")
    parser.add_argument("--env-name", required=True, help="OGBench env / dataset name.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for model init and example-batch sampling.")
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Optional OGBench dataset dir. If set, exports OGBENCH_DATASET_DIR for this process.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in key=value form. Repeat as needed.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Optional explicit target module param count. If omitted, the target config is instantiated.",
    )
    parser.add_argument(
        "--primary-module",
        default=None,
        help="Override the module counted for matching. Default depends on --agent.",
    )
    parser.add_argument("--depth-width", type=int, default=None, help="Fixed width used for the depth-only search.")
    parser.add_argument("--width-depth", type=int, default=None, help="Fixed depth used for the width-only search.")
    parser.add_argument("--min-depth", type=int, default=1, help="Minimum layer count in the depth-only search.")
    parser.add_argument("--max-depth", type=int, default=24, help="Maximum layer count in the depth-only search.")
    parser.add_argument("--min-width", type=int, default=64, help="Minimum width in the width-only search.")
    parser.add_argument("--max-width", type=int, default=2048, help="Maximum width in the width-only search.")
    parser.add_argument("--width-step", type=int, default=16, help="Width grid spacing for the width-only search.")
    parser.add_argument("--top-k", type=int, default=5, help="How many best matches to print for each search.")
    args = parser.parse_args()

    if args.dataset_dir:
        os.environ["OGBENCH_DATASET_DIR"] = args.dataset_dir

    overrides = [_parse_override(spec) for spec in args.overrides]
    spec = AGENT_SPECS[args.agent]
    primary_module = args.primary_module or spec["primary_module"]
    hidden_dims_key = spec["hidden_dims_key"]
    backbone_key = spec["backbone_key"]

    base_config = _load_agent_config(args.agent)
    target_config = _apply_overrides(base_config, overrides)
    if hidden_dims_key not in target_config:
        raise KeyError(f"Missing hidden dims key {hidden_dims_key!r} in config.")
    target_dims = tuple(int(x) for x in target_config[hidden_dims_key])

    _, example_batch = _prepare_example_batch(args.env_name, target_config, args.seed)

    target_counts = None
    if args.target_count is None:
        target_counts = _evaluate_config(args.agent, target_config, example_batch, args.seed)
        target_count = int(target_counts[primary_module])
    else:
        target_count = int(args.target_count)

    baseline_config = dict(target_config)
    baseline_config[backbone_key] = "mlp"

    depth_width = int(args.depth_width or _default_width(target_dims))
    width_depth = int(args.width_depth or _default_depth(target_dims))

    depth_matches = _search_depth_matches(
        args.agent,
        baseline_config,
        example_batch,
        args.seed,
        hidden_dims_key,
        primary_module,
        target_count,
        width=depth_width,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    width_matches = _search_width_matches(
        args.agent,
        baseline_config,
        example_batch,
        args.seed,
        hidden_dims_key,
        primary_module,
        target_count,
        depth=width_depth,
        min_width=args.min_width,
        max_width=args.max_width,
        width_step=args.width_step,
    )

    print(f"agent={args.agent} env={args.env_name} primary_module={primary_module}")
    if target_counts is not None:
        print("target_counts:")
        for name, count in sorted(target_counts.items()):
            if count > 0:
                print(f"  {name}: {count}")
    print(f"target_module_count={target_count}")
    print(f"target_hidden_dims={_format_dims_py(target_dims)}")
    print(f"target_backbone={target_config.get(backbone_key)!r}")
    print()

    print(f"best_depth_matches fixed_width={depth_width}")
    for cand in depth_matches[: args.top_k]:
        print(
            f"  depth={len(cand.hidden_dims):2d} "
            f"dims={_format_dims_py(cand.hidden_dims):<140} "
            f"count={cand.module_count:<10d} diff={cand.diff:<10d}"
        )
    depth_warning = _boundary_warning(
        "depth",
        depth_matches[0],
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_width=args.min_width,
        max_width=args.max_width,
    )
    if depth_warning is not None:
        print(depth_warning)
    print(f"depth_best_slurm={_slurm_hint(args.agent, depth_matches[0].hidden_dims)}")
    print()

    print(f"best_width_matches fixed_depth={width_depth}")
    for cand in width_matches[: args.top_k]:
        print(
            f"  width={cand.hidden_dims[0]:4d} "
            f"dims={_format_dims_py(cand.hidden_dims):<140} "
            f"count={cand.module_count:<10d} diff={cand.diff:<10d}"
        )
    width_warning = _boundary_warning(
        "width",
        width_matches[0],
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_width=args.min_width,
        max_width=args.max_width,
    )
    if width_warning is not None:
        print(width_warning)
    print(f"width_best_slurm={_slurm_hint(args.agent, width_matches[0].hidden_dims)}")


if __name__ == "__main__":
    main()
