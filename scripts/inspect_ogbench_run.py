#!/usr/bin/env python3
"""
Inspect one or more OGBench run directories (folders containing flags.json/train.csv/eval.csv).

Examples:
  python scripts/inspect_ogbench_run.py exp/OGBench/P2_TrainOnly_Depth6/sd000_*
  python scripts/inspect_ogbench_run.py --csv exp/OGBench/P2_TrainOnly_Depth6/sd000_*
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _get_nested(d: Optional[Dict[str, Any]], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _infer_run_dir(p: Path) -> Optional[Path]:
    if p.is_dir():
        if (p / "flags.json").exists() or (p / "train.csv").exists() or (p / "eval.csv").exists():
            return p
        return None
    if p.name in {"flags.json", "train.csv", "eval.csv"}:
        return p.parent
    return None


def _last_step(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    try:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                if row.get("step"):
                    last = row["step"]
            return int(last) if last is not None else None
    except Exception:
        return None


def _row_for_run_dir(run_dir: Path) -> Dict[str, Any]:
    flags = _safe_read_json(run_dir / "flags.json")
    return dict(
        run_dir=str(run_dir),
        seed=_get_nested(flags, ["seed"]),
        run_group=_get_nested(flags, ["run_group"]),
        env_name=_get_nested(flags, ["env_name"]),
        critic_backbone=_get_nested(flags, ["agent", "critic_backbone"]),
        critic_resnet_depth=_get_nested(flags, ["agent", "critic_resnet_depth"]),
        critic_recur_iters=_get_nested(flags, ["agent", "critic_recur_iters"]),
        critic_backbone_hidden_dim=_get_nested(flags, ["agent", "critic_backbone_hidden_dim"]),
        critic_eval_num_iters=_get_nested(flags, ["agent", "critic_eval_num_iters"]),
        train_last_step=_last_step(run_dir / "train.csv"),
        eval_last_step=_last_step(run_dir / "eval.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run dirs or globs (e.g., exp/.../sd000_*).")
    parser.add_argument("--csv", action="store_true", help="Print as CSV (default: table).")
    args = parser.parse_args()

    expanded: List[Path] = []
    for pat in args.paths:
        matches = glob.glob(pat)
        if not matches:
            expanded.append(Path(pat))
        else:
            expanded.extend(Path(m) for m in matches)

    run_dirs: List[Path] = []
    for p in expanded:
        rd = _infer_run_dir(p)
        if rd is not None:
            run_dirs.append(rd)

    run_dirs = sorted(set(run_dirs))
    rows = [_row_for_run_dir(d) for d in run_dirs]

    cols = [
        "run_dir",
        "seed",
        "critic_backbone",
        "critic_resnet_depth",
        "critic_recur_iters",
        "critic_backbone_hidden_dim",
        "critic_eval_num_iters",
        "train_last_step",
        "eval_last_step",
    ]

    if args.csv:
        print(",".join(cols))
        for r in rows:
            print(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
        return

    # Simple TSV table for terminal.
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))


if __name__ == "__main__":
    main()
