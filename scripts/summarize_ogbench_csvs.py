#!/usr/bin/env python3
"""
Summarize OGBench impls runs by scanning for `eval.csv` / `train.csv` and `flags.json`.

Example:
  python scripts/summarize_ogbench_csvs.py --roots exp logs --out runs.csv
  python scripts/summarize_ogbench_csvs.py --roots logs/phase_0/exp --print
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _to_number(x: str) -> Any:
    s = (x or "").strip()
    if s == "":
        return None
    try:
        if any(c in s for c in [".", "e", "E"]):
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            parsed = {k: _to_number(v) for k, v in row.items()}
            rows.append(parsed)
    return rows


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_restore_dir(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    try:
        candidates = glob.glob(path_str)
        if len(candidates) == 1:
            return Path(candidates[0])
    except Exception:
        pass
    p = Path(path_str).expanduser()
    return p if p.exists() else None


def _get_nested(d: Optional[Dict[str, Any]], keys: Iterable[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _last_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in reversed(rows):
        if any(v is not None and v != "" for v in row.values()):
            return row
    return None


def _best_metric(rows: List[Dict[str, Any]], key: str) -> Tuple[Optional[float], Optional[int]]:
    best_val: Optional[float] = None
    best_step: Optional[int] = None
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        try:
            v_f = float(v)
        except Exception:
            continue
        step = row.get("step")
        step_i = int(step) if step is not None else None
        if best_val is None or v_f > best_val:
            best_val = v_f
            best_step = step_i
    return best_val, best_step


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if not xs:
        return None, None
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return mean, math.sqrt(var)


@dataclass
class RunSummary:
    run_dir: str
    env_name: Optional[str]
    seed: Optional[int]
    run_group: Optional[str]
    agent_name: Optional[str]
    critic_backbone: Optional[str]
    critic_resnet_depth: Optional[int]
    critic_recur_iters: Optional[int]
    critic_partial_groups: Optional[int]
    critic_partial_iters_per_group: Optional[int]
    critic_partial_per_group_film: Optional[bool]
    critic_backbone_hidden_dim: Optional[int]
    critic_eval_num_iters: Optional[int]

    train_last_step: Optional[int]

    eval_last_step: Optional[int]
    eval_last_overall_success: Optional[float]
    eval_best_overall_success: Optional[float]
    eval_best_step: Optional[int]

    alpha: Optional[float]
    discount: Optional[float]

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RunSummary":
        flags_path = run_dir / "flags.json"
        train_path = run_dir / "train.csv"
        eval_path = run_dir / "eval.csv"

        flags = _safe_read_json(flags_path)
        agent_cfg = flags.get("agent") if isinstance(flags, dict) else None

        # If this is an eval-only run, flags.json may reflect the *default* agent config (because the run
        # restores from a checkpoint). In that case, read the restored run's flags.json and use that to
        # infer the correct backbone/depth/seed/env for summarization.
        restore_flags = None
        if isinstance(flags, dict):
            restore_dir = _resolve_restore_dir(flags.get("restore_path"))
            if restore_dir is not None:
                restore_flags = _safe_read_json(restore_dir / "flags.json")
                if isinstance(restore_flags, dict):
                    agent_cfg = restore_flags.get("agent") if isinstance(restore_flags.get("agent"), dict) else agent_cfg
                    # Prefer the restored run's seed/env_name for grouping.
                    if restore_flags.get("seed") is not None:
                        flags["seed"] = restore_flags.get("seed")
                    if restore_flags.get("env_name") is not None:
                        flags["env_name"] = restore_flags.get("env_name")

        env_name = flags.get("env_name") if isinstance(flags, dict) else None
        seed = flags.get("seed") if isinstance(flags, dict) else None
        run_group = flags.get("run_group") if isinstance(flags, dict) else None
        agent_name = _get_nested(flags, ["agent", "agent_name"])

        # Prefer restored agent_cfg if present.
        critic_backbone = _get_nested(agent_cfg, ["critic_backbone"]) if isinstance(agent_cfg, dict) else None
        critic_resnet_depth = _get_nested(agent_cfg, ["critic_resnet_depth"]) if isinstance(agent_cfg, dict) else None
        critic_recur_iters = _get_nested(agent_cfg, ["critic_recur_iters"]) if isinstance(agent_cfg, dict) else None
        critic_partial_groups = _get_nested(agent_cfg, ["critic_partial_groups"]) if isinstance(agent_cfg, dict) else None
        critic_partial_iters_per_group = (
            _get_nested(agent_cfg, ["critic_partial_iters_per_group"]) if isinstance(agent_cfg, dict) else None
        )
        critic_partial_per_group_film = (
            _get_nested(agent_cfg, ["critic_partial_per_group_film"]) if isinstance(agent_cfg, dict) else None
        )
        critic_backbone_hidden_dim = (
            _get_nested(agent_cfg, ["critic_backbone_hidden_dim"]) if isinstance(agent_cfg, dict) else None
        )
        critic_eval_num_iters = _get_nested(agent_cfg, ["critic_eval_num_iters"]) if isinstance(agent_cfg, dict) else None
        if critic_backbone_hidden_dim in (0, 0.0, "0"):
            critic_backbone_hidden_dim = None

        alpha = _get_nested(agent_cfg, ["alpha"]) if isinstance(agent_cfg, dict) else None
        discount = _get_nested(agent_cfg, ["discount"]) if isinstance(agent_cfg, dict) else None
        try:
            alpha = float(alpha) if alpha is not None else None
        except Exception:
            alpha = None
        try:
            discount = float(discount) if discount is not None else None
        except Exception:
            discount = None

        train_last_step = None
        if train_path.exists():
            train_rows = _read_csv_rows(train_path)
            last = _last_row(train_rows)
            if last is not None and last.get("step") is not None:
                train_last_step = int(last["step"])

        eval_last_step = None
        eval_last_overall_success = None
        eval_best_overall_success = None
        eval_best_step = None
        if eval_path.exists():
            eval_rows = _read_csv_rows(eval_path)
            last = _last_row(eval_rows)
            if last is not None:
                if last.get("step") is not None:
                    eval_last_step = int(last["step"])
                v = last.get("evaluation/overall_success")
                try:
                    eval_last_overall_success = float(v) if v is not None else None
                except Exception:
                    eval_last_overall_success = None
            eval_best_overall_success, eval_best_step = _best_metric(eval_rows, "evaluation/overall_success")

        return cls(
            run_dir=str(run_dir),
            env_name=env_name,
            seed=int(seed) if seed is not None else None,
            run_group=run_group,
            agent_name=str(agent_name) if agent_name is not None else None,
            critic_backbone=str(critic_backbone) if critic_backbone is not None else None,
            critic_resnet_depth=int(critic_resnet_depth) if critic_resnet_depth is not None else None,
            critic_recur_iters=int(critic_recur_iters) if critic_recur_iters is not None else None,
            critic_partial_groups=int(critic_partial_groups) if critic_partial_groups is not None else None,
            critic_partial_iters_per_group=(
                int(critic_partial_iters_per_group) if critic_partial_iters_per_group is not None else None
            ),
            critic_partial_per_group_film=(
                bool(critic_partial_per_group_film) if critic_partial_per_group_film is not None else None
            ),
            critic_backbone_hidden_dim=(
                int(critic_backbone_hidden_dim) if critic_backbone_hidden_dim is not None else None
            ),
            critic_eval_num_iters=int(critic_eval_num_iters) if critic_eval_num_iters is not None else None,
            train_last_step=train_last_step,
            eval_last_step=eval_last_step,
            eval_last_overall_success=eval_last_overall_success,
            eval_best_overall_success=eval_best_overall_success,
            eval_best_step=eval_best_step,
            alpha=alpha,
            discount=discount,
        )


def discover_run_dirs(roots: List[Path]) -> List[Path]:
    run_dirs = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for eval_path in root.rglob("eval.csv"):
            run_dir = eval_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["exp", "logs"],
        help="Directories to recursively scan for eval.csv (default: exp logs).",
    )
    parser.add_argument("--out", default="runs.csv", help="Output CSV path.")
    parser.add_argument("--print", action="store_true", help="Print a small table to stdout.")
    parser.add_argument("--aggregate", action="store_true", help="Also write an aggregate CSV by (run_group, env_name).")
    args = parser.parse_args()

    roots = [Path(p) for p in args.roots]
    run_dirs = discover_run_dirs(roots)
    summaries = [RunSummary.from_run_dir(d) for d in run_dirs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_dir",
        "run_group",
        "env_name",
        "seed",
        "agent_name",
        "critic_backbone",
        "critic_resnet_depth",
        "critic_recur_iters",
        "critic_partial_groups",
        "critic_partial_iters_per_group",
        "critic_partial_per_group_film",
        "critic_backbone_hidden_dim",
        "critic_eval_num_iters",
        "alpha",
        "discount",
        "train_last_step",
        "eval_last_step",
        "eval_last_overall_success",
        "eval_best_overall_success",
        "eval_best_step",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: getattr(s, k) for k in fieldnames})

    if args.aggregate:
        grouped: Dict[Tuple[str, str], List[RunSummary]] = {}
        for s in summaries:
            if s.run_group is None or s.env_name is None:
                continue
            grouped.setdefault((s.run_group, s.env_name), []).append(s)

        agg_path = out_path.with_name(out_path.stem + "_agg.csv")
        with agg_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_group",
                    "env_name",
                    "n",
                    "best_overall_success_mean",
                    "best_overall_success_std",
                    "last_overall_success_mean",
                    "last_overall_success_std",
                ],
            )
            writer.writeheader()
            for (run_group, env_name), items in sorted(grouped.items()):
                best_vals = [s.eval_best_overall_success for s in items if s.eval_best_overall_success is not None]
                last_vals = [s.eval_last_overall_success for s in items if s.eval_last_overall_success is not None]
                best_mean, best_std = _mean_std([float(v) for v in best_vals])
                last_mean, last_std = _mean_std([float(v) for v in last_vals])
                writer.writerow(
                    dict(
                        run_group=run_group,
                        env_name=env_name,
                        n=len(items),
                        best_overall_success_mean=best_mean,
                        best_overall_success_std=best_std,
                        last_overall_success_mean=last_mean,
                        last_overall_success_std=last_std,
                    )
                )

    if args.print:
        lines = []
        header = ["run_group", "env_name", "seed", "best", "last", "dir"]
        lines.append("\t".join(header))
        for s in summaries:
            lines.append(
                "\t".join(
                    [
                        s.run_group or "",
                        s.env_name or "",
                        "" if s.seed is None else str(s.seed),
                        "" if s.eval_best_overall_success is None else f"{s.eval_best_overall_success:.3f}",
                        "" if s.eval_last_overall_success is None else f"{s.eval_last_overall_success:.3f}",
                        s.run_dir,
                    ]
                )
            )
        print("\n".join(lines))


if __name__ == "__main__":
    main()
