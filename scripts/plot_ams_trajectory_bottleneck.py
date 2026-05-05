#!/usr/bin/env python3
"""Plot matched AMS MLP/recurrent trajectories for the paper bottleneck figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


MEDIUM_MAZE = np.asarray(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int32,
)
MAZE_UNIT = 4.0
OFFSET_X = 4.0
OFFSET_Y = 4.0


def ij_to_xy(ij: tuple[int, int]) -> tuple[float, float]:
    i, j = ij
    return j * MAZE_UNIT - OFFSET_X, i * MAZE_UNIT - OFFSET_Y


def load_task_trajs(path: Path, task_name: str) -> dict[int, dict[str, np.ndarray]]:
    trajs = {}
    for npz_path in sorted(path.glob(f"*_{task_name}_ep*.npz")):
        data = dict(np.load(npz_path, allow_pickle=True))
        episode_idx = int(data["episode_idx"])
        trajs[episode_idx] = data
    if not trajs:
        raise FileNotFoundError(f"No trajectories for {task_name} under {path}")
    return trajs


def choose_episode(mlp_trajs, recur_trajs):
    shared = sorted(set(mlp_trajs) & set(recur_trajs))
    if not shared:
        raise ValueError("No matched episode indices between the two trajectory directories.")
    for episode_idx in shared:
        mlp_success = float(mlp_trajs[episode_idx]["final_success"])
        recur_success = float(recur_trajs[episode_idx]["final_success"])
        if mlp_success < 0.5 and recur_success >= 0.5:
            return episode_idx
    return shared[0]


def xy_from_traj(traj):
    obs = np.asarray(traj["observation"])
    if obs.ndim != 2 or obs.shape[1] < 2:
        raise ValueError(f"Expected state observations with xy in the first two dims, got shape {obs.shape}")
    return obs[:, :2]


def goal_xy_from_traj(traj):
    goals = np.asarray(traj["goal"])
    if goals.ndim == 2 and goals.shape[1] >= 2:
        return goals[0, :2]
    return None


def draw_maze(ax):
    for i in range(MEDIUM_MAZE.shape[0]):
        for j in range(MEDIUM_MAZE.shape[1]):
            x, y = ij_to_xy((i, j))
            if MEDIUM_MAZE[i, j] == 1:
                ax.add_patch(
                    Rectangle(
                        (x - MAZE_UNIT / 2, y - MAZE_UNIT / 2),
                        MAZE_UNIT,
                        MAZE_UNIT,
                        facecolor="#d9dde3",
                        edgecolor="white",
                        linewidth=0.7,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (x - MAZE_UNIT / 2, y - MAZE_UNIT / 2),
                        MAZE_UNIT,
                        MAZE_UNIT,
                        facecolor="#fbfbf8",
                        edgecolor="#eeeeea",
                        linewidth=0.45,
                    )
                )
    ax.set_aspect("equal")
    ax.set_xlim(-6, 26)
    ax.set_ylim(-6, 26)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_path(ax, xy, color, label, alpha=1.0):
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.8, alpha=alpha, label=label)
    every = max(8, len(xy) // 10)
    ax.scatter(xy[::every, 0], xy[::every, 1], s=12, color=color, alpha=alpha, zorder=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_dir", required=True, type=Path)
    parser.add_argument("--recur_dir", required=True, type=Path)
    parser.add_argument("--task", default="task3")
    parser.add_argument("--episode", default="auto", help="'auto' or an integer episode index.")
    parser.add_argument("--out", default="paper/figures/ams_trajectory_bottleneck.pdf", type=Path)
    args = parser.parse_args()

    mlp_trajs = load_task_trajs(args.mlp_dir, args.task)
    recur_trajs = load_task_trajs(args.recur_dir, args.task)
    episode_idx = choose_episode(mlp_trajs, recur_trajs) if args.episode == "auto" else int(args.episode)

    mlp = mlp_trajs[episode_idx]
    recur = recur_trajs[episode_idx]
    mlp_xy = xy_from_traj(mlp)
    recur_xy = xy_from_traj(recur)
    goal_xy = goal_xy_from_traj(recur)
    start_xy = recur_xy[0]

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    draw_maze(ax)
    plot_path(ax, mlp_xy, "#6b7280", f"MLP actor ({'success' if float(mlp['final_success']) >= 0.5 else 'fail'})", alpha=0.9)
    plot_path(
        ax,
        recur_xy,
        "#2563eb",
        f"Recurrent critic actor ({'success' if float(recur['final_success']) >= 0.5 else 'fail'})",
    )
    ax.scatter([start_xy[0]], [start_xy[1]], marker="o", s=80, color="#16a34a", edgecolor="white", linewidth=1.0, zorder=6, label="start")
    if goal_xy is not None:
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", s=180, color="#dc2626", edgecolor="white", linewidth=0.9, zorder=7, label="goal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=8)
    ax.set_title(f"AMS {args.task}, matched episode {episode_idx}", fontsize=10, pad=18)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
