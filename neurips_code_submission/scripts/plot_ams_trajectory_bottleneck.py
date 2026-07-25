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


def plot_path(ax, xy, color, alpha=1.0):
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.8, alpha=alpha)
    every = max(8, len(xy) // 10)
    ax.scatter(xy[::every, 0], xy[::every, 1], s=12, color=color, alpha=alpha, zorder=4)


def mark_start_goal(ax, start_xy, goal_xy):
    ax.scatter(
        [start_xy[0]],
        [start_xy[1]],
        marker="o",
        s=80,
        color="#16a34a",
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    if goal_xy is not None:
        ax.scatter(
            [goal_xy[0]],
            [goal_xy[1]],
            marker="*",
            s=180,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.9,
            zorder=7,
        )


def task_title(task_name):
    if task_name.startswith("task"):
        suffix = task_name.removeprefix("task")
        if suffix.isdigit():
            return f"Task {int(suffix)}"
    return task_name


def parse_tasks(args):
    if args.tasks:
        return [task.strip() for task in args.tasks.split(",") if task.strip()]
    return [args.task]


def draw_task_row(axes, task_name, episode_arg, mlp_dir, recur_dir):
    mlp_trajs = load_task_trajs(mlp_dir, task_name)
    recur_trajs = load_task_trajs(recur_dir, task_name)
    episode_idx = choose_episode(mlp_trajs, recur_trajs) if episode_arg == "auto" else int(episode_arg)

    mlp = mlp_trajs[episode_idx]
    recur = recur_trajs[episode_idx]
    mlp_xy = xy_from_traj(mlp)
    recur_xy = xy_from_traj(recur)
    goal_xy = goal_xy_from_traj(recur)
    start_xy = recur_xy[0]

    draw_maze(axes[0])
    mark_start_goal(axes[0], start_xy, goal_xy)
    axes[0].set_title(f"{task_title(task_name)} layout", fontsize=10)

    draw_maze(axes[1])
    plot_path(axes[1], mlp_xy, "#6b7280", alpha=0.9)
    mark_start_goal(axes[1], start_xy, goal_xy)
    axes[1].set_title("MLP baseline", fontsize=10)

    draw_maze(axes[2])
    plot_path(axes[2], recur_xy, "#2563eb")
    mark_start_goal(axes[2], start_xy, goal_xy)
    axes[2].set_title("Recurrent (ours)", fontsize=10)

    return episode_idx


def draw_task_column(axes, task_name, episode_arg, mlp_dir, recur_dir):
    mlp_trajs = load_task_trajs(mlp_dir, task_name)
    recur_trajs = load_task_trajs(recur_dir, task_name)
    episode_idx = choose_episode(mlp_trajs, recur_trajs) if episode_arg == "auto" else int(episode_arg)

    mlp = mlp_trajs[episode_idx]
    recur = recur_trajs[episode_idx]
    mlp_xy = xy_from_traj(mlp)
    recur_xy = xy_from_traj(recur)
    goal_xy = goal_xy_from_traj(recur)
    start_xy = recur_xy[0]

    draw_maze(axes[0])
    mark_start_goal(axes[0], start_xy, goal_xy)
    axes[0].set_title(f"{task_title(task_name)} layout", fontsize=10)

    draw_maze(axes[1])
    plot_path(axes[1], mlp_xy, "#6b7280", alpha=0.9)
    mark_start_goal(axes[1], start_xy, goal_xy)
    axes[1].set_title("MLP baseline", fontsize=10)

    draw_maze(axes[2])
    plot_path(axes[2], recur_xy, "#2563eb")
    mark_start_goal(axes[2], start_xy, goal_xy)
    axes[2].set_title("Recurrent (ours)", fontsize=10)

    return episode_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_dir", required=True, type=Path)
    parser.add_argument("--recur_dir", required=True, type=Path)
    parser.add_argument("--task", default="task3")
    parser.add_argument("--tasks", default="", help="Comma-separated task names, e.g. task3,task4.")
    parser.add_argument("--episode", default="auto", help="'auto' or an integer episode index.")
    parser.add_argument("--out", default="outputs/ams_trajectory_bottleneck.pdf", type=Path)
    parser.add_argument("--layout", choices=("separate", "overlay", "task_columns"), default="separate")
    args = parser.parse_args()

    tasks = parse_tasks(args)

    if args.layout == "overlay":
        task_name = tasks[0]
        mlp_trajs = load_task_trajs(args.mlp_dir, task_name)
        recur_trajs = load_task_trajs(args.recur_dir, task_name)
        episode_idx = choose_episode(mlp_trajs, recur_trajs) if args.episode == "auto" else int(args.episode)

        mlp = mlp_trajs[episode_idx]
        recur = recur_trajs[episode_idx]
        mlp_xy = xy_from_traj(mlp)
        recur_xy = xy_from_traj(recur)
        goal_xy = goal_xy_from_traj(recur)
        start_xy = recur_xy[0]

        fig, ax = plt.subplots(figsize=(5.8, 4.8))
        draw_maze(ax)
        plot_path(ax, mlp_xy, "#6b7280", alpha=0.9)
        plot_path(ax, recur_xy, "#2563eb")
        mark_start_goal(ax, start_xy, goal_xy)
        ax.set_title(f"{task_title(task_name)}", fontsize=10, pad=18)
    elif args.layout == "task_columns":
        fig, axes = plt.subplots(3, len(tasks), figsize=(3.8 * len(tasks), 11.4), constrained_layout=True)
        if len(tasks) == 1:
            axes = np.expand_dims(axes, axis=1)
        for col_index, task_name in enumerate(tasks):
            draw_task_column(axes[:, col_index], task_name, args.episode, args.mlp_dir, args.recur_dir)
    else:
        fig, axes = plt.subplots(len(tasks), 3, figsize=(11.4, 3.8 * len(tasks)), constrained_layout=True)
        if len(tasks) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_index, task_name in enumerate(tasks):
            draw_task_row(axes[row_index], task_name, args.episode, args.mlp_dir, args.recur_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
