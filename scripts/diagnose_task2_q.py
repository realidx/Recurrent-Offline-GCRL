"""Diagnostic: dump per-step Q-values and actor actions during Task 2 eval episodes.

Run from the impls/ directory:
    cd third_party/ogbench/impls
    conda run -n recurrent python ../../../scripts/diagnose_task2_q.py \
        --checkpoint_dir /path/to/exp/OGBench/P3_RecurTied_DynFiLM/sd000_ktrain4_a01_lrd1000000 \
        --checkpoint_epoch 1000000

Output: a CSV (task2_q_diagnostic.csv) with columns:
    episode, step, x, y, goal_x, goal_y, euclid_dist, Q_min, Q_1, Q_2,
    actor_action_0..3, success
"""

import argparse
import csv
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Minimal OGBench / agent bootstrap (mirrors main.py's setup)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # impls/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # ogbench root

import ogbench
from agents.crl import CRLAgent, get_config
from utils.flax_utils import restore_agent
from utils.evaluation import supply_rng


def _make_agent(env, train_ds, config):
    """Instantiate a CRLAgent with the given config (random params — will be overwritten)."""
    ex_obs = train_ds['observations'][:1]
    ex_act = train_ds['actions'][:1]
    return CRLAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', required=True, help='Run directory containing params_*.pkl')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='Epoch to load (None = latest)')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--k_test', type=int, default=None, help='Override K_test (None = use K_train)')
    parser.add_argument('--output', type=str, default='task2_q_diagnostic.csv')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Environment & dataset
    # -----------------------------------------------------------------------
    dataset_dir = os.environ.get('OGBENCH_DATASET_DIR', '')
    env, train_ds, _ = ogbench.make_env_and_datasets(
        'antmaze-large-stitch-v0',
        compact_dataset=True,
        **(dict(dataset_dir=dataset_dir) if dataset_dir else {}),
    )

    # -----------------------------------------------------------------------
    # Agent config  — match the Phase 3 RecurTied setup exactly
    # -----------------------------------------------------------------------
    config = get_config().to_dict()
    config.update(dict(
        critic_backbone='recur_tied',
        critic_recur_iters=4,
        critic_recur_max_iters=16,
        critic_recur_tied_ln=True,
        critic_layerscale_init=1e-2,
        actor_p_randomgoal=0.5,
        actor_p_trajgoal=0.5,
        lr_decay_steps=1000000,
        lr_min=1e-5,
    ))
    if args.k_test is not None:
        config['critic_eval_num_iters'] = args.k_test

    agent = _make_agent(env, train_ds, config)

    # -----------------------------------------------------------------------
    # Restore checkpoint
    # -----------------------------------------------------------------------
    agent = restore_agent(agent, args.checkpoint_dir, args.checkpoint_epoch)

    # -----------------------------------------------------------------------
    # JIT the critic and actor once
    # -----------------------------------------------------------------------
    critic_eval_num_iters = config.get('critic_eval_num_iters', None)

    @jax.jit
    def _eval_critic(obs, goal, action):
        """Return (Q_min_over_ensemble, Q_1, Q_2) for a single (obs, goal, action)."""
        v = agent.network.select('critic')(
            obs[None, ...],
            goal[None, ...],
            action[None, ...],
            num_iters=critic_eval_num_iters,
        )
        # v shape: (2, 1) for ensemble=True
        if hasattr(v, 'ndim') and v.ndim == 2:
            q1, q2 = float(v[0, 0]), float(v[1, 0])
            return min(q1, q2), q1, q2
        return float(v[0]), float(v[0]), float(v[0])

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(42))

    # -----------------------------------------------------------------------
    # Task 2 eval loop with Q logging
    # -----------------------------------------------------------------------
    # BFS path cells for reference (printed to stdout)
    path_ij = [(5,4),(6,4),(7,4),(7,5),(7,6),(6,6),(5,6),(4,6),(3,6),
               (3,5),(3,4),(3,3),(3,2),(3,1),(4,1),(5,1),(5,2),(6,2),(7,2),(7,1)]
    path_xy = [(j*4-4, i*4-4) for i,j in path_ij]
    goal_xy = (0.0, 24.0)  # (7,1) in continuous coords

    print("Task 2 BFS path (20 cells, U-turn apex at step 8 = (3,6) = xy(20,8)):")
    print(f"  Start xy = (12, 16), Goal xy = {goal_xy}")
    print(f"  Apex xy  = (20, 8)  — farthest from goal, dist = {math.hypot(20-0, 8-24):.2f}")
    print(f"  Start dist to goal  = {math.hypot(12-0, 16-24):.2f}")
    print()

    rows = []
    for ep in range(args.num_episodes):
        obs, info = env.reset(options=dict(task_id=2))
        goal = info['goal']
        done = False
        step = 0
        while not done:
            action = np.array(actor_fn(observations=obs, goals=goal, temperature=0))
            action = np.clip(action, -1, 1)

            # Evaluate critic Q at current (obs, action, goal)
            q_min, q1, q2 = _eval_critic(jnp.array(obs), jnp.array(goal), jnp.array(action))

            # Also evaluate Q with zero action (baseline)
            q_min_zero, _, _ = _eval_critic(jnp.array(obs), jnp.array(goal), jnp.zeros(action.shape))

            x, y = float(obs[0]), float(obs[1])
            gx, gy = float(goal[0]), float(goal[1])
            euclid = math.hypot(x - gx, y - gy)

            # Determine which path cell we're closest to
            closest_cell = -1
            closest_dist = float('inf')
            for ci, (cx, cy) in enumerate(path_xy):
                d = math.hypot(x - cx, y - cy)
                if d < closest_dist:
                    closest_dist = d
                    closest_cell = ci

            rows.append(dict(
                episode=ep,
                step=step,
                x=round(x, 3),
                y=round(y, 3),
                goal_x=round(gx, 3),
                goal_y=round(gy, 3),
                euclid_dist=round(euclid, 3),
                closest_path_cell=closest_cell,
                closest_cell_dist=round(closest_dist, 3),
                Q_min=round(q_min, 6),
                Q_1=round(q1, 6),
                Q_2=round(q2, 6),
                Q_min_zero_action=round(q_min_zero, 6),
                action_0=round(float(action[0]), 4),
                action_1=round(float(action[1]), 4),
                action_2=round(float(action[2]), 4),
                action_3=round(float(action[3]), 4),
            ))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs
            step += 1

        success = info.get('success', 0)
        # Tag all rows in this episode with success
        for r in rows[-step:]:
            r['success'] = int(success)
        print(f"  Episode {ep}: {step} steps, success={success}, "
              f"final_dist={math.hypot(float(obs[0])-goal_xy[0], float(obs[1])-goal_xy[1]):.2f}")

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")

    # -----------------------------------------------------------------------
    # Summary: print Q profile at the first episode's trajectory
    # -----------------------------------------------------------------------
    print("\n--- Q-value profile (episode 0) ---")
    print(f"{'step':>4} {'x':>6} {'y':>6} {'euclid':>7} {'cell':>4} {'Q_min':>9} {'Q_zero':>9}")
    for r in rows:
        if r['episode'] != 0:
            break
        print(f"{r['step']:4d} {r['x']:6.1f} {r['y']:6.1f} {r['euclid_dist']:7.2f} "
              f"{r['closest_path_cell']:4d} {r['Q_min']:9.4f} {r['Q_min_zero_action']:9.4f}")


if __name__ == '__main__':
    main()
