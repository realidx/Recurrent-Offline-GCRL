from collections import defaultdict

import jax
import numpy as np
import jax.numpy as jnp
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    refine_steps=0,
    refine_lr=0.05,
    refine_l2=0.0,
    episode_seed_base=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_seed = 0 if episode_seed_base is None else int(episode_seed_base) + int(task_id or 0) * 1_000_000
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(actor_seed))
    trajs = []
    stats = defaultdict(list)

    def _cfg_get(key, default=None):
        if config is None:
            return default
        try:
            return config.get(key, default)
        except Exception:
            try:
                return config[key]
            except Exception:
                return default

    critic_eval_k = _cfg_get('critic_eval_k', None)
    modules = getattr(getattr(agent.network, 'model_def', None), 'modules', {})
    has_critic = hasattr(modules, 'keys') and 'critic' in modules
    has_actor = hasattr(modules, 'keys') and 'actor' in modules

    def _is_finite(x) -> bool:
        try:
            return bool(np.all(np.isfinite(np.asarray(x))))
        except Exception:
            return False

    def _critic_score(ob, goal, action):
        if has_critic:
            # CRL critic outputs an ensemble of bilinear scores; match actor loss aggregation (min over ensemble).
            v = agent.network.select('critic')(
                ob[None, ...],
                goal[None, ...],
                action[None, ...],
                num_iters=critic_eval_k,
            )
        else:
            raise ValueError('Action refinement requires a critic module.')
        if hasattr(v, 'ndim') and v.ndim >= 2:
            # v shape: (E, B) or (B,). For B=1, take min over ensemble.
            if v.ndim == 2:
                return jnp.min(v[:, 0])
            return v[0]
        return jnp.asarray(v)

    # ---------------------------------------------------------------------------
    # Jitted value + grad + ||grad|| over the critic.  Defined once here so that
    # XLA compiles it on the first call and reuses the compiled kernel for every
    # subsequent env step.  ob / goal / a / a0 are explicit args → no
    # recompilation when their values change.  refine_l2 is a Python scalar
    # resolved at trace time.
    # ---------------------------------------------------------------------------
    @jax.jit
    def _jit_value_grad_norm(ob, goal, a, a0):
        def obj(action):
            score = _critic_score(ob, goal, action)
            if refine_l2 and refine_l2 > 0:
                score = score - refine_l2 * jnp.sum((action - a0) ** 2)
            return score
        val, grad = jax.value_and_grad(obj)(a)
        return val, grad, jnp.linalg.norm(grad)

    def _refine_action(ob_np, goal_np, action0_np):
        if config.get('discrete') or refine_steps <= 0:
            return action0_np, None

        ob = jnp.asarray(ob_np)
        goal = jnp.asarray(goal_np)
        a0 = jnp.asarray(action0_np)

        grad_eps = 1e-6
        q_eps = 1e-5

        # Initial value + grad at a0.  The grad is reused as iteration 0's
        # gradient (a == a0 at that point), so no redundant forward pass.
        q_pre, g, g_norm_raw = _jit_value_grad_norm(ob, goal, a0, a0)
        q_pre_finite = _is_finite(q_pre)
        q_prev = q_pre
        a = a0

        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        steps_taken = 0
        nonfinite = 0
        early_stop_reason = "max_steps"

        for _ in range(int(refine_steps)):
            g_norm = g_norm_raw + 1e-8

            # Check 1: NaN/Inf in gradient.
            g_finite = _is_finite(g) and _is_finite(g_norm)
            if not g_finite:
                nonfinite = 1
                early_stop_reason = "nonfinite"
                break

            gn = float(g_norm)

            # Check 2: gradient too small (converged).
            if gn < grad_eps:
                early_stop_reason = "grad_vanished"
                break

            # Take step.
            a = jnp.clip(a + float(refine_lr) * (g / g_norm), -1.0, 1.0)

            steps_taken += 1
            grad_norm_sum += gn
            grad_norm_max = max(grad_norm_max, gn)

            # Check 3: negligible improvement in objective.
            # value_and_grad also produces the grad for the next iteration at no
            # extra cost; it is simply unused if we break here.
            q_curr, g, g_norm_raw = _jit_value_grad_norm(ob, goal, a, a0)
            q_curr_finite = _is_finite(q_curr)
            q_prev_finite = _is_finite(q_prev)
            if not q_curr_finite:
                nonfinite = 1
                early_stop_reason = "nonfinite"
                break

            if q_prev_finite:
                if float(q_curr - q_prev) < q_eps:
                    early_stop_reason = "q_plateau"
                    q_prev = q_curr
                    break
            q_prev = q_curr

        q_post = q_prev
        q_post_finite = _is_finite(q_post)
        if not (q_pre_finite and q_post_finite):
            nonfinite = 1
            early_stop_reason = "nonfinite"
        delta = jnp.linalg.norm(a - a0)

        return np.array(a), dict(
            q_pre=float(q_pre) if q_pre_finite else float('nan'),
            q_post=float(q_post) if q_post_finite else float('nan'),
            delta_a=float(delta) if _is_finite(delta) else float('nan'),
            q_improve=(float(q_post - q_pre) if (q_pre_finite and q_post_finite) else float('nan')),
            grad_norm_mean=(grad_norm_sum / max(1, steps_taken)) if steps_taken > 0 else 0.0,
            grad_norm_max=grad_norm_max,
            steps_taken=int(steps_taken),
            nonfinite=int(nonfinite),
            early_stop_reason=str(early_stop_reason),
            q_eps=float(q_eps),
            grad_eps=float(grad_eps),
        )

    @jax.jit
    def _jit_actor_mode(observation, goal):
        dist = agent.network.select('actor')(observation, goal, temperature=0.0)
        action = dist.mode()
        if not config.get('discrete'):
            action = jnp.clip(action, -1, 1)
        return action

    def _eval_action(observation, goal):
        # For evaluation, temperature=0 should mean greedy action selection. Keep the
        # deterministic path jitted so evaluation speed stays comparable to sample_actions.
        if has_actor and eval_temperature == 0:
            return np.array(_jit_actor_mode(observation, goal))
        action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
        if isinstance(action, tuple):
            action = action[0]
        return np.array(action)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        reset_kwargs = dict(options=dict(task_id=task_id, render_goal=should_render))
        if episode_seed_base is not None:
            # Make evaluation reproducible and comparable across separate jobs/models.
            # Use distinct seeds per (task, episode) so each model sees the same episode distribution.
            reset_kwargs['seed'] = int(episode_seed_base) + int(task_id or 0) * 100_000 + int(i)
        try:
            observation, info = env.reset(**reset_kwargs)
        except TypeError:
            # Backwards-compat: env.reset may not accept a seed kwarg.
            reset_kwargs.pop('seed', None)
            observation, info = env.reset(**reset_kwargs)
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        refine_step_stats = []
        while not done:
            action0 = _eval_action(observation, goal)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action0 = np.random.normal(action0, eval_gaussian)
                action0 = np.clip(action0, -1, 1)

            action = action0
            if refine_steps and refine_steps > 0:
                action, refine_info = _refine_action(observation, goal, action0)
                if refine_info is not None:
                    refine_step_stats.append(refine_info)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                goal=goal,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            if refine_step_stats:
                info = dict(info)
                reasons = [x.get('early_stop_reason', 'unknown') for x in refine_step_stats]
                info['refine'] = dict(
                    q_pre=float(np.mean([x['q_pre'] for x in refine_step_stats])),
                    q_post=float(np.mean([x['q_post'] for x in refine_step_stats])),
                    delta_a=float(np.mean([x['delta_a'] for x in refine_step_stats])),
                    q_improve=float(np.mean([x.get('q_improve', float('nan')) for x in refine_step_stats])),
                    grad_norm_mean=float(np.mean([x.get('grad_norm_mean', 0.0) for x in refine_step_stats])),
                    grad_norm_max=float(np.max([x.get('grad_norm_max', 0.0) for x in refine_step_stats])),
                    steps_taken_mean=float(np.mean([x.get('steps_taken', 0) for x in refine_step_stats])),
                    nonfinite_frac=float(np.mean([x.get('nonfinite', 0) for x in refine_step_stats])),
                    grad_vanished_frac=float(np.mean([r == 'grad_vanished' for r in reasons])),
                    q_plateau_frac=float(np.mean([r == 'q_plateau' for r in reasons])),
                    max_steps_frac=float(np.mean([r == 'max_steps' for r in reasons])),
                )
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
