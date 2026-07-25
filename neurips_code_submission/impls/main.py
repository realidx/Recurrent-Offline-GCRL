import glob
import hashlib
import re
import json
import os
import platform
import random
import signal
import socket
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from ogbench.utils import DEFAULT_DATASET_DIR
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


def _is_recurrent_backbone(backbone):
    return str(backbone) == 'recur'


FLAGS = flags.FLAGS


@contextmanager
def _maybe_preserve_np_random_state(enabled=True):
    state = np.random.get_state() if enabled else None
    try:
        yield
    finally:
        if state is not None:
            np.random.set_state(state)

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('exp_name', None, 'Experiment name override (useful for resuming into the same directory).')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer(
    'validation_log_interval',
    None,
    'Validation logging interval (defaults to log_interval).',
)
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_string('save_steps', '', 'Explicit save steps. Accepts comma-, colon-, or whitespace-separated integers. If set, save at these steps in addition to any save_interval behavior.')
flags.DEFINE_bool('disable_tqdm', False, 'Disable tqdm progress bars (reduces stderr spam).')
flags.DEFINE_integer('refine_probe_size', 256, 'Held-out probe batch size for recurrent value refinement diagnostics.')
flags.DEFINE_string(
    'refine_dump_steps',
    '',
    'Explicit refinement-dump steps. Accepts comma-, colon-, or whitespace-separated integers. Empty defaults to early/mid/final dump targets.',
)

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('eval_refine_steps', 0, 'Action refinement steps during evaluation (0 disables).')
flags.DEFINE_float('eval_refine_lr', 0.05, 'Action refinement step size.')
flags.DEFINE_float('eval_refine_l2', 0.0, 'Action refinement L2 penalty coefficient to keep actions near actor output.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')
flags.DEFINE_bool('dump_eval_trajs', False, 'Save evaluation trajectories as compressed NPZ files.')
flags.DEFINE_string(
    'eval_traj_dir',
    '',
    'Directory for --dump_eval_trajs. Empty defaults to <save_dir>/eval_trajs.',
)
flags.DEFINE_bool(
    'preserve_training_rng_on_eval',
    True,
    'Preserve the global NumPy RNG around evaluation/validation diagnostics so logging does not change training batches.',
)
flags.DEFINE_bool('eval_only', False, 'Skip training and run evaluation only (requires --restore_path).')
flags.DEFINE_bool(
    'restore_use_flags',
    True,
    'If restore_path points to a run dir with flags.json, merge agent config/env_name from it (helps eval/resume).',
)

config_flags.DEFINE_config_file('agent', 'agents/crl.py', lock_config=False)


def _parse_step_spec(spec: str) -> set[int]:
    if not spec:
        return set()
    try:
        tokens = [x for x in re.split(r'[\s,:;]+', spec) if x]
        return {int(x) for x in tokens}
    except ValueError as e:
        raise ValueError(
            f'Invalid step specification {spec!r}; expected comma-, colon-, or whitespace-separated integers.'
        ) from e


def _count_params(param_tree) -> int:
    if param_tree is None:
        return 0
    total = 0
    for leaf in jax.tree_util.tree_leaves(param_tree):
        total += int(np.asarray(leaf).size)
    return total


def _sha256_file(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_git_commit(repo_dir: str) -> str:
    try:
        return (
            subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_dir, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return 'unknown'


def _dataset_paths_for_env(env_name: str) -> tuple[str, str, str]:
    dataset_dir = os.path.expanduser(os.environ.get('OGBENCH_DATASET_DIR') or DEFAULT_DATASET_DIR)
    return (
        dataset_dir,
        os.path.join(dataset_dir, f'{env_name}.npz'),
        os.path.join(dataset_dir, f'{env_name}-val.npz'),
    )


def _device_summary() -> dict[str, str | int]:
    try:
        devices = jax.devices()
    except Exception:
        devices = []
    return dict(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        jax_backend=jax.default_backend(),
        jax_device_count=len(devices),
        jax_devices=' | '.join(str(d) for d in devices) if devices else 'unknown',
    )


def main(_):
    save_steps = _parse_step_spec(FLAGS.save_steps)
    refine_dump_steps = _parse_step_spec(FLAGS.refine_dump_steps)

    terminate_requested = {'flag': False, 'sig': None}

    def _handle_terminate(sig, frame):
        terminate_requested['flag'] = True
        terminate_requested['sig'] = sig

    try:
        signal.signal(signal.SIGTERM, _handle_terminate)
        signal.signal(signal.SIGINT, _handle_terminate)
    except Exception:
        pass

    # If restoring from a previous run, load its flags.json early so W&B / flags.json reflect
    # the actual checkpoint architecture (instead of defaults like mlp/3/4).
    if FLAGS.restore_use_flags and FLAGS.restore_path is not None:
        candidates = glob.glob(FLAGS.restore_path)
        restore_dir = (
            candidates[0]
            if len(candidates) == 1
            else (FLAGS.restore_path if os.path.isdir(FLAGS.restore_path) else None)
        )
        if restore_dir is not None and os.path.isfile(restore_dir) and restore_dir.endswith('.pkl'):
            restore_dir = os.path.dirname(restore_dir)
        if restore_dir is not None:
            flags_path = os.path.join(restore_dir, 'flags.json')
            if os.path.exists(flags_path):
                try:
                    restore_flags = json.load(open(flags_path, 'r'))
                except Exception:
                    restore_flags = None
                if isinstance(restore_flags, dict):
                    # Prefer checkpoint env_name for correctness, unless explicitly overridden on the command line.
                    try:
                        env_flag = flags.FLAGS['env_name']
                        env_was_set = bool(getattr(env_flag, 'present', False))
                    except Exception:
                        env_was_set = False
                    if not env_was_set and restore_flags.get('env_name') is not None:
                        FLAGS.env_name = restore_flags.get('env_name')

                    # Prefer checkpoint seed unless explicitly overridden.
                    try:
                        seed_flag = flags.FLAGS['seed']
                        seed_was_set = bool(getattr(seed_flag, 'present', False))
                    except Exception:
                        seed_was_set = False
                    if not seed_was_set and restore_flags.get('seed') is not None:
                        FLAGS.seed = restore_flags.get('seed')

                    # Merge agent config needed to reconstruct the exact parameter tree.
                    # Avoid overriding eval-only knobs that are useful to change at test time.
                    restore_agent_cfg = restore_flags.get('agent', {})
                    if isinstance(restore_agent_cfg, dict):
                        cfg = FLAGS.agent
                        for k, v in restore_agent_cfg.items():
                            if k == 'critic_eval_k':
                                continue
                            if k in cfg:
                                cfg[k] = v

    # Set up logger.
    exp_name = FLAGS.exp_name or get_exp_name(FLAGS.seed)
    wandb_project = os.environ.get('WANDB_PROJECT', 'OGBench')
    wandb_entity = os.environ.get('WANDB_ENTITY')
    setup_wandb(entity=wandb_entity, project=wandb_project, group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Initialize RNGs early so env/dataset creation (and their initial resets) are reproducible.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config['frame_stack'], seed=FLAGS.seed
    )

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    def _dataset_sample_kwargs():
        return {}

    def _sample_dataset(dataset, batch_size, *, idxs=None, evaluation=False):
        sample_kwargs = dict(idxs=idxs, evaluation=evaluation)
        sample_kwargs.update(_dataset_sample_kwargs())
        return dataset.sample(batch_size, **sample_kwargs)

    # Initialize agent.
    example_batch = _sample_dataset(train_dataset, 1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    elif FLAGS.eval_only:
        raise ValueError('--eval_only requires --restore_path and --restore_epoch.')

    def _extract_module_params(params, module_name: str):
        if not isinstance(params, dict):
            return None
        if module_name in params:
            return params[module_name]
        alt_keys = [f'modules_{module_name}', f'module_{module_name}', module_name]
        for k in alt_keys:
            if k in params:
                return params[k]
        for k, v in params.items():
            if module_name in str(k):
                return v
        return None

    def _collect_named_leaves(tree, target_key: str):
        leaves = []

        def _walk(node):
            if hasattr(node, 'items'):
                for k, v in node.items():
                    if str(k) == target_key:
                        leaves.append(np.asarray(v))
                    _walk(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    _walk(v)

        _walk(tree)
        return leaves

    def _abs_param_stats(param_tree):
        if param_tree is None:
            return None, None
        leaves = jax.tree_util.tree_leaves(param_tree)
        if not leaves:
            return None, None
        abs_sum = 0.0
        count = 0
        abs_max = 0.0
        for x in leaves:
            x = jnp.asarray(x)
            ax = jnp.abs(x)
            abs_sum = abs_sum + ax.sum()
            count += ax.size
            abs_max = jnp.maximum(abs_max, ax.max())
        abs_mean = abs_sum / max(1, count)
        return float(abs_mean), float(abs_max)

    def _actor_module_names():
        if _extract_module_params(agent.network.params, 'actor') is not None:
            return ['actor']
        names = []
        if _extract_module_params(agent.network.params, 'low_actor') is not None:
            names.append('low_actor')
        if _extract_module_params(agent.network.params, 'high_actor') is not None:
            names.append('high_actor')
        return names

    train_dataset_dir, train_dataset_path, val_dataset_path = _dataset_paths_for_env(FLAGS.env_name)
    device_summary = _device_summary()
    value_module_name = 'value' if _extract_module_params(agent.network.params, 'value') is not None else None
    critic_module_name = 'critic' if _extract_module_params(agent.network.params, 'critic') is not None else None
    target_value_module_name = 'target_value' if _extract_module_params(agent.network.params, 'target_value') is not None else None
    target_critic_module_name = 'target_critic' if _extract_module_params(agent.network.params, 'target_critic') is not None else None
    actor_module_names = _actor_module_names()
    has_goal_rep_module = _extract_module_params(agent.network.params, 'goal_rep') is not None

    def _filter_metrics_for_wandb(metrics, stage: str):
        """Keep W&B compact while preserving full CSV logs on disk."""
        if stage == 'eval':
            exact_keys = {
                'video',
                'evaluation/best_so_far_success',
                'evaluation/final_success',
                'evaluation/best_so_far_episode.return',
                'evaluation/final_episode.return',
                'evaluation/best_so_far_normalized_score',
                'evaluation/final_normalized_score',
                'params/critic_count',
                'params/value_count',
                'params/actor_count',
                'time/update_count',
                'time/samples_seen',
                'time/eval_count',
                'time/hours_elapsed',
                'time/samples_per_second',
            }
            prefixes = (
                'evaluation/00_',
                'evaluation/01_',
                'evaluation/02_',
                'evaluation/10_',
                'evaluation/11_',
                'evaluation/12_',
                'evaluation/13_',
                'evaluation/14_',
                'evaluation/15_',
                'evaluation/16_',
                'evaluation/17_',
                'evaluation/18_',
                'evaluation/19_',
                'evaluation/20_',
                'evaluation/21_',
                'evaluation/22_',
                'evaluation/23_',
                'evaluation/24_',
                'evaluation/30_',
                'evaluation/31_',
                'evaluation/32_',
                'evaluation/33_',
                'evaluation/34_',
                'evaluation/40_',
                'evaluation/41_',
                'evaluation/42_',
                'evaluation/43_',
                'evaluation/44_',
            )
            filtered = {}
            for key, value in metrics.items():
                if key in exact_keys:
                    filtered[key] = value
                    continue
                if key.startswith(prefixes):
                    filtered[key] = value
                    continue
                if key.startswith('evaluation/task') and key.endswith('_success'):
                    filtered[key] = value
                    continue
            return filtered

        if stage == 'train':
            exact_keys = {
                'training/critic/contrastive_loss',
                'training/critic/categorical_accuracy',
                'training/critic/score_margin',
                'training/critic/logits_pos',
                'training/critic/logits_neg',
                'training/critic/ensemble_disagreement',
                'training/critic/hidden_drift_mean',
                'training/actor/actor_loss',
                'training/actor/bc_loss',
                'training/actor/q_loss',
                'training/actor/q_mean',
                'training/actor/adv_mean',
                'training/actor/adv_std',
                'training/actor/adv_p10',
                'training/actor/adv_p50',
                'training/actor/adv_p90',
                'training/actor/positive_adv_frac',
                'training/actor/awr_weight_mean',
                'training/actor/awr_weight_max',
                'training/actor/awr_weight_ess_frac',
                'training/actor/bc_log_prob',
                'training/actor/mse',
                'training/low_actor/adv_mean',
                'training/low_actor/adv_std',
                'training/low_actor/positive_adv_frac',
                'training/low_actor/awr_weight_mean',
                'training/low_actor/awr_weight_max',
                'training/low_actor/awr_weight_ess_frac',
                'training/low_actor/bc_log_prob',
                'training/low_actor/mse',
                'training/high_actor/adv_mean',
                'training/high_actor/adv_std',
                'training/high_actor/positive_adv_frac',
                'training/high_actor/awr_weight_mean',
                'training/high_actor/awr_weight_max',
                'training/high_actor/awr_weight_ess_frac',
                'training/high_actor/bc_log_prob',
                'training/high_actor/mse',
                'training/grad/global_norm',
                'training/grad/critic_global_norm',
                'training/grad/value_global_norm',
                'training/grad/actor_global_norm',
                'action/policy_behavior_mse',
                'params/critic_abs_mean',
                'time/sps',
                'time/step_time',
                'time/update_count',
                'time/samples_seen',
                'time/eval_count',
                'time/hours_elapsed',
                'time/samples_per_second',
            }
            filtered = {}
            for key, value in metrics.items():
                if key in exact_keys or key.startswith('validation/'):
                    filtered[key] = value
            return filtered

        return dict(metrics)

    static_metadata = {
        'meta/task': FLAGS.env_name,
        'meta/agent_name': str(config.get('agent_name')),
        'meta/dataset_class': str(config.get('dataset_class')),
        'meta/git_commit': _get_git_commit(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ),
        'meta/dataset_dir': train_dataset_dir,
        'meta/train_dataset_path': train_dataset_path,
        'meta/val_dataset_path': val_dataset_path,
        'meta/train_dataset_sha256': _sha256_file(train_dataset_path) or 'missing',
        'meta/val_dataset_sha256': _sha256_file(val_dataset_path) or 'missing',
        'meta/hostname': device_summary['hostname'],
        'meta/platform': device_summary['platform'],
        'meta/python_version': device_summary['python_version'],
        'meta/jax_backend': device_summary['jax_backend'],
        'meta/jax_devices': device_summary['jax_devices'],
        'meta/jax_device_count': int(device_summary['jax_device_count']),
    }
    static_param_metrics = {
        'params/total_count': _count_params(
            {
                k: v
                for k, v in agent.network.params.items()
                if k not in ('modules_target_value', 'modules_target_critic')
            }
        ),
        'params/value_count': _count_params(_extract_module_params(agent.network.params, value_module_name))
        if value_module_name is not None
        else 0,
        'params/actor_count': sum(_count_params(_extract_module_params(agent.network.params, name)) for name in actor_module_names),
        'params/critic_count': _count_params(_extract_module_params(agent.network.params, 'critic')),
        'params/target_value_count': _count_params(_extract_module_params(agent.network.params, 'target_value'))
        if target_value_module_name is not None
        else 0,
        'params/target_critic_count': _count_params(_extract_module_params(agent.network.params, 'target_critic'))
        if target_critic_module_name is not None
        else 0,
    }
    maze_env = getattr(env, 'unwrapped', env)
    maze_distance_cache = {}
    source_labels = {
        0: 'current',
        1: 'traj',
        2: 'random',
    }

    def _reduced_scores(raw_scores, module_name: str):
        arr = np.asarray(raw_scores)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            if module_name == 'critic':
                return np.min(arr, axis=axis)
            return np.mean(arr, axis=axis)
        return arr.reshape(arr.shape[0], -1).mean(axis=-1)

    def _negative_goal_batch(batch, goal_key='value_goals'):
        negative_goals = batch.get('value_random_goals')
        if negative_goals is not None:
            return negative_goals
        goals = batch.get(goal_key)
        if goals is None:
            return None
        return jax.tree_util.tree_map(lambda arr: np.roll(np.asarray(arr), shift=1, axis=0), goals)

    def _goal_bucket_masks(batch, prefix='value'):
        source = batch.get(f'{prefix}_goal_source')
        horizon = batch.get(f'{prefix}_goal_horizon')
        if source is None or horizon is None:
            return {}
        source = np.asarray(source)
        horizon = np.asarray(horizon)
        return {
            'current': source == 0,
            'traj': source == 1,
            'traj_short': (source == 1) & (horizon >= 1) & (horizon <= 5),
            'traj_medium': (source == 1) & (horizon >= 6) & (horizon <= 20),
            'traj_long': (source == 1) & (horizon >= 21),
            'random': source == 2,
        }

    def _extract_xy_batch(observations):
        if not hasattr(maze_env, 'xy_to_ij'):
            return None
        try:
            arr = np.asarray(observations)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[-1] < 2:
            return None
        if np.issubdtype(arr.dtype, np.integer) and arr.dtype.itemsize == 1:
            return None
        return arr[:, :2].astype(np.float32)

    def _maze_dist_map(goal_ij):
        key = tuple(int(x) for x in goal_ij)
        if key in maze_distance_cache:
            return maze_distance_cache[key]
        maze_map = np.asarray(getattr(maze_env, 'maze_map'))
        dist_map = np.full_like(maze_map, fill_value=-1, dtype=np.int32)
        if (
            key[0] < 0
            or key[0] >= maze_map.shape[0]
            or key[1] < 0
            or key[1] >= maze_map.shape[1]
            or maze_map[key[0], key[1]] != 0
        ):
            maze_distance_cache[key] = dist_map
            return dist_map
        queue = [key]
        dist_map[key[0], key[1]] = 0
        head = 0
        while head < len(queue):
            i, j = queue[head]
            head += 1
            next_dist = dist_map[i, j] + 1
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < maze_map.shape[0]
                    and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and dist_map[ni, nj] < 0
                ):
                    dist_map[ni, nj] = next_dist
                    queue.append((ni, nj))
        maze_distance_cache[key] = dist_map
        return dist_map

    def _maze_distance_metadata(batch, goal_key='value_goals'):
        if not hasattr(maze_env, 'xy_to_ij') or not hasattr(maze_env, 'maze_map'):
            return None
        obs_xy = _extract_xy_batch(batch.get('observations'))
        goal_xy = _extract_xy_batch(batch.get(goal_key))
        if obs_xy is None or goal_xy is None or obs_xy.shape != goal_xy.shape:
            return None
        euclidean = np.linalg.norm(goal_xy - obs_xy, axis=-1)
        path_len = np.full(obs_xy.shape[0], -1, dtype=np.int32)
        for idx, (start_xy, target_xy) in enumerate(zip(obs_xy, goal_xy)):
            try:
                start_ij = maze_env.xy_to_ij(start_xy)
                goal_ij = maze_env.xy_to_ij(target_xy)
            except Exception:
                continue
            dist_map = _maze_dist_map(goal_ij)
            si, sj = int(start_ij[0]), int(start_ij[1])
            if 0 <= si < dist_map.shape[0] and 0 <= sj < dist_map.shape[1]:
                path_len[idx] = int(dist_map[si, sj])
        return {
            'euclidean': euclidean,
            'path_len': path_len,
        }

    def _pearson_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if len(a) < 2:
            return np.nan
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        if denom <= 0:
            return np.nan
        return float(np.sum(a * b) / denom)

    def _rankdata_average_ties(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind='mergesort')
        ranks = np.empty(len(x), dtype=np.float64)
        i = 0
        while i < len(x):
            j = i + 1
            while j < len(x) and x[order[j]] == x[order[i]]:
                j += 1
            avg_rank = 0.5 * (i + j - 1)
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    def _spearman_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if len(a) < 2:
            return np.nan
        return _pearson_corr(_rankdata_average_ties(a), _rankdata_average_ties(b))

    def _log_path_geometry_metrics(metrics, prefix, scores, batch, goal_prefix='value'):
        maze_meta = _maze_distance_metadata(batch, goal_key=f'{goal_prefix}_goals')
        if maze_meta is None:
            return
        path_len = np.asarray(maze_meta['path_len'])
        scores = np.asarray(scores)
        finite_mask = path_len >= 0
        if np.sum(finite_mask) < 2:
            return
        finite_scores = scores[finite_mask]
        finite_path = path_len[finite_mask].astype(np.float64)
        neg_path = -finite_path
        metrics[f'{prefix}/count'] = int(np.sum(finite_mask))
        metrics[f'{prefix}/graph_corr'] = _pearson_corr(finite_scores, neg_path)
        metrics[f'{prefix}/graph_spearman'] = _spearman_corr(finite_scores, neg_path)
        if len(finite_path) < 3:
            return
        hard_threshold = float(np.quantile(finite_path, 2.0 / 3.0))
        hard_mask = finite_path >= hard_threshold
        metrics[f'{prefix}/hard_threshold'] = hard_threshold
        metrics[f'{prefix}/hard_count'] = int(np.sum(hard_mask))
        if np.sum(hard_mask) < 2:
            return
        hard_scores = finite_scores[hard_mask]
        hard_neg_path = -finite_path[hard_mask]
        metrics[f'{prefix}/hard_graph_corr'] = _pearson_corr(hard_scores, hard_neg_path)
        metrics[f'{prefix}/hard_graph_spearman'] = _spearman_corr(hard_scores, hard_neg_path)

    def _log_signal_metrics(metrics, prefix, pos_scores, neg_scores, batch=None, goal_prefix='value'):
        pos_scores = np.asarray(pos_scores)
        neg_scores = np.asarray(neg_scores)
        delta = pos_scores - neg_scores
        metrics[f'{prefix}/pos_mean'] = float(np.mean(pos_scores))
        metrics[f'{prefix}/pos_std'] = float(np.std(pos_scores))
        metrics[f'{prefix}/neg_mean'] = float(np.mean(neg_scores))
        metrics[f'{prefix}/neg_std'] = float(np.std(neg_scores))
        metrics[f'{prefix}/margin_mean'] = float(np.mean(delta))
        metrics[f'{prefix}/margin_std'] = float(np.std(delta))
        metrics[f'{prefix}/rank_acc'] = float(np.mean(delta > 0))
        pooled_std = np.sqrt(np.var(pos_scores) + np.var(neg_scores) + 1e-8)
        metrics[f'{prefix}/separation_z'] = float(np.mean(delta) / pooled_std)
        if batch is None:
            return
        bucket_masks = _goal_bucket_masks(batch, prefix=goal_prefix)
        if not bucket_masks:
            return
        source = batch.get(f'{goal_prefix}_goal_source')
        horizon = batch.get(f'{goal_prefix}_goal_horizon')
        if source is not None:
            source = np.asarray(source)
            for source_id, source_name in source_labels.items():
                metrics[f'{prefix}/source_{source_name}_frac'] = float(np.mean(source == source_id))
        if horizon is not None:
            horizon = np.asarray(horizon)
            traj_mask = source == 1 if source is not None else np.zeros_like(horizon, dtype=bool)
            if np.any(traj_mask):
                metrics[f'{prefix}/traj_horizon_mean'] = float(np.mean(horizon[traj_mask]))
                metrics[f'{prefix}/traj_horizon_std'] = float(np.std(horizon[traj_mask]))
        for bucket_name, mask in bucket_masks.items():
            mask = np.asarray(mask, dtype=bool)
            metrics[f'{prefix}/{bucket_name}_frac'] = float(np.mean(mask))
            if not np.any(mask):
                continue
            bucket_delta = delta[mask]
            metrics[f'{prefix}/{bucket_name}_margin_mean'] = float(np.mean(bucket_delta))
            metrics[f'{prefix}/{bucket_name}_rank_acc'] = float(np.mean(bucket_delta > 0))
            metrics[f'{prefix}/{bucket_name}_pos_mean'] = float(np.mean(pos_scores[mask]))
            metrics[f'{prefix}/{bucket_name}_neg_mean'] = float(np.mean(neg_scores[mask]))
        maze_meta = _maze_distance_metadata(batch, goal_key=f'{goal_prefix}_goals')
        if maze_meta is None:
            return
        euclidean = np.asarray(maze_meta['euclidean'])
        path_len = np.asarray(maze_meta['path_len'])
        finite_path = path_len >= 0
        metrics[f'{prefix}/maze_xy_distance_mean'] = float(np.mean(euclidean))
        metrics[f'{prefix}/maze_xy_distance_std'] = float(np.std(euclidean))
        metrics[f'{prefix}/maze_path_available_frac'] = float(np.mean(finite_path))
        if np.any(finite_path):
            metrics[f'{prefix}/maze_path_len_mean'] = float(np.mean(path_len[finite_path]))
            metrics[f'{prefix}/maze_path_len_std'] = float(np.std(path_len[finite_path]))
        maze_masks = {
            'maze_xy_short': euclidean <= 4.0,
            'maze_xy_medium': (euclidean > 4.0) & (euclidean <= 12.0),
            'maze_xy_long': euclidean > 12.0,
            'maze_path_short': finite_path & (path_len <= 3),
            'maze_path_medium': finite_path & (path_len >= 4) & (path_len <= 8),
            'maze_path_long': finite_path & (path_len >= 9),
        }
        for bucket_name, mask in maze_masks.items():
            metrics[f'{prefix}/{bucket_name}_frac'] = float(np.mean(mask))
            if not np.any(mask):
                continue
            bucket_delta = delta[mask]
            metrics[f'{prefix}/{bucket_name}_margin_mean'] = float(np.mean(bucket_delta))
            metrics[f'{prefix}/{bucket_name}_rank_acc'] = float(np.mean(bucket_delta > 0))

    def _all_bucket_masks(batch, prefix='value'):
        masks = dict(_goal_bucket_masks(batch, prefix=prefix))
        maze_meta = _maze_distance_metadata(batch, goal_key=f'{prefix}_goals')
        if maze_meta is None:
            return masks
        euclidean = np.asarray(maze_meta['euclidean'])
        path_len = np.asarray(maze_meta['path_len'])
        finite_path = path_len >= 0
        masks.update(
            {
                'maze_xy_short': euclidean <= 4.0,
                'maze_xy_medium': (euclidean > 4.0) & (euclidean <= 12.0),
                'maze_xy_long': euclidean > 12.0,
                'maze_path_short': finite_path & (path_len <= 3),
                'maze_path_medium': finite_path & (path_len >= 4) & (path_len <= 8),
                'maze_path_long': finite_path & (path_len >= 9),
            }
        )
        return masks

    def _log_task_distribution(metrics, batch, prefix='batch'):
        candidate_keys = (
            'task_id',
            'task_ids',
            'value_task_id',
            'value_task_ids',
            'dataset_task_id',
            'dataset_task_ids',
        )
        task_ids = None
        for key in candidate_keys:
            if key in batch:
                task_ids = np.asarray(batch[key]).reshape(-1)
                break
        if task_ids is None or task_ids.size == 0:
            return

        try:
            valid_mask = np.isfinite(task_ids)
        except Exception:
            valid_mask = np.ones(task_ids.shape, dtype=bool)
        task_ids = task_ids[valid_mask]
        if task_ids.size == 0:
            return

        unique_task_ids = np.unique(task_ids)
        metrics[f'{prefix}/task_id_unique_count'] = int(unique_task_ids.size)
        for task_id in unique_task_ids:
            try:
                task_label = int(task_id)
            except Exception:
                continue
            metrics[f'{prefix}/task_id_{task_label}_frac'] = float(np.mean(task_ids == task_id))

    def _log_scalar_metrics(metrics, prefix, values, batch=None, goal_prefix='value'):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        metrics[f'{prefix}_mean'] = float(np.mean(values))
        metrics[f'{prefix}_std'] = float(np.std(values))
        metrics[f'{prefix}_p10'] = float(np.quantile(values, 0.10))
        metrics[f'{prefix}_p50'] = float(np.quantile(values, 0.50))
        metrics[f'{prefix}_p90'] = float(np.quantile(values, 0.90))
        if batch is None:
            return
        for bucket_name, mask in _all_bucket_masks(batch, prefix=goal_prefix).items():
            mask = np.asarray(mask, dtype=bool)
            metrics[f'{prefix}/{bucket_name}_frac'] = float(np.mean(mask))
            if not np.any(mask):
                continue
            bucket_values = values[mask]
            metrics[f'{prefix}/{bucket_name}_mean'] = float(np.mean(bucket_values))
            metrics[f'{prefix}/{bucket_name}_p90'] = float(np.quantile(bucket_values, 0.90))

    def _log_pair_preference_metrics(
        metrics,
        prefix,
        preferred_scores,
        reference_scores,
        *,
        batch=None,
        goal_prefix='value',
        preferred_name='preferred',
        reference_name='reference',
        delta_name='delta',
        prefer_name='prefer_preferred_frac',
    ):
        preferred_scores = np.asarray(preferred_scores, dtype=np.float64)
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        delta = preferred_scores - reference_scores
        metrics[f'{prefix}/{preferred_name}_mean'] = float(np.mean(preferred_scores))
        metrics[f'{prefix}/{reference_name}_mean'] = float(np.mean(reference_scores))
        metrics[f'{prefix}/{delta_name}_mean'] = float(np.mean(delta))
        metrics[f'{prefix}/{delta_name}_std'] = float(np.std(delta))
        metrics[f'{prefix}/{delta_name}_p10'] = float(np.quantile(delta, 0.10))
        metrics[f'{prefix}/{delta_name}_p50'] = float(np.quantile(delta, 0.50))
        metrics[f'{prefix}/{delta_name}_p90'] = float(np.quantile(delta, 0.90))
        metrics[f'{prefix}/{prefer_name}'] = float(np.mean(delta > 0))
        if batch is None:
            return
        for bucket_name, mask in _all_bucket_masks(batch, prefix=goal_prefix).items():
            mask = np.asarray(mask, dtype=bool)
            metrics[f'{prefix}/{bucket_name}_frac'] = float(np.mean(mask))
            if not np.any(mask):
                continue
            bucket_delta = delta[mask]
            metrics[f'{prefix}/{bucket_name}_{delta_name}_mean'] = float(np.mean(bucket_delta))
            metrics[f'{prefix}/{bucket_name}_{delta_name}_p90'] = float(np.quantile(bucket_delta, 0.90))
            metrics[f'{prefix}/{bucket_name}_{prefer_name}'] = float(np.mean(bucket_delta > 0))

    def _reduce_aux_per_example(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
        reduce_axes = tuple(range(arr.ndim - 1))
        return np.mean(arr, axis=reduce_axes)

    def _reduce_aux_stepwise(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim < 2:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        step_axis = arr.ndim - 2
        batch_axis = arr.ndim - 1
        reduce_axes = tuple(i for i in range(arr.ndim) if i not in (step_axis, batch_axis))
        return np.mean(arr, axis=reduce_axes)

    def _value_forward(eval_agent, observations, goals, module_name='value', *, num_iters=None):
        value_call = eval_agent.network.select(module_name)
        if _is_recurrent_backbone(value_backbone) and num_iters is not None:
            return value_call(observations, goals, num_iters=num_iters)
        return value_call(observations, goals)

    def _score_value_batch(eval_agent, observations, goals, module_name='value', *, num_iters=None):
        values = _value_forward(eval_agent, observations, goals, module_name=module_name, num_iters=num_iters)
        return _reduced_scores(values, 'value')

    def _critic_forward(eval_agent, observations, goals, actions, *, num_iters=None, return_aux=False):
        critic_num_iters = config.get('critic_eval_k', None) if num_iters is None else num_iters
        critic_call = eval_agent.network.select('critic')
        if _is_recurrent_backbone(critic_backbone):
            return critic_call(
                observations,
                goals,
                actions,
                num_iters=critic_num_iters,
                return_aux=return_aux,
            )
        if return_aux:
            values = critic_call(observations, goals, actions)
            return values, {}
        return critic_call(observations, goals, actions)

    def _score_critic_batch(eval_agent, observations, goals, actions, *, num_iters=None):
        values = _critic_forward(eval_agent, observations, goals, actions, num_iters=num_iters)
        return _reduced_scores(values, 'critic')

    def _add_compute_metrics(metrics, step: int, total_time: float):
        samples_seen = int(step) * int(config['batch_size'])
        total_count = int(static_param_metrics.get('params/total_count', 0) or 0)
        critic_count = int(static_param_metrics.get('params/critic_count', 0) or 0)
        value_count = int(static_param_metrics.get('params/value_count', 0) or 0)
        metrics['time/hours_elapsed'] = float(total_time / 3600.0)
        metrics['time/samples_per_second'] = float(
            metrics.get('time/sps', samples_seen / max(total_time, 1e-8))
        )
        metrics['compute/total_param_updates'] = float(total_count * int(step))
        metrics['compute/critic_param_updates'] = float(critic_count * int(step))
        metrics['compute/value_param_updates'] = float(value_count * int(step))
        metrics['compute/total_param_samples'] = float(total_count * samples_seen)
        metrics['compute/critic_param_samples'] = float(critic_count * samples_seen)
        metrics['compute/value_param_samples'] = float(value_count * samples_seen)

    def _maybe_log_actor_quality(metrics, eval_agent, eval_batch):
        try:
            if not config.get('discrete', False) and 'actor' in actor_module_names and 'actions' in eval_batch:
                actor_goals = eval_batch.get('actor_goals', eval_batch.get('value_goals'))
                if actor_goals is not None:
                    actor_goal_prefix = 'actor' if f'actor_goal_source' in eval_batch else 'value'
                    dist = eval_agent.network.select('actor')(eval_batch['observations'], actor_goals)
                    actor_actions = np.asarray(dist.mode())
                    behavior_actions = np.asarray(eval_batch['actions'])
                    diff = actor_actions - behavior_actions
                    per_example_mse = np.mean(diff**2, axis=-1)
                    per_example_l2 = np.linalg.norm(diff, axis=-1)
                    metrics['evaluation/actor_behavior_log_prob'] = float(np.mean(np.asarray(dist.log_prob(eval_batch['actions']))))
                    _log_scalar_metrics(
                        metrics,
                        'evaluation/policy_behavior/mse',
                        per_example_mse,
                        batch=eval_batch,
                        goal_prefix=actor_goal_prefix,
                    )
                    _log_scalar_metrics(
                        metrics,
                        'evaluation/policy_behavior/l2',
                        per_example_l2,
                        batch=eval_batch,
                        goal_prefix=actor_goal_prefix,
                    )
                    if critic_module_name is not None:
                        q_actor = _score_critic_batch(
                            eval_agent, eval_batch['observations'], actor_goals, actor_actions
                        )
                        q_behavior = _score_critic_batch(
                            eval_agent, eval_batch['observations'], actor_goals, behavior_actions
                        )
                        _log_pair_preference_metrics(
                            metrics,
                            'evaluation/actor_critic',
                            q_actor,
                            q_behavior,
                            batch=eval_batch,
                            goal_prefix=actor_goal_prefix,
                            preferred_name='q_actor',
                            reference_name='q_behavior',
                            delta_name='q_delta',
                            prefer_name='prefer_actor_frac',
                        )
        except Exception:
            pass

        try:
            if not config.get('discrete', False) and has_goal_rep_module and 'low_actor' in actor_module_names and 'low_actor_goals' in eval_batch:
                goal_reps = eval_agent.network.select('goal_rep')(
                    jnp.concatenate([eval_batch['observations'], eval_batch['low_actor_goals']], axis=-1)
                )
                dist = eval_agent.network.select('low_actor')(eval_batch['observations'], goal_reps, goal_encoded=True)
                mode = np.asarray(dist.mode())
                behavior_actions = np.asarray(eval_batch['actions'])
                diff = mode - behavior_actions
                metrics['evaluation/low_actor_behavior_mse'] = float(np.mean(diff**2))
                metrics['evaluation/low_actor_behavior_l2'] = float(np.mean(np.linalg.norm(diff, axis=-1)))
                metrics['evaluation/low_actor_behavior_log_prob'] = float(np.mean(np.asarray(dist.log_prob(eval_batch['actions']))))
        except Exception:
            pass

        try:
            if has_goal_rep_module and 'high_actor' in actor_module_names and 'high_actor_goals' in eval_batch and 'high_actor_targets' in eval_batch:
                dist = eval_agent.network.select('high_actor')(eval_batch['observations'], eval_batch['high_actor_goals'])
                target = eval_agent.network.select('goal_rep')(
                    jnp.concatenate([eval_batch['observations'], eval_batch['high_actor_targets']], axis=-1)
                )
                mode = np.asarray(dist.mode())
                target = np.asarray(target)
                diff = mode - target
                cosine = np.sum(mode * target, axis=-1) / (
                    np.linalg.norm(mode, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-8
                )
                metrics['evaluation/high_actor_target_mse'] = float(np.mean(diff**2))
                metrics['evaluation/high_actor_target_l2'] = float(np.mean(np.linalg.norm(diff, axis=-1)))
                metrics['evaluation/high_actor_target_cosine'] = float(np.mean(cosine))
                metrics['evaluation/high_actor_target_log_prob'] = float(np.mean(np.asarray(dist.log_prob(target))))
        except Exception:
            pass

    modules = getattr(getattr(agent.network, 'model_def', None), 'modules', {})
    has_value = hasattr(modules, 'keys') and 'value' in modules
    has_actor = hasattr(modules, 'keys') and 'actor' in modules
    value_backbone = str(config.get('value_backbone', ''))
    critic_backbone = str(config.get('critic_backbone', ''))
    probe_dataset = val_dataset if val_dataset is not None else train_dataset
    probe_batch_cache = None
    probe_idxs_cache = None
    best_eval_metrics = {
        'success': -np.inf,
        'episode.return': -np.inf,
        'normalized_score': -np.inf,
    }
    suppressed_eval_metric_keys = {
        'evaluation/overall_normalized_score',
        'evaluation/overall_episode.return',
        'evaluation/normalized_score',
        'evaluation/final_success',
        'evaluation/final_normalized_score',
        'evaluation/final_episode.return',
        'evaluation/best_so_far_normalized_score',
        'evaluation/best_so_far_episode.return',
        # Low-signal debug metrics that are not part of the current mechanism story.
        'evaluation/critic_q1_abs_max',
        'evaluation/critic_q2_abs_max',
        'evaluation/low_actor_behavior_mse',
        'evaluation/low_actor_behavior_l2',
        'evaluation/low_actor_behavior_log_prob',
        'evaluation/high_actor_target_mse',
        'evaluation/high_actor_target_l2',
        'evaluation/high_actor_target_cosine',
        'evaluation/high_actor_target_log_prob',
    }
    eval_count = 0
    last_eval_step = None
    last_completed_step = 0
    dump_targets = sorted(refine_dump_steps or {1, max(1, FLAGS.train_steps // 2), FLAGS.train_steps})
    completed_dump_targets = set()

    def _get_probe_batch():
        nonlocal probe_batch_cache, probe_idxs_cache
        if probe_batch_cache is not None:
            return probe_batch_cache, probe_idxs_cache
        raw_dataset = getattr(probe_dataset, 'dataset', probe_dataset)
        valid_idxs = getattr(raw_dataset, 'valid_idxs', np.arange(raw_dataset.size))
        if len(valid_idxs) == 0:
            probe_idxs = np.arange(min(FLAGS.refine_probe_size, raw_dataset.size))
        else:
            rng = np.random.RandomState(FLAGS.seed + 17)
            replace = len(valid_idxs) < FLAGS.refine_probe_size
            probe_idxs = rng.choice(valid_idxs, size=min(FLAGS.refine_probe_size, len(valid_idxs) if not replace else FLAGS.refine_probe_size), replace=replace)
        np_state = np.random.get_state()
        try:
            np.random.seed(FLAGS.seed + 23)
            probe_batch = _sample_dataset(probe_dataset, len(probe_idxs), idxs=probe_idxs, evaluation=True)
        finally:
            np.random.set_state(np_state)
        if config['discrete']:
            probe_batch['actions'] = np.full_like(probe_batch['actions'], env.action_space.n - 1)
        probe_batch_cache = probe_batch
        probe_idxs_cache = np.asarray(probe_idxs)
        return probe_batch_cache, probe_idxs_cache

    def _maybe_log_fixed_probe_metrics(eval_metrics, eval_agent):
        """Log comparable fixed-probe diagnostics for both MLP and recurrent models."""
        probe_batch, _ = _get_probe_batch()
        probe_goals = probe_batch.get('value_goals', probe_batch.get('goals'))
        if probe_goals is None:
            return
        probe_negative_goals = _negative_goal_batch(probe_batch, goal_key='value_goals')
        behavior_actions = np.asarray(probe_batch['actions']) if 'actions' in probe_batch else None

        has_probe_critic = critic_module_name is not None
        if has_probe_critic and behavior_actions is not None:
            critic_values = _critic_forward(
                eval_agent,
                probe_batch['observations'],
                probe_goals,
                behavior_actions,
            )
            critic_values = np.asarray(critic_values)
            q_pos = _reduced_scores(critic_values, 'critic')
            eval_metrics['evaluation/probe/critic_mean'] = float(np.mean(critic_values))
            eval_metrics['evaluation/probe/critic_std'] = float(np.std(critic_values))
            if probe_negative_goals is not None:
                q_neg = _score_critic_batch(
                    eval_agent,
                    probe_batch['observations'],
                    probe_negative_goals,
                    behavior_actions,
                )
                _log_signal_metrics(
                    eval_metrics,
                    'evaluation/probe/critic_signal',
                    q_pos,
                    q_neg,
                    batch=probe_batch,
                    goal_prefix='value',
                )
            _log_path_geometry_metrics(
                eval_metrics,
                'evaluation/probe/critic_geometry',
                q_pos,
                probe_batch,
                goal_prefix='value',
            )

        if has_value:
            value_values = eval_agent.network.select('value')(
                probe_batch['observations'],
                probe_goals,
            )
            value_values = np.asarray(value_values)
            v_pos = _reduced_scores(value_values, 'value')
            eval_metrics['evaluation/probe/value_mean'] = float(np.mean(value_values))
            eval_metrics['evaluation/probe/value_std'] = float(np.std(value_values))
            if probe_negative_goals is not None:
                v_neg = _score_value_batch(eval_agent, probe_batch['observations'], probe_negative_goals)
                _log_signal_metrics(
                    eval_metrics,
                    'evaluation/probe/value_signal',
                    v_pos,
                    v_neg,
                    batch=probe_batch,
                    goal_prefix='value',
                )
            _log_path_geometry_metrics(
                eval_metrics,
                'evaluation/probe/value_geometry',
                v_pos,
                probe_batch,
                goal_prefix='value',
            )

        if config.get('discrete', False) or behavior_actions is None or 'actor' not in actor_module_names:
            return

        actor_goals = probe_batch.get('actor_goals', probe_goals)
        if actor_goals is None:
            return
        try:
            dist = eval_agent.network.select('actor')(probe_batch['observations'], actor_goals)
            actor_actions = np.asarray(dist.mode())
            diff = actor_actions - behavior_actions
            _log_scalar_metrics(
                eval_metrics,
                'evaluation/probe/policy_behavior/mse',
                np.mean(diff**2, axis=-1),
                batch=probe_batch,
                goal_prefix='actor' if 'actor_goal_source' in probe_batch else 'value',
            )
            _log_scalar_metrics(
                eval_metrics,
                'evaluation/probe/policy_behavior/l2',
                np.linalg.norm(diff, axis=-1),
                batch=probe_batch,
                goal_prefix='actor' if 'actor_goal_source' in probe_batch else 'value',
            )
            eval_metrics['evaluation/probe/actor_behavior_log_prob'] = float(
                np.mean(np.asarray(dist.log_prob(probe_batch['actions'])))
            )
        except Exception:
            return

        actor_goal_prefix = 'actor' if 'actor_goal_source' in probe_batch else 'value'
        if has_probe_critic:
            q_actor = _score_critic_batch(
                eval_agent,
                probe_batch['observations'],
                actor_goals,
                actor_actions,
            )
            q_behavior = _score_critic_batch(
                eval_agent,
                probe_batch['observations'],
                actor_goals,
                behavior_actions,
            )
            _log_pair_preference_metrics(
                eval_metrics,
                'evaluation/probe/actor_critic',
                q_actor,
                q_behavior,
                batch=probe_batch,
                goal_prefix=actor_goal_prefix,
                preferred_name='q_actor',
                reference_name='q_behavior',
                delta_name='q_delta',
                prefer_name='prefer_actor_frac',
            )

        if has_value and has_probe_critic:
            # Advantage-weighted actor extraction uses behavior-action advantages to weight BC.
            v_actor_goal = _score_value_batch(eval_agent, probe_batch['observations'], actor_goals)
            q_behavior = _score_critic_batch(
                eval_agent,
                probe_batch['observations'],
                actor_goals,
                behavior_actions,
            )
            adv = q_behavior - v_actor_goal
            scaled_adv = np.minimum(float(config.get('alpha', 1.0)) * adv, np.log(100.0))
            awr_weight = np.exp(scaled_adv)
            weight_sum = np.sum(awr_weight)
            ess = (weight_sum**2) / (np.sum(awr_weight**2) + 1e-8)
            _log_scalar_metrics(
                eval_metrics,
                'evaluation/probe/actor_value/adv',
                adv,
                batch=probe_batch,
                goal_prefix=actor_goal_prefix,
            )
            _log_scalar_metrics(
                eval_metrics,
                'evaluation/probe/actor_value/awr_weight',
                awr_weight,
                batch=probe_batch,
                goal_prefix=actor_goal_prefix,
            )
            eval_metrics['evaluation/probe/actor_value/positive_adv_frac'] = float(np.mean(adv > 0))
            eval_metrics['evaluation/probe/actor_value/awr_weight_ess_frac'] = float(
                ess / max(1, len(awr_weight))
            )

    def _maybe_log_recurrent_refinement(eval_metrics, eval_agent, *, step: int):
        has_recurrent_value = value_module_name is not None and _is_recurrent_backbone(value_backbone)
        has_recurrent_critic = critic_module_name is not None and _is_recurrent_backbone(critic_backbone)
        if not has_recurrent_value and not has_recurrent_critic:
            return
        probe_batch, probe_idxs = _get_probe_batch()
        probe_goals = probe_batch.get('value_goals', probe_batch.get('goals'))
        if probe_goals is None:
            return
        probe_negative_goals = _negative_goal_batch(probe_batch, goal_key='value_goals')
        actor_probe_goals = probe_batch.get('actor_goals', probe_goals)
        actor_goal_prefix = 'actor' if 'actor_goal_source' in probe_batch else 'value'
        maze_meta_value = _maze_distance_metadata(probe_batch, goal_key='value_goals')
        maze_meta_actor = _maze_distance_metadata(probe_batch, goal_key='actor_goals')
        value_steps = []
        hidden_norm_steps = []
        critic_pos_steps = []
        critic_neg_steps = []
        critic_margin_steps = []
        critic_hidden_norm_steps = []
        actor_critic_deltas = []
        critic_final_aux = None

        if has_recurrent_value:
            total_iters = int(config.get('value_k', 1))
            target_values = None
            target_value_dump_key = None
            for k in range(1, total_iters + 1):
                values_k, aux_k = eval_agent.network.select('value')(
                    probe_batch['observations'],
                    probe_goals,
                    return_aux=True,
                    num_iters=k,
                )
                values_k = np.asarray(values_k)
                value_steps.append(values_k)
                hidden_norm = aux_k.get('final_hidden_norm') if isinstance(aux_k, dict) else None
                hidden_norm_steps.append(None if hidden_norm is None else np.asarray(hidden_norm))
                eval_metrics[f'evaluation/refine/value_step_{k}_mean'] = float(values_k.mean())
                eval_metrics[f'evaluation/refine/value_step_{k}_std'] = float(values_k.std())
                if probe_negative_goals is not None:
                    neg_values_k = np.asarray(
                        eval_agent.network.select('value')(
                            probe_batch['observations'],
                            probe_negative_goals,
                            num_iters=k,
                        )
                    )
                    _log_signal_metrics(
                        eval_metrics,
                        f'evaluation/refine/value_step_{k}_signal',
                        _reduced_scores(values_k, 'value'),
                        _reduced_scores(neg_values_k, 'value'),
                        batch=probe_batch,
                        goal_prefix='value',
                    )
                _log_path_geometry_metrics(
                    eval_metrics,
                    f'evaluation/refine/value_step_{k}_geometry',
                    _reduced_scores(values_k, 'value'),
                    probe_batch,
                    goal_prefix='value',
                )
                if hidden_norm is not None:
                    eval_metrics[f'evaluation/refine/value_hidden_norm_step_{k}_mean'] = float(np.asarray(hidden_norm).mean())
                if k > 1:
                    delta_k = np.asarray(value_steps[-1] - value_steps[-2])
                    eval_metrics[f'evaluation/refine/value_delta_step_{k}_mean_abs'] = float(np.mean(np.abs(delta_k)))

            final_values = value_steps[-1]
            if target_value_module_name is not None:
                target_values = np.asarray(
                    eval_agent.network.select('target_value')(
                        probe_batch['observations'],
                        probe_goals,
                    )
                )
                target_value_dump_key = 'target_value'
                eval_metrics['evaluation/refine/target_value_mean'] = float(target_values.mean())
                eval_metrics['evaluation/refine/target_value_std'] = float(target_values.std())
                eval_metrics['evaluation/refine/final_target_mse'] = float(np.mean((final_values - target_values) ** 2))

            elif target_critic_module_name is not None and 'actions' in probe_batch:
                target_q = np.asarray(
                    eval_agent.network.select('target_critic')(
                        probe_batch['observations'],
                        probe_goals,
                        probe_batch['actions'],
                    )
                )
                target_values = _reduced_scores(target_q, 'critic')
                target_value_dump_key = 'target_critic_on_data'
                eval_metrics['evaluation/refine/target_critic_mean'] = float(target_values.mean())
                eval_metrics['evaluation/refine/target_critic_std'] = float(target_values.std())
                eval_metrics['evaluation/refine/final_target_critic_mse'] = float(np.mean((final_values - target_values) ** 2))

            if (
                target_value_module_name is not None
                and 'rewards' in probe_batch
                and 'masks' in probe_batch
                and 'next_observations' in probe_batch
            ):
                next_target_values = np.asarray(
                    eval_agent.network.select('target_value')(
                        probe_batch['next_observations'],
                        probe_goals,
                    )
                )
                td_target = np.asarray(probe_batch['rewards']) + float(config['discount']) * np.asarray(
                    probe_batch['masks']
                ) * next_target_values
                eval_metrics['evaluation/refine/td_target_mean'] = float(td_target.mean())
                eval_metrics['evaluation/refine/final_td_target_mse'] = float(np.mean((final_values - td_target) ** 2))

        if has_recurrent_critic and 'actions' in probe_batch:
            total_iters = int(config.get('critic_k', 1))
            behavior_actions = np.asarray(probe_batch['actions'])
            actor_actions = None
            if not config.get('discrete', False) and 'actor' in actor_module_names and actor_probe_goals is not None:
                try:
                    actor_dist = eval_agent.network.select('actor')(probe_batch['observations'], actor_probe_goals)
                    actor_actions = np.asarray(actor_dist.mode())
                except Exception:
                    actor_actions = None

            for k in range(1, total_iters + 1):
                critic_values_k, critic_aux_k = _critic_forward(
                    eval_agent,
                    probe_batch['observations'],
                    probe_goals,
                    behavior_actions,
                    num_iters=k,
                    return_aux=True,
                )
                critic_values_k = np.asarray(critic_values_k)
                q_pos_k = _reduced_scores(critic_values_k, 'critic')
                critic_pos_steps.append(q_pos_k)
                eval_metrics[f'evaluation/refine/critic_step_{k}_mean'] = float(np.mean(critic_values_k))
                eval_metrics[f'evaluation/refine/critic_step_{k}_std'] = float(np.std(critic_values_k))
                if probe_negative_goals is not None:
                    neg_values_k = np.asarray(
                        _critic_forward(
                            eval_agent,
                            probe_batch['observations'],
                            probe_negative_goals,
                            behavior_actions,
                            num_iters=k,
                        )
                    )
                    q_neg_k = _reduced_scores(neg_values_k, 'critic')
                    critic_neg_steps.append(q_neg_k)
                    _log_signal_metrics(
                        eval_metrics,
                        f'evaluation/refine/critic_step_{k}_signal',
                        q_pos_k,
                        q_neg_k,
                        batch=probe_batch,
                        goal_prefix='value',
                    )
                    critic_margin_steps.append(q_pos_k - q_neg_k)
                _log_path_geometry_metrics(
                    eval_metrics,
                    f'evaluation/refine/critic_step_{k}_geometry',
                    q_pos_k,
                    probe_batch,
                    goal_prefix='value',
                )
                hidden_norm_total = None
                if isinstance(critic_aux_k, dict):
                    hidden_norm_total = critic_aux_k.get('final_hidden_norm_total')
                    if hidden_norm_total is None:
                        phi_hidden = critic_aux_k.get('final_hidden_norm_phi')
                        psi_hidden = critic_aux_k.get('final_hidden_norm_psi')
                        if phi_hidden is not None and psi_hidden is not None:
                            hidden_norm_total = np.sqrt(np.asarray(phi_hidden) ** 2 + np.asarray(psi_hidden) ** 2)
                critic_hidden_norm_steps.append(None if hidden_norm_total is None else np.asarray(hidden_norm_total))
                if hidden_norm_total is not None:
                    eval_metrics[f'evaluation/refine/critic_hidden_norm_step_{k}_mean'] = float(
                        np.asarray(hidden_norm_total).mean()
                    )
                if isinstance(critic_aux_k, dict):
                    hidden_drift = critic_aux_k.get('hidden_drift_mean')
                    if hidden_drift is not None:
                        hidden_drift = _reduce_aux_per_example(hidden_drift)
                        eval_metrics[f'evaluation/refine/critic_hidden_drift_step_{k}_mean'] = float(
                            np.mean(hidden_drift)
                        )
                    if k == total_iters:
                        critic_final_aux = {
                            key: np.asarray(value)
                            for key, value in critic_aux_k.items()
                            if value is not None
                        }
                if k > 1:
                    delta_k = critic_pos_steps[-1] - critic_pos_steps[-2]
                    eval_metrics[f'evaluation/refine/critic_delta_step_{k}_mean_abs'] = float(np.mean(np.abs(delta_k)))

                if actor_actions is not None and actor_probe_goals is not None:
                    q_actor_k = _score_critic_batch(
                        eval_agent,
                        probe_batch['observations'],
                        actor_probe_goals,
                        actor_actions,
                        num_iters=k,
                    )
                    q_behavior_k = _score_critic_batch(
                        eval_agent,
                        probe_batch['observations'],
                        actor_probe_goals,
                        behavior_actions,
                        num_iters=k,
                        )
                    actor_critic_deltas.append(q_actor_k - q_behavior_k)
                    _log_pair_preference_metrics(
                        eval_metrics,
                        f'evaluation/refine/critic_step_{k}_actor_critic',
                        q_actor_k,
                        q_behavior_k,
                        batch=probe_batch,
                        goal_prefix=actor_goal_prefix,
                        preferred_name='q_actor',
                        reference_name='q_behavior',
                            delta_name='q_delta',
                            prefer_name='prefer_actor_frac',
                        )

            if len(critic_margin_steps) >= 2:
                margin_gain = critic_margin_steps[-1] - critic_margin_steps[0]
                _log_scalar_metrics(
                    eval_metrics,
                    f'evaluation/refine/critic_margin_gain_1_to_{total_iters}',
                    margin_gain,
                    batch=probe_batch,
                    goal_prefix='value',
                )
            if len(actor_critic_deltas) >= 2:
                actor_q_delta_gain = actor_critic_deltas[-1] - actor_critic_deltas[0]
                _log_scalar_metrics(
                    eval_metrics,
                    f'evaluation/refine/critic_actor_q_delta_gain_1_to_{total_iters}',
                    actor_q_delta_gain,
                    batch=probe_batch,
                    goal_prefix=actor_goal_prefix,
                )
            if critic_final_aux is not None:
                scalar_aux_metrics = {
                    'evaluation/refine/critic_hidden_drift': 'hidden_drift_mean',
                }
                for metric_prefix, aux_key in scalar_aux_metrics.items():
                    values = critic_final_aux.get(aux_key)
                    if values is None:
                        continue
                    _log_scalar_metrics(
                        eval_metrics,
                        metric_prefix,
                        _reduce_aux_per_example(values),
                        batch=probe_batch,
                        goal_prefix='value',
                    )

        should_dump = False
        for target in dump_targets:
            if step >= target and target not in completed_dump_targets:
                completed_dump_targets.add(target)
                should_dump = True
        if not should_dump:
            return

        refine_dir = os.path.join(FLAGS.save_dir, 'refine_probes')
        os.makedirs(refine_dir, exist_ok=True)
        dump_dict = {
            'probe_idxs': probe_idxs,
            'step': np.asarray([step], dtype=np.int32),
        }
        for key in ['value_goal_source', 'value_goal_horizon', 'actor_goal_source', 'actor_goal_horizon']:
            if key in probe_batch:
                dump_dict[key] = np.asarray(probe_batch[key])
        if maze_meta_value is not None:
            dump_dict['value_goal_maze_xy_distance'] = np.asarray(maze_meta_value['euclidean'])
            dump_dict['value_goal_maze_path_len'] = np.asarray(maze_meta_value['path_len'])
        if maze_meta_actor is not None:
            dump_dict['actor_goal_maze_xy_distance'] = np.asarray(maze_meta_actor['euclidean'])
            dump_dict['actor_goal_maze_path_len'] = np.asarray(maze_meta_actor['path_len'])
        if has_recurrent_value and value_steps:
            if target_values is not None and target_value_dump_key is not None:
                dump_dict[target_value_dump_key] = target_values
            for idx, values_k in enumerate(value_steps, start=1):
                dump_dict[f'value_step_{idx}'] = values_k
                if idx > 1:
                    dump_dict[f'value_delta_step_{idx}'] = values_k - value_steps[idx - 2]
                hidden_norm = hidden_norm_steps[idx - 1]
                if hidden_norm is not None:
                    dump_dict[f'value_hidden_norm_step_{idx}'] = hidden_norm
        if has_recurrent_critic and critic_pos_steps:
            for idx, q_pos_k in enumerate(critic_pos_steps, start=1):
                dump_dict[f'critic_step_{idx}_pos'] = q_pos_k
                if idx > 1:
                    dump_dict[f'critic_delta_step_{idx}'] = q_pos_k - critic_pos_steps[idx - 2]
                if idx - 1 < len(critic_neg_steps):
                    q_neg_k = critic_neg_steps[idx - 1]
                    dump_dict[f'critic_step_{idx}_neg'] = q_neg_k
                    dump_dict[f'critic_step_{idx}_margin'] = q_pos_k - q_neg_k
                hidden_norm = critic_hidden_norm_steps[idx - 1]
                if hidden_norm is not None:
                    dump_dict[f'critic_hidden_norm_step_{idx}'] = hidden_norm
                if idx - 1 < len(actor_critic_deltas):
                    dump_dict[f'critic_step_{idx}_actor_q_delta'] = actor_critic_deltas[idx - 1]
            if len(critic_margin_steps) >= 2:
                dump_dict[f'critic_margin_gain_1_to_{len(critic_margin_steps)}'] = critic_margin_steps[-1] - critic_margin_steps[0]
            if len(actor_critic_deltas) >= 2:
                dump_dict[f'critic_actor_q_delta_gain_1_to_{len(actor_critic_deltas)}'] = (
                    actor_critic_deltas[-1] - actor_critic_deltas[0]
                )
            if critic_final_aux is not None:
                final_dump_scalars = {
                    'critic_hidden_drift_final': 'hidden_drift_mean',
                }
                for dump_key, aux_key in final_dump_scalars.items():
                    values = critic_final_aux.get(aux_key)
                    if values is not None:
                        dump_dict[dump_key] = _reduce_aux_per_example(values)
        np.savez_compressed(os.path.join(refine_dir, f'step_{step:09d}.npz'), **dump_dict)

    def _run_evaluation_impl(*, step: int, eval_agent, mark_final: bool = False):
        nonlocal eval_count, last_eval_step
        eval_start = time.time()
        renders = []
        eval_count += 1
        last_eval_step = step
        eval_metrics = dict(static_metadata)
        eval_metrics.update(static_param_metrics)
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
        for task_id in tqdm.trange(1, num_tasks + 1, disable=FLAGS.disable_tqdm):
            task_name = task_infos[task_id - 1]['task_name']
            eval_info, trajs, cur_renders = evaluate(
                agent=eval_agent,
                env=env,
                task_id=task_id,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
                refine_steps=FLAGS.eval_refine_steps,
                refine_lr=FLAGS.eval_refine_lr,
                refine_l2=FLAGS.eval_refine_l2,
                episode_seed_base=FLAGS.seed,
            )
            if FLAGS.dump_eval_trajs:
                traj_dir = FLAGS.eval_traj_dir or os.path.join(FLAGS.save_dir, 'eval_trajs')
                os.makedirs(traj_dir, exist_ok=True)
                for episode_idx, traj in enumerate(trajs):
                    infos = traj.get('info', [])
                    successes = np.asarray(
                        [
                            float(info.get('success', np.nan)) if isinstance(info, dict) else np.nan
                            for info in infos
                        ],
                        dtype=np.float32,
                    )
                    final_success = float(successes[-1]) if successes.size else np.nan
                    np.savez_compressed(
                        os.path.join(traj_dir, f'step_{step:09d}_{task_name}_ep{episode_idx:03d}.npz'),
                        task_id=np.asarray(task_id, dtype=np.int32),
                        task_name=np.asarray(task_name),
                        episode_idx=np.asarray(episode_idx, dtype=np.int32),
                        observation=np.asarray(traj.get('observation', [])),
                        next_observation=np.asarray(traj.get('next_observation', [])),
                        action=np.asarray(traj.get('action', [])),
                        goal=np.asarray(traj.get('goal', [])),
                        reward=np.asarray(traj.get('reward', []), dtype=np.float32),
                        done=np.asarray(traj.get('done', []), dtype=np.bool_),
                        success=successes,
                        final_success=np.asarray(final_success, dtype=np.float32),
                    )
            renders.extend(cur_renders)
            metric_names = ['success', 'episode.return']
            if FLAGS.eval_refine_steps and FLAGS.eval_refine_steps > 0:
                metric_names += [
                    'refine.q_pre',
                    'refine.q_post',
                    'refine.delta_a',
                    'refine.q_improve',
                    'refine.grad_norm_mean',
                    'refine.grad_norm_max',
                    'refine.steps_taken_mean',
                    'refine.nonfinite_frac',
                    'refine.grad_vanished_frac',
                    'refine.q_plateau_frac',
                    'refine.max_steps_frac',
                ]
            eval_metrics.update({f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)
        for k, v in overall_metrics.items():
            eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
        eval_metrics.setdefault('evaluation/overall_success', np.nan)
        eval_metrics.setdefault('evaluation/overall_episode.return', np.nan)
        # W&B panels sort metrics lexicographically within a section. Mirror the main
        # headline metrics under zero-padded aliases so they appear at the top of the
        # evaluation panel without breaking existing metric names or plots.
        eval_metrics['evaluation/00_overall_success'] = eval_metrics['evaluation/overall_success']
        eval_metrics['evaluation/01_overall_episode.return'] = eval_metrics['evaluation/overall_episode.return']
        normalized_score = None
        if 'episode.return' in overall_metrics:
            get_normalized_score = getattr(env.unwrapped, 'get_normalized_score', None)
            if callable(get_normalized_score):
                try:
                    normalized = [float(get_normalized_score(v)) for v in overall_metrics['episode.return']]
                    normalized_score = float(np.mean(normalized))
                    eval_metrics['evaluation/overall_normalized_score'] = normalized_score
                    eval_metrics['evaluation/normalized_score'] = normalized_score
                except Exception:
                    normalized_score = None
        if normalized_score is None:
            eval_metrics['evaluation/overall_normalized_score'] = np.nan
            eval_metrics['evaluation/normalized_score'] = np.nan
        eval_metrics['evaluation/02_overall_normalized_score'] = eval_metrics['evaluation/overall_normalized_score']

        if FLAGS.video_episodes > 0:
            video = get_wandb_video(renders=renders, n_cols=num_tasks)
            eval_metrics['video'] = video

        # Log "critic score" stats on a fixed batch (debugging / drift detection).
        eval_batch = _sample_dataset(train_dataset, config['batch_size'], evaluation=True)
        if config['discrete']:
            eval_batch['actions'] = np.full_like(eval_batch['actions'], env.action_space.n - 1)
        has_critic = hasattr(modules, 'keys') and 'critic' in modules
        q_vals = None
        if has_critic:
            critic_eval_k = config.get('critic_eval_k', None)
            q_vals = _critic_forward(
                eval_agent,
                eval_batch['observations'],
                eval_batch['value_goals'],
                eval_batch['actions'],
                num_iters=critic_eval_k,
            )
            q_vals = np.asarray(q_vals)
            eval_metrics['evaluation/Q_mean'] = float(q_vals.mean())
            eval_metrics['evaluation/Q_std'] = float(q_vals.std())
            eval_metrics['evaluation/Q_abs_max'] = float(np.abs(q_vals).max())
            negative_goals = _negative_goal_batch(eval_batch, goal_key='value_goals')
            if negative_goals is not None:
                q_neg = _score_critic_batch(eval_agent, eval_batch['observations'], negative_goals, eval_batch['actions'])
                q_pos = _reduced_scores(q_vals, 'critic')
                _log_signal_metrics(
                    eval_metrics,
                    'evaluation/critic_signal',
                    q_pos,
                    q_neg,
                    batch=eval_batch,
                    goal_prefix='value',
                )
                _log_path_geometry_metrics(
                    eval_metrics,
                    'evaluation/critic_geometry',
                    q_pos,
                    eval_batch,
                    goal_prefix='value',
                )
        elif has_value:
            v_vals = eval_agent.network.select('value')(
                eval_batch['observations'],
                eval_batch['value_goals'],
            )
            v_vals = np.asarray(v_vals)
            eval_metrics['evaluation/value_mean'] = float(v_vals.mean())
            eval_metrics['evaluation/value_std'] = float(v_vals.std())
            eval_metrics['evaluation/value_abs_max'] = float(np.abs(v_vals).max())
            negative_goals = _negative_goal_batch(eval_batch, goal_key='value_goals')
            if negative_goals is not None:
                v_neg = _score_value_batch(eval_agent, eval_batch['observations'], negative_goals)
                v_pos = _reduced_scores(v_vals, 'value')
                _log_signal_metrics(
                    eval_metrics,
                    'evaluation/value_signal',
                    v_pos,
                    v_neg,
                    batch=eval_batch,
                    goal_prefix='value',
                )
                _log_path_geometry_metrics(
                    eval_metrics,
                    'evaluation/value_geometry',
                    v_pos,
                    eval_batch,
                    goal_prefix='value',
                )

        # ---------------------------------------------------------------------
        # Extra eval diagnostics (helps debug instability / overfitting).
        # ---------------------------------------------------------------------
        # Ensemble disagreement / pessimism (actor uses min over ensemble).
        try:
            if q_vals.ndim >= 2 and q_vals.shape[0] == 2:
                q1, q2 = q_vals[0], q_vals[1]
                q_min = np.minimum(q1, q2)
                eval_metrics['evaluation/critic_q1_mean'] = float(np.mean(q1))
                eval_metrics['evaluation/critic_q2_mean'] = float(np.mean(q2))
                eval_metrics['evaluation/critic_q_min_mean'] = float(np.mean(q_min))
                eval_metrics['evaluation/critic_ensemble_disagreement'] = float(np.mean(np.std(q_vals, axis=0)))
                eval_metrics['evaluation/critic_q1_abs_max'] = float(np.max(np.abs(q1)))
                eval_metrics['evaluation/critic_q2_abs_max'] = float(np.max(np.abs(q2)))
            elif q_vals.ndim >= 2 and q_vals.shape[-1] == 2:
                q1, q2 = q_vals[..., 0], q_vals[..., 1]
                q_min = np.minimum(q1, q2)
                eval_metrics['evaluation/critic_q1_mean'] = float(np.mean(q1))
                eval_metrics['evaluation/critic_q2_mean'] = float(np.mean(q2))
                eval_metrics['evaluation/critic_q_min_mean'] = float(np.mean(q_min))
                eval_metrics['evaluation/critic_ensemble_disagreement'] = float(np.mean(np.std(q_vals, axis=-1)))
                eval_metrics['evaluation/critic_q1_abs_max'] = float(np.max(np.abs(q1)))
                eval_metrics['evaluation/critic_q2_abs_max'] = float(np.max(np.abs(q2)))
        except Exception:
            pass

        _maybe_log_actor_quality(eval_metrics, eval_agent, eval_batch)
        _maybe_log_fixed_probe_metrics(eval_metrics, eval_agent)
        _maybe_log_recurrent_refinement(eval_metrics, eval_agent, step=step)

        final_critic_refine_step = int(config.get('critic_k', 1))
        top_eval_aliases = {
            'evaluation/10_critic_margin_mean': 'evaluation/critic_signal/margin_mean',
            'evaluation/11_critic_rank_acc': 'evaluation/critic_signal/rank_acc',
            'evaluation/12_critic_separation_z': 'evaluation/critic_signal/separation_z',
            'evaluation/13_traj_medium_margin_mean': 'evaluation/critic_signal/traj_medium_margin_mean',
            'evaluation/14_maze_path_medium_margin_mean': 'evaluation/critic_signal/maze_path_medium_margin_mean',
            'evaluation/15_actor_q_delta_mean': 'evaluation/actor_critic/q_delta_mean',
            'evaluation/16_actor_prefer_actor_frac': 'evaluation/actor_critic/prefer_actor_frac',
            'evaluation/17_policy_behavior_mse_mean': 'evaluation/policy_behavior/mse_mean',
            'evaluation/18_policy_behavior_l2_mean': 'evaluation/policy_behavior/l2_mean',
            'evaluation/19_graph_corr': 'evaluation/critic_geometry/graph_corr',
            'evaluation/20_refine_margin_gain_mean': f'evaluation/refine/critic_margin_gain_1_to_{final_critic_refine_step}_mean',
            'evaluation/22_refine_hidden_drift_mean': 'evaluation/refine/critic_hidden_drift_mean',
            'evaluation/30_refine_critic_step1_margin_mean': 'evaluation/refine/critic_step_1_signal/margin_mean',
            'evaluation/31_refine_critic_step2_margin_mean': 'evaluation/refine/critic_step_2_signal/margin_mean',
            'evaluation/32_refine_critic_step3_margin_mean': 'evaluation/refine/critic_step_3_signal/margin_mean',
            'evaluation/33_refine_critic_step4_margin_mean': 'evaluation/refine/critic_step_4_signal/margin_mean',
            'evaluation/34_refine_final_actor_q_delta_mean': f'evaluation/refine/critic_step_{final_critic_refine_step}_actor_critic/q_delta_mean',
            'evaluation/40_probe_critic_margin_mean': 'evaluation/probe/critic_signal/margin_mean',
            'evaluation/41_probe_value_margin_mean': 'evaluation/probe/value_signal/margin_mean',
            'evaluation/42_probe_actor_q_delta_mean': 'evaluation/probe/actor_critic/q_delta_mean',
            'evaluation/43_probe_actor_value_adv_mean': 'evaluation/probe/actor_value/adv_mean',
            'evaluation/44_probe_policy_behavior_mse_mean': 'evaluation/probe/policy_behavior/mse_mean',
        }
        for alias_key, source_key in top_eval_aliases.items():
            if source_key in eval_metrics:
                eval_metrics[alias_key] = eval_metrics[source_key]

        summary_values = {
            'success': eval_metrics.get('evaluation/overall_success'),
            'episode.return': eval_metrics.get('evaluation/overall_episode.return'),
            'normalized_score': normalized_score,
        }
        for metric_name, metric_value in summary_values.items():
            if metric_value is None or not np.isfinite(metric_value):
                eval_metrics[f'evaluation/best_so_far_{metric_name}'] = np.nan
                eval_metrics[f'evaluation/final_{metric_name}'] = np.nan
            else:
                best_eval_metrics[metric_name] = max(best_eval_metrics[metric_name], float(metric_value))
                eval_metrics[f'evaluation/best_so_far_{metric_name}'] = float(best_eval_metrics[metric_name])
                eval_metrics[f'evaluation/final_{metric_name}'] = float(metric_value) if mark_final else np.nan

        for metric_key in suppressed_eval_metric_keys:
            eval_metrics.pop(metric_key, None)

        eval_metrics['time/update_count'] = int(step)
        eval_metrics['time/samples_seen'] = int(step) * int(config['batch_size'])
        eval_metrics['time/eval_count'] = int(eval_count)
        eval_metrics['time/eval_time'] = time.time() - eval_start
        eval_metrics['time/total_time'] = time.time() - first_time
        eval_metrics['time/train_wall_clock'] = eval_metrics['time/total_time']
        _add_compute_metrics(eval_metrics, step=step, total_time=eval_metrics['time/total_time'])
        wandb.log(_filter_metrics_for_wandb(eval_metrics, stage='eval'), step=step)
        eval_logger.log(eval_metrics, step=step)

    def run_evaluation(*, step: int, eval_agent, mark_final: bool = False):
        with _maybe_preserve_np_random_state(FLAGS.preserve_training_rng_on_eval):
            return _run_evaluation_impl(step=step, eval_agent=eval_agent, mark_final=mark_final)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    last_log_step = 0
    validation_log_interval = FLAGS.validation_log_interval or FLAGS.log_interval

    if FLAGS.eval_only:
        eval_step = int(FLAGS.restore_epoch) if FLAGS.restore_epoch is not None else 0
        eval_agent = agent
        if FLAGS.eval_on_cpu:
            try:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            except Exception:
                print(
                    'Warning: eval_on_cpu requested but CPU backend unavailable; '
                    f'JAX_PLATFORMS={os.environ.get("JAX_PLATFORMS")!r}. '
                    'Evaluating on current device.'
                )
        run_evaluation(step=eval_step, eval_agent=eval_agent, mark_final=True)
        train_logger.close()
        eval_logger.close()
        return

    # Resume-aware training loop: interpret `--train_steps` as the total desired step count.
    # `TrainState.step` starts at 1 and increments after each update, so the next update should use `i=network.step`.
    start_i = int(getattr(agent.network, 'step', 1))
    last_log_step = start_i - 1
    if start_i > FLAGS.train_steps:
        train_logger.close()
        eval_logger.close()
        return

    for i in tqdm.tqdm(
        range(start_i, FLAGS.train_steps + 1),
        smoothing=0.1,
        dynamic_ncols=True,
        disable=FLAGS.disable_tqdm,
    ):
        last_completed_step = i
        # Update agent.
        batch = _sample_dataset(train_dataset, config['batch_size'])
        agent, update_info = agent.update(batch)

        # One-time debug: dump param/grad key structure so we can fix per-module grad norms.
        if i == start_i:
            def _key_structure(tree, prefix=''):
                if hasattr(tree, 'keys'):
                    return {prefix + k: _key_structure(v, prefix + k + '/') for k, v in tree.items()}
                return 'leaf'
            print('DEBUG param keys:', _key_structure(agent.network.params), flush=True)
            print('DEBUG update_info keys:', sorted(update_info.keys()), flush=True)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = dict(static_metadata)
            train_metrics.update(static_param_metrics)
            train_metrics.update({f'training/{k}': v for k, v in update_info.items()})
            if val_dataset is not None and i % validation_log_interval == 0:
                with _maybe_preserve_np_random_state(FLAGS.preserve_training_rng_on_eval):
                    val_batch = _sample_dataset(val_dataset, config['batch_size'], evaluation=True)
                    _, val_info = agent.total_loss(val_batch, grad_params=None)
                # Validation metrics are agent-specific, so log every scalar-like entry.
                validation_metrics = {}
                for k, v in val_info.items():
                    try:
                        if np.asarray(v).ndim == 0:
                            validation_metrics[f'validation/{k}'] = v
                    except Exception:
                        continue
                train_metrics.update(validation_metrics)
            steps_since = max(1, i - last_log_step)
            log_now = time.time()
            epoch_time = (log_now - last_time) / steps_since
            train_metrics['time/sps'] = steps_since / (epoch_time + 1e-8)
            train_metrics['time/step_time'] = epoch_time
            train_metrics['time/update_count'] = int(i)
            train_metrics['time/samples_seen'] = int(i) * int(config['batch_size'])
            train_metrics['time/eval_count'] = int(eval_count)
            train_metrics['time/total_time'] = log_now - first_time
            train_metrics['time/train_wall_clock'] = train_metrics['time/total_time']
            last_time = log_now
            last_log_step = i
            _add_compute_metrics(train_metrics, step=i, total_time=train_metrics['time/total_time'])

            # Parameter magnitude stats (drift detection).
            critic_params = _extract_module_params(agent.network.params, 'critic')
            critic_abs_mean, _ = _abs_param_stats(critic_params)
            if critic_abs_mean is not None:
                train_metrics['params/critic_abs_mean'] = critic_abs_mean
            input_lambda_leaves = _collect_named_leaves(critic_params, 'input_lambda')
            if input_lambda_leaves:
                lambda_rows = []
                for leaf in input_lambda_leaves:
                    leaf = np.asarray(leaf, dtype=np.float64)
                    if leaf.ndim == 1:
                        lambda_rows.append(leaf[None, :])
                    else:
                        lambda_rows.append(leaf.reshape(-1, leaf.shape[-1]))
                lambda_matrix = np.concatenate(lambda_rows, axis=0)
                train_metrics['training/critic/input_lambda_abs_mean'] = float(np.mean(np.abs(lambda_matrix)))
                active_steps = int(config.get('critic_k', lambda_matrix.shape[-1]))
                for step_idx, value in enumerate(np.mean(lambda_matrix, axis=0)[:active_steps], start=1):
                    train_metrics[f'training/critic/input_lambda_step_{step_idx}'] = float(value)

            if 'value_goal_source' in batch:
                source = np.asarray(batch['value_goal_source'])
                horizon = np.asarray(batch.get('value_goal_horizon', np.full_like(source, -1)))
                for source_id, source_name in source_labels.items():
                    train_metrics[f'batch/value_goal_source_{source_name}_frac'] = float(np.mean(source == source_id))
                traj_mask = source == 1
                if np.any(traj_mask):
                    train_metrics['batch/value_goal_horizon_mean'] = float(np.mean(horizon[traj_mask]))
                    train_metrics['batch/value_goal_horizon_std'] = float(np.std(horizon[traj_mask]))
                maze_meta = _maze_distance_metadata(batch, goal_key='value_goals')
                if maze_meta is not None:
                    euclidean = np.asarray(maze_meta['euclidean'])
                    path_len = np.asarray(maze_meta['path_len'])
                    finite_path = path_len >= 0
                    train_metrics['batch/value_goal_maze_xy_distance_mean'] = float(np.mean(euclidean))
                    train_metrics['batch/value_goal_maze_xy_distance_std'] = float(np.std(euclidean))
                    train_metrics['batch/value_goal_maze_path_available_frac'] = float(np.mean(finite_path))
                    if np.any(finite_path):
                        train_metrics['batch/value_goal_maze_path_len_mean'] = float(np.mean(path_len[finite_path]))
                        train_metrics['batch/value_goal_maze_path_len_std'] = float(np.std(path_len[finite_path]))
                        train_metrics['batch/value_goal_maze_path_short_frac'] = float(np.mean(path_len[finite_path] <= 3))
                        train_metrics['batch/value_goal_maze_path_medium_frac'] = float(
                            np.mean((path_len[finite_path] >= 4) & (path_len[finite_path] <= 8))
                        )
                        train_metrics['batch/value_goal_maze_path_long_frac'] = float(np.mean(path_len[finite_path] >= 9))
            _log_task_distribution(train_metrics, batch, prefix='batch')

            # Policy vs behavior action divergence (offline RL sanity).
            if not config.get('discrete'):
                try:
                    actor_goals = batch.get('actor_goals', batch.get('low_actor_goals'))
                    if has_actor and actor_goals is not None:
                        dist = agent.network.select('actor')(
                            batch['observations'],
                            actor_goals,
                            temperature=1.0,
                        )
                        policy_actions = dist.mode()
                        behavior_actions = jnp.asarray(batch['actions'])
                        diff = policy_actions - behavior_actions
                        train_metrics['action/policy_behavior_mse'] = float(jnp.mean(diff**2))
                except Exception:
                    pass

            wandb.log(_filter_metrics_for_wandb(train_metrics, stage='train'), step=i)
            train_logger.log(train_metrics, step=i)

        # If the job is about to be killed (time limit / preemption), save once and exit cleanly.
        if terminate_requested['flag']:
            try:
                save_agent(agent, FLAGS.save_dir, i)
            finally:
                print(f"Terminate signal received (sig={terminate_requested['sig']}), saved params_{i}.pkl and exiting.")
            break

        # Evaluate agent.
        if FLAGS.eval_interval and (i == 1 or i % FLAGS.eval_interval == 0):
            eval_agent = agent
            if FLAGS.eval_on_cpu:
                try:
                    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
                except Exception:
                    print(
                        'Warning: eval_on_cpu requested but CPU backend unavailable; '
                        f'JAX_PLATFORMS={os.environ.get("JAX_PLATFORMS")!r}. '
                        'Evaluating on current device.'
                    )
            run_evaluation(step=i, eval_agent=eval_agent, mark_final=(i == FLAGS.train_steps))

        # Save agent.
        should_save = False
        if FLAGS.save_interval and i % FLAGS.save_interval == 0:
            should_save = True
        if i in save_steps:
            should_save = True
        if should_save:
            save_agent(agent, FLAGS.save_dir, i)

    if (
        not terminate_requested['flag']
        and last_eval_step != last_completed_step
    ):
        eval_agent = agent
        if FLAGS.eval_on_cpu:
            try:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            except Exception:
                print(
                    'Warning: eval_on_cpu requested but CPU backend unavailable; '
                    f'JAX_PLATFORMS={os.environ.get("JAX_PLATFORMS")!r}. '
                    'Evaluating on current device.'
                )
        run_evaluation(step=last_completed_step, eval_agent=eval_agent, mark_final=True)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
