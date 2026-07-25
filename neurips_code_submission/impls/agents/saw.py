import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCRecurrentValue, GCValue, Identity, LengthNormalize, MLP


class SAWAgent(flax.struct.PyTreeNode):
    """Subgoal Advantage-Weighted (SAW) policy bootstrapping agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def _is_recurrent_backbone(backbone):
        return str(backbone) == 'recur'

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the expectile value loss."""
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        info = {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }
        return value_loss, info

    def target_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss used to bootstrap the flat policy."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('low_actor')(batch['observations'], batch['low_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def actor_loss(self, batch, grad_params):
        """Compute flat actor loss with respect to high-actor goals."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv1, nv2 = self.network.select('value')(batch['next_observations'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['awr_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        log_prob = dist.log_prob(batch['actions'])
        awr_loss = -(exp_a * log_prob).mean()

        awr_info = {
            'awr_loss': awr_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            awr_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return awr_loss, awr_info

    def waypoint_loss(self, batch, grad_params):
        """Compute the waypoint bootstrapping loss."""
        v1, v2 = self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        wv1, wv2 = self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        v = (v1 + v2) / 2
        wv = (wv1 + wv2) / 2
        wadv = wv - v

        exp_w = jnp.exp(wadv * self.config['kl_alpha'])
        exp_w = jnp.minimum(exp_w, 100.0)

        dist = self.network.select('actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)
        w_dist = self.network.select('low_actor')(batch['observations'], batch['high_actor_targets'])
        if self.config['const_std']:
            w_mode = jax.lax.stop_gradient(w_dist.mode())
            kld = jnp.sum((dist.mode() - w_mode) ** 2, axis=-1)
        else:
            kld = w_dist.kl_divergence(dist)
        waypoint_loss = (exp_w * kld).mean()

        waypoint_info = {
            'waypoint_loss': waypoint_loss,
            'kld': kld.mean(),
            'wadv': wadv.mean(),
        }

        return waypoint_loss, waypoint_info

    def total_loss(self, batch, grad_params=None, rng=None):
        """Compute the total loss."""
        del rng
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.target_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        waypoint_loss, waypoint_info = self.waypoint_loss(batch, grad_params)
        for k, v in waypoint_info.items():
            info[f'waypoint/{k}'] = v

        loss = value_loss + low_actor_loss + actor_loss + waypoint_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with info."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)
        actor_goal_rep_def = goal_rep_def if config['share_goal_rep'] else copy.deepcopy(goal_rep_def)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            encoders['target_value'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            encoders['actor'] = GCEncoder(state_encoder=encoder_module(), concat_encoder=actor_goal_rep_def)
            encoders['low_actor'] = GCEncoder(concat_encoder=encoder_module())
        else:
            encoders['value'] = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            encoders['target_value'] = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            encoders['actor'] = GCEncoder(state_encoder=Identity(), concat_encoder=actor_goal_rep_def)

        value_backbone = str(config.get('value_backbone', 'mlp'))
        if value_backbone == 'mlp':
            value_kwargs = dict(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                backbone=value_backbone,
            )
            value_def = GCValue(
                gc_encoder=encoders.get('value'),
                **value_kwargs,
            )
            target_value_def = GCValue(
                gc_encoder=encoders.get('target_value'),
                **value_kwargs,
            )
        elif cls._is_recurrent_backbone(value_backbone):
            recur_value_kwargs = dict(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                backbone=value_backbone,
                recur_num_iters=int(config.get('value_k', 4)),
                recur_num_dense_layers=int(config.get('value_m', 2)),
                layerscale_init=float(config.get('value_layerscale_init', 1e-2)),
            )
            value_def = GCRecurrentValue(gc_encoder=encoders.get('value'), **recur_value_kwargs)
            target_value_def = GCRecurrentValue(gc_encoder=encoders.get('target_value'), **recur_value_kwargs)
        else:
            raise ValueError(f'Unsupported value_backbone: {value_backbone}')

        actor_kwargs = dict(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            gc_encoder=encoders.get('actor'),
            layer_norm=config['layer_norm'],
        )
        low_actor_kwargs = dict(
            hidden_dims=config['low_actor_hidden_dims'],
            action_dim=action_dim,
            gc_encoder=encoders.get('low_actor'),
            layer_norm=config['layer_norm'],
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                **actor_kwargs,
            )
            low_actor_def = GCDiscreteActor(
                **low_actor_kwargs,
            )
        else:
            actor_def = GCActor(
                state_dependent_std=False,
                const_std=config['const_std'],
                **actor_kwargs,
            )
            low_actor_def = GCActor(
                state_dependent_std=False,
                const_std=config['const_std'],
                **low_actor_kwargs,
            )

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            goal_rep=(goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
        )
        if not config['share_goal_rep']:
            network_info.update(
                actor_goal_rep_def=(actor_goal_rep_def, jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='saw',
            lr=3e-4,
            batch_size=1024,
            low_actor_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            value_backbone='mlp',  # 'mlp' | 'recur'
            value_k=4,
            value_m=2,
            value_layerscale_init=1e-2,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,
            awr_alpha=3.0,
            kl_alpha=3.0,
            subgoal_steps=25,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            share_goal_rep=False,
            rep_dim=10,
            dataset_class='HGCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            actor_geom_discount=0.99,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
