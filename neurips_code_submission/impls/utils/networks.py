from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


RECURRENT_BACKBONES = frozenset({'recur'})


def is_recurrent_backbone(backbone):
    return str(backbone) in RECURRENT_BACKBONES


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class ResNetBlock(nn.Module):
    hidden_dim: int
    layer_norm: bool = True
    layerscale_init: float = 1e-2
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, h):
        h1 = h
        u = h1
        for layer_idx in range(4):
            u = nn.Dense(self.hidden_dim, name=f'fc_{layer_idx + 1}')(u)
            if self.layer_norm:
                u = nn.LayerNorm(name=f'ln_{layer_idx + 1}')(u)
            u = self.activations(u)
        return h + u


class DeepResNetBackbone(nn.Module):
    hidden_dim: int
    out_dim: int
    num_blocks: int
    layer_norm: bool = True
    layerscale_init: float = 1e-2
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim)(x)
        for block_idx in range(int(self.num_blocks)):
            h = ResNetBlock(
                self.hidden_dim,
                layer_norm=self.layer_norm,
                layerscale_init=self.layerscale_init,
                activations=self.activations,
                name=f'resblock_{block_idx + 1}',
            )(h)
        return nn.Dense(self.out_dim)(h)


class PreNormResidualDenseBlock(nn.Module):
    hidden_dim: int
    num_layers: int = 4
    use_inner_layer_norm: bool = True
    norm_mode: str = 'pre_block'
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, h):
        num_layers = int(self.num_layers)
        if num_layers < 2:
            raise ValueError(f'num_layers must be at least 2. Got {num_layers}.')
        if self.norm_mode not in ('pre_block', 'per_layer', 'pre_block_per_layer'):
            raise ValueError(
                f"Unsupported norm_mode={self.norm_mode!r}. Use 'pre_block', 'per_layer', or 'pre_block_per_layer'."
            )

        u = nn.LayerNorm(name='pre_ln')(h) if self.norm_mode in ('pre_block', 'pre_block_per_layer') else h
        for layer_index in range(num_layers):
            is_last_layer = layer_index + 1 == num_layers
            u = nn.Dense(self.hidden_dim, name=f'fc_{layer_index + 1}')(u)
            if not is_last_layer:
                if self.use_inner_layer_norm:
                    if self.norm_mode in ('per_layer', 'pre_block_per_layer') or layer_index > 0:
                        u = nn.LayerNorm(name=f'ln_{layer_index + 1}')(u)
                u = self.activations(u)
        return h + u


class PreNormResidualDenseBackbone(nn.Module):
    hidden_dim: int
    out_dim: int
    num_blocks: int
    block_num_layers: int = 4
    use_inner_layer_norm: bool = True
    norm_mode: str = 'pre_block'
    activations: Any = nn.gelu

    @nn.compact
    def __call__(self, x):
        num_blocks = int(self.num_blocks)
        if num_blocks <= 0:
            raise ValueError(f'num_blocks must be positive. Got {num_blocks}.')

        h = nn.Dense(self.hidden_dim, name='input_proj')(x)
        for block_index in range(num_blocks):
            h = PreNormResidualDenseBlock(
                hidden_dim=self.hidden_dim,
                num_layers=int(self.block_num_layers),
                use_inner_layer_norm=bool(self.use_inner_layer_norm),
                norm_mode=str(self.norm_mode),
                activations=self.activations,
                name=f'prenorm_resblock_{block_index + 1}',
            )(h)
        return nn.Dense(self.out_dim, name='output_proj')(h)


class ResidualMLPBackbone(nn.Module):
    """Residual MLP with Dense -> GeLU -> LayerNorm and a skip every 4 layers."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True
    residual_stride: int = 4

    @nn.compact
    def __call__(self, x):
        if len(self.hidden_dims) == 0:
            return x

        stride = int(self.residual_stride)
        if stride <= 0:
            raise ValueError(f'residual_stride must be positive. Got {stride}.')

        h = x
        block_input = h
        for layer_index, size in enumerate(self.hidden_dims):
            if layer_index % stride == 0:
                block_input = h

            h = nn.Dense(size, kernel_init=self.kernel_init, name=f'fc_{layer_index + 1}')(h)
            h = self.activations(h)
            if self.layer_norm:
                h = nn.LayerNorm(name=f'ln_{layer_index + 1}')(h)

            if (layer_index + 1) % stride == 0:
                residual = block_input
                if residual.shape[-1] != size:
                    residual = nn.Dense(size, kernel_init=self.kernel_init, name=f'resid_proj_{(layer_index + 1) // stride}')(
                        residual
                    )
                h = residual + h

        return h


class RecurrentBackbone(nn.Module):
    """Single recurrent design: K iterations, each with m residual SwiGLU layers."""

    hidden_dim: int
    out_dim: int
    k: int
    m: int = 2
    layer_norm: bool = True
    layerscale_init: float = 1e-2

    @nn.compact
    def __call__(self, x, num_iters=None, return_aux=False, context=None):
        del context
        iters = self.k if num_iters is None else int(num_iters)
        num_layers = int(self.m)
        if iters <= 0:
            raise ValueError(f'K must be positive. Got {iters}.')
        if num_layers <= 0:
            raise ValueError(f'm must be positive. Got {num_layers}.')

        h = nn.Dense(self.hidden_dim, name='input_state_proj')(x)
        initial_h = h

        step_embeddings = self.param('step_embedding', nn.initializers.normal(stddev=0.02), (int(self.k), self.hidden_dim))
        alpha = self.param('alpha', nn.initializers.constant(self.layerscale_init), (int(self.k),))
        inner_layer_norms = (
            [nn.LayerNorm(name=f'inner_ln_{layer_index + 1}') for layer_index in range(num_layers)]
            if self.layer_norm
            else None
        )
        swiglu_gate_layers = [nn.Dense(self.hidden_dim, name=f'swiglu_fc_a_{layer_index + 1}') for layer_index in range(num_layers)]
        swiglu_value_layers = [nn.Dense(self.hidden_dim, name=f'swiglu_fc_b_{layer_index + 1}') for layer_index in range(num_layers)]
        swiglu_out_layers = [nn.Dense(self.hidden_dim, name=f'swiglu_fc_o_{layer_index + 1}') for layer_index in range(num_layers)]
        output_proj = nn.Dense(self.out_dim, name='output_proj')

        hidden_steps = [] if return_aux else None
        hidden_drift_steps = []
        if iters > int(self.k):
            raise ValueError(f'num_iters={iters} exceeds configured K={int(self.k)} for step-indexed parameters.')

        for step_index in range(iters):
            h_inner = h
            step_embedding = step_embeddings[step_index]
            for layer_index in range(num_layers):
                u = h_inner
                if inner_layer_norms is not None:
                    u = inner_layer_norms[layer_index](u)
                u = u + step_embedding
                a_k = swiglu_gate_layers[layer_index](u)
                b_k = swiglu_value_layers[layer_index](u)
                g_k = jax.nn.silu(a_k) * b_k
                delta = swiglu_out_layers[layer_index](g_k)
                h_inner = h_inner + delta

            h_next = h + alpha[step_index] * (h_inner - h)

            initial_h_norm = jnp.linalg.norm(initial_h, axis=-1)
            hidden_drift = jnp.linalg.norm(h_next - initial_h, axis=-1) / (initial_h_norm + 1e-8)
            hidden_drift_steps.append(hidden_drift)
            h = h_next
            if hidden_steps is not None:
                hidden_steps.append(h_next)

        outputs = output_proj(h)
        if return_aux:
            hidden_stack = jnp.stack(hidden_steps, axis=0)
            hidden_drift_steps = jnp.stack(hidden_drift_steps, axis=0)
            return outputs, {
                'value_step_outputs': output_proj(hidden_stack),
                'hidden_drift_steps': hidden_drift_steps,
                'hidden_drift': hidden_drift_steps[-1],
                'final_hidden_norm': jnp.linalg.norm(h, axis=-1),
            }
        return outputs


def recurrent_backbone_cls_and_kwargs(
    backbone,
    *,
    hidden_dim,
    out_dim,
    k,
    m,
    layer_norm,
    layerscale_init,
):
    backbone = str(backbone)
    if backbone == 'recur':
        return RecurrentBackbone, dict(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            k=int(k),
            m=int(m),
            layer_norm=layer_norm,
            layerscale_init=layerscale_init,
        )
    raise ValueError(f'Unsupported backbone: {backbone}')


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    backbone: str = 'mlp'  # 'mlp' | 'residual_mlp'

    def setup(self):
        backbone = str(self.backbone)
        if backbone == 'mlp':
            self.actor_net = MLP(self.hidden_dims, activate_final=True)
        elif backbone == 'residual_mlp':
            self.actor_net = ResidualMLPBackbone(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                activations=nn.gelu,
            )
        else:
            raise ValueError(f'Unsupported actor backbone: {backbone!r}')
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    backbone: str = 'mlp'  # 'mlp' | 'residual_mlp'

    def setup(self):
        backbone = str(self.backbone)
        if backbone == 'mlp':
            self.actor_net = MLP(self.hidden_dims, activate_final=True)
        elif backbone == 'residual_mlp':
            self.actor_net = ResidualMLPBackbone(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                activations=nn.gelu,
            )
        else:
            raise ValueError(f'Unsupported actor backbone: {backbone!r}')
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None
    backbone: str = 'mlp'

    def setup(self):
        if self.backbone == 'mlp':
            value_module = MLP
            value_kwargs = dict(
                hidden_dims=(*self.hidden_dims, 1),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        else:
            raise ValueError(f'Unsupported value backbone: {self.backbone!r}')

        if self.ensemble:
            value_module = ensemblize(value_module, 2)
        self.value_net = value_module(**value_kwargs)

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCRecurrentValue(nn.Module):
    """Goal-conditioned scalar value function using the single recurrent backbone."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None
    backbone: str = 'recur'
    recur_num_iters: int = 4
    recur_num_dense_layers: int = 2
    layerscale_init: float = 1e-2

    def setup(self):
        hidden_dim = int(self.hidden_dims[-1])
        value_module, value_kwargs = recurrent_backbone_cls_and_kwargs(
            self.backbone,
            hidden_dim=hidden_dim,
            out_dim=1,
            k=self.recur_num_iters,
            m=self.recur_num_dense_layers,
            layer_norm=self.layer_norm,
            layerscale_init=self.layerscale_init,
        )
        if self.ensemble:
            value_module = ensemblize(value_module, 2)
        self.value_net = value_module(**value_kwargs)

    def __call__(self, observations, goals=None, actions=None, return_aux=False, num_iters=None):
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        if return_aux:
            values, aux = self.value_net(inputs, num_iters, True, inputs)
            return values.squeeze(-1), aux
        return self.value_net(inputs, num_iters, False, inputs).squeeze(-1)


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    backbone: str = 'mlp'  # 'mlp' | 'recur'
    recur_num_iters: int = 4
    recur_num_dense_layers: int = 2
    layerscale_init: float = 1e-2

    def setup(self):
        backbone = self.backbone
        if backbone == 'mlp':
            module_cls = MLP
            phi_kwargs = dict(hidden_dims=(*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
            psi_kwargs = dict(hidden_dims=(*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        elif is_recurrent_backbone(backbone):
            hidden_dim = int(self.hidden_dims[-1])
            module_cls, phi_kwargs = recurrent_backbone_cls_and_kwargs(
                backbone,
                hidden_dim=hidden_dim,
                out_dim=self.latent_dim,
                k=self.recur_num_iters,
                m=self.recur_num_dense_layers,
                layer_norm=self.layer_norm,
                layerscale_init=self.layerscale_init,
            )
            _, psi_kwargs = recurrent_backbone_cls_and_kwargs(
                backbone,
                hidden_dim=hidden_dim,
                out_dim=self.latent_dim,
                k=self.recur_num_iters,
                m=self.recur_num_dense_layers,
                layer_norm=self.layer_norm,
                layerscale_init=self.layerscale_init,
            )
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        if self.ensemble:
            module_cls = ensemblize(module_cls, 2)

        self.phi = module_cls(**phi_kwargs)
        self.psi = module_cls(**psi_kwargs)

    def __call__(self, observations, goals, actions=None, info=False, *, num_iters=None, return_aux=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
            num_iters: Override iteration count (only for iterative backbones).
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        aux = {}
        if is_recurrent_backbone(self.backbone) and return_aux:
            # IMPORTANT: nn.vmap does not support kwargs; pass iterative args positionally.
            phi, phi_aux = self.phi(phi_inputs, num_iters, True, phi_inputs)
            psi, psi_aux = self.psi(goals, num_iters, True, goals)
            phi_hidden_norm = phi_aux.get('final_hidden_norm')
            psi_hidden_norm = psi_aux.get('final_hidden_norm')
            phi_hidden_drift = phi_aux.get('hidden_drift')
            psi_hidden_drift = psi_aux.get('hidden_drift')
            phi_hidden_drift_steps = phi_aux.get('hidden_drift_steps')
            psi_hidden_drift_steps = psi_aux.get('hidden_drift_steps')
            if phi_hidden_norm is not None:
                aux['final_hidden_norm_phi'] = phi_hidden_norm
            if psi_hidden_norm is not None:
                aux['final_hidden_norm_psi'] = psi_hidden_norm
            if phi_hidden_norm is not None and psi_hidden_norm is not None:
                aux['final_hidden_norm_total'] = jnp.sqrt(phi_hidden_norm**2 + psi_hidden_norm**2)
            if phi_hidden_drift is not None:
                aux['hidden_drift_phi'] = phi_hidden_drift
            if psi_hidden_drift is not None:
                aux['hidden_drift_psi'] = psi_hidden_drift
            if phi_hidden_drift is not None and psi_hidden_drift is not None:
                aux['hidden_drift_mean'] = 0.5 * (phi_hidden_drift + psi_hidden_drift)
            if phi_hidden_drift_steps is not None:
                aux['hidden_drift_steps_phi'] = phi_hidden_drift_steps
            if psi_hidden_drift_steps is not None:
                aux['hidden_drift_steps_psi'] = psi_hidden_drift_steps
        elif is_recurrent_backbone(self.backbone):
            # IMPORTANT: nn.vmap does not support kwargs; pass iterative args positionally.
            phi = self.phi(phi_inputs, num_iters, False, phi_inputs)
            psi = self.psi(goals, num_iters, False, goals)
        else:
            phi = self.phi(phi_inputs)
            psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info and return_aux:
            return v, phi, psi, aux
        if info:
            return v, phi, psi
        if return_aux:
            return v, aux
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False, *, num_iters=None, return_aux=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info, num_iters=num_iters, return_aux=return_aux)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None
    backbone: str = 'mlp'  # 'mlp' | 'recur'
    recur_num_iters: int = 4
    recur_num_dense_layers: int = 2
    layerscale_init: float = 1e-2

    def setup(self):
        backbone = self.backbone
        if backbone == 'mlp':
            module_cls = MLP
            phi_kwargs = dict(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        elif is_recurrent_backbone(backbone):
            hidden_dim = int(self.hidden_dims[-1])
            module_cls, phi_kwargs = recurrent_backbone_cls_and_kwargs(
                backbone,
                hidden_dim=hidden_dim,
                out_dim=self.latent_dim,
                k=self.recur_num_iters,
                m=self.recur_num_dense_layers,
                layer_norm=self.layer_norm,
                layerscale_init=self.layerscale_init,
            )
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.phi = module_cls(**phi_kwargs)

    def __call__(self, observations, goals, is_phi=False, info=False, *, num_iters=None):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            if is_recurrent_backbone(self.backbone):
                phi_s = self.phi(observations, num_iters=num_iters, return_aux=False, context=observations)
                phi_g = self.phi(goals, num_iters=num_iters, return_aux=False, context=goals)
            else:
                phi_s = self.phi(observations)
                phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None
    backbone: str = 'mlp'  # 'mlp' | 'recur'
    recur_num_iters: int = 4
    recur_num_dense_layers: int = 2
    layerscale_init: float = 1e-2

    def setup(self):
        backbone = self.backbone
        if backbone == 'mlp':
            module_cls = MLP
            phi_kwargs = dict(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        elif is_recurrent_backbone(backbone):
            hidden_dim = int(self.hidden_dims[-1])
            module_cls, phi_kwargs = recurrent_backbone_cls_and_kwargs(
                backbone,
                hidden_dim=hidden_dim,
                out_dim=self.latent_dim,
                k=self.recur_num_iters,
                m=self.recur_num_dense_layers,
                layer_norm=self.layer_norm,
                layerscale_init=self.layerscale_init,
            )
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        self.phi = module_cls(**phi_kwargs)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False, *, num_iters=None):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            if is_recurrent_backbone(self.backbone):
                phi_s = self.phi(observations, num_iters=num_iters, return_aux=False, context=observations)
                phi_g = self.phi(goals, num_iters=num_iters, return_aux=False, context=goals)
            else:
                phi_s = self.phi(observations)
                phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v
