"""Qwen2 model."""

import dataclasses
from typing import Tuple, Dict, TypedDict
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping

K_MASK = -2.3819763e38


from models.utils import typechecked
from models import MODE
from .configs import ShardingConfig, Qwen2ModelConfig
from models.types import (
    POSITION_IDS_TYPE, 
    ATTENTION_MASK_TYPE,
    Q_PROJ_VALUES_TYPE,
    KV_PROJ_VALUES_TYPE,
    ATTN_HIDDEN_STATES_TYPE,
    Cache,
    LayerCache
)


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == 'cpu':
        return x

    return jax.lax.with_sharding_constraint(
        x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
    )


class Einsum(nnx.Module):
    r"""
    Einsum is a convenience module for parameterized tensor multiplication.
    Args:
        einsum_str (str): format input shape, weight shape -> output shape
    """

    def __init__(
            self,
            einsum_str: str,
            shape: flax.typing.Shape,
            *,
            rngs: nnx.Rngs,
            sharding: Tuple[str | None, ...],
        )->None:

        self.einsum_str = einsum_str
        self.shape = shape
        self.w = nnx.Param(
            nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
        )
    
    @jax.named_scope('einsum')
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.einsum(self.einsum_str, x, self.w.value)


class EinsumBias(Einsum):
    r"""
    Einsum for multiplication with bias
    Args:
        einsum_str (str): format input shape, weight shape -> output shape
    """
    def __init__(
            self,
            einsum_str: str,
            shape: flax.typing.Shape,
            bias_shape: flax.typing.Shape,
            *,
            rngs: nnx.Rngs,
            sharding: Tuple[str | None, ...],
        )->None:
        super().__init__(
            einsum_str= einsum_str, 
            shape= shape , 
            rngs= rngs, 
            sharding= sharding
        )
        
        self.b = nnx.Param(
            nnx.initializers.normal()(rngs.params(), bias_shape), sharding=sharding
        )

    @jax.named_scope('einsum')
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:        
        return super().__call__(x= x) + self.b


class Embedder(nnx.Module):
    """Embedder module."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        *,
        rngs: nnx.Rngs,
        shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
    ):
        self.input_embedding = nnx.Param(
            nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
            sharding=shd_config.emb_vd,
        )
        self.shd_config = shd_config

    @typechecked(mode=MODE)
    @jax.named_scope('embedder_encode')
    def encode(self, x: jaxtyping.Int[jaxtyping.Array,"B S"]) -> ATTN_HIDDEN_STATES_TYPE:
        x = self.input_embedding[(x,)]
        x = shard(x, self.shd_config.act_btd)
        return x

    @jax.named_scope('embedder_decode')
    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.dot(x, self.input_embedding.value.T)

@typechecked(mode= MODE)
def apply_rope(
        projected: Q_PROJ_VALUES_TYPE|KV_PROJ_VALUES_TYPE,
        positions: POSITION_IDS_TYPE,
        head_dim: int,
        rope_theta: int = 1_000_000,
    ) -> Q_PROJ_VALUES_TYPE|KV_PROJ_VALUES_TYPE:
    r"""
    Applies RoPE for projected query or key
    """
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(projected, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(projected.dtype)


class RMSNorm(nnx.Module):
    """RMSNorm layer."""

    def __init__(
            self,
            dim: int,
            *,
            norm_eps: float = 1e-06,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.w = nnx.Param(
            nnx.initializers.ones_init()(rngs.params(), dim),
            sharding=shd_config.rms_norm_weight,
        )
        self.norm_eps = norm_eps

    @jax.named_scope('rms_norm')
    def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
        dtype = x.dtype
        rms = jnp.sqrt(
            jnp.mean(jnp.astype(x, jnp.float32) ** 2, axis=-1, keepdims=True)
            + self.norm_eps
        )
        return jnp.astype(self.w * x / rms, dtype)


class Attention(nnx.Module):
    r"""
    Attention module.
    In Qwen2, there are no q_norm and k_norm for q_proj and k_proj compared to Qwen3, see
    implement on [huggingface](https://github.com/huggingface/transformers/blob/b937d474550cb282b304b2d27ef58a306b2fd512/src/transformers/models/qwen2/modeling_qwen2.py#L124-L137) 

    """

    def __init__(
            self,
            config: Qwen2ModelConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.shd_config = shd_config
        self.q_proj = EinsumBias(
            einsum_str='BTD,DNH->BTNH',
            shape=(config.embed_dim, config.num_heads, config.head_dim),
            bias_shape = (config.num_heads, config.head_dim),
            rngs=rngs,
            sharding=shd_config.q_weight_ndh,
        )
        self.k_proj = EinsumBias(
            einsum_str='BSD,DKH->BSKH',
            shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
            bias_shape = (config.num_kv_heads, config.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )
        self.v_proj = EinsumBias(
            einsum_str='BSD,DKH->BSKH',
            shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
            bias_shape = (config.num_kv_heads, config.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )
        self.o_proj = Einsum(
            einsum_str='BTNH,NHD->BTD',
            shape=(config.num_heads, config.head_dim, config.embed_dim),
            rngs=rngs,
            sharding=shd_config.o_weight_nhd,
        )

        self.n_rep = config.num_heads // config.num_kv_heads
        self.scale = self.head_dim**-0.5

    @typechecked(mode=MODE)
    @jax.named_scope('attention')
    def __call__(
            self,
            x: ATTN_HIDDEN_STATES_TYPE,
            position_ids: POSITION_IDS_TYPE,
            layer_cache: LayerCache | None,
            attn_mask: ATTENTION_MASK_TYPE | None,
        ) -> tuple[LayerCache | None, ATTN_HIDDEN_STATES_TYPE]:
        seq_len = x.shape[1]

        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)

        query_proj = shard(query_proj, self.shd_config.act_btnh)
        key_proj = shard(key_proj, self.shd_config.act_btnh)
        value_proj = shard(value_proj, self.shd_config.act_btnh)

        query_proj = apply_rope(
            query_proj,
            position_ids,
            head_dim=self.head_dim,
        )
        key_proj = apply_rope(
            key_proj,
            position_ids,
            head_dim=self.head_dim,
        )

        if layer_cache is not None:
            end_index = layer_cache['end_index'][0]
            slice_indices = (0, end_index % layer_cache['v'].shape[1], 0, 0)
            value_proj: KV_PROJ_VALUES_TYPE = jax.lax.dynamic_update_slice(
                layer_cache['v'],
                value_proj,
                slice_indices,
            )
            key_proj: KV_PROJ_VALUES_TYPE = jax.lax.dynamic_update_slice(
                layer_cache['k'], key_proj, slice_indices
            )

        b, t, n, h = query_proj.shape
        _, s, k, _ = key_proj.shape

        # GQA
        query_proj = query_proj.reshape((b, t, k, n//k, h))

        # NOTE: n//k == G
        attn = jnp.einsum('BTKGH,BSKH->BKGTS', query_proj, key_proj) * self.scale
        # [B, K, G, T, S] -> [B, K*G, T, S] <-> [B, N, T, S]
        attn = attn.reshape((b, n, t, s))

        if attn_mask is not None:
            attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
            key_proj.dtype
        )

        attn = attn.reshape((b, k, n//k, t, s))
        qkv = jnp.einsum('BKGTS,BSKH->BTKGH', attn, value_proj)
        # [B, T, K, G, H] -> [B, T, K*G, H] <-> [B, T, N, H]
        qkv = qkv.reshape((b, t, n, h))

        outputs = self.o_proj(qkv)
        outputs = shard(outputs, self.shd_config.act_btd)

        if layer_cache is not None:
            new_cache = {
                'v': value_proj,
                'k': key_proj,
                'end_index': layer_cache['end_index'] + seq_len,
            }
        else:
            new_cache = None

        return new_cache, outputs

    @property
    def head_dim(self):
        return self.o_proj.shape[1]

    @property
    def num_heads(self):
        return self.q_proj.shape[0]

    @property
    def num_kv_heads(self):
        return self.k_proj.shape[1]


class MoELayer(nnx.Module):
    """MoE layer."""

    def __init__(
            self,
            config: Qwen2ModelConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.shd_config = shd_config
        self.experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.router = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.num_experts,
            use_bias=False,
            rngs=rngs,
        )
        self.gate_proj = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(),
                (config.num_experts, config.embed_dim, config.hidden_dim),
            ),
            sharding=shd_config.exp_weight_cdf,
        )
        self.up_proj = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(),
                (config.num_experts, config.embed_dim, config.hidden_dim),
            ),
            sharding=shd_config.exp_weight_cdf,
        )
        self.down_proj = nnx.Param(
            nnx.initializers.normal()(
                rngs.params(),
                (config.num_experts, config.hidden_dim, config.embed_dim),
            ),
            sharding=shd_config.exp_weight_cfd,
        )

    def __call__(self, x):
        scores = self.router(x).astype(jnp.float32)  # [B,T,E]
        routing_weights, routing_idx = jax.lax.top_k(
            jax.nn.softmax(scores, axis=-1), self.experts_per_tok
        )
        routing_weights = (
            routing_weights / jnp.sum(routing_weights, axis=-1, keepdims=True)
        ).astype(x.dtype)

        dispatch_mask = jax.nn.one_hot(
            routing_idx, num_classes=self.num_experts, dtype=x.dtype
        )  # [B, T, K, E]
        dispatch_mask = jnp.swapaxes(dispatch_mask, -1, -2)  # [B, T, E, K]

        dispatched_input = jnp.einsum(
            'BTID,BTEK->BTED', x[:, :, None, :], dispatch_mask
        ).astype(x.dtype)

        expert_outputs = []
        for i in range(self.num_experts):
            expert_input = dispatched_input[:, :, i, :]
            activations = nnx.silu(
                jnp.einsum('BTD,DF->BTF', expert_input, self.gate_proj[i])
            ) * jnp.einsum('BTD,DF->BTF', expert_input, self.up_proj[i])
            activations = shard(activations, self.shd_config.act_btf)
            expert_output = jnp.einsum('BTF,FD->BTD', activations, self.down_proj[i])
            expert_outputs.append(expert_output)

        stacked_outputs = jnp.stack(expert_outputs, axis=2)  # [B, T, E, D]
        routing_weights = jnp.tile(
            routing_weights[:, :, None, :], (1, 1, self.num_experts, 1)
        )  # [B, T, E, K]
        routing_weights = dispatch_mask * routing_weights  # [B, T, E, K]

        output = jnp.einsum('BTED,BTEK->BTD', stacked_outputs, routing_weights)
        return output


class MLP(nnx.Module):
    """MLP module."""

    def __init__(
            self,
            config: Qwen2ModelConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):

        self.shd_config = shd_config
        kernel_init_fn = nnx.initializers.zeros_init()

        self.gate_proj = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                kernel_init_fn, shd_config.ffw_weight_df
            ),
        )
        
        self.up_proj = nnx.Linear(
            in_features=config.embed_dim,
            out_features=config.hidden_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                kernel_init_fn, shd_config.ffw_weight_df
            ),
        )

        self.down_proj = nnx.Linear(
            in_features=config.hidden_dim,
            out_features=config.embed_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                kernel_init_fn, shd_config.ffw_weight_fd
            ),
        )

    @jax.named_scope('feed_forward')
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
        activations = shard(activations, self.shd_config.act_btf)
        outputs = self.down_proj(activations)
        return outputs


class DecoderLayer(nnx.Module):
    """DecoderLayer."""

    def __init__(
            self,
            config: Qwen2ModelConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.input_layernorm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.attn = Attention(
            config=config,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.post_attention_layernorm = RMSNorm(
            config.embed_dim,
            norm_eps=config.norm_eps,
            rngs=rngs,
            shd_config=shd_config,
        )
        if config.num_experts is None:
            self.mlp = MLP(
                config=config,
                rngs=rngs,
                shd_config=shd_config,
            )
        else:
            self.mlp = MoELayer(
                config=config,
                rngs=rngs,
                shd_config=shd_config,
            )

    @typechecked(mode=MODE)
    def __call__(
            self,
            x: ATTN_HIDDEN_STATES_TYPE,
            position_ids: POSITION_IDS_TYPE,
            layer_cache: LayerCache | None,
            attn_mask: ATTENTION_MASK_TYPE | None,
        ) -> tuple[LayerCache | None, jaxtyping.Array]:
        
        inputs_normalized = self.input_layernorm(x)
        
        layer_cache, attn_output = self.attn(
            inputs_normalized,
            position_ids,
            layer_cache,
            attn_mask,
        )

        attn_output += x
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return layer_cache, outputs


class Qwen2ForCausalLM(nnx.Module):
    """
    Qwen2ForCausalLM Model.
    """

    def __init__(
            self,
            config: Qwen2ModelConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):

        self.config = config
        self.embedder = Embedder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            rngs=rngs,
            shd_config=shd_config,
        )
        self.layers = [
            DecoderLayer(config=config, rngs=rngs, shd_config=shd_config)
            for _ in range(config.num_layers)
        ]
        self.final_norm = RMSNorm(
            config.embed_dim,
            rngs=rngs,
            norm_eps=config.norm_eps,
            shd_config=shd_config,
        )

    @typechecked(mode=MODE)
    def __call__(
            self,
            input_embedd: ATTN_HIDDEN_STATES_TYPE,
            position_ids: POSITION_IDS_TYPE,
            cache: Cache | None,  # (sequence length L')
            attention_mask: ATTENTION_MASK_TYPE
        ) -> tuple[jaxtyping.Array, Cache | None]:
        r"""Qwen2ForCausalLM model.
        Args:
            input_tokens: input sequence of tokens.
            positions: input absolute positions.
            cache: Attention KV cache or None.
            attention_mask: transformer input mask.
            output_hidden_states: whether to output the hidden states.

        Returns:
            tuple of predicted_logits, new_cache

            predicted_logits: output logits predicted by the model
            new_cache: updated cache if the input cache is not None, None elsewhere.
        """
        new_cache = None if cache is None else {}
        x = input_embedd

        for i, layer in enumerate(self.layers):
            layer_name = f'layer_{i}'
            layer_cache = cache[layer_name] if cache else None
            layer_cache, x = layer(
                x = x,
                position_ids = position_ids,
                cache = layer_cache,
                attention_mask = attention_mask,
            )
            if cache is not None:
                new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        x = self.final_norm(x)
        
        return x, new_cache

