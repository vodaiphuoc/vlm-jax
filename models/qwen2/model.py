"""Qwen2 model."""

import dataclasses
from typing import Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping


K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]

from models.utils import typechecked
from models import MODE
from .types import TEXT_EMBEDDING_OUT_TYPE
from models.internvl.types import POSITION_IDS_TYPE, ATTENTION_MASK_TYPE

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for Qwen2 model."""

    emb_vd: Tuple[str | None, ...]
    emb_dv: Tuple[str | None, ...]
    q_weight_ndh: Tuple[str | None, ...]
    kv_weight_ndh: Tuple[str | None, ...]
    o_weight_nhd: Tuple[str | None, ...]
    ffw_weight_df: Tuple[str | None, ...]
    ffw_weight_fd: Tuple[str | None, ...]
    rms_norm_weight: Tuple[str | None, ...]
    act_btd: Tuple[str | None, ...]
    act_btf: Tuple[str | None, ...]
    act_btnh: Tuple[str | None, ...]
    exp_weight_cdf: Tuple[str | None, ...]
    exp_weight_cfd: Tuple[str | None, ...]

    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = 'fsdp' if not is_sampling else None

        return ShardingConfig(
            emb_vd=('tp', fsdp),
            emb_dv=(fsdp, 'tp'),
            q_weight_ndh=('tp', fsdp, None),
            kv_weight_ndh=('tp', fsdp, None),
            o_weight_nhd=('tp', None, fsdp),
            ffw_weight_df=(fsdp, 'tp'),
            ffw_weight_fd=('tp', fsdp),
            rms_norm_weight=('tp',),
            act_btd=('fsdp', None, None if is_sampling else 'tp'),
            act_btf=('fsdp', None, 'tp'),
            act_btnh=('fsdp', None, 'tp', None),
            exp_weight_cdf=('fsdp', None, 'tp'),
            exp_weight_cfd=('fsdp', 'tp', None),
        )


@dataclasses.dataclass(frozen=True)
class Qwen2ModelConfig:
    r"""
    Configuration for the Qwen2 model
    
    Args:
        num_layers (int): alias of `num_hidden_layers` on hf config  
        vocab_size (int): vocab size of vocab embedding
        embed_dim (int): alias of `hidden_size` on hf config 
        hidden_dim (int): alias of `intermediate_size` on hf config 
        num_heads (int): alias of `num_attention_heads` on hf config
        head_dim (int): Qwen2's config doesnt have `head_dim`. So it will be:
            ```math
            embed_dim/num_heads
            ```
            or in hf
            ```math
            hidden_size/num_attention_heads
            ```
        see more in [here](https://github.com/huggingface/transformers/blob/6c3f27ba6186897d072b87e9e6e7c63d97f0fe99/src/transformers/models/qwen2/modeling_qwen2.py#L128C1-L128C102)
        
        num_kv_heads (int): alias of `num_key_value_heads` on hf config
        norm_eps (float): alias of `rms_norm_eps` on hf config
        rope_theta (int): rope_theta for ROPE
    """

    num_layers: int  # num_hidden_layers
    vocab_size: int
    embed_dim: int   # hidden_size
    hidden_dim: int  # intermediate_size
    num_heads: int   # num_attention_heads
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()

    @classmethod
    def qwen2_0_5_b(cls):  # qwen2-0.5B
        return cls(
            num_layers=24,
            vocab_size=151936,
            embed_dim=896,
            hidden_dim=4864,
            num_heads=14,
            head_dim= 896//14, # 64
            num_kv_heads=2,
            norm_eps=1e-06,
            rope_theta=1_000_000,
        )



def shard(x: jnp.ndarray, s: Tuple[str, ...]):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty or jax.devices()[0].platform == 'cpu':
        return x

    return jax.lax.with_sharding_constraint(
        x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
    )


class Einsum(nnx.Module):
    """Einsum is a convenience module for parameterized tensor multiplication."""

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
    def encode(self, x: jaxtyping.Int[jaxtyping.Array,"B S"]) -> TEXT_EMBEDDING_OUT_TYPE:
        x = self.input_embedding[(x,)]
        x = shard(x, self.shd_config.act_btd)
        return x

    @jax.named_scope('embedder_decode')
    def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        return jnp.dot(x, self.input_embedding.value.T)


def apply_rope(
        inputs: jaxtyping.Array,  # [B, L]
        positions: jaxtyping.Array,  # [B, L]
        head_dim: int,
        rope_theta: int = 1_000_000,
    ) -> jaxtyping.Array:
    """Applies RoPE."""
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta**fraction

    sinusoid_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


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

    @jax.named_scope('attention')
    def __call__(
            self,
            x: jaxtyping.Array,
            segment_pos: jaxtyping.Array,
            cache: LayerCache | None,
            attn_mask: jaxtyping.Array | None,
        ) -> tuple[LayerCache | None, jaxtyping.Array]:
        seq_len = x.shape[1]

        query_proj = self.q_proj(x)
        key_proj = self.k_proj(x)
        value_proj = self.v_proj(x)

        query_proj = shard(query_proj, self.shd_config.act_btnh)
        key_proj = shard(key_proj, self.shd_config.act_btnh)
        value_proj = shard(value_proj, self.shd_config.act_btnh)

        query_proj = apply_rope(
            query_proj,
            segment_pos,
            head_dim=self.head_dim,
        )
        key_proj = apply_rope(
            key_proj,
            segment_pos,
            head_dim=self.head_dim,
        )

        if cache is not None:
            end_index = cache['end_index'][0]
            slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
            value_proj = jax.lax.dynamic_update_slice(
                cache['v'],
                value_proj,
                slice_indices,
            )
            key_proj = jax.lax.dynamic_update_slice(
                cache['k'], key_proj, slice_indices
            )

        b, t, qh, d = query_proj.shape
        _, s, kh, _ = key_proj.shape

        # GQA
        query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
        attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
        attn = attn.reshape((b, qh, t, s))

        if attn_mask is not None:
            attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
            key_proj.dtype
        )

        attn = attn.reshape((b, kh, qh // kh, t, s))
        qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
        qkv = qkv.reshape((b, t, qh, d))

        outputs = self.o_proj(qkv)
        outputs = shard(outputs, self.shd_config.act_btd)

        if cache is not None:
            new_cache = {
                'v': value_proj,
                'k': key_proj,
                'end_index': cache['end_index'] + seq_len,
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
        return self.kv_proj.shape[1]


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

    def __call__(
            self,
            x: TEXT_EMBEDDING_OUT_TYPE,
            segment_pos: jaxtyping.Array,
            cache: LayerCache | None,
            attn_mask: jaxtyping.Array,
        ) -> tuple[LayerCache | None, jaxtyping.Array]:
        
        inputs_normalized = self.input_layernorm(x)
        
        cache, attn_output = self.attn(
            inputs_normalized,
            segment_pos,
            cache,
            attn_mask,
        )

        attn_output += x
        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        outputs = self.mlp(attn_output)
        outputs = residual + outputs
        return cache, outputs


class Qwen2Text(nnx.Module):
    """
    Qwen2 Text Model.
    Intent for loading qwen2 0.5 only, so there is no `lm_head`
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

    def __call__(
            self,
            input_embedd: TEXT_EMBEDDING_OUT_TYPE,
            position_ids: POSITION_IDS_TYPE,
            cache: Cache | None,  # (sequence length L')
            attention_mask: ATTENTION_MASK_TYPE
        ) -> tuple[jaxtyping.Array, Cache | None]:
        r"""Qwen2 model.
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
                x,
                position_ids,
                layer_cache,
                attention_mask,
            )
            if cache is not None:
                new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

        x = self.final_norm(x)
        
        return x, new_cache

