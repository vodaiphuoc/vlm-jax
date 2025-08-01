"""InternVLVisionModel model."""

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

NAMED_SCOPE_PREFIX = "internvl"

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for model."""

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
class InternVLVisionConfig:
    r"""
    Configuration for InternVLVision
    Ref: []()
    
    """
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    attention_bias: bool
    use_qk_norm: bool
    intermediate_size:int
    hidden_act: str
    hidden_dropout_prob: float
    attention_dropout: float
    projection_dropout: float
    initializer_range: float
    norm_type:str
    layer_norm_eps:str
    image_size: Tuple[int]
    patch_size: Tuple[int]
    num_channels:int
    use_mask_token: bool
    use_absolute_position_embeddings: bool
    layer_scale_init_value: float
    use_mean_pooling: bool
    
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()

    @classmethod
    def internvl3_1b_hf(cls):
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            attention_bias=True,
            use_qk_norm=False,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_dropout=0.0,
            projection_dropout=0.0,
            initializer_range=1e-10,
            norm_type="layer_norm",
            layer_norm_eps=1e-06,
            image_size=[448, 448],
            patch_size=[14, 14],
            num_channels=3,
            use_mask_token=False,
            use_absolute_position_embeddings=True,
            layer_scale_init_value=0.1,
            use_mean_pooling=True,
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


class InternVLVisionPatchEmbeddings(nnx.Module):
    def __init__(
            self, 
            image_size: Tuple[int], 
            patch_size: Tuple[int],
            num_channels: int, 
            hidden_size: int,
            rngs: nnx.Rngs,
        ):
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape
        
        self.projection = nnx.Conv(
            in_features=num_channels,
            out_features= hidden_size, 
            kernel_size=patch_size,
            stride=patch_size,
            rngs= rngs
        )
    
    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_patch_embedding')
    def __call__(
            self, 
            pixel_values: jaxtyping.Array
        )->Tuple[jaxtyping.Array, Tuple[int]]:
        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


class InternVLVisionEmbeddings(nnx.Module):
    """Vision embedding module."""

    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        
        self.shd_config = shd_config

        self.cls_token = nnx.Param(
            value=jnp.zeros(shape=(1, 1, config.hidden_size)),
            sharding=shd_config
        )
        
        if config.use_mask_token:
            self.mask_token = nnx.Param(
                value=jnp.zeros(shape=(1, 1, config.hidden_size)),
                sharding=shd_config
            )
            
        else:
            self.mask_token = None

        self.patch_embeddings = InternVLVisionPatchEmbeddings(
            image_size = config.image_size, 
            patch_size = config.patch_size,
            num_channels = config.num_channels, 
            hidden_size = config.hidden_size,
            rngs = rngs
        )
        self.patch_size = config.patch_size
        self.image_size = config.image_size

        num_patches = self.patch_embeddings.num_patches


        if config.use_absolute_position_embeddings:
            self.position_embeddings = nnx.Param(
                value=jnp.zeros(shape=(1, num_patches + 1, config.hidden_size)),
                sharding=shd_config
            )

        else:
            self.position_embeddings = None
        
        self.dropout = nnx.Dropout(
            rate= config.hidden_dropout_prob, 
            rngs= rngs
        )

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_embedding')
    def __call__(
            self, 
            pixel_values: jaxtyping.Array
        ):
        r"""
        bool_masked_pos is None based on implement of huggingface
        so [the section]() is remove
        """
        
        _, _, height, width = pixel_values.shape
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = jnp.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)

    def interpolate_pos_encoding(
            self, 
            embeddings: jaxtyping.Array, 
            height: int, 
            width: int
        ) -> jaxtyping.Array:

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = jnp.power(num_positions,0.5).astype(jnp.int8)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = jax.image.resize(
            image= patch_pos_embed, 
            shape= (new_height, new_width), 
            method="cubic"
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return jnp.cat((class_pos_embed, patch_pos_embed), dim=1)


class InternVLVisionAttention(nnx.Module):
    r"""
    Attention module.
    """
    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.shd_config = shd_config

        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = EinsumBias(
            einsum_str='BTD,DNH->BTNH',
            shape=(config.hidden_size, config.num_attention_heads, self.head_dim),
            bias_shape = (config.num_attention_heads, self.head_dim),
            rngs=rngs,
            sharding=shd_config.q_weight_ndh,
        )
        self.k_proj = EinsumBias(
            einsum_str='BSD,DKH->BSKH',
            shape=(config.hidden_size, config.num_attention_heads, self.head_dim),
            bias_shape = (config.num_attention_heads, self.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )
        self.v_proj = EinsumBias(
            einsum_str='BSD,DKH->BSKH',
            shape=(config.hidden_size, config.num_attention_heads, self.head_dim),
            bias_shape = (config.num_attention_heads, self.head_dim),
            rngs=rngs,
            sharding=shd_config.kv_weight_ndh,
        )

        kernel_init_fn = nnx.initializers.zeros_init()
        self.projection_layer = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            use_bias=True,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                kernel_init_fn, shd_config.ffw_weight_df
            ),
        )
       
        self.scale = self.head_dim**-0.5

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_attention')
    def __call__(
            self,
            hidden_states: jaxtyping.Array,
            attention_mask: jaxtyping.Array,
        ) -> tuple[LayerCache | None, jaxtyping.Array]:
        r"""
        - The following modules are obmitted due to `use_qk_norm` is False:
            - self.projection_dropout
            - self.q_norm
            - self.k_norm 
        """

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = shard(query_states, self.shd_config.act_btnh)
        key_states = shard(key_states, self.shd_config.act_btnh)
        value_states = shard(value_states, self.shd_config.act_btnh)

        b, t, qh, d = query_states.shape
        _, s, kh, _ = key_states.shape

        # GQA
        query_states = query_states.reshape((b, t, kh, qh // kh, d))
        attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_states, key_states) * self.scale
        attn = attn.reshape((b, qh, t, s))

        if attention_mask is not None:
            attn = jnp.where((jnp.expand_dims(attention_mask, -3)), attn, K_MASK)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
            key_states.dtype
        )

        attn = attn.reshape((b, kh, qh // kh, t, s))
        qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_states)
        qkv = qkv.reshape((b, t, qh, d))

        outputs = self.o_proj(qkv)
        outputs = shard(outputs, self.shd_config.act_btd)
        return outputs

    @property
    def head_dim(self):
        return self.o_proj.shape[1]

    @property
    def num_heads(self):
        return self.q_proj.shape[0]

    @property
    def num_kv_heads(self):
        return self.kv_proj.shape[1]


class InternVLVisionMLP(nnx.Module):
    """InternVLVisionMLP module."""

    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):

        if config.hidden_act != "gelu":
            raise NotImplementedError
        else:
            self.activation_fn = nnx.gelu

        self.fc1 = nnx.Linear(
            in_features=config.hidden_size, 
            out_features=config.intermediate_size,
            use_bias=True,
            rngs=rngs
        )
        
        self.fc2 = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            use_bias=True,
            rngs=rngs
        )

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_mlp')
    def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states, approximate=False)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class InternVLVisionLayer(nnx.Module):
    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):

        self.seq_len_dim = 1
        self.attention = InternVLVisionAttention(config)
        self.mlp = InternVLVisionMLP(config)

        # use `layer_norm` for target model only
        if config.norm_type != "layer_norm":
            raise NotImplementedError
        
        self.layernorm_before = nnx.LayerNorm(
            num_features = config.hidden_size,
            epsilon = config.layer_norm_eps,
            use_bias = True,
            use_scale = True,
            rngs = rngs
        )
        self.layernorm_after = nnx.LayerNorm(
            num_features = config.hidden_size,
            epsilon = config.layer_norm_eps,
            use_bias = True,
            use_scale = True,
            rngs = rngs
        )

        
        self.lambda_1 = nnx.Param(
            value= config.layer_scale_init_value * jnp.ones(config.hidden_size),
            sharding=shd_config
            )
        self.lambda_2 = nnx.Param(
            value= config.layer_scale_init_value * jnp.ones(config.hidden_size),
            sharding=shd_config
            )
        
        self.dropout = nnx.Dropout(
            rate= config.hidden_dropout_prob, 
            rngs= rngs
        )

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_layer')
    def __call__(
            self, 
            hidden_states: jaxtyping.ArrayLike,
        ) -> jaxtyping.Array:
        
        attention_output = self.attention(
            self.layernorm_before(hidden_states)
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in InternVLVision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = layer_output + hidden_states

        return layer_output

class InternVLVisionEncoder(nnx.Module):
    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):
        self.layer = [
            InternVLVisionLayer(
                config = config, 
                rngs= rngs, 
                shd_config= shd_config
            ) 
            for _ in range(config.num_hidden_layers)
        ]

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_encode')
    def __call__(
            self, 
            hidden_states: jaxtyping.Array,
        ) -> jaxtyping.Array:
        r"""
        Based on model config, dont output attentions and hidden states
        Returns
            hidden_states : last_hidden_states
        """
        for layer_module in self.layer:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs[0]

        return hidden_states

class InternVLVisionModel(nnx.Module):
    """
    InternVLVisionModel model.
    """

    def __init__(
            self,
            config: InternVLVisionConfig,
            *,
            rngs: nnx.Rngs,
            shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
        ):

        self.config = config

        self.embeddings = InternVLVisionEmbeddings(config)
        self.encoder = InternVLVisionEncoder(
            config = config, 
            rngs= rngs, 
            shd_config= shd_config
        )

    @jax.named_scope(f'{NAMED_SCOPE_PREFIX}_vision_forward')
    def __call__(
            self, 
            pixel_values: jaxtyping.Array,
        ) -> jaxtyping.Array:
        r"""
        - Forward implement of InternVLVisionModel with input arguments:
            - bool_masked_pos
            - output_attentions
            - output_hidden_states
        are omitted

        - self.layernorm is removed since `use_mean_pooling` is True
        """
        embedding_output, _ = self.embeddings(pixel_values)
        return self.encoder(
            hidden_states = embedding_output
        )

