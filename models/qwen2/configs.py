import dataclasses
from typing import Tuple

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
    repo_id:str
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
            repo_id = "Qwen/Qwen2-0.5B",
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

