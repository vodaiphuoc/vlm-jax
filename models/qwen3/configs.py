import dataclasses
from typing import Tuple

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for Qwen3 model."""

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
class Qwen3Config:
    """Configuration for the Qwen3 model."""

    num_layers: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()

    @classmethod
    def qwen3_0_6_b(cls):  # qwen3-0.6B
        return cls(
            num_layers=28,
            vocab_size=151936,
            embed_dim=1024,
            hidden_dim=3072,
            num_heads=16,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
        )

    @classmethod
    def qwen3_1_7_b(cls):  # qwen3-1.7B
        return cls(
            num_layers=28,
            vocab_size=151936,
            embed_dim=2048,
            hidden_dim=6144,
            num_heads=16,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
        )

    @classmethod
    def qwen3_14_b(cls):  # qwen3-14B
        return cls(
            num_layers=40,
            vocab_size=151936,
            embed_dim=5120,
            hidden_dim=17408,
            num_heads=40,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-06,
            rope_theta=1_000_000,
        )

    @classmethod
    def qwen3_30_b(cls):  # qwen3-30B
        return cls(
            num_layers=48,
            vocab_size=151936,
            embed_dim=2048,
            hidden_dim=768,
            num_heads=32,
            head_dim=128,
            num_kv_heads=4,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            num_experts=128,
            num_experts_per_tok=8,
        )
