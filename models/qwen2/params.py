"""Utils for loading and converting Qwen2 PT weights."""
from typing import Tuple
import re
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import safetensors.flax as safetensors
import tqdm
from .model import Qwen2ForCausalLM
from .configs import Qwen2ModelConfig
from models.utils import download_hf_repo

def _stack_experts(params: dict[str, jax.Array]):
    """Stack experts in the loaded pytorch params."""
    key_fn = lambda x: int(re.match(r"(.*?)experts\.([0-9]+)\..*", x).group(2))  # pytype: disable=attribute-error
    new_params = dict(params).copy()
    
    for kw in ["gate", "up", "down"]:
        pattern = r"(.*?)experts\.(.*?)\.{}_proj\.(.*)".format(kw)
        keys = [k for k in params.keys() if re.match(pattern, k)]
        prefix_groups = set([re.match(pattern, k).group(1) for k in keys])  # pytype: disable=attribute-error
        for prefix in prefix_groups:
            keys_to_merge = list(
                sorted([k for k in keys if k.startswith(prefix)], key=key_fn)
            )
        for k in keys_to_merge:
            del new_params[k]
        
        suffix = re.match(pattern, keys_to_merge[0]).group(3)  # pytype: disable=attribute-error
        with jax.default_device(jax.devices("cpu")[0]):
            new_params[f"{prefix}experts.{kw}_proj.{suffix}"] = jnp.stack(
                [params[k] for k in keys_to_merge], 0
            )
    return new_params


def _get_key_and_transform_mapping(cfg: Qwen2ModelConfig):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
    return {
        r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
        # attention projection weights
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
            r"layers.\1.attn.q_proj.b",
            (None, (cfg.num_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.attn.k_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
            r"layers.\1.attn.k_proj.b",
            (None, (cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.attn.v_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
            r"layers.\1.attn.v_proj.b",
            (None, (cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
        ),
        # mlp
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            ((1, 0), None),
        ),
        # moe
        r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
            r"layers.\1.mlp.router.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj",
            ((0, 2, 1), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj",
            ((0, 2, 1), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.experts\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj",
            ((0, 2, 1), None),
        ),
        # norms
        r"model\.norm\.weight": ("final_norm.w", None),
        r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
            r"layers.\1.attn.q_norm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
            r"layers.\1.attn.k_norm.w",
            None,
        ),
        # layer norms (pre/post attention)
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.w",
            None,
        )
    }


def _torch_key_to_jax_key(mapping, source_key):
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        raise ValueError(f"Only one key should be found: {subs}, len: {len(subs)}")
    else:
        return subs[0]


def _assign_weights(keys, tensor: jax.Array, state_dict, torch_key, transform: Tuple[Tuple[int],None]|None):
    """Convert weights and assign to nnx state_dict."""
    key = keys[0]
    if len(keys) == 1:
        try:
            if transform is not None:
                permute, reshape = transform
                tensor = tensor.transpose(permute) if permute else tensor
                tensor = tensor.reshape(reshape) if reshape else tensor
        except Exception as e:
            raise RuntimeError(
                f"Failed to transform tensor {torch_key} with shape"
                f" {tensor.shape}: {e}"
            ) from e

        if tensor.shape != state_dict[key].shape:
            raise ValueError(
                f"shape must match for {torch_key}, got {tensor.shape} vs"
                f" {state_dict[key].shape}"
            )
        state_dict[key] = tensor
        return state_dict
    
    else:
        if key not in state_dict:
            raise ValueError(f"Unfound key {key} in {state_dict}")
        _assign_weights(keys[1:], tensor, state_dict[key], torch_key, transform)
        return state_dict


def _stoi(s):
    try:
        return int(s)
    except ValueError:
        return s


def create_model_from_safe_tensors(
    config: Qwen2ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
) -> Qwen2ForCausalLM:
    file_dir = download_hf_repo(repo_id= config.repo_id)
    """Load tensors from the safetensors file and create a Qwen2 model."""
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")

    tensor_dict = {}
    for f in tqdm.tqdm(files):
        tensor_dict |= safetensors.load_file(f)

    if config.num_experts is not None:
        tensor_dict = _stack_experts(tensor_dict)

    qwen2 = nnx.eval_shape(
        lambda: Qwen2ForCausalLM(config, rngs=nnx.Rngs(params=0))
    )

    graph_def, abs_state = nnx.split(qwen2)
    state_dict = abs_state.to_pure_dict()

    for k in tensor_dict.keys():
        print(k)

    with jax.default_device(jax.devices("cpu")[0]):
        for k, v in tqdm.tqdm(tensor_dict.items()):
            try:
                jax_key, transform = _torch_key_to_jax_key(
                    mapping = _get_key_and_transform_mapping(config), 
                    source_key = k
                )
            except Exception as e:
                print('key failed: ', k)
                raise e

            jax_keys = [_stoi(s) for s in jax_key.split(".")]
            _assign_weights(jax_keys, v, state_dict, k, transform)

    if mesh is not None:
        sharding = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
        state_dict = jax.device_put(state_dict, sharding)
    else:
        state_dict = jax.device_put(state_dict, jax.devices()[0])

    nnx.replace_by_pure_dict(abs_state, state_dict)
    return nnx.merge(graph_def, abs_state)
