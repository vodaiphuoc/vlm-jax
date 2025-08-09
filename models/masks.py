from models.types import INPUT_IDS_TYPE, INPUT_MASK_TYPE, ATTENTION_MASK_TYPE, POSITION_IDS_TYPE
from models.utils import typechecked
from models import MODE
import jax.numpy as jnp
import jaxtyping

@typechecked(mode= MODE)
def make_causal_attn_mask(
        input_ids: INPUT_IDS_TYPE,
        input_mask: INPUT_MASK_TYPE,
        start_img_token_id:int,
        context_img_token_id:int,
        end_img_token_id:int
    )->ATTENTION_MASK_TYPE:
    """
    Makes a causal attention mask.
    Args:
        input_mask (`INPUT_MASK_TYPE`): Input mask for the input. True 
        for non-padded tokens only, else False.
    Returns:
        Attention mask (`ATTENTION_MASK_TYPE`)
    """
    seq_len = input_mask.shape[-1]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
    attn_mask = input_mask[..., None, :]
    attn_mask *= causal_mask[None, ...]

    bidirectional_mask = (input_ids == start_img_token_id) | \
        (input_ids == context_img_token_id) | \
        (input_ids == end_img_token_id)


    q_block_indices = _make_block_mask_indices(bidirectional_mask)
    kv_block_indices = q_block_indices

    attn_mask = attn_mask | (
        (kv_block_indices[:, None, :] == q_block_indices[..., None])
        & (q_block_indices[..., None] > 0)
    )

    return attn_mask

@typechecked(mode= MODE)
def _make_block_mask_indices(
        bidirectional_mask: INPUT_MASK_TYPE,
    ) -> jaxtyping.Int[jaxtyping.Array, "B L"]:
    """Creates block mask identifying segments based on a bidirectional mask.
    Args:
        bidirectional_mask: boolean mask, e.g. [011110011010].

    Returns:
        block mask for segments, e.g. [011110022030].
    """
    # Left pad 0.
    padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
    boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
    numbered_boundary = jnp.cumsum(boundary, axis=-1)
    return bidirectional_mask * numbered_boundary

@typechecked(mode= MODE)
def get_positions_from_mask(input_mask: INPUT_MASK_TYPE) -> POSITION_IDS_TYPE:
    """Computes the `positions` from the `input_mask`.

    Args:
        input_mask: The tokens `input_mask`, True for non-padded tokens only.

    Returns:
        The indices to use for RoPE and absolute position encodings for the given
        input mask.
    """
    positions = jnp.cumsum(input_mask, axis=-1)
    # Subtract one for all positions from the first valid one as they are
    # 0-indexed
    return positions - (positions >= 1)
