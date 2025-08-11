"""Utility functions related to attention mask, position for sampler."""

import functools
import jax
import jaxtyping
from jax import lax
import jax.numpy as jnp

from models.utils import typechecked
from models import MODE
from models.types import (
    INPUT_IDS_TYPE, 
    INPUT_MASK_TYPE, 
    ATTENTION_MASK_TYPE, 
    POSITION_IDS_TYPE
)

def build_positions_from_mask(input_mask: INPUT_MASK_TYPE) -> POSITION_IDS_TYPE:
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


def compute_attention_masks(
        time_step: int, 
        seq_len: int, 
        input_mask: jax.Array
    ) -> jax.Array:
    """Computes causal attention mask."""
    batch_size = input_mask.shape[0]
    batch_time_step = jnp.full((batch_size, 1), time_step, dtype=jnp.uint32)
    causal_padding = jnp.greater(
        jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
    )
    max_seq_len = min(input_mask.shape[-1], seq_len)
    input_mask = jax.lax.dynamic_slice(
        input_mask,
        (0, jnp.maximum(time_step - seq_len + 1, 0)),
        (batch_size, max_seq_len),
    )
    input_mask = (
        jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
        .at[:, :max_seq_len]
        .set(input_mask)
    )

    causal_padding = jnp.logical_or(causal_padding, input_mask)
    attention_mask = causal_padding[:, jnp.newaxis, :].astype(jnp.bool_)

    return ~attention_mask

@typechecked(mode= MODE)
def make_causal_attn_mask(
        input_mask: INPUT_MASK_TYPE,
        cache_size: int
    ) -> ATTENTION_MASK_TYPE:
    """Create standard causal attention mask for prefill.

    The causal attention mask during prefill phase is having shape
    (B, L, CACHE_SIZE).

    Args:
        input_mask (`INPUT_MASK_TYPE`): Mask for the input
        cache_size (`int`): KV cache size

    Returns:
        `ATTENTION_MASK_TYPE`: causal attention mask
    """
    seq_len = input_mask.shape[-1]
    attn_mask = input_mask[..., None, :]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    attn_mask *= causal_mask[None, ...]
    padding = cache_size - seq_len
    assert padding >= 0
    attn_mask = jnp.pad(
        attn_mask, (*((0, 0) for _ in range(attn_mask.ndim - 1)), (0, padding))
    )
    return attn_mask


@typechecked(mode= MODE)
def make_causal_attn_mask_for_documents(
        input_mask: INPUT_MASK_TYPE,
        position_ids: POSITION_IDS_TYPE,
        cache_size: int
    )->ATTENTION_MASK_TYPE:
    """
    Makes a causal attention mask between documents, usefull in \
    pre-training where tokens in each document only attend together

    Args:
        - input_mask (`INPUT_MASK_TYPE`): Input mask for the input. True 
        for non-padded tokens only, else False.
        - position_ids (`POSITION_IDS_TYPE`): position ids indice documents \
        with start from 0 to some ID value, it may contains 0 or pad token id \ 
        e.g [[0,1,2,3,4,0,1,2,0,0],[0,1,2,0,1,2,3,0,0,0]]
    Returns:
        Attention mask (`ATTENTION_MASK_TYPE`)
    
    Example:
        - args:
            - input_mask = jnp.array(
                [[1,1,1,1,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,0,0,0]], 
                dtype = jnp.bool_
                )

            - position_ids = jnp.array(
                [[0,1,2,3,4,0,1,2,0,0],
                [0,1,2,0,1,2,3,0,0,0]]
                , dtype = jnp.int8
            )
        - returns:
            [[[1 0 0 0 0 0 0 0 0 0]  # element 1 in batch
            [1 1 0 0 0 0 0 0 0 0]
            [1 1 1 0 0 0 0 0 0 0]
            [1 1 1 1 0 0 0 0 0 0]
            [1 1 1 1 1 0 0 0 0 0]
            [0 0 0 0 0 1 0 0 0 0]
            [0 0 0 0 0 1 1 0 0 0]
            [0 0 0 0 0 1 1 1 0 0]
            [0 0 0 0 0 0 0 0 0 0]    # 2 last pad tokens are ignored
            [0 0 0 0 0 0 0 0 0 0]]

            [[1 0 0 0 0 0 0 0 0 0]   # element 2 in batch
            [1 1 0 0 0 0 0 0 0 0]
            [1 1 1 0 0 0 0 0 0 0]
            [0 0 0 1 0 0 0 0 0 0]
            [0 0 0 1 1 0 0 0 0 0]
            [0 0 0 1 1 1 0 0 0 0]
            [0 0 0 1 1 1 1 0 0 0]
            [0 0 0 0 0 0 0 0 0 0]   # 3 last pad tokens are ignored
            [0 0 0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0 0 0]]]
    """
    attn_mask = make_causal_attn_mask(input_mask,cache_size)

    # count for pad token if pad token id is zero
    # will be removed later by `attn_mask`
    mask = (position_ids == 0)

    # use cumsum, tokens in same document will have
    # same value
    # e.g: 
    # [1 1 1 1 1 2 2 2 3 4]
    # [1 1 1 2 2 2 2 3 4 5]] 
    mask = jnp.cumsum(mask, axis= -1)

    # True where it have same value
    # e.g:
    # [[[1 1 1 1 1 0 0 0 0 0]
    # [1 1 1 1 1 0 0 0 0 0]
    # [1 1 1 1 1 0 0 0 0 0]
    # [1 1 1 1 1 0 0 0 0 0]
    # [1 1 1 1 1 0 0 0 0 0]
    # [0 0 0 0 0 1 1 1 0 0]
    # [0 0 0 0 0 1 1 1 0 0]
    # [0 0 0 0 0 1 1 1 0 0]
    # [0 0 0 0 0 0 0 0 1 0]
    # [0 0 0 0 0 0 0 0 0 1]]

    # [[1 1 1 0 0 0 0 0 0 0]
    # [1 1 1 0 0 0 0 0 0 0]
    # [1 1 1 0 0 0 0 0 0 0]
    # [0 0 0 1 1 1 1 0 0 0]
    # [0 0 0 1 1 1 1 0 0 0]
    # [0 0 0 1 1 1 1 0 0 0]
    # [0 0 0 1 1 1 1 0 0 0]
    # [0 0 0 0 0 0 0 1 0 0]
    # [0 0 0 0 0 0 0 0 1 0]
    # [0 0 0 0 0 0 0 0 0 1]]]

    docmask = mask[:, None, :] == mask[..., None]
    return attn_mask & docmask


@typechecked(mode= MODE)
def make_causal_attn_mask_for_vision(
        input_ids: INPUT_IDS_TYPE,
        input_mask: INPUT_MASK_TYPE,
        cache_size: int, 
        start_img_token_id:int,
        context_img_token_id:int,
        end_img_token_id:int
    )->ATTENTION_MASK_TYPE:
    """
    Makes a causal attention mask for vision-language model where \
    text tokens are attented in causal way and vision tokens can \
    attented in bidirectinal way.

    Args:
        - input_ids (`INPUT_IDS_TYPE`): input token of texts
        - input_mask (`INPUT_MASK_TYPE`): Input mask for the input. True 
        for non-padded tokens only, else False.
        - start_img_token_id (int): taken from tokenizer
        - context_img_token_id (int): taken from tokenizer
        - end_img_token_id (int): taken from tokenizer
    Returns:
        Attention mask (`ATTENTION_MASK_TYPE`)

    NOTE: this implement rely on sepecial tokens in `input_ids`
    """
    attn_mask = make_causal_attn_mask(input_mask, cache_size)

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
