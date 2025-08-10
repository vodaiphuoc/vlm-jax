from models.types import (
    INPUT_IDS_TYPE, 
    INPUT_MASK_TYPE, 
    ATTENTION_MASK_TYPE, 
    POSITION_IDS_TYPE
)
from models.utils import typechecked
from models import MODE
import jax.numpy as jnp
import jaxtyping

@typechecked(mode= MODE)
def make_causal_attn_mask(
        input_mask: INPUT_MASK_TYPE,
    )->ATTENTION_MASK_TYPE:
    """
    Makes standar causal attention mask

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
    return attn_mask


@typechecked(mode= MODE)
def make_causal_attn_mask_for_documents(
        input_mask: INPUT_MASK_TYPE,
        position_ids: POSITION_IDS_TYPE
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
    attn_mask = make_causal_attn_mask(input_mask)

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
    attn_mask = make_causal_attn_mask(input_mask)

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
