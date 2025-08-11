# - annotations:
#     - B: batch size
#     - L: sequence length of text input also sequence length in Q_proj
#     - S: sequence length in K_proj and V_proj or Cache size
#     - D: embedding dim
#     - H: head dim
#     - N: num attention heads
#     - K: num key-value groups

import jaxtyping
from typing import List, TypedDict, Dict

class INPUT_IDS_TYPE(
        jaxtyping.Int[jaxtyping.Array,"B L"],
        jaxtyping.Array
    ):
    r"""
    Type of input ids (after tokenized) with:
        - dtype: int
        - shape: (B, L) where B=Batch, L=Sequence length
    """

class INPUT_MASK_TYPE(
        jaxtyping.Bool[jaxtyping.Array,"B L"],
        jaxtyping.Array
    ):
    r"""
    Type of `attention_mask` from HF tokeninzer, be used as `input_mask`
        - dtype: boolen
        - shape: (B, L) where B=Batch, L=Sequence length
    """

class ATTENTION_MASK_TYPE(
        jaxtyping.Bool[jaxtyping.Array,"B L S"],
        jaxtyping.Array
    ):
    r"""
    Type of attention_mask input to attention layer
        - dtype: boolen
        - shape: (B, L, S) where B=Batch, S=Cache size
    """

class POSITION_IDS_TYPE(
        jaxtyping.Int[jaxtyping.Array, "B L"],
        jaxtyping.Array
    ):
    r"""
    Type of position ids
        - dtype: int
        - shape: (B, L)
    """

class INPUT_IMAGES_TYPE(
        jaxtyping.UInt8[jaxtyping.Array,"B_I H W C"],
        jaxtyping.Array
    ):
    r"""
    Type of input images in channels-last convention
        - dtype: uint8
        - shape: (B_I, H, W ,C) with B_I is batch of all images
         (with `crop_to_patches`) in batch of inputs
    """

class PATCH_EMBEDDING_OUT_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I S D"],
        jaxtyping.Array
    ):
    r"""
    Type of output of `vision.model.InternVLVisionPatchEmbeddings` class
        - dtype: bfloat16
        - shape: (B_I, S, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - D is hidden size
            - S is sequence length aka num_patches
    """

class VISION_HIDDEN_STATES_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I S+1 D"],
        jaxtyping.Array
    ):
    r"""
    Type of output embedding shape is also hidden states input of attention
        - dtype: bfloat16
        - shape: (B_I, S+1, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - S is sequence length, plus one for cls token
            - D is hidden size
    """

class PIXEL_SHUFFLE_INPUT_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I H_no_scalse W_no_scale D"],
        jaxtyping.Array    
    ):
    r"""
    Type of input of `INternVL3.pixel_shuffle` method
        - dtype: bfloat16
        - shape: (B_I, H_no_scalse, W_no_scale, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - H_no_scalse, W_no_scale: shape of images
            - - D is hidden size
    """


class PIXEL_SHUFFLE_OUTPUT_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I H_scale W_scale 4096"],
        jaxtyping.Array
    ):
    r"""
    Type of output of `INternVL3.pixel_shuffle` method
        - dtype: bfloat16
        - shape: (B_I, H_no_scalse, W_no_scale, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - H_no_scalse, W_no_scale: shape of images
            - D is hidden size == 4096
    """

class MM_PROJ_INPUT_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I _ 4096"],
        jaxtyping.Array
    ):
    r"""
    Type of input of `mm_proj.InternVLMultiModalProjector.__call__` method
        - dtype: bfloat16
        - shape: (B_I, * , D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - Any middle shape
            - D is hidden size == 4096
    """
    
class MM_PROJ_OUTPUT_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B_I 256 896"],
        jaxtyping.Array
    ):
    r"""
    Type of output of `mm_proj.InternVLMultiModalProjector.__call__` method
        - dtype: bfloat16
        - shape: (B_I, mm_seq_length , D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - 256 is image sequence length
            - 896 is text embedding shape, see Qwen2ModelConfig.embed_dim
    """

class ATTN_HIDDEN_STATES_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B L D"],
        jaxtyping.Array
    ):
    r"""
    Type of hidden state input to attention also type of output \
of `vision.model.InternVLVisionPatchEmbeddings` class.

        - dtype: bfloat16
        - shape: (B, L, D) where:
            - B is batch of input text
            - L is sequence length of input text
            - D is text embedding shape, see Qwen2ModelConfig.embed_dim
    """


class Q_PROJ_VALUES_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B L N H"],
        jaxtyping.Array
    ):
    r"""
    Type of projected values after passed through q_proj
        - dtype: bloat16
        - shape: (B, L, N, H) where L can be sequence
            length of input in prefill phase or 1 in 
            decode phase
    """

class KV_PROJ_VALUES_TYPE(
        jaxtyping.BFloat16[jaxtyping.Array,"B S K H"],
        jaxtyping.Array
    ):
    r"""
    Type of projected values after passed through k_proj, v_proj
        - dtype: bloat16
        - shape: (B, S, K, H) where S can be sequence
            length of input if cache is None, else is full
            size of cache
    """


class SYS_INST_PART(TypedDict):
    content: str
    role:str = "system"

class TEXT_PART(TypedDict):
    text: str
    type:str = "text"

class IMG_PART(TypedDict):
    url: str
    type:str = "image"

class USER_PART(TypedDict):
    content: List[TEXT_PART|IMG_PART]
    role:str = "user"

CONVERSATION = List[SYS_INST_PART| USER_PART]


class LayerCache(TypedDict):
    r"""Type hint for cache of each layer"""
    k: KV_PROJ_VALUES_TYPE
    v: KV_PROJ_VALUES_TYPE
    end_index: jaxtyping.Int32[jaxtyping.Array, "B"]
    
Cache = Dict[str, LayerCache]