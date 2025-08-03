import jaxtyping
from typing import List, TypedDict

class INPUT_IDS_TYPE(jaxtyping.Int[jaxtyping.Array,"B S"]):
    r"""
    Type of input ids (after tokenized) with:
        - dtype: int
        - shape: (Batch, Sequence length)
    """

class INPUT_IMAGES_TYPE(jaxtyping.UInt8[jaxtyping.Array,"B_I H W C"]):
    r"""
    Type of input images in channels-last convention
        - dtype: uint8
        - shape: (B_I, H, W ,C) with B_I is batch of all images
         (with `crop_to_patches`) in batch of inputs
    """

class PATCH_EMBEDDING_OUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I S D"]):
    r"""
    Type of output of `vision.model.InternVLVisionPatchEmbeddings` class
        - dtype: bfloat16
        - shape: (B_I, S, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - D is hidden size
            - S is sequence length aka num_patches
    """

class HIDDEN_STATES_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I S+1 D"]):
    r"""
    Type of output embedding shape is also hidden states input of attention
        - dtype: bfloat16
        - shape: (B_I, S+1, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - S is sequence length, plus one for cls token
            - D is hidden size
    """

class PIXEL_SHUFFLE_INPUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I H_no_scalse W_no_scale D"]):
    r"""
    Type of input of `INternVL3.pixel_shuffle` method
        - dtype: bfloat16
        - shape: (B_I, H_no_scalse, W_no_scale, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - H_no_scalse, W_no_scale: shape of images
            - - D is hidden size
    """


class PIXEL_SHUFFLE_OUTPUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I H_scale W_scale 4096"]):
    r"""
    Type of output of `INternVL3.pixel_shuffle` method
        - dtype: bfloat16
        - shape: (B_I, H_no_scalse, W_no_scale, D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - H_no_scalse, W_no_scale: shape of images
            - D is hidden size == 4096
    """

class MM_PROJ_INPUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I _ 4096"]):
    r"""
    Type of input of `mm_proj.InternVLMultiModalProjector.__call__` method
        - dtype: bfloat16
        - shape: (B_I, * , D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - Any middle shape
            - D is hidden size == 4096
    """
    
class MM_PROJ_OUTPUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B_I 256 896"]):
    r"""
    Type of output of `mm_proj.InternVLMultiModalProjector.__call__` method
        - dtype: bfloat16
        - shape: (B_I, * , D) where:
            - B_I is batch of all images (with `crop_to_patches`) 
            in batch of inputs
            - 256 is image sequence length
            - 896 is text embedding shape, see Qwen2ModelConfig.embed_dim
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
