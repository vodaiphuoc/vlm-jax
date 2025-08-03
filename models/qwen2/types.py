import jaxtyping
from typing import List

class TEXT_EMBEDDING_OUT_TYPE(jaxtyping.BFloat16[jaxtyping.Array,"B S 896"]):
    r"""
    Type of output of `vision.model.InternVLVisionPatchEmbeddings` class
        - dtype: bfloat16
        - shape: (B, S, D) where:
            - B is batch of input text
            - S is sequence length of input text
            - 896 is text embedding shape, see Qwen2ModelConfig.embed_dim
    """

