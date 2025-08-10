"""InternVL3 1B hf"""
from functools import partial
import dataclasses
from typing import Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping


from models.internvl.configs import InternVL3Config
from models.internvl.mm_proj import InternVLMultiModalProjector
from models.internvl.vision.model import InternVLVisionModel

from models.utils import typechecked
from models import MODE
from models.masks import make_causal_attn_mask_for_vision
from models.types import (
    INPUT_IDS_TYPE,
    POSITION_IDS_TYPE,
    INPUT_MASK_TYPE,
    INPUT_IMAGES_TYPE,
    PIXEL_SHUFFLE_INPUT_TYPE,
    PIXEL_SHUFFLE_OUTPUT_TYPE,
    MM_PROJ_OUTPUT_TYPE,
    ATTN_HIDDEN_STATES_TYPE,
    Cache
)

from models.qwen2.model import Qwen2ForCausalLM



@partial(jax.jit,static_argnames = "context_img_token_id")
@typechecked(mode = MODE)
def merge_embeddings(
        input_ids: INPUT_IDS_TYPE, 
        input_embedd: ATTN_HIDDEN_STATES_TYPE,
        image_features: MM_PROJ_OUTPUT_TYPE,
        context_img_token_id:int
    )->ATTN_HIDDEN_STATES_TYPE:
    r"""
    Merge image features into input embedded
    """
    N, SI, D = image_features.shape
    image_features = image_features.reshape((N*SI,D))

    # where to replace context image ids with image features
    x_img_ids, y_img_ids = jnp.where(
        input_ids == context_img_token_id, 
        size = N*SI
    )

    # merging
    merged_input_embedd = input_embedd.at[x_img_ids, y_img_ids, : ].set(image_features)

    return merged_input_embedd


class INternVL3(nnx.Module):
    """
    Main implementation for intervl3 1B hf
    """
    def __init__(
            self,
            config: InternVL3Config,
            *,
            rngs: nnx.Rngs
        ):
        self.config = config
        self.vision_tower = InternVLVisionModel(
            config = config.vision_config,
            rngs= rngs
        )
        self.multi_modal_projector = InternVLMultiModalProjector(
            config = config,
            rngs = rngs
        )
        self.language_model = Qwen2ForCausalLM(
            config = config.text_config,
            rngs = rngs
        )

    @typechecked(mode=MODE)
    def pixel_shuffle(
            self, 
            vision_features: PIXEL_SHUFFLE_INPUT_TYPE, 
            scale_factor: float = 0.5
        )->PIXEL_SHUFFLE_OUTPUT_TYPE:
        """Perform pixel shuffle downsampling on vision features.

        Args:
            vision_features (jaxtyping.Array):
                Input tensor of shape (batch_size, width, height, channels).
            scale_factor (`float`, *optional*, defaults to `0.5`):
                Factor by which to downsample. Default is 0.5, which halves the dimensions.

        Returns:
            vision_features (jaxtyping.Array):
                Downsampled tensor of shape 
                (batch_size, height*scale_factor, width*scale_factor, channels/(scale_factor^2)).
        """
        batch_size, width, height, channels = vision_features.size()

        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError("Height and width must be divisible by scale_factor for proper downsampling.")

        # Reshape to allow downsampling
        vision_features = vision_features.view(
            batch_size, width, int(height * scale_factor), int(channels / scale_factor)
        )
        # Permute dimensions to align downsampled axis correctly
        vision_features = vision_features.transpose(0, 2, 1, 3).contiguous()

        # Reshape to achieve final downsampled dimensions
        vision_features = vision_features.view(
            batch_size, int(height * scale_factor), int(width * scale_factor), int(channels / (scale_factor**2))
        )

        # Swap height and width back for proper orientation
        vision_features = vision_features.transpose(0, 2, 1, 3).contiguous()

        return vision_features

    @typechecked(mode=MODE)
    def get_image_features(
            self,
            pixel_values: INPUT_IMAGES_TYPE
        )->MM_PROJ_OUTPUT_TYPE:
        r"""
        Get image features from batch of list images.
        Args:
            pixel_values (jaxtyping.UInt8[jaxtyping.Array,"B_I H W C"]): with `B_I` is batch
            of all images in B batch
        """
        vision_features = self.vision_tower(pixel_values=pixel_values)
        vision_features = vision_features[:, 1:, :]

        channels = vision_features.shape[1]
        feature_size = int(channels**0.5)
        batch_size = vision_features.shape[0]

        # Reshape tensor to spatial dimensions
        vision_features = vision_features.reshape(batch_size, feature_size, feature_size, -1)

        # Apply downsampling using pixel shuffle
        vision_features = self.pixel_shuffle(vision_features, scale_factor=self.config.downsample_ratio)

        # Reshape tensor to prepare for projection
        vision_features = vision_features.reshape(batch_size, -1, vision_features.shape[-1])

        # Project features through multi-modal projector
        vision_features = self.multi_modal_projector(vision_features)
        return vision_features

    @typechecked(mode=MODE)
    def __call__(
            self,
            input_ids: INPUT_IDS_TYPE,
            position_ids: POSITION_IDS_TYPE,
            cache: Cache | None, 
            input_mask: INPUT_MASK_TYPE,
            pixel_values: INPUT_IMAGES_TYPE|None,
        ):
        r"""
        Args:
            input_ids: input sequence of tokens.
            positions: input absolute positions.
            cache: Attention KV cache or None.
            input_mask: transformer input mask.
        NOTE: input_ids after apply tokenizer has been 
        inserted [...,start_image_token_id, context_image_token_id*image_seq_length*num_patches, end_image_token_id...]
        with image_seq_length is 256, num_patches of a image is 13

        """
        # get text features from text embedding
        inputs_embeds: ATTN_HIDDEN_STATES_TYPE = self.language_model.embedder(input_ids)

        # get image features
        image_features: MM_PROJ_OUTPUT_TYPE = self.get_image_features(pixel_values=pixel_values)
        
        # image mask
        image_mask = input_ids == self.config.image_token_id

        # check equal
        n_image_features = image_features.shape[0] * image_features.shape[1]
        n_image_tokens = image_mask.sum()
        assert n_image_features == n_image_tokens, \
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"

        # merge image feautres with input embeds
        merged_input_embedd = merge_embeddings(
            input_ids = input_ids, 
            input_embedd = inputs_embeds,
            image_features = image_features,
            context_img_token_id = self.config.processsor_config.context_image_token_id
        )

        attention_mask = make_causal_attn_mask_for_vision(
            input_ids = input_ids,
            input_mask = input_mask,
            start_img_token_id = self.config.processsor_config.start_image_token_id,
            context_img_token_id = self.config.processsor_config.context_image_token_id,
            end_img_token_id = self.config.processsor_config.end_image_token_id
        )

        outputs = self.language_model(
            input_embedd = merged_input_embedd, 
            position_ids = position_ids,
            cache = cache,
            attention_mask = attention_mask
        )

        return outputs, cache
