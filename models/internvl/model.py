"""InternVL3 1B hf"""

import dataclasses
from typing import Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]

from .mm_proj import InternVLMultiModalProjector
from models.internvl.vision.model import InternVLVisionModel, InternVLVisionConfig
from models.internvl.types import (
    INPUT_IDS_TYPE, 
    INPUT_IMAGES_TYPE,
    PIXEL_SHUFFLE_INPUT_TYPE,
    PIXEL_SHUFFLE_OUTPUT_TYPE,
    MM_PROJ_OUTPUT_TYPE
)
from models.qwen2.types import TEXT_EMBEDDING_OUT_TYPE
from models.qwen2.model import Qwen2, Qwen2ModelConfig
from models.utils import typechecked
from models import MODE


@dataclasses.dataclass(slots=True, frozen=True)
class InternVL3Config:
    downsample_ratio: float
    projector_hidden_act: str
    image_token_id: int
    vision_feature_select_strategy: str
    vision_config: InternVLVisionConfig
    text_config: Qwen2ModelConfig

    @classmethod
    def internvl3_1b_hf(cls):
        return cls(
            downsample_ratio = 0.5,
            projector_hidden_act = "gelu",
            image_token_id = 151667,
            vision_feature_select_strategy = "default",
            vision_config =  InternVLVisionConfig.internvl3_1b_hf_vision(),
            text_config = Qwen2ModelConfig.qwen2_0_5_b()
        )

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
        self.language_model = Qwen2(
            config = config.text_config,
            rngs= rngs
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
            position_ids: jaxtyping.Array,
            cache: Cache | None, 
            attention_mask: jaxtyping.Array,
            pixel_values: INPUT_IMAGES_TYPE,
        ):
        r"""
        Args:
            input_ids: input sequence of tokens.
            positions: input absolute positions.
            cache: Attention KV cache or None.
            attention_mask: transformer input mask.
        """
        # get text features from text embedding
        inputs_embeds: TEXT_EMBEDDING_OUT_TYPE = self.language_model.embedder(input_ids)

        # get image features
        image_features: MM_PROJ_OUTPUT_TYPE = self.get_image_features(pixel_values=pixel_values)

        # image mask
        image_mask = input_ids == self.config.image_token_id

        # merge image feautres with input embeds
        


        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return outputs, cache
