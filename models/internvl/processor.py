from typing import Dict, List
import jax.numpy as jnp
import dataclasses
from transformers import AutoProcessor
from models.internvl.types import CONVERSATION

@dataclasses.dataclass(slots=True, frozen=True)
class InternVLProcessorConfig(object):
    model_id: str = "OpenGVLab/InternVL3-1B-hf"
    start_image_token: str = "<img>"
    start_image_token_id: int = 151665
    end_image_token: str = "</img>"
    end_image_token_id: int = 151666
    context_image_token:str = "<IMG_CONTEXT>"
    context_image_token_id: int = 151667

class InternVLProcessor(object):
    def __init__(self, config:InternVLProcessorConfig = InternVLProcessorConfig()):
        r"""
        Args:
            model_id (str): model id on hf or local dir
        """
        self.config = config
        self._processor = AutoProcessor.from_pretrained(config.model_id)

        # check id miss-match
        assert self._processor.tokenizer.start_image_token == self.config.start_image_token
        assert self._processor.tokenizer.start_image_token_id == self.config.start_image_token_id
        assert self._processor.tokenizer.end_image_token == self.config.end_image_token
        assert self._processor.tokenizer.end_image_token_id == self.config.end_image_token_id
        assert self._processor.tokenizer.context_image_token == self.config.context_image_token
        assert self._processor.tokenizer.context_image_token_id == self.config.context_image_token_id
    
    def __call__(self, message: CONVERSATION):
        inputs = self._processor.apply_chat_template(
            message, 
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        return (inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values'])

    @property
    def start_image_token(self)->str:
        return self.config.start_image_token

    @property
    def start_image_token_id(self)->int:
        return self.staconfig.rt_image_token_id
    
    @property
    def end_image_token(self)->str:
        return self.config.end_image_token
    
    @property
    def end_image_token_id(self)->int:
        return self.config.end_image_token_id

    @property
    def context_image_token(self)->str:
        return self.config.context_image_token
    
    @property
    def context_image_token_id(self)->int:
        return self.config.context_image_token_id
    