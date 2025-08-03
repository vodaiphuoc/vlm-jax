
from typing import Dict, List
import jax.numpy as jnp

from transformers import AutoProcessor
from models.internvl.types import CONVERSATION


class InternVLProcessor(object):
    def __init__(self, model_id:str = "OpenGVLab/InternVL3-1B-hf"):
        r"""
        Args:
            model_id (str): model id on hf or local dir
        """
        self._processor = AutoProcessor.from_pretrained(model_id)
    
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
        return self._processor.tokenizer.start_image_token

    @property
    def start_image_token_id(self)->int:
        return self._processor.tokenizer.start_image_token_id
    
    @property
    def end_image_token(self)->str:
        return self._processor.tokenizer.end_image_token
    
    @property
    def end_image_token_id(self)->int:
        return self._processor.tokenizer.end_image_token_id

    @property
    def context_image_token(self)->str:
        return self._processor.tokenizer.context_image_token
    
    @property
    def context_image_token_id(self)->int:
        return self._processor.tokenizer.context_image_token_id
    