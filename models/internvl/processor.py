
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
        return inputs['input_ids'], inputs['attention_mask'], inputs['pixel_values']
