import dataclasses

from models.internvl.processor import InternVLProcessorConfig
from models.internvl.vision.model import InternVLVisionConfig
from models.qwen2.model import Qwen2ModelConfig

@dataclasses.dataclass(slots=True, frozen=True)
class InternVL3Config:
    downsample_ratio: float
    projector_hidden_act: str
    image_token_id: int
    vision_feature_select_strategy: str
    processsor_config: InternVLProcessorConfig
    vision_config: InternVLVisionConfig
    text_config: Qwen2ModelConfig

    @classmethod
    def internvl3_1b_hf(cls):
        return cls(
            downsample_ratio = 0.5,
            projector_hidden_act = "gelu",
            image_token_id = 151667,
            vision_feature_select_strategy = "default",
            processsor_config = InternVLProcessorConfig(),
            vision_config =  InternVLVisionConfig.internvl3_1b_hf_vision(),
            text_config = Qwen2ModelConfig.qwen2_0_5_b()
        )
