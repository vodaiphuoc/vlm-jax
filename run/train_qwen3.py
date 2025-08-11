from models.qwen3.params import create_model_from_safe_tensors
from models.qwen3.configs import Qwen3Config

config = Qwen3Config.qwen3_1_7_b()

model = create_model_from_safe_tensors(config)