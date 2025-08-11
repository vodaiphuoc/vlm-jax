from models.qwen3.params import create_model_from_safe_tensors
from models.qwen3.configs import Qwen3Config

import jax
# from jax.sharding import Mesh, PartitionSpec, NamedSharding

MESH = [(1, 8), ("fsdp", "tp")]
config = Qwen3Config.qwen3_1_7_b()

mesh = jax.make_mesh(*MESH)

model = create_model_from_safe_tensors(config = config, mesh = mesh)

print(model)