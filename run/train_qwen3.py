import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
    sys.path.append(project_root)


from models.qwen3.params import create_model_from_safe_tensors
from models.qwen3.configs import Qwen3Config
from pipelines.generate import sampler

import jax
# from jax.sharding import Mesh, PartitionSpec, NamedSharding

MESH = [(1, 8), ("fsdp", "tp")]
config = Qwen3Config.qwen3_1_7_b()

mesh = jax.make_mesh(*MESH)

model, tokenizer = create_model_from_safe_tensors(config = config, mesh = mesh)


model_sampler = sampler.Sampler(
    model, 
    tokenizer,
    cache_config= sampler.CacheConfig(
        cache_size=256, 
        num_layers= config.num_layers, 
        num_kv_heads=config.num_kv_heads, 
        head_dim=config.head_dim
    )
)

def templatize(prompts):
    out = []
    for p in prompts:
        out.append(
            tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        )
    return out

inputs = templatize(
    [
        "which is larger 9.9 or 9.11?",
        "讲几句人话",
        "tell me your name, respond in Chinese",
    ]
)

print(model_sampler.transformer)

out = model_sampler(inputs, total_generation_steps=128, echo=True)

for t in out.text:
    print(t)
    print('*' * 30)