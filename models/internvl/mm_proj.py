from flax import nnx
import jax
import jaxtyping
from .model import InternVL3Config
from models.utils import typechecked
from models import MODE
from models.internvl.types import MM_PROJ_INPUT_TYPE, MM_PROJ_OUTPUT_TYPE

class InternVLMultiModalProjector(nnx.Module):
    def __init__(
            self,
            config: InternVL3Config,
            *,
            rngs: nnx.Rngs
        ):
        self.layer_norm = nnx.LayerNorm(
            num_features = config.vision_config.hidden_size*int(1/config.downsample_ratio)**2,
            epsilon = 1e-05,
            use_bias = True,
            use_scale = True,
            rngs = rngs
        )

        self.linear_1 = nnx.Linear(
            in_features=config.vision_config.hidden_size*int(1 / config.downsample_ratio)**2, 
            out_features=config.text_config.embed_dim,
            use_bias=True,
            rngs=rngs
        )

        if config.projector_hidden_act != "gelu":
            raise NotImplementedError
        else:
            self.act = nnx.gelu
        
        self.linear_2 = nnx.Linear(
            in_features=config.text_config.embed_dim, 
            out_features=config.text_config.embed_dim,
            use_bias=True,
            rngs=rngs
        )

    @typechecked(mode= MODE)
    @jax.named_scope('mm_projecter')
    def __call__(self, image_features: MM_PROJ_INPUT_TYPE) -> MM_PROJ_OUTPUT_TYPE:
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
