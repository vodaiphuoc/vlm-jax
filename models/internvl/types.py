import jaxtyping

INPUT_IDS_TYPE = jaxtyping.Int[jaxtyping.Array,"B S"]

# channels-last convention
INPUT_IMAGES_TYPE = jaxtyping.UInt8[jaxtyping.Array,"B_I H W C"] 

# D is hidden size, S is sequence length aka num_patches
PATCH_EMBEDDING_OUT_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I S D"]

# output embedding shape is also hidden states input of attention
HIDDEN_STATES_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I S+1 D"]

PIXEL_SHUFFLE_INPUT_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I * * D"]
PIXEL_SHUFFLE_OUTPUT_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I H_scale W_scale 4096"]

# ED is text embedding shape, see Qwen2ModelConfig.embed_dim
MM_PROJ_INPUT_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I * 4096"]
MM_PROJ_OUTPUT_TYPE = jaxtyping.BFloat16[jaxtyping.Array,"B_I * ED"]