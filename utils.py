from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2-0.5B"
snapshot_download(model_id, local_dir="./.hf_cache/" + model_id,revision="main")