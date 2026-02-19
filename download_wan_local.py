from huggingface_hub import snapshot_download
import os

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
local_dir = "Wan2.1-T2V-1.3B"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Download complete.")
