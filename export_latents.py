
import torch
import numpy as np
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
import os

def export_latents():
    print("Loading VAE (Float32)...")
    model_id = "e:/Projects/textModel/Wan2.1-T2V-1.3B"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    vae.enable_slicing()
    vae.enable_tiling()
    vae.to("cuda")

    print("Loading latents from latents_checkpoint.pt...")
    if not os.path.exists("latents_checkpoint.pt"):
        print("Error: latents_checkpoint.pt not found!")
        return
        
    # Load on CUDA
    latents = torch.load("latents_checkpoint.pt", map_location="cuda").to(dtype=torch.float32)
    print(f"Latents shape: {latents.shape}")

    # No need to move VAE to CPU
    
    # CRITICAL FIX: The checkpoint IS ALREADY denormalized if saved after arithmetic.
    # Check bounds to confirm
    print(f"Latents Min: {latents.min()}, Max: {latents.max()}")
    
    print("SKIPPING normalization (checkpoint already processed).")
    
    print("Decoding latents on CUDA...")
    with torch.no_grad():
        video_tensor = vae.decode(latents, return_dict=False)[0]
    
    print(f"Video Tensor Min: {video_tensor.min()}, Max: {video_tensor.max()}")

    
    # Un-normalize [0, 1]
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    
    video_np = video_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video_uint8 = (video_np * 255).round().astype(np.uint8)

    path_rgb = "output_cat_final_rgb.mp4"
    print(f"Saving GPU decoded video (RGB) to {path_rgb}...")
    export_to_video(video_uint8, path_rgb, fps=15)
    
    path_bgr = "output_cat_final_bgr.mp4"
    print(f"Saving GPU decoded video (BGR) to {path_bgr}...")
    video_bgr = video_uint8[..., ::-1]
    export_to_video(video_bgr, path_bgr, fps=15)
    
    print("Done!")

if __name__ == "__main__":
    export_latents()
