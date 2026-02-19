
import torch
import numpy as np
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
import os

def debug_colors():
    print("Loading VAE...")
    model_id = "e:/Projects/textModel/Wan2.1-T2V-1.3B"
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    vae.to("cuda")
    vae.enable_slicing()
    vae.enable_tiling()

    print("Loading latents from checkpoint...")
    if not os.path.exists("latents_checkpoint.pt"):
        print("Error: latents_checkpoint.pt not found!")
        return
        
    latents = torch.load("latents_checkpoint.pt", map_location="cuda").to(dtype=torch.float32)
    
    # Manual Un-normalization (Same as before)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    latents = latents / latents_std + latents_mean

    print("Decoding latents...")
    with torch.no_grad():
        video_tensor = vae.decode(latents, return_dict=False)[0]
    
    # Un-normalize [0, 1]
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    
    # Convert to standard format [F, H, W, C]
    # shape: [1, 3, 81, 480, 832] -> [81, 480, 832, 3]
    video_np = video_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
    video_uint8 = (video_np * 255).round().astype(np.uint8)

    # 1. Save as is (RGB assumption) - likely what we just did
    print("Saving output_rgb_test.mp4...")
    export_to_video(video_uint8, "output_rgb_test.mp4", fps=15)

    # 2. Save with BGR flip (assuming model output was BGR, but we treated as RGB, or vice versa)
    # If video is blueish, maybe Red and Blue are swapped.
    print("Saving output_bgr_test.mp4...")
    video_bgr = video_uint8[..., ::-1] # Flip channels
    export_to_video(video_bgr, "output_bgr_test.mp4", fps=15)

    print("Done! Check both files.")

if __name__ == "__main__":
    debug_colors()
