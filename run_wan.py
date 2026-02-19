
import torch
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, AutoModel
import os
import sys
import numpy as np
import gc

def generate_video(prompt, output_path="output_wan.mp4", num_frames=81):
    # Redirect stdout/stderr to file for logging
    sys.stdout = open("output.log", "w", encoding="utf-8")
    sys.stderr = sys.stdout

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return

    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    
    # Path to local model
    model_id = "e:/Projects/textModel/Wan2.1-T2V-1.3B"
    
    # --- Step 1: Load Models ---
    print("\n[1/4] Loading VAE (Float32)...")
    # Load VAE in Float32 to avoid artifacts
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    vae.enable_slicing()
    vae.enable_tiling()
    
    # Check for Dual GPU
    device_count = torch.cuda.device_count()
    print(f"GPUs detected: {device_count}")
    
    if device_count > 1:
        print("Dual-GPU Mode Detected! Distributing models...")
        # VAE on GPU 0
        vae.to("cuda:0")
        transformer_device = "cuda:1" # Transformer on GPU 1
    else:
        print("Single GPU Mode.")
        vae.to("cuda")
        transformer_device = "cuda"

    print(f"[2/4] Loading Transformer (BF16) on {transformer_device}...")
    # Use BF16 for Transformer stability
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(transformer_device)

    # --- Step 2: Generate Embeddings (Fresh on CPU) ---
    print("\n[2/4] Generating Embeddings (Fresh on CPU)...")
    
    # Load Text Encoder on CPU to save VRAM
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(
        model_id, 
        subfolder="text_encoder", 
        torch_dtype=torch.bfloat16
    ).to("cpu")
    
    print("Encoding prompt...")
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        prompt_embeds = text_encoder.encoder(
            text_inputs.input_ids.to("cpu"),
            attention_mask=text_inputs.attention_mask.to("cpu"),
        )[0]
    
    print("Encoding negative prompt...")
    negative_prompt = "low quality, bad video, distorted, blurred, pixelated"
    neg_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        neg_embeds = text_encoder.encoder(
            neg_inputs.input_ids.to("cpu"),
            attention_mask=neg_inputs.attention_mask.to("cpu"),
        )[0]
        
    print("Embeddings generated.")
    
    # Cleanup Text Encoder
    del text_encoder
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Move embeddings to Transformer device
    prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16, device=transformer_device)
    neg_embeds = neg_embeds.to(dtype=torch.bfloat16, device=transformer_device)

    # --- Step 3: Build Pipeline ---
    print("\n[3/4] Building pipeline...")
    pipe = WanPipeline.from_pretrained(
        model_id,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    
    print(f"Pipeline ready. Transformer on {transformer.device}, VAE on {vae.device}")
    
    # --- Step 4: Generate Latents ---
    print(f"\n[4/4] Generating Latents for prompt: '{prompt}'")
    
    # Generate latents only
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        num_frames=num_frames,
        height=480,
        width=832,
        guidance_scale=5.0,
        output_type="latent"
    )
    
    latents = output.frames
    print(f"Latents shape: {latents.shape}")

    # --- Step 5: Manual VAE Decoding (Float32) ---
    print("\n[5/5] Manually decoding with VAE (Float32)...")
    
    # Clean up pipeline to free VRAM for decoding
    del pipe
    del transformer
    torch.cuda.empty_cache()
    
    # Move latents to VAE device
    latents = latents.to(device=vae.device, dtype=torch.float32)
    
    # Un-normalize latents (Wan specific)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    latents = latents / latents_std + latents_mean
    
    # Decode
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]
    
    # Post-process video
    # Un-normalize from [-1, 1] to [0, 1]
    video = (video / 2 + 0.5).clamp(0, 1)
    
    # Convert to Uint8 numpy
    video_uint8 = (video.cpu().float().numpy() * 255).round().astype(np.uint8)
    
    # Remove batch dimension if present
    if video_uint8.ndim == 5:
        video_uint8 = video_uint8[0]
        
    # Permute [C, T, H, W] -> [T, H, W, C]
    video_uint8 = np.transpose(video_uint8, (1, 2, 3, 0))
    print(f"Final video shape: {video_uint8.shape}")
    
    # Save RGB Version (Standard)
    print(f"\nSaving RGB video to {output_path}")
    export_to_video(video_uint8, output_path, fps=15)
    
    # Save BGR Version (Alternative) - helpful for debugging color issues
    output_bgr = output_path.replace(".mp4", "_bgr.mp4")
    print(f"Saving BGR video to {output_bgr}")
    video_bgr = video_uint8[..., ::-1] # Flip RGB to BGR
    export_to_video(video_bgr, output_bgr, fps=15)
    
    print("Done!")

if __name__ == "__main__":
    prompt = "A cute cartoon cat playing, kids video style, vivid colors"
    # 33 frames is approx 2 seconds at 15fps, optimized for VAE tiling
    generate_video(prompt, num_frames=33)
