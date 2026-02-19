
import torch
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from diffusers.utils import export_to_video
import os
import sys
import numpy as np

def generate_video(prompt, output_path="output_wan.mp4", num_frames=81):
    sys.stdout = open("output_debug.log", "w", encoding="utf-8")
    sys.stderr = sys.stdout

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return

    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    
    model_id = "e:/Projects/textModel/Wan2.1-T2V-1.3B"
    
    # --- Step 1: Load Models ---
    print("\n[1/4] Loading VAE (Float32)...")
    # VAE on GPU 0 (or CPU offload)
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.float32
    )
    vae.enable_slicing()
    vae.enable_tiling()
    
    # Check for 2 GPUs
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
    # Using BF16 for better precision stability than FP16
    transformer = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(transformer_device)

    # --- Step 2: Generate Embeddings (Fresh) ---
    print("\n[2/4] Generating Embeddings (Fresh on CPU)...")
    
    # Check for cache file to compare or delete?
    # No, let's regenerate to be safe.
    
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(
        model_id, 
        subfolder="text_encoder", 
        torch_dtype=torch.bfloat16
    ).to("cpu") # Keep on CPU
    
    print("Encoding prompt...")
    # Tokenize
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    
    # Encode (No Grad)
    with torch.no_grad():
        # UMT5 is an encoder-decoder. We only need the encoder output.
        prompt_embeds = text_encoder.encoder(
            text_inputs.input_ids.to("cpu"),
            attention_mask=text_inputs.attention_mask.to("cpu"),
        )[0]
    
    print("Encoding negative prompt...")
    negative_prompt = "low quality, bad video, distorted"
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
    
    # Save cache for inspection
    torch.save({"prompt_embeds": prompt_embeds, "neg_embeds": neg_embeds}, "prompt_embeds_fresh.pt")
    
    # Free memory
    del text_encoder
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Move to Transformer Device
    prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16).to(transformer_device)
    neg_embeds = neg_embeds.to(dtype=torch.bfloat16).to(transformer_device)

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
    
    # Pipeline .to() might mess up manual placement, so skip it or be careful.
    # We already placed models.
    # But pipe needs to know where things are.
    # Actually, if we pass pre-loaded models, pipe.to("cuda") typically moves everything to "cuda:0".
    # So we should NOT call pipe.to("cuda") if we want split GPUs.
    # Instead, we rely on the models being on correct devices.
    
    print(f"Pipeline ready. Transformer on {transformer.device}, VAE on {vae.device}")
    
    # Ensure Embeddings on the SAME DEVICE as the Transformer
    print(f"Moving embeddings to {transformer.device}...")
    prompt_embeds = prompt_embeds.to(transformer.device)
    neg_embeds = neg_embeds.to(transformer.device)

    # --- Step 4: Generate Latents ---
    print(f"\n[4/4] Generating Latents for prompt: '{prompt}'")
    
    # Only generate latents first (output_type="latent")
    # This skips the VAE decode step in the pipeline
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
    print(f"Latents shape: {latents.shape}, dtype: {latents.dtype}")
    
    # Inspect Latents
    if torch.isnan(latents).any():
        print("!!! ERROR: Latents contain NaNs !!!")
    else:
        print("Latents look clean (no NaNs).")
        
    print(f"Latents Min: {latents.min()}, Max: {latents.max()}")

    # --- Step 5: Manual VAE Decoding (Strict Float32) ---
    print("\n[5/5] Manually decoding with VAE (Float32)...")
    
    # Release pipeline memory to make room for VAE decode
    del pipe
    del transformer
    torch.cuda.empty_cache()
    
    # Move VAE to its device (should be already there, but safe check)
    # If dual GPU, vae is on cuda:0. If single, cuda.
    
    # Move Latents to VAE device
    latents = latents.to(device=vae.device, dtype=torch.float32)
    
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latents.device, torch.float32)
    
    latents = latents / latents_std + latents_mean
    
    # Decode
    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0] 
        # video shape: [1, 3, 81, 480, 832]
    
    # Save Latents for safety
    torch.save(latents.cpu(), "latents_checkpoint.pt")
    print("Latents saved to latents_checkpoint.pt")

    # Post-process (VideoProcessor logic simplified)
    # Un-normalize from [-1, 1] to [0, 1]
    video = (video / 2 + 0.5).clamp(0, 1)
    
    # Fix save bug: Convert to CPU numpy before astype
    video_uint8 = (video.cpu().float().numpy() * 255).round().astype(np.uint8)
    
    # (Optional) Permute to [T, H, W, C] if needed for export_to_video?
    # video output from VAE is [B, C, T, H, W] -> [1, 3, 33, 480, 832]
    # export_to_video usually expects [T, H, W, C] (numpy) or [C, T, H, W]?
    # diffusers.utils.export_to_video expects List[PIL] or np.array [T, H, W, C]
    
    # Check shape
    print(f"Video shape before permute: {video_uint8.shape}")
    if video_uint8.ndim == 5:
        video_uint8 = video_uint8[0] # remove batch
    # video_uint8 is [C, T, H, W] -> need [T, H, W, C]
    video_uint8 = np.transpose(video_uint8, (1, 2, 3, 0))
    print(f"Video shape after permute: {video_uint8.shape}")
    
    # Save RGB Version
    path_rgb = "output_cat_rgb.mp4"
    print(f"\nSaving RGB video to {path_rgb}")
    export_to_video(video_uint8, path_rgb, fps=15)
    
    # Save BGR Version (Fix for Blue/Brown swap)
    path_bgr = "output_cat_bgr.mp4"
    print(f"Saving BGR video to {path_bgr}")
    video_bgr = video_uint8[..., ::-1]
    export_to_video(video_bgr, path_bgr, fps=15)
    
    print("Done! Generated both RGB and BGR versions.")

if __name__ == "__main__":
    # User Request: 2s video, "kids video with cat", not 4k
    # 2s at ~15fps = 30 frames. VAE needs (f-1)%4==0. 33 frames is close (33-1=32, 32/4=8).
    prompt = "A cute cartoon cat playing, kids video style, vivid colors"
    generate_video(prompt, num_frames=33)
