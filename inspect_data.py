
import torch
import os

def inspect_data():
    print("--- Inspecting Prompt Embeddings ---")
    if os.path.exists("prompt_embeds.pt"):
        data = torch.load("prompt_embeds.pt", map_location="cpu")
        pe = data["prompt_embeds"]
        ne = data["neg_embeds"]
        print(f"Prompt Embeds Shape: {pe.shape}")
        print(f"Prompt Embeds Mean: {pe.mean()}, Std: {pe.std()}")
        print(f"Prompt Embeds Min: {pe.min()}, Max: {pe.max()}")
        if pe.std() < 0.001:
            print("WARNING: Prompt embeddings seem flat/empty!")
        
        print(f"Neg Embeds Shape: {ne.shape}")
        print(f"Neg Embeds Mean: {ne.mean()}, Std: {ne.std()}")
    else:
        print("prompt_embeds.pt NOT FOUND.")

    print("\n--- Inspecting Saved Latents ---")
    if os.path.exists("latents_checkpoint.pt"):
        latents = torch.load("latents_checkpoint.pt", map_location="cpu")
        print(f"Latents Shape: {latents.shape}")
        print(f"Latents Mean: {latents.mean()}, Std: {latents.std()}")
        print(f"Latents Min: {latents.min()}, Max: {latents.max()}")
        # Check if they look like standard Gaussian (before potential double norm)
        # or if they are shifted.
    else:
        print("latents_checkpoint.pt NOT FOUND.")

if __name__ == "__main__":
    inspect_data()
