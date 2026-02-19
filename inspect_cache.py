
import torch

try:
    data = torch.load("e:/Projects/textModel/prompt_embeds.pt", map_location="cpu")
    print("Keys:", data.keys())
    for k, v in data.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}, mean={v.mean()}")
        if torch.isnan(v).any():
            print(f"!!! {k} contains NaNs !!!")
        if (v == 0).all():
            print(f"!!! {k} is all zeros !!!")
except Exception as e:
    print(f"Error loading cache: {e}")
