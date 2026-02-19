# Wan2.1 Dual-GPU Video Generator

This project implements a high-performance video generation pipeline using the **Wan2.1-T2V-1.3B** model from Wan-AI. It is optimized for systems with **Dual GPUs**, leveraging distributed inference to handle large model components efficiently.

## Features

-   **Dual-GPU Support**: Automatically distributes the VAE (Visual Autoencoder) to `cuda:0` and the Transformer to `cuda:1` to maximize memory efficiency.
-   **BF16 Precision**: Utilizes `bfloat16` precision for the Transformer to ensure stability and reduce VRAM usage without compromising quality.
-   **CPU Offloading**: Aggressively offloads the massive **UMT5 Text Encoder (~22GB)** to the CPU, performing embedding generation "fresh" on the CPU to save precious GPU memory.
-   **Manual VAE Decoding**: Implements a custom VAE decoding loop in `float32` to prevent color artifacts (green/white screens) often seen with lower precision decoding.

## Requirements

To run this pipeline successfully, your system should meet the following specifications:

-   **OS**: Windows or Linux
-   **Python**: 3.10+
-   **GPU**: 
    -   Minimum: 1x NVIDIA GPU with 24GB+ VRAM (Single GPU mode)
    -   Recommended: 2x NVIDIA GPUs (e.g., RTX 3090/4090 or A100s) for Dual-GPU mode.
-   **System RAM**: **32GB+ (64GB Recommended)**. The text encoder requires significant system RAM when offloaded.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/[REPO_NAME].git
    cd [REPO_NAME]
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate # Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download the Model:**
    Ensure you have the `Wan-AI/Wan2.1-T2V-1.3B` model downloaded locally or accessible via Hugging Face Hub. Update the `model_id` path in `run_wan.py` if necessary.

2.  **Run the Generator:**
    ```bash
    python run_wan.py
    ```

    The script will automatically detect your GPU configuration and start generating the video.

## Output

The generated video will be saved as `output_wan.mp4` in the project directory. An additional debug version `output_wan_bgr.mp4` is also saved to help verify color channel correctness.

## Troubleshooting

-   **OOM Errors**: If you encounter Out-of-Memory errors, ensure you have enough System RAM (not just VRAM) for the text encoder offloading.
-   **Slow Generation**: Initial text encoding on the CPU can be slow; this is a trade-off for reducing GPU VRAM usage.
