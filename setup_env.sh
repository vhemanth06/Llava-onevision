#!/bin/bash
# 1) Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4) Install LLaVA / Hugging Face packages
pip install -U "transformers>=4.40.0" accelerate huggingface_hub
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
