#!/bin/bash

# 1) Create conda environment
conda create -n llava python=3.10 -y
conda activate llava

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3.5) Install transformers version 4.40.0 only, since it's the only one being supported
pip show transformers
pip uninstall -y transformers && conda remove -y transformers
pip install git+https://github.com/huggingface/transformers.git@v4.40.0

# 4) Dependent packages to be used
pip install einops open_clip_torch decord opencv-python

# 4) Install LLaVA / Hugging Face packages
pip install accelerate huggingface_hub

# 5) Install Llava-Next for acessing the models
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .
cd ..
