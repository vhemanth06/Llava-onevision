<!-- # Replication og Llava-onevision results

--- 
sources: [official blog](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)

HF : [search](https://huggingface.co/models?search=Llava-onevision), 
out of the available options, tryt use qwen2, 7B param , suffix "-si" $\larr$ i presume -->
# LLaVA-OneVision Assignment

  This repository contains code and setup for testing the LLaVA-OneVision model
  for multi-modal image/video-to-text tasks. The goal is to replicate the model’s
  inference results and save outputs for analysis.

### folder_structure:
  - llava_onevision/:
      description: "All LLaVA scripts"
      files:
        - test_llava_onevision.py: "Inference script (image/video)"
        - video_test_llava.py: "Run inference on a video frame"
        - setup_env.sh: "Environment setup script"
        - other_scripts.py: "Any other helper scripts"
  - results/:
      description: "Outputs from scripts (ignored by Git)"
      files:
        - output.json
        - video_frame_output.json
  - .gitignore: "Ignore results, cache, env"
  - README.md: "This file"

### environment_setup:
  description: "Python 3.10, PyTorch, Hugging Face Transformers"
  steps:
    - bash setup_env.sh
    - conda activate llava
  notes: >
    setup_env.sh will create a Python 3.10 environment, install PyTorch (with CUDA support
    if available), install Hugging Face Transformers, Accelerate, LLaVA-NeXT, and lmms-eval.

### video_inference:
  steps:
    - Place a video file in the repo folder or provide a path.
    - Run the script: python video_test_llava.py
    - The script will:
        - Extract a specific frame from the video
        - Run LLaVA-OneVision inference
        - Save output in results/video_frame_output.json
  tip: "Modify frame_number in the script to test different frames"

### notes_recommendations:
  - Use 0.5B or 7B LLaVA-OneVision models for single-GPU setups.
  - Use device_map="auto" and torch_dtype=torch.float16 to save VRAM.
  - Full training reproduction requires large GPU resources.
  - Save outputs in results/ to keep the repo clean.

### references:
  - LLaVA-NeXT GitHub: "https://github.com/LLaVA-VL/LLaVA-NeXT"
  - Hugging Face LLaVA-OneVision: "https://huggingface.co/llava-hf"
  - LMMS Eval GitHub: "https://github.com/EvolvingLMMs-Lab/lmms-eval"
  - LLaVA Paper on arXiv: "https://arxiv.org/abs/2308.09492"

author: "Vuppula Hemanth Reddy"
date: 2025

