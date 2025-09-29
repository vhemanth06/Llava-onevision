# LLaVA-OneVision Assignment

This repository contains code and setup for testing the **LLaVA-OneVision** model for multi-modal image/video-to-text tasks.  
The goal is to replicate the model’s inference results and save outputs for analysis.

---

## Prerequisites

- **Conda**: Install Miniconda or Anaconda for Python 3.10. See https://docs.conda.io/en/latest/miniconda.html  
- **GPU**: Recommended for 7B or higher models; CUDA installed if using GPU acceleration  
- **Hugging Face account**: Required to download models via Hugging Face Hub  

---

## Folder Structure
```markdown
llava_onevision_assignment/
├── llava_onevision/            # All LLaVA scripts
│   ├── test_llava_onevision.py   # Inference script (image/video)
│   ├── video_test_llava.py       # Run inference on a video frame
│   ├── setup_env.sh              # Environment setup script
│   └── other_scripts.py          # Any other helper scripts
├── results/                     # Outputs from scripts (ignored by Git)
│   ├── output.json
│   └── video_frame_output.json
├── .gitignore                   # Ignore results, cache, env
└── README.md                    # This file
```
---

## Environment Setup

**Python 3.10, PyTorch, Hugging Face Transformers**

1. Run the setup script:
   ```bash
   bash setup_env.sh
   ```
2. Activate the conda environment:
   ```bash
   conda activate llava
   ```

3. Log in to Hugging Face via Bash:
   ```bash
   pip install --upgrade huggingface_hub
   huggingface-cli login
   ```

- Paste your **Personal Access Token (PAT)** from https://huggingface.co/settings/tokens when prompted.

**Notes:**  
`setup_env.sh` will create a Python 3.10 environment, install PyTorch (with CUDA support if available), Hugging Face Transformers, Accelerate, LLaVA-NeXT, and lmms-eval.

---

## Running Inference on Video Frames

1. Place a video file in the repo folder or provide a path.  
2. Run the script:
```bash
python3 video_test_llava.py
```
3. The script will:  

- Extract a specific frame from the video  
- Run LLaVA-OneVision inference  
- Save output in results/video_frame_output.json

**Tip:** Modify `frame_number` in the script to test different frames.

---

## Notes & Recommendations

- Use **0.5B or 7B** LLaVA-OneVision models for single-GPU setups.  
- Use `device_map="auto"` and `torch_dtype=torch.float16` to save VRAM.  
- Full training reproduction requires large GPU resources.  
- Save outputs in results/ to keep the repo clean.  

---

## References

- LLaVA-NeXT GitHub: https://github.com/LLaVA-VL/LLaVA-NeXT  
- Hugging Face LLaVA-OneVision: https://huggingface.co/llava-hf  
- LMMS Eval GitHub: https://github.com/EvolvingLMMs-Lab/lmms-eval  
- LLaVA Paper on arXiv: https://arxiv.org/abs/2308.09492  

---

**Author:** Vuppula Hemanth Reddy  
**Date:** 2025-09-29

---

## Quick Start (Optional)

For a turnkey setup on a cloud GPU:

# Clone repo
```bash
git clone https://github.com/vhemanth06/Llava-onevision.git
cd Llava-onevision
```

# Setup environment
```bash
bash setup_env.sh
conda activate llava
```

# Login to Hugging Face
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

# Run a test video frame inference
```bash
python3 video_test_llava.py
```
