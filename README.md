# LLaVA-OneVision MLVU Action Ordering Evaluation

This repository contains code and comprehensive evaluation of the **LLaVA-OneVision** multimodal model for the **MLVU (Multi-Task Long Video Understanding)** test dataset, specifically focused on the **Action Ordering** task.

## Project Overview

The project evaluates LLaVA-OneVision's ability to predict the correct chronological sequence of actions in videos. The implementation includes:
- **Multiple model sizes**: 0.5B, 7B, 72B
- **Flexible frame sampling**: 8, 16, 32, 64 frames per video
- **Three prompt engineering approaches**: detailed, optimized, and minimal
- **Comprehensive result logging and analysis**: per-video predictions, accuracy metrics, and option distribution charts

For detailed technical documentation, see **[REPORT.md](REPORT.md)** which explains model loading, prompting strategy, implementation details, and results analysis.

---

## Prerequisites

- **Conda**: Install Miniconda or Anaconda for Python 3.10. See https://docs.conda.io/en/latest/miniconda.html  
- **GPU**: Required for inference
  - **0.5B model**: ~4GB VRAM
  - **7B model**: ~16-20GB VRAM
  - **72B model**: ~40-80GB VRAM
- **CUDA**: 11.8 or higher for GPU acceleration
- **Hugging Face account**: Required to download models (sign up at https://huggingface.co)

---

## Folder Structure

```markdown
Llava-onevision/
├── single_runner.py                    # Primary inference script (configurable, modular)
├── main72.py                           # Specialized 72B model inference (legacy)
├── gt_plot.py                          # Ground truth distribution visualization
├── test_llava_onevision.py            # HuggingFace pipeline test (deprecated)
├── setup_env.sh                        # Environment setup script
├── REPORT.md                           # Comprehensive project report
├── README.md                           # This file
├── LLaVA_OneVision_Tutorials.ipynb    # Official LLaVA-NeXT tutorial examples
├── complete.json                       # Full MLVU dataset (70 action ordering samples)
├── content.json                        # Subset of MLVU dataset (12 action ordering samples)
├── results/                            # Outputs from scripts (ignored by Git)
│   ├── output_*.txt                    # Per-video predictions and accuracy
│   ├── option_freq*.jpg                # Option distribution charts
│   └── gt_freq.jpg                     # Ground truth distribution
└── .gitignore                          # Ignore results, cache, env
```

---

## Environment Setup

### 1. Run the setup script

```bash
bash setup_env.sh
```

This will create a Python 3.10 conda environment with all required dependencies:
- PyTorch (with CUDA support if available)
- Hugging Face Transformers
- Accelerate
- LLaVA-NeXT
- lmms-eval
- decord (video processing)
- matplotlib (visualization)

### 2. Activate the conda environment

```bash
conda activate llava
```

### 3. Login to Hugging Face

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

Paste your **Personal Access Token (PAT)** from https://huggingface.co/settings/tokens when prompted. This is required to download the LLaVA-OneVision models.

---

## Running Inference

### Quick Start with Default Configuration

```bash
# Default: 0.5B model, 32 frames, my_prompt2 template
python single_runner.py
```

Output will be saved to `results/output_0.5b_32_my_prompt2.txt`

### Custom Configuration

```bash
# Test 7B model with 64 frames
python single_runner.py --model_size 7b --max_frames 64

# Test 72B model with custom prompt
python single_runner.py --model_size 72b --prompt_type my_prompt

# All 72B configurations
python single_runner.py --model_size 72b --max_frames 8 --prompt_type my_prompt
python single_runner.py --model_size 72b --max_frames 8 --prompt_type my_prompt2
python single_runner.py --model_size 72b --max_frames 16 --prompt_type my_prompt
python single_runner.py --model_size 72b --max_frames 32 --prompt_type my_prompt2
```

### Command-Line Arguments

```
usage: single_runner.py [-h] [--model_size {0.5b,7b,72b}] 
                        [--max_frames MAX_FRAMES]
                        [--prompt_type {my_prompt,my_prompt2,question}]

options:
  --model_size       Model size: 0.5b, 7b, or 72b (default: 0.5b)
  --max_frames       Number of frames to sample from video (default: 32)
  --prompt_type      Prompt template: my_prompt, my_prompt2, or question (default: my_prompt2)
```

---

## Output Files

Results are automatically saved to the `results/` directory:

### Text Results
- **Filename pattern**: `output_{model_size}_{frames}_{prompt_type}.txt`
- **Example**: `output_72b_32_my_prompt2.txt`
- **Content**: Per-video predictions, ground truth comparisons, overall accuracy

### Visualization
- **Filename pattern**: `option_freq{model_size}_{frames}_{prompt_type}.jpg`
- **Content**: Bar chart showing distribution of selected answer options
- **Baseline**: `gt_freq.jpg` shows ground truth option distribution

### Sample Output Format
```
Processing video: test_order_31.mp4
Predicted: baking cookies --> carving pumpkin --> riding mule --> pole vault
Ground Truth: baking cookies --> carving pumpkin --> riding mule --> pole vault
--------------------------------------------------
Processing video: test_order_36.mp4
Predicted: stomping grapes --> zumba --> carving pumpkin --> water sliding
Ground Truth: stomping grapes --> zumba --> carving pumpkin --> water sliding
--------------------------------------------------

Accuracy: XX.XX%
```

---

## Key Features

### Model Architecture
- **LLaVA-OneVision**: Unified vision-language model based on Qwen2
- **Vision Encoder**: LLaVA's vision tower for temporal video understanding
- **Language Model**: Qwen2 (0.5B, 7B, or 72B)
- **Video Support**: Processes multiple frames as a single video input

### Video Processing
- **Frame Extraction**: Uniform temporal sampling using Decord
- **Preprocessing**: Converts frames to tensors with FP16 precision for efficiency
- **Answer Matching**: Uses sequence similarity to match model output to candidates

### Prompt Templates
1. **my_prompt**: Detailed instruction with bullet-point candidates
2. **my_prompt2**: Optimized format with cleaner structure (recommended)
3. **question**: Minimal prompt with just the question

See [REPORT.md](REPORT.md) for detailed prompt specifications.

---

## Experimental Setup

The project evaluates **27+ configurations** combining:
- **3 model sizes**: 0.5B, 7B, 72B
- **4 frame counts**: 8, 16, 32, 64 frames
- **3 prompt types**: my_prompt, my_prompt2, question

### Recommended Configurations

| Use Case | Model | Frames | Prompt | VRAM |
|----------|-------|--------|--------|------|
| Edge devices | 0.5B | 16-32 | my_prompt2 | 4GB |
| Balanced | 7B | 32 | my_prompt2 | 16-20GB |
| Maximum accuracy | 72B | 32-64 | my_prompt2 | 40-80GB |

---

## Notes & Recommendations

- **VRAM Optimization**: Use `attn_implementation="sdpa"` (Scaled Dot-Product Attention) and FP16 precision to save memory
- **Model Selection**: Use 0.5B or 7B for single-GPU setups; 72B requires multi-GPU or high-VRAM hardware
- **Frame Count Trade-off**: More frames = better temporal coverage but higher compute cost. 32 frames is a good balance.
- **Prompt Engineering**: `my_prompt2` performed best in testing
- **Dataset**: The code is configured for MLVU Action Ordering task. Modify `json_path` and `dataset_path` to use different datasets

---

## Project Architecture

### `single_runner.py` (Primary Script)

**Main Components**:
1. **Argument Parser**: Flexible configuration for model, frames, and prompts
2. **Model Loading**: Loads pretrained LLaVA-OneVision from HuggingFace Hub
3. **Video Processing**: Extracts and preprocesses video frames
4. **Answer Generation**: Runs model inference on video + question + candidates
5. **Answer Matching**: Uses sequence similarity to find best candidate match
6. **Results Logging**: Saves predictions, ground truth, and accuracy metrics
7. **Visualization**: Creates option distribution charts

**Key Functions**:
- `load_video()`: Extracts uniformly-sampled frames from video
- `generate_answer()`: Runs full inference pipeline
- `optionchecker()`: Maps predicted answer to option number

---

## References

### Official Resources
- **LLaVA-NeXT GitHub**: https://github.com/LLaVA-VL/LLaVA-NeXT
- **HuggingFace Models**: https://huggingface.co/lmms-lab
- **MLVU Dataset**: https://huggingface.co/datasets/MLVU/MVLU

### Related Papers
- LLaVA Paper: https://arxiv.org/abs/2304.08485
- LLaVA-NeXT: https://llava-vl.github.io/blog/2024-04-30-llava-next-video/
- Qwen2: https://qwenlm.github.io/

### Tools & Libraries
- **Decord**: High-performance video reading https://github.com/dmlc/decord
- **PyTorch**: Deep learning framework https://pytorch.org
- **Hugging Face Transformers**: Model loading and inference https://huggingface.co/docs/transformers

---

<!-- ## FAQ

**Q: Which model should I use?**  
A: Start with 0.5B for testing, 7B for production on single GPUs, and 72B if you have high-end hardware and need maximum accuracy.

**Q: How do I change the dataset?**  
A: Modify the `json_path` and `dataset_path` variables in `single_runner.py` to point to your dataset.

**Q: Can I use this for other video understanding tasks?**  
A: Yes! The pipeline is generalizable. Modify the prompts and answer matching logic for your specific task.

**Q: Out of memory errors?**  
A: Try using a smaller model, fewer frames, or reduce `max_new_tokens` in the model.generate() call.

---

## Citation

If you use this project, please cite the original LLaVA-OneVision paper:

```bibtex
@article{llava_onevision,
  title={LLaVA-NeXT: A Strong Zero-shot Video Understanding Model for Temporally Long Videos},
  author={Liu, Haotian and Li, Chunyuan and Wu, Yuheng and Abbeel, Pieter},
  year={2024}
}
``` -->


