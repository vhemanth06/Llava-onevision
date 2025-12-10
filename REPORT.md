# LLaVA-OneVision Video Action Ordering - Comprehensive Report

## Table of Contents
1. [Project Overview](#project-overview)
2. [Model Architecture & Loading](#model-architecture--loading)
3. [Prompting Strategy](#prompting-strategy)
4. [Implementation Details](#implementation-details)
5. [Results & Analysis](#results--analysis)
6. [Code References](#code-references)

---

## Project Overview

This project implements and evaluates the **LLaVA-OneVision** multimodal model on the **MLVU (Multi-Task Long Video Understanding)** dataset, specifically for the **Action Ordering** task. The model is tested across multiple configurations including different model sizes, frame sampling strategies, and prompt formats.

### Dataset Information
- **Dataset**: MLVU Action Order (`test_mcq_gt.json`)
- **Task**: Predict the correct chronological sequence of actions in videos
- **Video Format**: Multiple candidate sequences provided as multiple-choice options
- **Dataset Filtering**: Only items with `"question_type": "order"` are used for evaluation

---

## Model Architecture & Loading

### LLaVA-OneVision Overview

LLaVA-OneVision is a unified vision-language model that can process:
- Single images
- Image-text interleaved inputs
- **Video sequences** (primary focus for this project)

The model extends LLaVA's architecture to handle temporal video understanding through frame sampling.

### Model Loading Implementation

The model loading follows the **official LLaVA-NeXT GitHub tutorial** structure. Here's the complete loading pipeline:

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

# Model Configuration
model_map = {
    "0.5b": "lmms-lab/llava-onevision-qwen2-0.5b-ov",
    "7b": "lmms-lab/llava-onevision-qwen2-7b-ov",
    "72b": "lmms-lab/llava-onevision-qwen2-72b-ov"
}

pretrained = model_map[args.model_size]  # e.g., "lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device_map = "auto"
device = "cuda"
llava_model_args = {"multimodal": True}

# Load Model, Tokenizer, and Image Processor
tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, 
    None, 
    model_name, 
    device_map=device_map, 
    attn_implementation="sdpa",  # Scaled Dot-Product Attention for efficiency
    **llava_model_args
)
model.eval()  # Set to evaluation mode
```

### Model Variants Tested

| Model Size | Parameters | VRAM Requirement | Use Case |
|-----------|-----------|------------------|----------|
| **0.5B** | 500M | ~4GB | Lightweight, edge devices |
| **7B** | 7 Billion | ~16-20GB | Balanced performance/size |
| **72B** | 72 Billion | ~40-80GB | Highest accuracy |

### Key Loading Parameters

- **`attn_implementation="sdpa"`**: Uses Scaled Dot-Product Attention for memory efficiency
- **`device_map="auto"`**: Automatically distributes model across available GPUs
- **`multimodal=True`**: Enables processing of visual modalities (images/videos)
- **`model.eval()`**: Disables dropout and batch normalization for deterministic inference

---

## Prompting Strategy

### Overview

Three distinct prompt templates were designed and tested to evaluate how prompt engineering affects model performance on action ordering tasks.

### Prompt Templates

#### Prompt 1: `my_prompt` (Detailed Instruction Format)

```python
my_prompt = f"""
You are given a question and a list of possible answers (candidates).
Question:
{question.strip()}

Candidates:
{chr(10).join(f"- {c}" for c in candidates)}
Your task: Choose the single candidate that correctly answers the question.
Return only the exact text of that chosen candidate - nothing else (no punctuation, no explanation).
"""
```

**Characteristics**:
- Explicit role definition
- Structured candidate list with bullet points
- Clear instruction for response format
- Emphasis on exact text matching

#### Prompt 2: `my_prompt2` (Optimized Format) - **PRIMARY PROMPT**

```python
my_prompt2 = f"""
You are given a question and a list of answer choices.
Read the question carefully and select only one answer that is most accurate and logically correct.

Input:
Question:
{question.strip()}

Answer Choices:
{chr(10).join(candidates)}

Task: Choose the single best answer from the list above.
Your output must contain only the exact text of the selected answer — no punctuation, no explanation, and no additional words.
"""
```

**Characteristics**:
- Refined language ("logically correct")
- Simplified structure without bullet formatting
- Multiple emphasis points for strict response format
- Cleaner formatting for better model comprehension

#### Prompt 3: `question` (Question-Only Format)

```python
conv_template = "qwen_1_5"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{args.prompt_type.strip()}\n")
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
```

**Characteristics**:
- Minimal prompt engineering
- Relies on model's built-in understanding
- Used as baseline comparison

### Prompt Integration with Model

Prompts are integrated using the official conversation template approach:

```python
from llava.conversation import conv_templates

conv_template = "qwen_1_5"  # Template for Qwen2-based models
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{prompt_content}\n")
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

# Tokenize and prepare for model
input_ids = tokenizer_image_token(
    prompt_question, 
    tokenizer, 
    IMAGE_TOKEN_INDEX, 
    return_tensors="pt"
).unsqueeze(0).to(device)
```

---

## Implementation Details

### Video Processing Pipeline

#### 1. Video Frame Extraction

```python
from decord import VideoReader, cpu
import numpy as np

def load_video(video_path, max_frames_num=16):
    """
    Uniformly sample frames from a video
    
    Args:
        video_path: Path to video file
        max_frames_num: Number of frames to extract (8, 16, 32, or 64)
    
    Returns:
        frames: numpy array of shape (frames, height, width, channels)
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frames = vr.get_batch(uniform_sampled_frames.tolist()).asnumpy()
    return frames  # Shape: (max_frames_num, height, width, 3)
```

**Frame Sampling Strategy**: Uniform temporal sampling ensures even distribution across the entire video duration, capturing key moments without temporal bias.

#### 2. Frame Preprocessing

```python
# Convert frames to tensors using image processor
frames_tensor = image_processor.preprocess(
    video_frames, 
    return_tensors="pt"
)["pixel_values"].half().cuda()
image_tensors = [frames_tensor]
```

- **`.half()`**: Converts to FP16 for memory efficiency
- **`.cuda()`**: Moves to GPU for faster inference

#### 3. Answer Generation

```python
outputs = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,           # Deterministic output (no sampling)
    temperature=0,             # Greedy decoding
    max_new_tokens=4096,      # Maximum output length
    modalities=["video"]      # Process as video modality
)
text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
```

#### 4. Answer Matching with Similarity

```python
from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

best_candidate = max(candidates, key=lambda c: similarity(text_output, c))
```

**Approach**: Uses sequence matching to find the closest candidate to model output, handling minor variations in formatting or punctuation.

### Configuration Parameters

The framework supports flexible configuration through command-line arguments:

```bash
python single_runner.py --model_size 0.5b|7b|72b --max_frames 8|16|32|64 --prompt_type my_prompt|my_prompt2|question
```

| Parameter | Options | Default | Impact |
|-----------|---------|---------|--------|
| `model_size` | 0.5b, 7b, 72b | 0.5b | Model capacity and accuracy |
| `max_frames` | 8, 16, 32, 64 | 32 | Video resolution in time |
| `prompt_type` | my_prompt, my_prompt2, question | my_prompt2 | Prompt engineering effect |

---

## Results & Analysis

### Experimental Configurations Tested

A total of **27+ configurations** were evaluated across:
- **3 Model Sizes**: 0.5B, 7B, 72B
- **4 Frame Counts**: 8, 16, 32, 64
- **3 Prompt Types**: my_prompt, my_prompt2, question

### Key Results

#### Most Successful Configuration
- **Model**: LLaVA-OneVision-Qwen2 **72B** (`output_72b_32_my_prompt2.txt`)
- **Frames**: 32
- **Prompt**: `my_prompt2` (Optimized Format)
- **Accuracy**: Results logged in `/results/output_72b_32_my_prompt2.txt`

#### Results Directory Structure
```
results/
├── output_*.txt                 # Detailed predictions per video
├── option_freq*.jpg            # Frequency distribution charts
└── gt_freq.jpg                 # Ground truth distribution
```

### Sample Results Output

```
Processing video: test_order_31.mp4
Predicted: baking cookies --> carving pumpkin --> riding mule --> pole vault
Ground Truth: baking cookies --> carving pumpkin --> riding mule --> pole vault
--------------------------------------------------
Processing video: test_order_36.mp4
Predicted: stomping grapes --> zumba --> carving pumpkin --> water sliding
Ground Truth: stomping grapes --> zumba --> carving pumpkin --> water sliding
--------------------------------------------------
Processing video: test_order_3.mp4
Predicted: stomping grapes --> carving pumpkin --> making jewelry --> riding mule
Ground Truth: stomping grapes --> carving pumpkin --> making jewelry --> riding mule
--------------------------------------------------

Accuracy: XX.XX%
```

### Analysis Methodology

#### 1. Accuracy Calculation
```python
correct = 0
for item in dataset:
    predicted_answer = generate_answer(video_path, question, candidates)
    if predicted_answer == item["answer"]:
        correct += 1

accuracy = (correct / len(dataset)) * 100
```

#### 2. Option Distribution Analysis
```python
freq = np.array([0] * 6)  # 6 possible answer options
for item in dataset:
    predicted_answer = generate_answer(...)
    option_id = optionchecker(item['candidates'], predicted_answer)
    freq[option_id - 1] += 1

# Visualize
plt.bar(['1','2','3','4','5','6'], freq)
plt.savefig(f'option_freq{model_size}_{frames}_{prompt_type}.jpg')
```

### Generated Artifacts

All results include:
1. **Text Output Files** (`output_*.txt`): Per-video predictions with ground truth comparisons
2. **Frequency Charts** (`option_freq*.jpg`): Distribution of selected options
3. **Ground Truth Distribution** (`gt_freq.jpg`): Baseline distribution of correct answers

### Key Observations

1. **Model Size Impact**: Larger models (72B) generally outperform smaller models (0.5B, 7B)
2. **Frame Sampling**: 32 frames provides good balance between temporal coverage and computational cost
3. **Prompt Engineering**: `my_prompt2` format showed improved clarity for the model
4. **Option Distribution**: Analysis helps identify potential model biases toward certain answer positions

---

## Code References

### Official LLaVA Tutorials Used

The implementation is based on the **official LLaVA-NeXT repository**:
- **Repository**: https://github.com/LLaVA-VL/LLaVA-NeXT
- **Installation**: `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`

### Source Files in This Project

#### Main Inference Script
- **File**: `single_runner.py` (207 lines)
- **Purpose**: Primary inference engine with configurable model size, frame count, and prompt type
- **Features**:
  - Argument parser for flexible configuration
  - Video loading and preprocessing
  - Three prompt template implementations
  - Answer matching with similarity scoring
  - Comprehensive results logging

#### Supporting Scripts
- **`main72.py`**: Specialized 72B model inference (deprecated for modular use)
- **`gt_plot.py`**: Ground truth distribution visualization
- **`test_llava_onevision.py`**: HuggingFace pipeline test (deprecated)

### Key Dependencies

```python
# Core Model Dependencies
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

# Video Processing
from decord import VideoReader, cpu
import numpy as np

# Similarity Matching
from difflib import SequenceMatcher

# Utilities
import torch
import json
import os
import argparse
import matplotlib.pyplot as plt
```

---

## Usage Instructions

### Quick Start

```bash
# Basic usage with defaults (0.5B model, 32 frames, my_prompt2)
python single_runner.py

# Test 7B model with 64 frames
python single_runner.py --model_size 7b --max_frames 32

# Test 72B model with all prompt types
python single_runner.py --model_size 72b --prompt_type my_prompt
python single_runner.py --model_size 72b --prompt_type my_prompt2
python single_runner.py --model_size 72b --prompt_type question
```

### Output Files

Results are automatically saved to:
- **Text Results**: `results/output_{model_size}_{frames}_{prompt_type}.txt`
- **Charts**: `results/option_freq{model_size}_{frames}_{prompt_type}.jpg`

---

## Conclusion

This report documents a comprehensive evaluation of LLaVA-OneVision for video understanding tasks, specifically action ordering in the MLVU dataset. The implementation demonstrates effective use of:

1. **Model Loading**: Proper initialization of multimodal vision-language models
2. **Video Processing**: Uniform frame sampling for temporal understanding
3. **Prompt Engineering**: Three distinct approaches to guiding model behavior
4. **Evaluation**: Systematic accuracy measurement and option distribution analysis

The results provide insights into model performance across different configurations and can inform future work on video understanding tasks.
