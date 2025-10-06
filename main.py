import os
import json
import torch
import copy
import warnings
import numpy as np
from decord import VideoReader, cpu
from difflib import SequenceMatcher
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

warnings.filterwarnings("ignore")

json_path="/DATA2/ai23mtech12002/DATASETS/MLVU/Action_order/5_order.json"
dataset_path="/DATA2/ai23mtech12002/DATASETS/MLVU/Action_order/video/5_order"

with open(json_path, "r") as f:
    dataset = json.load(f)
    
pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device_map = "auto"
device = "cuda"
llava_model_args = {"multimodal": True} 

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args
)
model.eval()

def load_video(video_path, max_frames_num=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frames = vr.get_batch(uniform_sampled_frames.tolist()).asnumpy()
    return frames  # (frames, height, width, channels)

def generate_answer(video_path, question, candidates, max_frames=16):
    video_frames = load_video(video_path, max_frames)
    frames_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors = [frames_tensor]

    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.shape[:2] for frame in video_frames]

    # Generate raw output
    outputs = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
        modalities=["video"]
    )
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    best_candidate = max(candidates, key=lambda c: similarity(text_output, c))
    return best_candidate

correct=0
for item in dataset:
    video_file = item["video"]
    video_path = os.path.join(dataset_path, video_file)
    print(f"Processing video: {video_file}")

    predicted_answer = generate_answer(video_path, item["question"], item["candidates"])
    print(f"Predicted: {predicted_answer}")
    print(f"Ground Truth: {item['answer']}\n{'-'*50}")

    if predicted_answer == item["answer"]:
        correct += 1
        
        
accuracy = correct / len(dataset)
print(f"Accuracy: {accuracy*100:.2f}%")

