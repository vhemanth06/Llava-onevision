import os
import json
import torch
import copy
import warnings
import numpy as np
import argparse
from decord import VideoReader, cpu
from difflib import SequenceMatcher
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

###########################
# Standard Argument Parser
###########################
parser = argparse.ArgumentParser(description="Running with different configurations")

parser.add_argument(
    "--model_size",
    type=str,
    default="0.5b",
    choices=["0.5b", "7b", "72b"],
    help="Choose the model size: 0.5b, 7b, or 72b"
)

parser.add_argument(
    "--max_frames",
    type=int,
    default=32,
    help="Number of frames to sample from each video (e.g., 8, 16, 32, 64)"
)

parser.add_argument(
    "--prompt_type",
    type=str,
    default="my_prompt2",
    choices=["my_prompt", "my_prompt2", "question"],
    help="Choose which prompt to use: my_prompt, my_prompt2, question"
)

args = parser.parse_args()


##########################
# model and dataset paths
##########################
json_path = "/DATA2/ai23mtech12002/DATASETS/MLVU/MLVU_Test/test-ground-truth/test_mcq_gt.json"
dataset_path = "/DATA2/ai23mtech12002/DATASETS/MLVU/MLVU_Test/MLVU_Test/Action_order"

# Map model size 
model_map = {
    "0.5b": "lmms-lab/llava-onevision-qwen2-0.5b-ov",
    "7b": "lmms-lab/llava-onevision-qwen2-7b-ov",
    "72b": "lmms-lab/llava-onevision-qwen2-72b-ov"
}

pretrained = model_map[args.model_size]
model_name = "llava_qwen"
device_map = "auto"
device = "cuda"
llava_model_args = {"multimodal": True}

with open(json_path, "r") as f:
    dataset = json.load(f)
dataset = [item for item in dataset if item.get("question_type") == "order"]


tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args
)
model.eval()

######################
# Utilizing Functions
######################

def load_video(video_path, max_frames_num=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frames = vr.get_batch(uniform_sampled_frames.tolist()).asnumpy()
    return frames  # (frames, height, width, channels)

def generate_answer(video_path, question, candidates, max_frames=args.max_frames):
    video_frames = load_video(video_path, max_frames)
    frames_tensor = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
    image_tensors = [frames_tensor]

    ###########################
    # Different Prompt Options
    ###########################
    my_prompt = f"""
    You are given a question and a list of possible answers (candidates).
    Question:
    {question.strip()}
    
    Candidates:
    {chr(10).join(f"- {c}" for c in candidates)}
    Your task: Choose the single candidate that correctly answers the question.
    Return only the exact text of that chosen candidate - nothing else (no punctuation, no explanation).
    """

    my_prompt2 = f"""
    You are given a question and a list of answer choices.
    Read the question carefully and select only one answer that is most accurate and logically correct.
    
    Input:
    Question:
    {question.strip()}
    
    Answer Choices:
    {chr(10).join(candidates)}
    
    Task:Choose the single best answer from the list above.
    Your output must contain only the exact text of the selected answer — no punctuation, no explanation, and no additional words.
    """



    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{args.prompt_type.strip()}\n")
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [frame.shape[:2] for frame in video_frames]

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

def optionchecker(candidates, answer):
    for id, c in enumerate(candidates):
        if answer == c:
            return id + 1
    return -1


#####################################
# RRR - Running, Results , Reporting
#####################################
output_folder = os.path.join(os.getcwd(), "results")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(
    output_folder,
    f"output_{args.model_size}_{args.max_frames}_{args.prompt_type}.txt"
)

correct = 0
freq = np.array([0] * 6)

with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        video_file = item["video"]
        video_path = os.path.join(dataset_path, video_file)

        print(f"Processing video: {video_file}")
        print(f"Processing video: {video_file}", file=f)
        predicted_answer = generate_answer(video_path, item["question"], item["candidates"])
        print(f"Predicted: {predicted_answer}", file=f)
        # print(f"Candidates: {item['candidates']}")
        
        print(f"Predicted: {predicted_answer}")
        freq[optionchecker(item['candidates'],predicted_answer)-1]+=1
        # print(f"option: {optionchecker(item['candidates'],predicted_answer)}")
        print(f"Ground Truth: {item['answer']}", file=f)
        print(f"Ground Truth: {item['answer']}")

        print("-" * 37)
        print("-" * 50, file=f)

        if predicted_answer == item["answer"]:
            correct += 1
            
        # if correct >2:
        #     break

    accuracy = correct / len(dataset)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Accuracy: {accuracy*100:.2f}%", file=f)
    
plt.bar(['1','2','3','4','5','6'],freq)
plt.xlabel('Options')
plt.ylabel('Frequency')
plt.title('Frequency of Selected Options')
plt.savefig(os.path.join(output_folder, f'option_freq{args.model_size}_{args.max_frames}_{args.prompt_type}.jpg'))
plt.show()
