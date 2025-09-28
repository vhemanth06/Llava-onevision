
from transformers import pipeline
pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
messages = [
    {
      "role": "user",
      "content": [
          {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"},
          {"type": "text", "text": "What does the label 10 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
        ],
    },
]
out = pipe(text=messages, max_new_tokens=20)
print(out)

import json
import os

os.makedirs("results", exist_ok=True)

with open("results/output.json", "w") as f:
    json.dump(out, f)
# import cv2
# import os
# import json
# from transformers import pipeline

# # -----------------------------
# # CONFIG
# video_path = "sample_video.mp4"  # your test video
# frame_number = 30                # which frame to extract (0 = first)
# results_folder = "results"
# os.makedirs(results_folder, exist_ok=True)

# # -----------------------------
# # 1) Extract frame from video
# cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# ret, frame = cap.read()
# if not ret:
#     raise ValueError(f"Could not read frame {frame_number} from {video_path}")

# # Save the frame as an image (optional, for debugging)
# frame_path = os.path.join(results_folder, "frame.jpg")
# cv2.imwrite(frame_path, frame)
# cap.release()

# # -----------------------------
# # 2) Run LLaVA-OneVision inference
# pipe = pipeline("image-text-to-text", model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

# # Prepare message (replace with a real prompt)
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": frame_path},
#             {"type": "text", "text": "Describe the scene in this frame in one sentence."}
#         ],
#     },
# ]

# out = pipe(text=messages, max_new_tokens=50)

# # -----------------------------
# # 3) Save output to JSON
# output_file = os.path.join(results_folder, "video_frame_output.json")
# with open(output_file, "w") as f:
#     json.dump(out, f, indent=2)

# print(f"Output saved to {output_file}")
