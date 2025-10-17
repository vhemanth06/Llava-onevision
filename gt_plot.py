import os
import json

import copy
import warnings
import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

json_path="/DATA1/ai23mtech12002/DATASETS/MLVU/MLVU_Test/test-ground-truth/test_mcq_gt.json"
dataset_path="/DATA1/ai23mtech12002/DATASETS/MLVU/MLVU_Test/MLVU_Test/Action_order"

with open(json_path, "r") as f:
    dataset = json.load(f)
    
dataset = [item for item in dataset if item.get("question_type") == "order"]

freq=np.array([0]*6)
def optionchecker(candidates, answer):
    for id,c in enumerate(candidates):
        if answer== c:
            return id+1
    return -1

for item in dataset:
    answer=item["answer"]
    candidates=item["candidates"]
    id=optionchecker(candidates, answer) -1
    freq[id]+=1
output_folder = os.path.join(os.getcwd(), "results")
plt.bar(range(1,7), freq)
plt.xticks(range(1,7))
plt.xlabel('Options')
plt.ylabel('Frequency')
plt.title('Frequency of Correct Options')
plt.savefig(os.path.join(output_folder, 'gt_freq.jpg'))
plt.show()