from transformers import AutoModel, AutoTokenizer
from data import build_test_dataloader
import torch
from utils import select_best_answer
from tqdm import tqdm
import json

test_dataloader = build_test_dataloader(test_json_file = "/workspace/geometry3k_test_with_true_false.json")

MODEL_PATH = "./weights"
model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_flash_attn=False,
        ).cuda()
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)

correct = 0
total = 0

result_test = []

for inputs in tqdm(test_dataloader):
    # input_test_data_format:
    # {"question": question, "image_path": image_path, "candidate":[1, 2, 3, 4], "true_false":[True, False, True, False]}
    with torch.no_grad():
        true_false, best_index,list_scores = select_best_answer(model, tokenizer, inputs, 'mean')
        correct += int(true_false)
    total += 1
    # inputs['list_scores'] = list_scores
    result_test.append(list_scores)
acc = correct / total
print(acc)
output_path = "/workspace/geometry3k_test_with_scores.json"
with open(output_path, "w") as f:
    json.dump(result_test, f, indent=4, ensure_ascii=False)