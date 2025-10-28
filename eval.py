

import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import base64
from io import BytesIO



# ======================================================
# ⚙️ 1. Load mô hình
# ======================================================
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

input_file = "test/geometry3k_test_metadata.json"          # file test
output_file = "/answer/geometry3k_test_results_qwen2-5-vl-7b.json" # file kết quả

print(f"Loading model: {model_name} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
processor = AutoProcessor.from_pretrained(model_name)

# ======================================================
# 📚 2. Hàm sinh lời giải đầy đủ
# ======================================================
def generate_full_solution(image, question, max_new_tokens=1024,
                           temperature=0.7, top_p=0.95, num_return_sequences=8):
    """
    Sinh lời giải hoàn chỉnh (từ Step 1 đến Final Answer).
    """
    base_prompt = (
        "You are an expert in solving multimodal mathematical problems. "
                    "I will provide a mathematical problem along with its corresponding image." 
                    "According to the problem and the image, please first conduct step-by-step reasoning, "
                    "Format:\nStep 1: ...\nStep 2: ...\n...\nAnswer: ...\n\n"
                    "and after your reasoning, please provide your final answer using the format: Answer: ... "
                    "Problem:"
                    "<Question>"
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a geometry reasoning assistant. Solve the problem step by step. "
                        "Show reasoning clearly and end with 'Final answer: <value>'.\n\n"
                        "Format:\nStep 1: ...\nStep 2: ...\n...\nFinal answer: ...\n\n"
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image}"},
                {"type": "text", "text": question},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)


    # Cắt phần prompt
   generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,               # bật sampling thay vì greedy search
        temperature=0.8,              # càng cao → kết quả càng đa dạng
        top_p=0.9,                    # nucleus sampling
        num_return_sequences=5,       # số câu trả lời muốn sinh
    )


    generated_ids_trimmed = [
    out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids
]
output_texts = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
    return output_text


# ======================================================
# 🔍 3. Hàm trích "Final answer"
# ======================================================
def extract_final_answer(text):
    """
    Tìm chuỗi 'Final answer: xxx' hoặc tương tự trong output.
    """
    match = re.search(r"final answer\s*[:：]\s*([^\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# ======================================================
# 🧪 4. Benchmark trên dataset Geometry3k
# ======================================================
results = []
total, correct = 0, 0

num_generations = 8  # số câu trả lời sinh cho mỗi câu hỏi

print(f"🚀 Running inference on {total} samples with {num_generations} generations each...")
with open(output_file, "w") as fout:
    for data in tqdm(test_data, desc="Processing"):
        image_path ='/workspace/PRM/test/' +  data["image_paths"][0]  # giả sử chỉ có 1 hình
        question = data["meta"]["problem"]
        correct_answer = str(data["meta"]["answer"]).strip()

        try:
            pixel_values = load_image(image_path)
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            continue

        # prompt cố định
        base_prompt = (
            "You are a geometry reasoning assistant. Solve the problem step by step.\n"
            "Show reasoning clearly and end with 'Final answer: <value>'.\n\n"
            f"<image>{question}"
        )

        multi_answers = []
        for i in range(num_generations):
            # Đặt seed để mỗi lần khác nhau
            torch.manual_seed(i * 13 + 7)
            try:
                response, _ = model.chat(
                    tokenizer,
                    pixel_values,
                    base_prompt,
                    gen_cfg,
                    history=None,
                    return_history=True
                )
            except Exception as e:
                response = f"ERROR: {e}"
            multi_answers.append(response)

        # Lưu kết quả
        result = {
            "image": image_path,
            "question": question,
            "correct_answer": correct_answer,
            "model_answers": multi_answers
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()
        results.append(result)

print(f"\n✅ Done! Saved multi-answer results to: {output_file}")
print(f"🧾 Each question has {num_generations} generated answers.")