import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import numpy as np
import wandb
from data import build_dataloader, create_dataset_mapping
from utils import *
import torch.utils.checkpoint as cp
from functools import partial
from model import compute_supervised_loss, input_processing, generate_target

cp.checkpoint = partial(cp.checkpoint, use_reentrant=False)

parser = argparse.ArgumentParser(description="Train reward model (lower only)")
parser.add_argument('--train_json_file', type=str, default="/workspace/PRM/data/geometry3k_en_20240402_extracted_open_ended_only_prm.json")
parser.add_argument('--meta_json_file', type=str, default="./data/meta.json")
parser.add_argument('--weights_path', type=str, default="./weights")
parser.add_argument("--reward_model", type=str, default="OpenGVLab/InternVL3-1B")
parser.add_argument("--iteration_num", type=int, default=1000)
parser.add_argument("--save_every_iterations", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="bf16")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--scheduler_type", type=str, default="cosine_schedule_with_warmup")
parser.add_argument("--scheduler_step_size", type=int, default=1000)
parser.add_argument("--scheduler_gamma", type=float, default=0.95)
parser.add_argument("--max_patch_num", type=int, default=6)
args = parser.parse_args()

print(args)
set_seed(args.seed)
device = torch.device(args.device)
wandb.init(project="DreamPRM-LowerOnly")

# === Load Dataset ===
domain_list = create_dataset_mapping(args.train_json_file)
train_dataloader = build_dataloader(
    train_json_file=args.train_json_file,
    meta_json_file=args.meta_json_file,
    train_batch_size=args.batch_size,
    meta_batch_size=args.batch_size,
    max_patch_num=args.max_patch_num,
)

# === Load Model ===
MODEL_PATH = args.reward_model
model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    use_flash_attn=False,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.scheduler_type == "cosine_schedule_with_warmup":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05 * args.iteration_num,
        num_training_steps=args.iteration_num
    )
elif args.scheduler_type == "step_lr":
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
    )
else:
    scheduler = None

criterion = nn.CrossEntropyLoss()

# === Training Loop ===
print("Start training lower model only ...")
model.train()
iteration = 0
accum_loss = []
best_loss = float("inf")

for epoch in range(1, 10001):  # run indefinitely until reaching iteration_num
    for batch in train_dataloader:
        if iteration >= args.iteration_num:
            break

        optimizer.zero_grad(set_to_none=True)

        prompt, pixel_values, _ = batch
        prompt = prompt[0]
        pixel_values = pixel_values[0]

        # input processing
        input_ids, attention_mask, image_flags, pixel_values = input_processing(
            model, tokenizer, prompt, pixel_values
        )

        # forward
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
        )
        logits = outputs.logits
        target = generate_target(input_ids, tokenizer, model.template)
        loss = compute_supervised_loss(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        accum_loss.append(loss.item())
        iteration += 1

        if iteration % 100 == 0:
            mean_loss = np.mean(accum_loss)
            wandb.log({"lower_loss": mean_loss, "lr": optimizer.param_groups[0]['lr']})
            print(f"[Iter {iteration}] loss={mean_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
            accum_loss.clear()

        if iteration % args.save_every_iterations == 0:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.save_pretrained(args.weights_path)
            print(f"✅ Saved model at iteration {iteration}")

        torch.cuda.empty_cache()

    print(f"Epoch {epoch} completed.")
    if iteration >= args.iteration_num:
        break

print("Training finished ✅")
model.save_pretrained(args.weights_path)
print(f"Final model saved to {args.weights_path}")
