import argparse
import os
import json
import yaml
import random
import numpy as np
import torch
from datasets import DatasetDict, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time  # ← 追加（上のimport群に入れてください）

# =====================
# シード値を固定
# =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================
# 引数定義
# =====================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Base model name or path")
parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
parser.add_argument("--file_name", type=str, required=True, help="Output file name prefix")
args = parser.parse_args()

# =====================
# config 読み込み
# =====================
with open(args.config, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

test_path = config["test_path"]
# 絶対パスを削除し、相対パス "results" をデフォルトに変更
output_dir = config.get("output_dir", "results")
reasoning_effort = config.get("reasoning_effort", "low")

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{args.file_name}.txt")

# =====================
# データ読み込み関数
# =====================
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list) and isinstance(data[0], dict):
        return Dataset.from_dict({k: [d[k] for d in data] for k in data[0]})
    else:
        raise ValueError("JSON must be a list of dictionaries")

# =====================
# テストデータ準備
# =====================
raw_datasets = DatasetDict({"test": load_data(test_path)})
print(f"Test size: {len(raw_datasets['test'])}")

# =====================
# モデルロード
# =====================
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
print("Base model loaded.")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.eval()
print("LoRA merged.")

# =====================
# Tokenize
# =====================

def tokenize_function(example):
    messages = [
        {"role": "system", "content": "入力文の文法的な誤りを正しい文法に修正してください。"},
        {"role": "user", "content": example["input"]},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        reasoning_effort=reasoning_effort,
        tokenize=False,
    )

    formatted_text += (
        "<|start|>assistant<|channel|>analysis<|message|><|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )

    return tokenizer(formatted_text)

print("Tokenizing...")
tokenized_dataset = raw_datasets["test"].map(tokenize_function)
print("Tokenization done.")

# =====================
# 推論
# =====================
outputs = []

print("Inference start...")

# 🔽 追加：計測用
token_speeds = []
total_tokens = 0
max_latency = 0
max_latency_tokens = 0

for i, item in enumerate(tokenized_dataset, 1):
    print(f"Progress: {i}/{len(tokenized_dataset)}")

    input_ids = torch.tensor([item["input_ids"]]).to(model.device)
    prompt_length = input_ids.shape[1]

    # 🔽 GPU同期（正確な時間測定）
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=model.config.max_position_embeddings,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start

    # 🔽 生成トークン取得
    generated_tokens = output[0][prompt_length:]
    num_generated_tokens = generated_tokens.shape[0]

    # 🔽 最遅ケース更新
    if elapsed > max_latency:
        max_latency = elapsed
        max_latency_tokens = num_generated_tokens

    # 🔽 tokens/sec
    if elapsed > 0:
        speed = num_generated_tokens / elapsed
        token_speeds.append(speed)

    total_tokens += num_generated_tokens

    generated_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    
    generated_text = generated_text.replace("\n", "\\n")

    print(generated_text, flush=True)
    outputs.append(generated_text)

# =====================
# 🔽 ここは元の保存処理（そのまま）
# =====================
with open(output_path, "w", encoding="utf-8") as f:
    for line in outputs:
        f.write(line + "\n")

print(f"Saved: {output_path}")

# =====================
# 🔽 統計出力（追加）
# =====================
avg_speed = sum(token_speeds) / len(token_speeds) if token_speeds else 0
max_speed = max(token_speeds) if token_speeds else 0

print("===== Inference Stats =====")
print(f"Total generated tokens: {total_tokens}")
print(f"Average tokens/sec: {avg_speed:.2f}")
print(f"Max tokens/sec: {max_speed:.2f}")
print(f"Max latency (slowest sample): {max_latency:.4f} sec")
print(f"Tokens at max latency: {max_latency_tokens}")