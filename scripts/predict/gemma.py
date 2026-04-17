import sys
import os
import random
import numpy as np
import torch

# シード値を固定
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# プロジェクトのルートディレクトリを sys.path に追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import yaml
import importlib
from utils.preprocess import load_data
from transformers import TrainingArguments
from torch.optim import AdamW
from datasets import DatasetDict  # 必要に応じてインポート
from models.model import Model  # Modelクラスをインポート
# from trl import DataCollatorForCompletionOnlyLM

def predict_prompt_template(examples, tokenizer):
    """
    チャットテンプレートを適用してプロンプトを生成します。
    """
    input = examples["input"]
    
    message = [
        {"role": "system", "content": "入力文の文法的な誤りを正しい文法に修正してください。"},
        {
            "role": "user",
            "content": input,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    
    return {"prompt": prompt}  # トークンIDを返す



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., bert, llm)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter file")
    parser.add_argument("--file_name", type=str, required=True, help="Path to file_path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Modelクラスを直接使用
    model = Model(
        config=config,
        model_checkpoint=args.model,
    )

    # データの読み込み
    raw_datasets = DatasetDict({
        "test": load_data(config["test_path"]),
    })
    
    # トークナイズ
    tokenizer = model.tokenizer
    
    # プロンプトテンプレートを適用
    tokenized_datasets = raw_datasets.map(
        lambda examples: predict_prompt_template(examples, tokenizer),
    )
    print(tokenized_datasets["test"])

    # トレーニングの実行
    model.predict(
        tokenized_datasets["test"]["prompt"], 
        "results",
        adapter_path=args.adapter,
        file_name=args.file_name
    )

