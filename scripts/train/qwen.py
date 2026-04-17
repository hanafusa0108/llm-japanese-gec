import os
import sys
import random

import numpy as np
import torch
import argparse
import yaml
from transformers import TrainingArguments
from datasets import DatasetDict

from utils.preprocess import load_data
from models.model import Model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class MyDataCollator:
    def __init__(self, tokenizer, response_template):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, batch):
        input_ids_list, attention_mask_list, labels_list = [], [], []

        for example in batch:
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            labels = input_ids.copy()

            # 出力部分の開始位置を探す
            start_index = None
            for i in range(len(input_ids) - len(self.response_template_ids) + 1):
                if input_ids[i : i + len(self.response_template_ids)] == self.response_template_ids:
                    start_index = i + len(self.response_template_ids)
                    break

            if start_index is None:
                labels = [-100] * len(input_ids)
            else:
                labels[:start_index] = [-100] * start_index

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        # ★ tokenizer.pad を使ってバッチ化（長さを揃える）
        batch_out = self.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            return_tensors="pt",
        )

        # labels も同じ長さに pad
        labels_padded = torch.full_like(batch_out["input_ids"], -100)
        for i, l in enumerate(labels_list):
            labels_padded[i, :len(l)] = torch.tensor(l, dtype=torch.long)

        batch_out["labels"] = labels_padded
        return batch_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-2,
        help="Local rank passed from distributed launcher."
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., bert, llm)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--response_template", type=str, required=True, help="Response template for the model")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate to override the one in config")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    if args.learning_rate is not None:
        config["training_args"]["learning_rate"] = args.learning_rate

    # Modelクラスを直接使用
    model = Model(
        config=config,
        model_checkpoint=args.model,
    )
    
    print("学習率")
    training_args_config = config["training_args"]
    print(training_args_config["learning_rate"])

    # データの読み込み
    raw_datasets = DatasetDict({
        "train": load_data(config["train_path"]),
        "dev": load_data(config["dev_path"]),
    })
    
    # トークナイズ
    tokenizer = model.tokenizer
    
    def prompt_template(examples, tokenizer):
        """
        チャットテンプレートを適用してプロンプトを生成します。
        """
        outputs = []
        for input, output in zip(examples["input"], examples["output"]):
            
            message = [
                {"role": "system", "content": "入力文の文法的な誤りを正しい文法に修正してください。"},
                {
                    "role": "user",
                    "content": input,
                },
                {
                    "role": "assistant",
                    "content": output,
                },
            ]
            
            prompt = tokenizer.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=False,
                enable_thinking=False  # True is the default value for enable_thinking
            )

            outputs.append(prompt)
    
        return outputs  # トークンIDを返す
    
    train_dataset = prompt_template(raw_datasets["train"], tokenizer)
    dev_dataset = prompt_template(raw_datasets["dev"], tokenizer)
    print(train_dataset[0])  # デバッグ用に最初のサンプルを表示
    
    def token_to_ids(example):
        encoded_example = tokenizer(
            example, max_length=512, padding=True, truncation=True
        )
        return encoded_example
    
    train_dataset = [token_to_ids(item) for item in train_dataset]
    dev_dataset = [token_to_ids(item) for item in dev_dataset]
    
    # response_templateは必須指定
    response_template = args.response_template
    
    # data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    data_collator = MyDataCollator(tokenizer, args.response_template)
    
    # デバッグ: batchを作って確認
    batch = data_collator(train_dataset[:1])
    print("デバッグ: input_ids の内容")
    print(batch["input_ids"][0])
    print("デバッグ: labels の内容")
    print(batch["labels"][0])
    print("デコード (-100以外):", tokenizer.decode([t.item() for t in batch["labels"][0] if t.item() != -100]))
    
    # YAML から TrainingArguments を読み込む
    training_args_config = config["training_args"]

    base_dir = training_args_config["output_dir"]
    
    # ディレクトリ名をパラメータから生成
    dir_name = f"{args.model}_lr{training_args_config['learning_rate']}"
    output_dir_path = os.path.join(base_dir, dir_name)
    
    
    training_args = TrainingArguments(
        output_dir=output_dir_path,
        eval_strategy =training_args_config["eval_strategy"],
        eval_steps=int(training_args_config["eval_steps"]),
        save_strategy=training_args_config["save_strategy"],
        save_steps=training_args_config["save_steps"],
        max_steps=training_args_config["max_steps"],
        learning_rate=float(training_args_config["learning_rate"]),
        per_device_train_batch_size=int(training_args_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_args_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_args_config["gradient_accumulation_steps"]),
        metric_for_best_model=training_args_config["metric_for_best_model"],
        load_best_model_at_end=training_args_config["load_best_model_at_end"],
        push_to_hub=training_args_config["push_to_hub"],
        save_total_limit=int(training_args_config["save_total_limit"]),
        report_to="none",
        # run_name=wandb.run.name,
        bf16=True
    )

    # トレーニングの実行
    model.train(
        train_dataset, 
        dev_dataset, 
        training_args, 
        data_collator=data_collator,
    )