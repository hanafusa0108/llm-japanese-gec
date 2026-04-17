import os
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from models.base_model import BaseModel

class Model(BaseModel):
    def __init__(self, config, model_checkpoint):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.config = config

    def train(self, train_data, dev_data, training_args, data_collator=None):
        """
        Trainer を使用してモデルをトレーニングします。
        """
        # LoRA の設定を適用
        if self.config.get("lora"):
            lora_config = self.config["lora"]
            peft_config = LoraConfig(
                task_type=lora_config["task_type"],
                inference_mode=lora_config["inference_mode"],
                r=lora_config["r"],
                lora_alpha=lora_config["lora_alpha"],
                target_modules=lora_config["target_modules"],
                lora_dropout=lora_config["lora_dropout"],         
            )
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            print("LoRA configuration applied to the model.")
            
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=dev_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config["training_args"]["early_stopping_patience"])],
        )
        trainer.train()

    
    def predict(self, test_data, output_dir, adapter_path=None, file_name=None):
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name + ".txt")

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        generated_texts = []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print("推論開始")

        token_speeds = []
        total_tokens = 0

        # 🔽 追加：最大遅延の記録用
        max_latency = 0
        max_latency_tokens = 0
        
        latency_records = []  # (elapsed, num_tokens, index)

        with torch.no_grad():
            for i in range(len(test_data)):
                inputs = self.tokenizer(
                    test_data[i],
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.model.config.max_position_embeddings
                    # max_length=12000
                ).to(device)

                inputs.pop("token_type_ids", None)

                input_ids = inputs["input_ids"]
                prompt_length = input_ids.shape[1]

                if device == "cuda":
                    torch.cuda.synchronize()
                
                start = time.time()

                outputs = self.model.generate(
                    **inputs,
                    max_length=10000,
                    do_sample=False
                )

                if device == "cuda":
                    torch.cuda.synchronize()
                    
                end = time.time()

                elapsed = end - start

                generated_tokens = outputs[0][prompt_length:]
                num_generated_tokens = generated_tokens.shape[0]
                
                latency_records.append((elapsed, num_generated_tokens, i))

                # 🔽 最大 latency 更新
                if elapsed > max_latency:
                    max_latency = elapsed
                    max_latency_tokens = num_generated_tokens

                # tokens/sec
                if elapsed > 0:
                    speed = num_generated_tokens / elapsed
                    token_speeds.append(speed)

                total_tokens += num_generated_tokens

                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                generated_text = generated_text.replace("\n", "\\n")

                print(generated_text, flush=True)
                generated_texts.append(generated_text)

        # 統計
        avg_speed = sum(token_speeds) / len(token_speeds) if token_speeds else 0
        max_speed = max(token_speeds) if token_speeds else 0

        # 保存
        with open(output_file, "w", encoding="utf-8") as f:
            for line in generated_texts:
                f.write(line + "\n")

        print(f"Generated texts saved to {output_file}")
        print(f"Total generated tokens: {total_tokens}")
        print(f"Average tokens/sec: {avg_speed:.2f}")
        print(f"Max tokens/sec: {max_speed:.2f}")

        # 🔽 追加出力
        print(f"Max latency (slowest sample): {max_latency:.4f} sec")
        print(f"Tokens at max latency: {max_latency_tokens}")
        
        # 🔽 上位5件の遅いサンプルを表示
        top_k = 5
        latency_records_sorted = sorted(latency_records, key=lambda x: x[0], reverse=True)

        print(f"\nTop {top_k} slowest samples:")
        for rank, (lat, tokens, idx) in enumerate(latency_records_sorted[:top_k], 1):
            print(f"{rank}位: sample_index={idx}, latency={lat:.4f} sec, tokens={tokens}")

        return generated_texts
    