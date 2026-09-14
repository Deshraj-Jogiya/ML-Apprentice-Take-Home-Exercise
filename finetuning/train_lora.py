"""Real LoRA fine-tuning of Qwen2.5-0.5B-Instruct on a real subset of the
Databricks Dolly-15k instruction dataset. This is the standalone script
version of the training run actually executed on a free Google Colab T4 GPU
(this repo has no local GPU) -- run this yourself on any CUDA machine to
reproduce it, or open the same steps in Colab.

Produces a LoRA adapter (a few MB) saved to finetuning/adapter/, which is
what's committed to this repo (not the full fine-tuned model weights).
"""
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter")


def load_dolly_subset(n=600, seed=42):
    raw = load_dataset("databricks/databricks-dolly-15k", split="train")
    raw = raw.filter(lambda x: x["context"] == "")
    return raw.shuffle(seed=seed).select(range(n))


def format_dataset(raw, tokenizer):
    to_text = lambda ex: {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": ex["response"]},
            ],
            tokenize=False,
        )
    }
    return raw.map(to_text, remove_columns=raw.column_names)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dolly_subset()
    formatted = format_dataset(raw, tokenizer)
    split = formatted.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir="./qwen-lora-dolly",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_length=512,
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds, eval_dataset=eval_ds, peft_config=lora_config)
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
