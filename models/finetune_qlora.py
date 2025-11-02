# models/finetune_qlora.py
"""
QLoRA SFT on Anthropic HH-RLHF (prompt->assistant) with Unsloth Mistral-7B-Instruct 4-bit.
- Reads config/training_config.json
- Trains only on assistant tokens (user tokens masked from loss)
"""

import json, os, gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

CONFIG_PATH = "config/training_config.json"

USER_TAG = "<|user|>\n"
ASSIST_TAG = "<|assistant|>\n"

def find_subsequence(seq, sub):
    """Return start index of sub in seq (first match) or -1."""
    L, l = len(seq), len(sub)
    for i in range(L - l + 1):
        if seq[i:i+l] == sub:
            return i
    return -1

def main(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]
    os.makedirs(adapter_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Preparing model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=None  # let PEFT infer for Mistral
    )
    model = get_peft_model(model, lora_config)
    print("PEFT (LoRA) model prepared.")

    # ===== Load CSV dataset =====
    data_path = cfg.get("train_csv", "datasets/hh_train.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found. Run: python datasets/prepare_dataset.py"
        )
    print(f"Loading dataset from {data_path} ...")
    dataset = load_dataset("csv", data_files={"train": data_path})["train"]

    # Precompute assistant tag tokens for masking
    assistant_tag_ids = tokenizer(ASSIST_TAG, add_special_tokens=False)["input_ids"]

    def build_text(prompt: str, response: str) -> str:
        return f"{USER_TAG}{prompt.strip()}\n\n{ASSIST_TAG}{response.strip()}"

    def tokenize_and_mask(example):
        text = build_text(example["prompt"], example["response"])
        toks = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg["max_seq_length"]
        )
        input_ids = toks["input_ids"]
        labels = input_ids.copy()

        # Find assistant tag; mask everything BEFORE the end of assistant tag
        idx = find_subsequence(input_ids, assistant_tag_ids)
        cutoff = idx + len(assistant_tag_ids) if idx != -1 else 0

        for i in range(cutoff):
            labels[i] = -100  # ignore user + assistant tag in loss

        # Also mask padded tokens
        pad_id = tokenizer.pad_token_id
        for i, tok in enumerate(input_ids):
            if tok == pad_id:
                labels[i] = -100

        toks["labels"] = labels
        return toks

    tokenized = dataset.map(tokenize_and_mask, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch")

    # ===== Training args =====
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="wandb",
        run_name="mistral_7b_qlora_safety"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    finally:
        print("Saving LoRA adapters to", adapter_dir)
        model.save_pretrained(adapter_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
