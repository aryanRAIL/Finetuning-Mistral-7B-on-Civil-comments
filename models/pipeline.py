# models/pipeline.py
"""
Interactive inference pipeline for your fine-tuned Mistral-7B LoRA model.
Uses the quantization level specified in config/eval_config.json (2-bit / 4-bit).
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load config
with open("config/eval_config.json", "r") as f:
    cfg = json.load(f)

USER_TAG = "<|user|>\n"
ASSIST_TAG = "<|assistant|>\n"

base_model = cfg["base_model"]
adapter_dir = cfg["adapter_dir"]
quant_bits = cfg.get("quant_bits", 4)
max_new_tokens = cfg.get("max_new_tokens", 128)
temperature = cfg.get("temperature", 0.0)
top_p = cfg.get("top_p", 0.95)

# Quantization config (same for baseline & fine-tuned)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf2" if quant_bits == 2 else "nf4",
)

print(f"Loading base model '{base_model}' with {quant_bits}-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto")
model.eval()

def wrap_prompt(p: str) -> str:
    """Format user prompt with role tags."""
    return f"{USER_TAG}{p.strip()}\n\n{ASSIST_TAG}"

def generate(prompt: str):
    """Generate model response."""
    text = wrap_prompt(prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("\n=== Interactive Inference (type 'exit' to quit) ===\n")
    while True:
        user_input = input("Prompt: ")
        if user_input.lower().strip() in ("exit", "quit"):
            break
        print("\n=== Response ===\n")
        print(generate(user_input))
        print("\n================\n")
