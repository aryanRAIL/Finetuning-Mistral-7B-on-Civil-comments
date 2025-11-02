# models/pipeline.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

with open("config/eval_config.json", "r") as f:
    cfg = json.load(f)

USER_TAG = "<|user|>\n"
ASSIST_TAG = "<|assistant|>\n"

base_model = cfg["base_model"]
adapter_dir = cfg["adapter_dir"]

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16
)
print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_dir, device_map="auto")
model.eval()

def wrap_prompt(p: str) -> str:
    return f"{USER_TAG}{p.strip()}\n\n{ASSIST_TAG}"

def generate(prompt, max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"], top_p=cfg["top_p"]):
    text = wrap_prompt(prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        p = input("Prompt (exit to quit): ")
        if p.lower() in ("exit", "quit"):
            break
        print("\n=== Response ===\n")
        print(generate(p))
        print("\n================\n")
