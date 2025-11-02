# datasets/prepare_dataset.py
import os
import csv
from datasets import load_dataset

def prepare_hh_rlhf(subset_size: int = 2000, out_path: str = "datasets/hh_train.csv") -> str:
    """
    Build a prompt/response CSV from Anthropic HH-RLHF 'chosen' conversations.
    We take the first user->assistant pair (harmless/helpful alignment style).
    Output schema: prompt, response
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading Anthropic/hh-rlhf (train split)...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    rows = []
    for item in ds:
        # Each item has "chosen" and "rejected" conversations (list of {role, content})
        chosen = item.get("chosen", [])
        if not isinstance(chosen, list) or len(chosen) < 2:
            continue

        # Expect alternating roles: "human" then "assistant"
        prompt = chosen[0].get("content", "").strip()
        response = chosen[1].get("content", "").strip()

        if prompt and response:
            rows.append({"prompt": prompt, "response": response})

        if len(rows) >= subset_size:
            break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "response"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved HH-RLHF dataset to {out_path} | Samples = {len(rows)}")
    return out_path


if __name__ == "__main__":
    prepare_hh_rlhf(subset_size=2000, out_path="datasets/hh_train.csv")
