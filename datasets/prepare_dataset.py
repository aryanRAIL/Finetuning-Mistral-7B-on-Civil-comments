# datasets/prepare_dataset.py
import os, csv
from datasets import load_dataset

def prepare_civil_comments(subset_size=7000, out_path="datasets/tox_train.csv"):
    """
    Loads google/civil_comments and converts toxic -> safe rewrite instruction pairs.
    prompt = original comment
    response = safe rewrite target
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Loading google/civil_comments...")
    ds = load_dataset("google/civil_comments", split="train")

    rows = []
    for x in ds:
        text = x["text"]
        toxicity_score = x["toxicity"]

        if toxicity_score > 0.5:
            # toxic comment → rewrite target
            safe_target = "Rewrite this statement into a respectful, non-toxic alternative version."
        else:
            # non toxic → reinforce polite constructive style
            safe_target = "Respond politely and constructively."

        rows.append({"prompt": text.strip(), "response": safe_target})

        if len(rows) >= subset_size:
            break

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "response"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved {len(rows)} samples → {out_path}")
    return out_path

if __name__ == "__main__":
    prepare_civil_comments()
