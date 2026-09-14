"""Real before/after comparison: base Qwen2.5-0.5B-Instruct vs the same
model with the real LoRA adapter, on real held-out instructions from the
Dolly eval split (never seen during training).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import generate
from train_lora import load_dolly_subset

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_results.json")


def get_eval_prompts(n=5):
    raw = load_dolly_subset()
    split = raw.train_test_split(test_size=0.1, seed=42)
    eval_raw = split["test"]
    return [eval_raw[i] for i in range(min(n, len(eval_raw)))]


def run_comparison(n=5):
    examples = get_eval_prompts(n)
    results = []
    for ex in examples:
        instruction = ex["instruction"]
        reference = ex["response"]
        base_output = generate(instruction, use_adapter=False)
        finetuned_output = generate(instruction, use_adapter=True)
        results.append({
            "instruction": instruction,
            "reference_response": reference,
            "base_output": base_output,
            "finetuned_output": finetuned_output,
        })
        print(f"\nInstruction: {instruction}")
        print(f"Reference:  {reference}")
        print(f"Base:       {base_output}")
        print(f"Fine-tuned: {finetuned_output}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    run_comparison()
