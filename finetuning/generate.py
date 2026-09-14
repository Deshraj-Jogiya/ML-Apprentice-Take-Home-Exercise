"""Shared helper: load the base Qwen2.5-0.5B-Instruct model, optionally with
the real LoRA adapter on top, and generate a response for a prompt.
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapter")

_tokenizer = None
_base_model = None
_finetuned_model = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_base_model():
    global _base_model
    if _base_model is None:
        _base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        _base_model.eval()
    return _base_model


def get_finetuned_model():
    global _finetuned_model
    if _finetuned_model is None:
        _finetuned_model = PeftModel.from_pretrained(get_base_model(), ADAPTER_DIR)
        _finetuned_model.eval()
    return _finetuned_model


def generate(instruction: str, use_adapter: bool, max_new_tokens: int = 150) -> str:
    tokenizer = get_tokenizer()
    model = get_finetuned_model() if use_adapter else get_base_model()
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
