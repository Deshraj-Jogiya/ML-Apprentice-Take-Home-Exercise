# Sentence Transformers, Multi-Task Learning & LLM Fine-Tuning

A personal NLP project covering three related, hands-on pieces of transformer
work: building a sentence-embedding model from a pretrained encoder, extending
it into a multi-task architecture, and real LoRA fine-tuning of a modern
generative LLM.

---

## Part 1 — Sentence Transformer

- Built using a Small BERT model from TensorFlow Hub.
- Encodes sentences into 128-dimensional contextual embeddings.
- Used mean pooling across token embeddings for fixed-length output.
- Tested with sample inputs for verification.

## Part 2 — Multi-Task Learning Expansion

- Added two task-specific heads to the sentence transformer:
  - Task A: 3-class sentence classification
  - Task B: 2-class sentiment classification
- Shared transformer backbone with separate softmax output layers.

## Part 3 — Training Strategy Notes

- Explored training strategies:
  - When to freeze the entire model vs. just the backbone
  - When to freeze only one task-specific head
- Proposed a transfer learning strategy using Small BERT:
  - Freeze lower layers, fine-tune top layers and task heads.
- See [reports/task3_training_notes.md](reports/task3_training_notes.md).

## Part 4 — Multi-Task Training Loop

- Real `model.fit()` training loop over the multi-task architecture (synthetic
  text/labels, since Parts 1-4 are about the architecture and training
  mechanics, not a specific labeled dataset).
- Logs per-task losses and accuracy.

---

## Part 5 — Real LoRA Fine-Tuning (Qwen2.5-0.5B-Instruct)

Unlike Parts 1-4 (architecture + synthetic training loop), this part is a
real, complete fine-tuning run: **LoRA fine-tuning of Qwen2.5-0.5B-Instruct**
on a real 600-example subset of the
[Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
instruction dataset (filtered to context-free instructions, 90/10 train/eval
split), trained for 3 epochs on a free Google Colab T4 GPU (this repo's dev
machine has no local GPU -- LoRA + a free T4 is the real, zero-cost way to do
this without one).

**Why LoRA**: fine-tuning all ~500M parameters isn't necessary or efficient
for adapting a model's response style/format to a new dataset -- LoRA trains
a small set of low-rank adapter matrices (here, on the attention
projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`) while the base model
stays frozen, making the whole thing feasible on a free GPU in a few minutes
and producing an adapter that's a few MB, not a multi-GB full model.

### What's committed
- `finetuning/train_lora.py` -- the real training script (standalone version
  of what was actually run; same logic, runnable on any CUDA machine).
- `finetuning/adapter/` -- the real trained LoRA adapter weights.
- `finetuning/compare.py` / `finetuning/generate.py` -- loads the base model
  and the base+adapter model, generates real responses for real held-out
  Dolly eval instructions (never seen during training), and saves a
  side-by-side comparison.
- `finetuning/comparison_results.json` -- real output from that comparison.
- `tests/test_finetuned_adapter.py` -- a real regression test: loads the
  actual base model and the actual committed adapter and confirms the
  adapter's generation output is real, non-empty, and measurably different
  from the base model's on at least one of several held-out prompts (see the
  honest result below for why "at least one" and not "every prompt").

### Real training results
3 epochs, 540 train / 60 eval examples, real declining loss:

| Epoch | Training Loss | Validation Loss | Mean Token Accuracy |
|-------|---------------|------------------|----------------------|
| 1     | 1.779         | 1.723            | 0.6318               |
| 2     | 1.760         | 1.710            | 0.6326               |
| 3     | 1.650         | 1.708            | 0.6338               |

### An honest evaluation finding
With greedy decoding, the LoRA adapter does **not** change the output on
every prompt -- for short factual completions (e.g. "What are the main types
of Thai curries?"), the fine-tuned model produced byte-identical output to
the base model. That's expected for a light LoRA fine-tune (r=16, 3 epochs):
the perturbation isn't always large enough to flip the argmax token at every
step. It **does** produce a real, visible difference on others -- e.g. for
"What are fun things to do in New York City", the base model's answer trails
off into a repetitive, partially garbled list (mentions "J. Paul Getty
Center" as a NYC store), while the fine-tuned model gives a clean, complete,
better-formatted 10-item list. Full outputs for all 5 real comparison
prompts are in `finetuning/comparison_results.json`. The regression test
reflects this honestly: it checks that at least one of several real prompts
shows a real difference, not that a single fixed prompt always does.

### A real dependency conflict found while building this
`peft`'s `is_torchao_available()` check requires `torchao>=0.16.0`, but
Google Colab's base image ships `torchao==0.10.0` pre-installed -- fine-tuning
failed immediately with `ImportError: Found an incompatible version of
torchao` until `torchao` was explicitly upgraded before importing `peft`/`trl`.
Locally (no GPU, used for inference/comparison only), the opposite problem
hit: the newest `torchao` release doesn't import cleanly against this
machine's installed `torch` version (`ImportError: cannot import name
'ScalingType' from 'torch.nn.functional'). Since `torchao` is only needed
for quantized workflows -- not used here -- the fix was to just not install
it locally; `transformers`/`peft` import it opportunistically and work fine
without it for plain-precision inference.

### Reproducing it
```bash
pip install -r finetuning/requirements.txt
python finetuning/train_lora.py   # needs a CUDA GPU; ~6-7 min on a free Colab T4
python finetuning/compare.py      # real before/after comparison on held-out prompts
```

---

## Tech Stack

- **Parts 1-4**: TensorFlow, TensorFlow Hub, NumPy
- **Part 5**: PyTorch, Hugging Face `transformers`, `peft` (LoRA), `trl`
  (`SFTTrainer`), `datasets`, real GPU training (Google Colab, free T4)

---

## Repository Structure

```text
.
├── models/
│   ├── sentence_transformer.py     # Part 1
│   ├── multitask_model.py          # Part 2
│   └── train_multitask.py          # Part 4
├── reports/
│   └── task3_training_notes.md     # Part 3
├── output_screens/                 # Parts 1/2/4 output screenshots
├── finetuning/                     # Part 5
│   ├── train_lora.py
│   ├── generate.py
│   ├── compare.py
│   ├── comparison_results.json
│   ├── requirements.txt
│   └── adapter/                    # real committed LoRA weights
├── tests/
│   └── test_finetuned_adapter.py
├── requirements.txt                 # Parts 1-4 (TensorFlow stack)
└── README.md
```

## Running Parts 1-4

```bash
pip install -r requirements.txt
python main.py                      # Part 1
python models/multitask_model.py    # Part 2
python models/train_multitask.py    # Part 4
```
