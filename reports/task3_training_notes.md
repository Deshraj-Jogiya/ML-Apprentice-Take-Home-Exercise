# Task 3: Training Considerations and Transfer Learning Strategy

# Freezing Scenarios

# 1. Entire Network Frozen
- **What It Means**: No layers are updated during training.
- **Use Case**: Only useful for using embeddings as static feature extractors.
- **Pros**:
  - Extremely fast
  - Avoids overfitting on small data
- **Cons**:
  - No learning happens — model can't adapt to new tasks

# 2. Only Transformer Backbone Frozen
- **What It Means**: Only task-specific heads are trainable.
- **Use Case**: When using a general-purpose pre-trained transformer and customizing for downstream tasks.
- **Pros**:
  - Retains rich language knowledge from pretraining
  - Reduces compute cost
- **Cons**:
  - Task-specific learning is limited to shallow layers

# 3. Only One Task Head Frozen
- **What It Means**: One task adapts; the other remains fixed.
- **Use Case**: When one task is already well-trained and shouldn’t be affected by multi-task updates.
- **Pros**:
  - Protects performance of a stable task
- **Cons**:
  - Shared backbone still changes, so frozen head’s output may drift

---

# Transfer Learning Approach

# Scenario: Adapting BERT to Multi-task Learning (Sentence Classification + Sentiment Analysis)

# 1. Choice of Pre-trained Model
- **Model**: `bert-en-uncased-l-10-h-128-a-2/2`
- **Why?**:
  - Lightweight (faster training)
  - Still retains meaningful language knowledge
  - Hosted on TensorFlow Hub for easy loading

# 2. Layer Freezing Strategy
- **Freeze**: Lower transformer layers (e.g., first 3 of 6)
- **Train**:
  - Upper transformer layers (to adapt contextual understanding)
  - All task-specific heads

# 3. Rationale
- Lower BERT layers often capture syntax and grammar (language universal)
- Upper layers capture task-specific signals
- Keeping task heads trainable ensures flexibility across both tasks

---

#  Key Takeaways

- Smart freezing can stabilize training, especially in multi-task setups
- Full fine-tuning is powerful but needs good regularization/data
- BERT-like transformers benefit heavily from task-aware fine-tuning

---