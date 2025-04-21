ML Apprentice Take-Home Project
===============================

Project Focus:
This project demonstrates the design, implementation, and explanation of a Sentence Transformer and its extension into a Multi-Task Learning (MTL) architecture using TensorFlow.

Task Breakdown:
------------------

Task 1 – Sentence Transformer
- Built using a Small BERT model from TensorFlow Hub.
- Encodes sentences into 128-dimensional contextual embeddings.
- Used mean pooling across token embeddings for fixed-length output.
- Tested with sample inputs for verification.

Task 2 – Multi-Task Learning Expansion
- Added two task-specific heads to the sentence transformer:
  • Task A: 3-class sentence classification
  • Task B: 2-class sentiment classification
- Shared transformer backbone with separate softmax output layers.

Task 3 – Training Considerations
- Explored training strategies:
  • When to freeze the entire model vs. just the backbone
  • When to freeze only one task-specific head
- Proposed a transfer learning strategy using Small BERT:
  • Freeze lower layers, fine-tune top layers and task heads.

Task 4 – Training Loop Simulation (Bonus)
- Used synthetic text and randomly assigned labels.
- Encoded labels as one-hot vectors.
- Trained model using model.fit() to simulate multi-task learning.
- Logged per-task losses and accuracy.

---

Tech Stack:
--------------
- TensorFlow & TensorFlow Hub
- Python (core scripting)
- NumPy (synthetic data generation)

---

How to Run:
--------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run Sentence Transformer (Task 1):
   python main.py

3. multiTask_model (Task 2):
    python multiTask_model.py

4. Simulate Multi-Task Training (Task 4):
   python train_multitask.py

---

Files Included:
------------------
- models/
  • sentence_transformer.py
  • multitask_model.py
  • train_multitask.py
- reports/
  • task3_training_notes.md
- output_screens/
  • Task-1 output
  • Task-2 output
  • Task-4 output
- README.md
- rquirements.txt

Final Notes:
---------------
This project showcases strong foundational knowledge in NLP, transformers, and multi-task learning. The architecture is modular, explainable, and production-ready for scaling or real-data fine-tuning.

Thank you for reviewing my submission!
