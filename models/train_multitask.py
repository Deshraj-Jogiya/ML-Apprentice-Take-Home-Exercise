import tensorflow as tf
import numpy as np
from multitask_model import build_multitask_model


def generate_fake_data(num_samples=100):
    """
    Generates dummy input-output pairs for training the multi-task model.
    Returns tokenized text data and two sets of classification labels.
    """
    sample_sentences = [
        "The movie was great!",
        "I love programming in Python.",
        "This is a boring day.",
        "AI is transforming the future.",
        "The food was terrible.",
        "This is ML_Apprentice take home challage.",
        "Completeing challanges is fun activity.",
        "I made pancakes for breakfast.",
        "He runs every morning before work.",
        "She painted a beautiful sunset.",
        "They watched a documentary last night.",
        "I’m learning how to play chess.",
        "The sunset looked amazing yesterday.",
        "He enjoys hiking in the mountains.",
        "She bought a new pair of shoes.",
        "We visited a museum over the weekend.",
        "I drink coffee every morning.",
        "They adopted a kitten last month.",
        "She writes in her journal every day.",
        "He fixed the broken chair.",
        "I’m planning a trip to Italy.",
        "The flowers are blooming nicely.",
        "The weather is perfect today.",
        "She makes delicious cookies.",
        "I enjoy reading mystery novels.",
        "He plays the guitar really well.",
        "They live in a small town.",
        "We had fun at the beach.",
        "I like listening to jazz music.",
        "The book was fascinating.",
        "My dog loves going for walks.",
        "It’s cold outside today."
    ]
    # Repeat and shuffle to simulate data
    texts = np.random.choice(sample_sentences, size=num_samples)
    
    # Task A: 3-class dummy labels (e.g., category: tech, movie, general)
    labels_task_a = np.random.randint(0, 3, size=(num_samples,))
    
    # Task B: 2-class dummy labels (e.g., sentiment: positive/negative)
    labels_task_b = np.random.randint(0, 2, size=(num_samples,))
    
    return texts, labels_task_a, labels_task_b


def prepare_dataset(texts, labels_a, labels_b, batch_size=11):
    """
    Prepares a TensorFlow dataset with proper formatting for multi-output models.
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        texts,
        {
            "task_a": tf.one_hot(labels_a, depth=3),
            "task_b": tf.one_hot(labels_b, depth=2)
        }
    ))
    return dataset.shuffle(100).batch(batch_size)


def compile_multitask_model():
    """
    Builds and compiles the multi-task model with appropriate losses and metrics.
    """
    model = build_multitask_model(num_classes_task_a=3, num_classes_task_b=2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss={
            "task_a": tf.keras.losses.CategoricalCrossentropy(),
            "task_b": tf.keras.losses.CategoricalCrossentropy()
        },
        metrics={
            "task_a": tf.keras.metrics.CategoricalAccuracy(),
            "task_b": tf.keras.metrics.CategoricalAccuracy()
        }
    )
    return model


def train_model():
    print("Generating synthetic dataset...")
    texts, labels_a, labels_b = generate_fake_data(num_samples=100)

    print("Preparing dataset...")
    dataset = prepare_dataset(texts, labels_a, labels_b)

    print("Building and compiling model...")
    model = compile_multitask_model()

    print("Training model (simulated)...")
    model.fit(dataset, epochs=3)

    print("Training complete!")


if __name__ == "__main__":
    train_model()
