import tensorflow as tf
from sentence_transformer import load_bert_layers, build_sentence_transformer

def build_multitask_model(num_classes_task_a=3, num_classes_task_b=2):
    """
    Builds a multi-task model with a shared sentence transformer backbone
    and two task-specific classification heads.

    Args:
        num_classes_task_a (int): Number of output classes for Task A (e.g., sentence classification).
        num_classes_task_b (int): Number of output classes for Task B (e.g., sentiment analysis).

    Returns:
        tf.keras.Model: Multi-task learning model.
    """
    # Load shared preprocessor and encoder
    preprocessor, encoder = load_bert_layers()

    # Get the base embedding model
    base_model = build_sentence_transformer(preprocessor, encoder)

    # Shared input
    input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name="input_text")

    # Shared embedding
    shared_embedding = base_model(input_text)

    # Task A: Sentence Classification head
    task_a_output = tf.keras.layers.Dense(128, activation="relu", name="task_a_dense")(shared_embedding)
    task_a_output = tf.keras.layers.Dense(num_classes_task_a, activation="softmax", name="task_a_classifier")(task_a_output)

    # Task B: Sentiment Analysis (or similar) head
    task_b_output = tf.keras.layers.Dense(128, activation="relu", name="task_b_dense")(shared_embedding)
    task_b_output = tf.keras.layers.Dense(num_classes_task_b, activation="softmax", name="task_b_classifier")(task_b_output)

    # Final model with two outputs
    multitask_model = tf.keras.Model(inputs=input_text, outputs={
        "task_a": task_a_output,
        "task_b": task_b_output
    }, name="multitask_sentence_transformer")

    return multitask_model

if __name__ == "__main__":
    model = build_multitask_model()
    model.summary()  
