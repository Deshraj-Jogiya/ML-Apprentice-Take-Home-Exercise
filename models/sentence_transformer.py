import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 

def load_bert_layers():
    """
    Loads the preprocessing and encoder models from TensorFlow Hub.
    Using Small BERT for a lightweight embedding encoder.
    """
    preprocess_url = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3"
    encoder_url = "https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-10-h-128-a-2/2"


    preprocessor = hub.KerasLayer(preprocess_url, name="text_preprocessor")
    encoder = hub.KerasLayer(encoder_url, trainable=True, name="bert_encoder")

    return preprocessor, encoder

def build_sentence_transformer(preprocessor, encoder):
    """
    Builds the sentence transformer model using BERT.
    
    Args:
        preprocessor: TensorFlow Hub preprocessing layer.
        encoder: TensorFlow Hub BERT encoder layer.

    Returns:
        tf.keras.Model: Sentence embedding model.
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
    encoder_inputs = preprocessor(text_input)
    outputs = encoder(encoder_inputs)

    # Mean pooling across the token embeddings
    token_embeddings = outputs['sequence_output']
    sentence_embedding = tf.reduce_mean(token_embeddings, axis=1)

    # Normalize embeddings
    normalized_embedding = tf.keras.layers.LayerNormalization(name="layer_norm")(sentence_embedding)

    return tf.keras.Model(inputs=text_input, outputs=normalized_embedding, name="sentence_transformer_model")
