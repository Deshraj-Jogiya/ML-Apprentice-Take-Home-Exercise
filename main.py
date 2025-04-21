import tensorflow as tf
from models.sentence_transformer import load_bert_layers, build_sentence_transformer


def display_sample_embeddings(model):
    """
    Feeds example sentences to the sentence transformer model
    and prints out the resulting embeddings.
    """
    sample_sentences = [
        "Machine learning is fun.",
        "Transformers are powerful models for NLP.",
        "TensorFlow makes model building easy.",
        "This is ML_Apprentice take home challage.",
        "Completeing challanges is fun activity."
    ]

    embeddings = model(tf.constant(sample_sentences))

    print("Embedding Shape:", embeddings.shape)
    for i, sentence in enumerate(sample_sentences):
        print(f"\nSentence {i+1}: {sentence}")
        print(f"Embedding (first 5 dimensions): {embeddings[i].numpy()[:5]}")


def main():
    print(" Loading BERT layers...")
    preprocessor, encoder = load_bert_layers()

    print(" Building sentence transformer model...")
    sentence_model = build_sentence_transformer(preprocessor, encoder)

    print(" Running test inference on sample sentences...")
    display_sample_embeddings(sentence_model)


if __name__ == "__main__":
    main()
