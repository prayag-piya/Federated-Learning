import tensorflow as tf
import keras
from keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore


def create_ngram_sequences(text, tokenizer=None, max_len=20):
    sentences = text.lower().split('\n')

    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)

    sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[:i+1]
            sequences.append(n_gram_seq)

    return sequences, tokenizer


def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = Add()([attention, inputs])
    attention = LayerNormalization(epsilon=1e-6)(attention)

    # Feed-forward layer
    ff = Dense(ff_dim, activation="relu")(attention)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Add()([ff, attention])
    ff = LayerNormalization(epsilon=1e-6)(ff)
    return ff

def build_transformer_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout: float = 0.1
) -> Model:
    inputs = Input(shape=(max_len,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    for _ in range(num_blocks):
        x = transformer_block(x, head_size=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )
    return model




# def predict_next(model, tokenizer, seed_text, max_len=20):
#     token_list = tokenizer.texts_to_sequences([seed_text])[0]
#     token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
#     predicted_logits = model(token_list, training=False)
#     predicted_id = tf.argmax(predicted_logits[:, -1, :], axis=-1).numpy()[0]
#     return tokenizer.index_word.get(predicted_id, '<UNK>')
