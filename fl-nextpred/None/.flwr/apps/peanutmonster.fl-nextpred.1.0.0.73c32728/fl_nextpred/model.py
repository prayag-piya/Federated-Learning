import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def create_ngram_sequences(text: str, vocab_size=10000, max_len=20):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    sentences = text.lower().split('\n')
    tokenizer.fit_on_texts(sentences)

    sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[:i+1]
            sequences.append(n_gram_seq)

    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
    return sequences, tokenizer


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        # Create look-ahead mask
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)  # Lower triangle
        attn_output = self.att(x, x, x, attention_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        return x


class TransformerLM(keras.Model):
    def __init__(self, vocab_size, max_len, d_model=128, num_heads=4, dff=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_len, d_model)
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout) for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout)
        self.final_layer = layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model
        )

        # Apply sin to even indices in the array (0, 2, 4, ...)
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Apply cos to odd indices in the array (1, 3, 5, ...)
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Interleave sines and cosines
        pos_encoding = tf.concat([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[tf.newaxis, ...]  # Add batch dimension
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.dropout(x, training=training)
        return self.final_layer(x)


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    sequences, tokenizer = create_ngram_sequences(text)
    X = sequences[:, :-1]
    y = sequences[:, 1:]
    x_train, y_train, x_test, y_test = train_test_split(X, y, random_state=42)
    return x_train, y_train, x_test, y_test, tokenizer

def load_model(vocab_size, max_len):
    model = TransformerLM(vocab_size=vocab_size, max_len=max_len)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


# def predict_next(model, tokenizer, seed_text, max_len=20):
#     token_list = tokenizer.texts_to_sequences([seed_text])[0]
#     token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
#     predicted_logits = model(token_list, training=False)
#     predicted_id = tf.argmax(predicted_logits[:, -1, :], axis=-1).numpy()[0]
#     return tokenizer.index_word.get(predicted_id, '<UNK>')
