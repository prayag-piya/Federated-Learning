
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore


import torch
import torch.nn as nn
import torch.nn.functional as F

def create_ngram_sequences(text, tokenizer=None, max_len=20):
    sentences = text.lower().split('\n')

    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(sentences)
    
    sequences = []
    for line in sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        # Create n-gram sequences
        for i in range(1, len(token_list)):
            n_gram_seq = token_list[:i+1]
            sequences.append(n_gram_seq)
    
    # Pad/truncate sequences
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='pre')

    # Split into features and labels
    X = sequences[:, :-1]
    y = sequences[:, -1]

    return X, y, tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class NextWordTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim=128, num_heads=4, ff_dim=256, num_blocks=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(max_len, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.pos_encoding(x)  # same shape

        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim) for nn.MultiheadAttention
        for block in self.transformer_blocks:
            x = block(x)
        x = x.transpose(0, 1)  # back to (batch, seq_len, embed_dim)

        x = x.mean(dim=1)  # GlobalAveragePooling over sequence length
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    
# def predict_next(model, tokenizer, seed_text, max_len=20):
#     token_list = tokenizer.texts_to_sequences([seed_text])[0]
#     token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
#     predicted_logits = model(token_list, training=False)
#     predicted_id = tf.argmax(predicted_logits[:, -1, :], axis=-1).numpy()[0]
#     return tokenizer.index_word.get(predicted_id, '<UNK>')
