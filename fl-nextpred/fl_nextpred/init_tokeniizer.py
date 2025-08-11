import os
import json
import pickle
from utils.path_finder import read_data
from model import create_ngram_sequences

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "D:\\Lakehead\\Semester4\\Cloud_Computing\\Federated_Learning\\dataset"
TOKENIZER_PATH = "model\\tokenizer.pkl"
CONFIG_PATH = "configs\\config.json"
MAX_SEQ_LENGTH = 20

# -----------------------------
# 1. Load raw data
# -----------------------------
print("[INIT] Reading dataset...")
raw_text = read_data(DATASET_PATH)

# -----------------------------
# 2. Create tokenizer and sequences
# -----------------------------
print("[INIT] Creating tokenizer and sequences...")
sequences, tokenizer = create_ngram_sequences(raw_text, max_len=MAX_SEQ_LENGTH)
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token

# -----------------------------
# 3. Save tokenizer
# -----------------------------
print(f"[INIT] Saving tokenizer to {TOKENIZER_PATH}")
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

# -----------------------------
# 4. Save config
# -----------------------------
print(f"[INIT] Saving config to {CONFIG_PATH}")
config = {
    "vocab_size": vocab_size,
    "max_len": MAX_SEQ_LENGTH
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

# -----------------------------
# Done
# -----------------------------
print(f"[INIT] Tokenizer and config saved.")
print(f"[INIT] vocab_size = {vocab_size}, max_len = {MAX_SEQ_LENGTH}")
