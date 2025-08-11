import sys
import logging
import json
import pickle
import tensorflow as tf
from typing import Tuple, Dict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flwr.client import NumPyClient, start_numpy_client
from flwr.common.typing import NDArrays, Scalar

from model import build_transformer_model, create_ngram_sequences
from utils.path_finder import read_data, load_dataset

logging.basicConfig(level=logging.INFO)

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[TensorFlow] GPU(s) available: {gpus}")
    except RuntimeError as e:
        print(f"[TensorFlow] GPU memory growth setup failed: {e}")
else:
    print("[TensorFlow] No GPU detected. Using CPU.")

# Constants
MAX_SEQ_LENGTH = 20

class NextWordPrediction(NumPyClient):
    def __init__(self, model, X_train, y_train, epochs, batch_size, verbose):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def get_parameters(self, config) -> NDArrays:
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.model.built:
            self.model(tf.zeros((1, self.X_train.shape[1]), dtype=tf.int32))
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        if not self.model.built:
            self.model(tf.zeros((1, self.X_train.shape[1]), dtype=tf.int32))
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return float(loss), len(self.X_train), {"accuracy": float(acc)}

def get_client() -> NextWordPrediction:
    dataset_path = "D:\\Lakehead\\Semester4\\Cloud_Computing\\Federated_Learning\\dataset"
    raw_text = read_data(dataset_path)

    # Load config.json
    with open("configs\\config.json", "r") as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    max_len = config["max_len"]

    # Load saved tokenizer instead of recreating
    with open("model\\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Now generate sequences using existing tokenizer
    # Modify create_ngram_sequences to accept tokenizer as an optional parameter
    sequences = create_ngram_sequences(raw_text, tokenizer=tokenizer, max_len=max_len)[0]

    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='pre')

    X_train, y_train = load_dataset(padded_sequences)
    input_len = X_train.shape[1]

    model = build_transformer_model(
        vocab_size=vocab_size,
        max_len=input_len,
        embed_dim=128,
        num_heads=4,
        ff_dim=128,
        num_blocks=2,
        dropout=0.1
    )
    model(tf.zeros((1, input_len), dtype=tf.int32))  # Build model

    return NextWordPrediction(model, X_train, y_train, epochs=1, batch_size=64, verbose=1)

def start_flower_client(server_address: str):
    client = get_client()
    print(f"[Client] Connecting to server at {server_address}...")
    start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() != "client":
        print("Usage: python script.py client <server_address>")
        sys.exit(1)

    server_addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:8080"
    start_flower_client(server_addr)