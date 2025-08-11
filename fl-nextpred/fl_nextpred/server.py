import os
import sys
import json
import flwr as fl
import tensorflow as tf
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, start_server
from flwr.server.client_manager import SimpleClientManager
from flwr.common import parameters_to_ndarrays
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout

from model import build_transformer_model, create_ngram_sequences

# -------------------------
# GPU CONFIGURATION
# -------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU found, training will use CPU.")

# -------------------------
# TRANSFORMER MODEL
# -------------------------
def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, dropout):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# -------------------------
# FLOWER STRATEGY TO SAVE MODEL
# -------------------------
class SaveModelStrategy(FedAvg):
    def __init__(self, save_path="final_model.h5", **kwargs):
        with open("configs\\config.json", "r") as f:
            config = json.load(f)
        self.save_path = save_path
        self.vocab_size = config["vocab_size"]
        self.max_len = config["max_len"]
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures):  # type: ignore
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            print(f"[INFO] Saving model after round {rnd} to {self.save_path}")

            # Convert Parameters to list of NumPy arrays
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)

            # Build and save model
            model = build_transformer_model(
                vocab_size=self.vocab_size,
                max_len=self.max_len,
                embed_dim=128,
                num_heads=4,
                ff_dim=128,
                num_blocks=2,
                dropout=0.1,
            )
            model(tf.zeros((1, self.max_len), dtype=tf.int32))  # Build model
            model.set_weights(aggregated_weights)
            model.save(self.save_path)
        return aggregated_parameters, metrics
# -------------------------
# FLOWER SERVER
# -------------------------
def start_flower_server():
    strategy = SaveModelStrategy(
        save_path="model\\final_model.h5",
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=1)  # change as needed
    client_manager = SimpleClientManager()
    server = fl.server.Server(strategy=strategy, client_manager=client_manager)
    start_server(server=server, config=config)

# -------------------------
# MAIN ENTRYPOINT
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        start_flower_server()
    else:
        print("Run with: python script.py server")