import json
from typing import Tuple

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar

from fl_nextpred.model import load_model, create_ngram_sequences
from .utils.path_finder import read_data, load_dataset
import os
import logging

logging.basicConfig(level=logging.DEBUG)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class NextWordPrediction(NumPyClient):
    def __init__(self, model, X_train, y_train, epochs, batch_size, verbose, learning_rate):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.learning_rate = learning_rate
    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict[str, Scalar]]:
        try:
            print("[INFO] Starting local training on client")
            self.model.set_weights(parameters)
            history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose if self.verbose is not None else 1
            )
            new_weights = self.model.get_weights()
            num_examples = len(self.X_train)
            metrics = {"loss": history.history["loss"][-1]}
            print("[INFO] Finished local training on client")
            return new_weights, num_examples, metrics
        except Exception as e:
            print(f"[ERROR] during local training: {e}")
            return self.model.get_weights(), 0, {"error": 1.0}

    def get_parameters(self, config: dict) -> NDArrays:
        return self.model.get_weights()

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict[str, Scalar]]:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"accuracy": accuracy}

def client_fn(context: Context):
    print("[DEBUG] Client function started")
    
    # Hyperparameters
    try:
        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose")
        learning_rate = context.run_config["learning-rate"]

        # Dataset/model loading
        print("[DEBUG] Loading dataset")
        dataset = "D:\\Lakehead\\Semester4\\Cloud_Computing\\Federated_Learning\\dataset"
        content = read_data(dataset)
        sequences, tokenizer = create_ngram_sequences(content)
        vocab_size = len(tokenizer.word_index) + 1
        max_lenght = 20
        model = load_model(vocab_size, max_lenght)
        X_train, y_train = load_dataset(sequences)

        print("[DEBUG] Model and data loaded")
        return NextWordPrediction(
            model, X_train, y_train, epochs, batch_size, verbose, learning_rate
        ).to_client()
    except Exception as e:
        print(f"[ERROR] Exception inside client_fn: {e}")
        raise




app = ClientApp(client_fn=client_fn)