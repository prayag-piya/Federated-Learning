import json
import flwr as fl
import torch
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, start_server
from flwr.server.client_manager import SimpleClientManager
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from model import NextWordTransformer

class SaveModelStrategy(FedAvg):
    def __init__(self, save_path="final_model.pkl", **kwargs):
        with open("configs/config.json", "r") as f:
            config = json.load(f)
        self.save_path = save_path
        self.vocab_size = config["vocab_size"]
        self.max_len = config["max_len"]
        super().__init__(**kwargs)

    def aggregate_fit(self, rnd, results, failures): #type: ignore
        aggregated_parameters, metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            print(f"[INFO] Saving model after round {rnd} to {self.save_path}")

            # Convert Parameters to list of numpy arrays
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters)

            # Load model and set weights
            model = NextWordTransformer(
                vocab_size=self.vocab_size,
                max_len=self.max_len,
                embed_dim=128,
                num_heads=4,
                ff_dim=128,
                num_blocks=2,
                dropout=0.1
            )
            # Convert numpy weights to PyTorch tensors and assign
            state_dict = model.state_dict()
            new_state_dict = {}
            for key, value in zip(state_dict.keys(), aggregated_weights):
                new_state_dict[key] = torch.tensor(value)
            model.load_state_dict(new_state_dict)

            # Save model weights
            torch.save(model.state_dict(), self.save_path)
        return aggregated_parameters, metrics

def start_flower_server():
    strategy = SaveModelStrategy(
        save_path="model/final_model.pkl",
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=3)
    client_manager = SimpleClientManager()
    server = fl.server.Server(strategy=strategy, client_manager=client_manager)
    start_server(server=server, config=config)

if __name__ == "__main__":
    start_flower_server()