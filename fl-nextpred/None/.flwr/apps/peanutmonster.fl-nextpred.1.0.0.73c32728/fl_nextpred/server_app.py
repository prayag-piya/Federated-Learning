from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .utils.path_finder import read_data
from fl_nextpred.model import create_ngram_sequences, load_model

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context):
    dataset = "D:\\Lakehead\\Semester4\\Cloud_Computing\\Federated_Learning\\dataset"
    content = read_data(dataset)
    _, tokenizer = create_ngram_sequences(content)
    vocab_size = len(tokenizer.word_index) + 1
    max_lenght = 20
    
    parameters = ndarrays_to_parameters(load_model(vocab_size, max_lenght).get_weights())

    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"], # type:ignore
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)