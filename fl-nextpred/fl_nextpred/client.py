# client_pytorch.py
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from flwr.client import NumPyClient, start_numpy_client
from typing import Tuple, Dict
from model import NextWordTransformer
from utils.path_finder import read_data, load_dataset  # Your existing data utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flwr.common.typing import NDArrays, Scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Pad/truncate sequences to max_len
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='pre')

    # Split into features and labels
    X_np = sequences[:, :-1]
    y_np = sequences[:, -1]

    # Convert to PyTorch tensors
    X = torch.tensor(X_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.long)

    return X, y, tokenizer


class NextWordPredictionClient(NumPyClient):
    def __init__(self, model, train_loader, epochs=20, verbose=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.epochs = epochs
        self.verbose = verbose
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Set model parameters from numpy arrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(1, self.epochs + 1):
            print(f"[Client] Training epoch {epoch}/{self.epochs}")
            
            batch_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for X_batch, y_batch in batch_iter:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                batch_iter.set_postfix(loss=loss.item())
        
        # Save model after training
        torch.save(self.model.state_dict(), f"client_model_epoch{self.epochs}.pt")
        print(f"[Client] Model saved to client_model_epoch{self.epochs}.pt")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss_sum += loss.item() * X_batch.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += X_batch.size(0)
        loss_avg = loss_sum / total
        accuracy = correct / total
        return loss_avg, total, {"accuracy": accuracy}

def start_client(server_address: str):
    dataset_path = "D:\\Lakehead\\Semester4\\Cloud_Computing\\Federated_Learning\\dataset"
    raw_text = read_data(dataset_path)

    with open("configs\\config.json", "r") as f:
        config = json.load(f)
    vocab_size = config["vocab_size"]
    max_len = config["max_len"]

    with open("model\\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    sequences = create_ngram_sequences(raw_text, tokenizer=tokenizer, max_len=max_len)[0]
    padded_sequences = torch.tensor(sequences, dtype=torch.long)
    X_train, y_train = load_dataset(padded_sequences.numpy())  # Assuming this returns np arrays

    # Convert to tensors and dataloader
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NextWordTransformer(vocab_size=vocab_size, max_len=max_len, embed_dim=128, num_heads=4, ff_dim=128, num_blocks=2, dropout=0.1)

    client = NextWordPredictionClient(model=model, train_loader=train_loader, epochs=30)
    start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() != "client":
        print("Usage: python client_pytorch.py client <server_address>")
        sys.exit(1)
    server_addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:8080"
    start_client(server_addr)
