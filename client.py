# client.py
import flwr as fl
import torch
from model import build_model_and_optimizer, get_dataloaders, train_one_epoch, evaluate

class MnistClient(fl.client.NumPyClient):
    def __init__(self, lr=0.01, device="cpu"):
        self.device = torch.device(device)
        self.model, self.optimizer = build_model_and_optimizer(lr=lr, device=self.device)
        self.train_loader, self.val_loader = get_dataloaders(batch_size=32, device=self.device)

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        # Load parameters into the local model
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(w).to(self.device)

        # Train for one local epoch (you can tune `config["local_epochs"]` if you wish)
        epoch_loss = train_one_epoch(
            self.model, self.train_loader, self.optimizer, self.device, epoch=1
        )

        # Return updated parameters and metadata
        return [p.cpu().numpy() for p in self.model.parameters()], len(self.train_loader), {"loss": epoch_loss}

    def evaluate(self, parameters, config):
        # Load parameters into the local model
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(w).to(self.device)

        val_loss, val_acc = evaluate(self.model, self.val_loader, self.device)
        return float(val_loss), len(self.val_loader.dataset), {"accuracy": val_acc}

if __name__ == "__main__":
    client = MnistClient(lr=0.01, device="cpu")
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)