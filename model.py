# model.py
"""
A minimal PyTorch model and training helpers for Flower federated learning.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- #
#   1. Model Definition
# --------------------------------------------------------------------------- #
class Net(nn.Module):
    """
    Very small CNN suitable for MNIST / Fashion‑MNIST.
    Architecture:
        Conv(1->32) -> ReLU -> Conv(32->64) -> ReLU -> MaxPool
        Flatten
        Linear(64*7*7 -> 128) -> ReLU
        Linear(128 -> 10)
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                              # 28x28 -> 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))          # [N, 32, 28, 28]
        x = F.relu(self.conv2(x))          # [N, 64, 28, 28]
        x = self.pool(x)                   # [N, 64, 14, 14]
        x = torch.flatten(x, 1)            # [N, 64*14*14]
        x = F.relu(self.fc1(x))            # [N, 128]
        x = self.fc2(x)                    # [N, 10]
        return x

# --------------------------------------------------------------------------- #
#   2. Dataset / DataLoader helpers
# --------------------------------------------------------------------------- #
def get_dataloaders(
    batch_size: int = 32,
    train_split: float = 0.8,
    root: str | None = None,
    download: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns training and validation DataLoaders for MNIST.
    - `train_split`: fraction of the training set used for training.
    - `root`: directory for dataset storage (defaults to ~/.torch/).
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    torch.manual_seed(seed)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    # Randomly split the training set
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# --------------------------------------------------------------------------- #
#   3. Training / Evaluation functions
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.
    Returns the average training loss.
    """
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model.
    Returns (average loss, accuracy).
    """
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return loss, accuracy

# --------------------------------------------------------------------------- #
#   4. Convenience factory for the client
# --------------------------------------------------------------------------- #
def build_model_and_optimizer(
    lr: float = 0.01,
    weight_decay: float = 0.0,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, optim.Optimizer]:
    """
    Returns a freshly‑initialized model and Adam optimizer.
    """
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer

# --------------------------------------------------------------------------- #
#   5. CLI demo (optional)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo training loop for Flower client.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, optimizer = build_model_and_optimizer(lr=args.lr, device=device)
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4%}")