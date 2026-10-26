from __future__ import annotations

import random
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Classifieur linéaire volontairement simple pour MNIST (784 -> 10)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(-1, 28 * 28))


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [v.detach().cpu().numpy().copy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: Iterable[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    arrays = list(parameters)
    if len(keys) != len(arrays):
        raise ValueError(f"Nombre de paramètres incompatible: {len(arrays)} reçu(s), {len(keys)} attendu(s).")
    state_dict = OrderedDict(
        (key, torch.tensor(array, dtype=model.state_dict()[key].dtype))
        for key, array in zip(keys, arrays)
    )
    model.load_state_dict(state_dict, strict=True)


def train_local(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 1,
    lr: float = 0.1,
    device: torch.device | str = "cpu",
) -> float:
    if epochs < 1:
        raise ValueError("epochs doit être >= 1")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    total_loss = 0.0
    total_examples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            n = y.size(0)
            total_loss += float(loss.item()) * n
            total_examples += n

    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        n = y.size(0)
        total_loss += float(loss.item()) * n
        total += n
        correct += int((out.argmax(dim=1) == y).sum().item())

    return total_loss / max(1, total), correct / max(1, total)
