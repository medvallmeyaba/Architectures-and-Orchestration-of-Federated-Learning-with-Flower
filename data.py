from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _partition_indices(n: int, partition_id: int, num_partitions: int, seed: int = 42) -> list[int]:
    if num_partitions < 1:
        raise ValueError("num_partitions doit être >= 1")
    if not 0 <= partition_id < num_partitions:
        raise ValueError(f"partition_id={partition_id} hors intervalle [0, {num_partitions - 1}]")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    base, rem = divmod(n, num_partitions)
    sizes = [base + (1 if i < rem else 0) for i in range(num_partitions)]
    start = sum(sizes[:partition_id])
    end = start + sizes[partition_id]
    return perm[start:end]


def prepare_mnist(data_dir: str | Path = "data") -> None:
    data_dir = str(data_dir)
    transform = transforms.ToTensor()
    datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    datasets.MNIST(data_dir, train=False, download=True, transform=transform)


def load_client_data(
    *,
    partition_id: int,
    num_partitions: int,
    data_dir: str | Path = "data",
    batch_size: int = 32,
    test_batch_size: int = 128,
    max_train_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(str(data_dir), train=True, download=download, transform=transform)
    testset = datasets.MNIST(str(data_dir), train=False, download=download, transform=transform)

    train_idx = _partition_indices(len(trainset), partition_id, num_partitions, seed)
    test_idx = _partition_indices(len(testset), partition_id, num_partitions, seed + 1)
    if max_train_samples is not None:
        if max_train_samples < 1:
            raise ValueError("max_train_samples doit être >= 1")
        train_idx = train_idx[: min(max_train_samples, len(train_idx))]

    trainloader = DataLoader(Subset(trainset, train_idx), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(Subset(testset, test_idx), batch_size=test_batch_size, shuffle=False)
    return trainloader, testloader


def load_global_test_data(
    *,
    data_dir: str | Path = "data",
    batch_size: int = 256,
    max_samples: int | None = None,
    download: bool = False,
) -> DataLoader:
    transform = transforms.ToTensor()
    testset = datasets.MNIST(str(data_dir), train=False, download=download, transform=transform)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples doit être >= 1")
        testset = Subset(testset, list(range(min(max_samples, len(testset)))))
    return DataLoader(testset, batch_size=batch_size, shuffle=False)
