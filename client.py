from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flwr as fl
import torch

from common.data import load_client_data
from common.model import Net, evaluate, get_parameters, seed_all, set_parameters, train_local


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Client Flower classique (sync ou async)")
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--server", type=str, default="127.0.0.1:8080")
    p.add_argument("--delay", type=float, default=0.0, help="Retard artificiel appliqué avant l'entraînement local")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        seed_all(args.seed + args.client_id)
        self.device = torch.device("cpu")
        self.model = Net().to(self.device)
        self.trainloader, self.testloader = load_client_data(
            partition_id=args.client_id,
            num_partitions=args.num_clients,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
            seed=args.seed,
            download=False,
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        if self.args.delay > 0:
            time.sleep(self.args.delay)
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", self.args.lr))
        train_loss = train_local(
            self.model,
            self.trainloader,
            epochs=local_epochs,
            lr=lr,
            device=self.device,
        )
        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "client_id": int(self.args.client_id),
                "delay_s": float(self.args.delay),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


if __name__ == "__main__":
    args = parse_args()
    if args.client_id < 0 or args.client_id >= args.num_clients:
        raise SystemExit("--client-id doit être compris entre 0 et num-clients-1")
    print(
        f"Client {args.client_id}/{args.num_clients} -> {args.server} | "
        f"delay={args.delay}s | max_train_samples={args.max_train_samples}"
    )
    fl.client.start_client(server_address=args.server, client=FlowerClient(args).to_client())
