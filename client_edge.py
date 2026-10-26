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
    p = argparse.ArgumentParser(description="Client rattaché à un serveur Edge")
    p.add_argument("--edge-id", type=int, required=True)
    p.add_argument("--local-id", type=int, required=True)
    p.add_argument("--num-edges", type=int, default=2)
    p.add_argument("--clients-per-edge", type=int, default=3)
    p.add_argument("--server", type=str, required=True)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class EdgeClient(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace, global_idx: int, total_clients: int) -> None:
        self.args = args
        self.global_idx = global_idx
        seed_all(args.seed + global_idx)
        self.device = torch.device("cpu")
        self.model = Net().to(self.device)
        self.trainloader, self.testloader = load_client_data(
            partition_id=global_idx,
            num_partitions=total_clients,
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
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", self.args.lr))
        train_loss = train_local(self.model, self.trainloader, epochs=epochs, lr=lr)
        return (
            get_parameters(self.model),
            len(self.trainloader.dataset),
            {
                "train_loss": float(train_loss),
                "edge_id": int(self.args.edge_id),
                "local_id": int(self.args.local_id),
                "global_client_id": int(self.global_idx),
                "delay_s": float(self.args.delay),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


if __name__ == "__main__":
    args = parse_args()
    if args.num_edges < 1 or args.clients_per_edge < 1:
        raise SystemExit("num-edges et clients-per-edge doivent être >= 1")
    if not 0 <= args.edge_id < args.num_edges:
        raise SystemExit("edge-id invalide")
    if not 0 <= args.local_id < args.clients_per_edge:
        raise SystemExit("local-id invalide")
    total_clients = args.num_edges * args.clients_per_edge
    global_idx = args.edge_id * args.clients_per_edge + args.local_id
    print(
        f"EdgeClient edge={args.edge_id} local={args.local_id} global={global_idx}/{total_clients} "
        f"-> {args.server} | delay={args.delay}s | max_train_samples={args.max_train_samples}"
    )
    fl.client.start_numpy_client(
        server_address=args.server,
        client=EdgeClient(args, global_idx, total_clients).to_client(),
    )
