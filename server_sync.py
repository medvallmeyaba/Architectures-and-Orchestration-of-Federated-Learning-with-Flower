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
from flwr.common import ndarrays_to_parameters

from common.data import load_global_test_data
from common.io_utils import append_csv, utc_timestamp
from common.model import Net, evaluate, get_parameters, seed_all, set_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serveur Flower synchrone")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--rounds", type=int,default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--log", type=str, default=str(ROOT / "logs" / "logs_sync.csv"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def weighted_train_loss(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"train_loss": 0.0}
    return {
        "train_loss": sum(n * float(m.get("train_loss", 0.0)) for n, m in metrics) / total
    }


class TimedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_start: dict[int, float] = {}
        self.round_duration: dict[int, float] = {}
        self.round_num_clients: dict[int, int] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start[server_round] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        self.round_duration[server_round] = time.perf_counter() - self.round_start.get(
            server_round, time.perf_counter()
        )
        self.round_num_clients[server_round] = len(results)
        return aggregated


if __name__ == "__main__":
    args = parse_args()
    if args.num_clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        raise SystemExit("num-clients, rounds et local-epochs doivent être >= 1")

    seed_all(args.seed)
    testloader = load_global_test_data(
        data_dir=args.data_dir,
        max_samples=args.eval_limit,
        download=False,
    )
    model = Net().to(torch.device("cpu"))
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    strategy = None

    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate(model, testloader)
        if server_round > 0 and strategy is not None:
            append_csv(
                args.log,
                ["round", "loss", "accuracy", "round_duration_s", "timestamp", "num_clients"],
                {
                    "round": server_round,
                    "loss": f"{loss:.8f}",
                    "accuracy": f"{acc:.8f}",
                    "round_duration_s": f"{strategy.round_duration.get(server_round, float('nan')):.6f}",
                    "timestamp": utc_timestamp(),
                    "num_clients": strategy.round_num_clients.get(server_round, 0),
                },
            )
            print(
                f"[SYNC] round={server_round} loss={loss:.4f} acc={acc:.4f} "
                f"duration={strategy.round_duration.get(server_round, 0.0):.2f}s"
            )
        return float(loss), {"accuracy": float(acc)}

    strategy = TimedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs, "lr": args.lr},
        fit_metrics_aggregation_fn=weighted_train_loss,
        evaluate_fn=evaluate_fn,
    )

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).unlink(missing_ok=True)
    address = f"{args.host}:{args.port}"
    print(f"Serveur synchrone sur {address}, {args.num_clients} clients, {args.rounds} rounds")
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
