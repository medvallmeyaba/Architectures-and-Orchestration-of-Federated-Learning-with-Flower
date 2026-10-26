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
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from common.data import load_global_test_data
from common.io_utils import append_csv, save_parameter_bundle, utc_timestamp
from common.model import Net, evaluate, get_parameters, seed_all, set_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serveur Edge Flower")
    p.add_argument("--edge-id", type=int, required=True)
    p.add_argument("--num-edge-clients", type=int, default=3)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--params-dir", type=str, default=str(ROOT / "params"))
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def weighted_train_loss(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"train_loss": 0.0}
    return {"train_loss": sum(n * float(m.get("train_loss", 0.0)) for n, m in metrics) / total}


class SavingTimedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *, edge_id: int, params_dir: str, model_keys: list[str], **kwargs):
        super().__init__(**kwargs)
        self.edge_id = edge_id
        self.params_dir = Path(params_dir)
        self.model_keys = model_keys
        self.round_start: dict[int, float] = {}
        self.round_duration: dict[int, float] = {}
        self.round_num_clients: dict[int, int] = {}
        self.round_examples: dict[int, int] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start[server_round] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self.round_duration[server_round] = time.perf_counter() - self.round_start.get(
            server_round, time.perf_counter()
        )
        self.round_num_clients[server_round] = len(results)
        total_examples = sum(int(res.num_examples) for _, res in results)
        self.round_examples[server_round] = total_examples
        if aggregated_parameters is not None:
            arrays = parameters_to_ndarrays(aggregated_parameters)
            save_parameter_bundle(
                self.params_dir / f"edge_{self.edge_id}_round{server_round}",
                arrays,
                self.model_keys,
                {
                    "edge_id": self.edge_id,
                    "round": server_round,
                    "num_examples": total_examples,
                    "num_clients": len(results),
                    "round_duration_s": self.round_duration[server_round],
                    "timestamp": utc_timestamp(),
                },
            )
        return aggregated_parameters, metrics


if __name__ == "__main__":
    args = parse_args()
    if args.num_edge_clients < 1 or args.rounds < 1 or args.local_epochs < 1:
        raise SystemExit("num-edge-clients, rounds et local-epochs doivent être >= 1")
    seed_all(args.seed + args.edge_id)
    log_path = args.log or str(ROOT / "logs" / f"edge_{args.edge_id}.csv")
    Path(log_path).unlink(missing_ok=True)

    testloader = load_global_test_data(
        data_dir=args.data_dir,
        max_samples=args.eval_limit,
        download=False,
    )
    model = Net().to(torch.device("cpu"))
    model_keys = list(model.state_dict().keys())
    initial_parameters = ndarrays_to_parameters(get_parameters(model))
    strategy = None

    def evaluate_fn(server_round, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate(model, testloader)
        if server_round > 0 and strategy is not None:
            append_csv(
                log_path,
                ["round", "loss", "accuracy", "round_duration_s", "num_examples", "num_clients", "timestamp"],
                {
                    "round": server_round,
                    "loss": f"{loss:.8f}",
                    "accuracy": f"{acc:.8f}",
                    "round_duration_s": f"{strategy.round_duration.get(server_round, float('nan')):.6f}",
                    "num_examples": strategy.round_examples.get(server_round, 0),
                    "num_clients": strategy.round_num_clients.get(server_round, 0),
                    "timestamp": utc_timestamp(),
                },
            )
            print(
                f"[EDGE {args.edge_id}] round={server_round} loss={loss:.4f} acc={acc:.4f} "
                f"examples={strategy.round_examples.get(server_round, 0)}"
            )
        return float(loss), {"accuracy": float(acc)}

    strategy = SavingTimedFedAvg(
        edge_id=args.edge_id,
        params_dir=args.params_dir,
        model_keys=model_keys,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_edge_clients,
        min_evaluate_clients=0,
        min_available_clients=args.num_edge_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs, "lr": args.lr},
        fit_metrics_aggregation_fn=weighted_train_loss,
        evaluate_fn=evaluate_fn,
    )

    address = f"{args.host}:{args.port}"
    print(
        f"EDGE {args.edge_id} sur {address} | clients={args.num_edge_clients} | rounds={args.rounds}"
    )
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
