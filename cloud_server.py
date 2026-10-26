from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.data import load_global_test_data
from common.io_utils import (
    append_csv,
    average_parameters,
    load_parameter_bundle,
    save_parameter_bundle,
    utc_timestamp,
)
from common.model import Net, evaluate, set_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agrégation Cloud des modèles Edge")
    p.add_argument("--num-edges", type=int, default=2)
    p.add_argument("--edge-round", type=int, default=5)
    p.add_argument("--weighting", choices=["uniform", "examples"], default="uniform")
    p.add_argument("--params-dir", type=str, default=str(ROOT / "params"))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--log", type=str, default=str(ROOT / "logs" / "cloud_results.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_edges < 1 or args.edge_round < 1:
        raise SystemExit("num-edges et edge-round doivent être >= 1")

    params_dir = Path(args.params_dir)
    keys_ref = None
    all_params = []
    example_weights = []
    edge_examples = []

    for eid in range(args.num_edges):
        base = params_dir / f"edge_{eid}_round{args.edge_round}"
        keys, params, meta = load_parameter_bundle(base)
        if keys_ref is None:
            keys_ref = keys
        elif keys != keys_ref:
            raise RuntimeError(f"Ordre des paramètres incompatible pour Edge {eid}")
        n = int(meta.get("num_examples", 0))
        if args.weighting == "examples" and n <= 0:
            raise RuntimeError(
                f"Métadonnée num_examples absente/invalide pour Edge {eid}; "
                "relancer edge_server.py avec le pack corrigé."
            )
        all_params.append(params)
        edge_examples.append(n)
        example_weights.append(float(n) if args.weighting == "examples" else 1.0)

    averaged = average_parameters(all_params, example_weights)
    out_name = args.out or f"global_round{args.edge_round}_{args.weighting}"
    out_base = params_dir / out_name
    save_parameter_bundle(
        out_base,
        averaged,
        keys_ref or [],
        {
            "edge_round": args.edge_round,
            "weighting": args.weighting,
            "num_edges": args.num_edges,
            "edge_examples": edge_examples,
            "total_examples": sum(edge_examples),
            "timestamp": utc_timestamp(),
        },
    )

    model = Net().to(torch.device("cpu"))
    set_parameters(model, averaged)
    testloader = load_global_test_data(
        data_dir=args.data_dir,
        max_samples=args.eval_limit,
        download=False,
    )
    loss, acc = evaluate(model, testloader)
    append_csv(
        args.log,
        ["edge_round", "weighting", "loss", "accuracy", "num_edges", "total_examples", "timestamp"],
        {
            "edge_round": args.edge_round,
            "weighting": args.weighting,
            "loss": f"{loss:.8f}",
            "accuracy": f"{acc:.8f}",
            "num_edges": args.num_edges,
            "total_examples": sum(edge_examples),
            "timestamp": utc_timestamp(),
        },
    )
    print(
        f"Cloud {args.weighting}: loss={loss:.4f} acc={acc:.4f} | "
        f"edge_examples={edge_examples} | sauvegarde={out_base}.npz"
    )
