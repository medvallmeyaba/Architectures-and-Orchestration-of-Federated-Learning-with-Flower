from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flwr as fl
import torch
from flwr.common import Code, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import History, Server, SimpleClientManager

from common.data import load_global_test_data
from common.io_utils import append_csv, blend_parameters, utc_timestamp
from common.model import Net, evaluate, get_parameters, seed_all, set_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serveur Flower asynchrone pédagogique")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--max-updates", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=None, help="Poids de mélange; défaut=1/num_clients")
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--log", type=str, default=str(ROOT / "logs" / "logs_async.csv"))
    p.add_argument("--wait-clients", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class PedagogicalAsyncServer(Server):
    """Asynchronisme simplifié: une mise à jour reçue est appliquée immédiatement.

    Chaque client a au plus une requête fit en cours. Lorsqu'il termine, sa mise à jour
    est mélangée au modèle global, puis le même client reçoit le modèle global courant.
    Les clients rapides peuvent donc contribuer plus souvent. Cette classe est destinée
    à comparer l'orchestration, pas à reproduire un algorithme FedAsync de recherche.
    """

    def __init__(
        self,
        *,
        client_manager,
        initial_parameters,
        num_clients: int,
        max_updates: int,
        local_epochs: int,
        lr: float,
        alpha: float,
        evaluate_fn,
        log_path: str,
        wait_clients: int,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=fl.server.strategy.FedAvg())
        self.parameters = initial_parameters
        self.num_clients = num_clients
        self.max_updates = max_updates
        self.local_epochs = local_epochs
        self.lr = lr
        self.alpha = alpha
        self.evaluate_fn = evaluate_fn
        self.log_path = log_path
        self.wait_clients = wait_clients

    @staticmethod
    def _fit_one(client, ins, timeout):
        try :
            return client, client.fit(ins, timeout=timeout, group_id=0)
        except TypeError:
                return client, client.fit(ins, timeout=timeout)

    def fit(self, num_rounds: int, timeout: float | None):
        del num_rounds  # l'arrêt est défini par max_updates
        history = History()
        if not self._client_manager.wait_for(self.num_clients, timeout=self.wait_clients):
            raise RuntimeError(
                f"Nombre de clients insuffisant: {self._client_manager.num_available()}/{self.num_clients}"
            )

        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_path).unlink(missing_ok=True)
        clients = list(self._client_manager.all().values())[: self.num_clients]
        start = time.perf_counter()
        accepted = 0
        version = 0

        def make_ins(base_version: int):
            return FitIns(
                self.parameters,
                {"local_epochs": self.local_epochs, "lr": self.lr, "base_version": base_version},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            pending: dict[concurrent.futures.Future, tuple[object, int, float]] = {}
            for client in clients:
                fut = executor.submit(self._fit_one, client, make_ins(version), timeout)
                pending[fut] = (client, version, time.perf_counter())

            while accepted < self.max_updates:
                if not pending:
                    raise RuntimeError("Aucun client actif alors que le budget de mises à jour n'est pas atteint")
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    client, base_version, sent_at = pending.pop(fut)
                    try:
                        _, fit_res = fut.result()
                    except Exception as exc:
                        print(f"[ASYNC] échec client {getattr(client, 'cid', '?')}: {exc}")
                        fit_res = None

                    if fit_res is not None and fit_res.status.code == Code.OK:
                        g = parameters_to_ndarrays(self.parameters)
                        c = parameters_to_ndarrays(fit_res.parameters)
                        staleness = max(0, version - base_version)
                        # Mélange constant volontairement simple; la staleness est mesurée, pas corrigée.
                        self.parameters = ndarrays_to_parameters(blend_parameters(g, c, self.alpha))
                        version += 1
                        accepted += 1
                        loss, metrics = self.evaluate_fn(accepted, parameters_to_ndarrays(self.parameters), {})
                        elapsed = time.perf_counter() - start
                        update_duration = time.perf_counter() - sent_at
                        client_id = fit_res.metrics.get("client_id", getattr(client, "cid", "?"))
                        append_csv(
                            self.log_path,
                            [
                                "update", "client", "loss", "accuracy", "elapsed_s",
                                "update_duration_s", "staleness", "num_examples", "timestamp",
                            ],
                            {
                                "update": accepted,
                                "client": client_id,
                                "loss": f"{float(loss):.8f}",
                                "accuracy": f"{float(metrics.get('accuracy', 0.0)):.8f}",
                                "elapsed_s": f"{elapsed:.6f}",
                                "update_duration_s": f"{update_duration:.6f}",
                                "staleness": staleness,
                                "num_examples": fit_res.num_examples,
                                "timestamp": utc_timestamp(),
                            },
                        )
                        history.add_loss_centralized(accepted, float(loss))
                        history.add_metrics_centralized(accepted, metrics)
                        print(
                            f"[ASYNC] update={accepted}/{self.max_updates} client={client_id} "
                            f"staleness={staleness} loss={float(loss):.4f} "
                            f"acc={float(metrics.get('accuracy', 0.0)):.4f}"
                        )

                    # Maintenir assez de travaux en vol pour pouvoir atteindre max_updates.
                    if accepted < self.max_updates and accepted + len(pending) < self.max_updates:
                        fut2 = executor.submit(self._fit_one, client, make_ins(version), timeout)
                        pending[fut2] = (client, version, time.perf_counter())

        return history, time.perf_counter() - start


if __name__ == "__main__":
    args = parse_args()
    if args.num_clients < 1 or args.max_updates < 1 or args.local_epochs < 1:
        raise SystemExit("num-clients, max-updates et local-epochs doivent être >= 1")
    alpha = args.alpha if args.alpha is not None else 1.0 / args.num_clients
    if not 0.0 < alpha <= 1.0:
        raise SystemExit("alpha doit appartenir à ]0,1]")

    seed_all(args.seed)
    model = Net().to(torch.device("cpu"))
    testloader = load_global_test_data(
        data_dir=args.data_dir,
        max_samples=args.eval_limit,
        download=False,
    )

    def evaluate_fn(step, parameters, config):
        set_parameters(model, parameters)
        loss, acc = evaluate(model, testloader)
        return float(loss), {"accuracy": float(acc)}

    manager = SimpleClientManager()
    server = PedagogicalAsyncServer(
        client_manager=manager,
        initial_parameters=ndarrays_to_parameters(get_parameters(model)),
        num_clients=args.num_clients,
        max_updates=args.max_updates,
        local_epochs=args.local_epochs,
        lr=args.lr,
        alpha=alpha,
        evaluate_fn=evaluate_fn,
        log_path=args.log,
        wait_clients=args.wait_clients,
    )
    address = f"{args.host}:{args.port}"
    print(
        f"Serveur asynchrone pédagogique sur {address} | clients={args.num_clients} "
        f"| updates={args.max_updates} | alpha={alpha:.4f}"
    )
    fl.server.start_server(
        server_address=address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=1),
    )
