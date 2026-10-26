# TP2 — Architectures and Orchestration of Federated Learning with Flower

Files for the **synchronous**, **pedagogical asynchronous**, and **hierarchical Edge–Cloud** experiments.

## 1. Recommended environment

- Python 3.10, 3.11, or 3.12
- A CPU is enough
- Flower `1.8.0`

Installation in a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python selfcheck.py
```

> This pack explicitly targets Flower 1.8.0. Do not automatically install the latest version of Flower for the session.

## 2. Prepare MNIST before the lab

To avoid six simultaneous downloads during the session:

```bash
python prepare_data.py
```

The `data/MNIST` folder is then reused by all local processes.

## 3. Synchronous experiment — 3 clients, 5 rounds

Server terminal:

```bash
python classical/server_sync.py --num-clients 3 --rounds 5 --port 8080 --log logs/sync_homogeneous.csv
```

Three client terminals:

```bash
python classical/client.py --client-id 0 --num-clients 3 --server 127.0.0.1:8080 --delay 0
python classical/client.py --client-id 1 --num-clients 3 --server 127.0.0.1:8080 --delay 0
python classical/client.py --client-id 2 --num-clients 3 --server 127.0.0.1:8080 --delay 0
```

### Straggler

Restart the server with a new log file, then:

```bash
python classical/client.py --client-id 0 --num-clients 3 --server 127.0.0.1:8080 --delay 0
python classical/client.py --client-id 1 --num-clients 3 --server 127.0.0.1:8080 --delay 1
python classical/client.py --client-id 2 --num-clients 3 --server 127.0.0.1:8080 --delay 3
```

## 4. Asynchronous experiment — comparable budget

The synchronous protocol `5 rounds × 3 clients` produces 15 local contributions. We therefore use 15 accepted updates on the asynchronous side.

```bash
python classical/server_async.py --num-clients 3 --max-updates 15 --port 8081 --log logs/async_straggler.csv
```

Clients:

```bash
python classical/client.py --client-id 0 --num-clients 3 --server 127.0.0.1:8081 --delay 0
python classical/client.py --client-id 1 --num-clients 3 --server 127.0.0.1:8081 --delay 1
python classical/client.py --client-id 2 --num-clients 3 --server 127.0.0.1:8081 --delay 3
```

The asynchronous server applies each update as soon as it arrives. This is a **simplified, pedagogical orchestration**: the mixing rule is `global <- (1-alpha) global + alpha client`, with `alpha=1/num_clients` by default. Staleness is recorded but not corrected.

## 5. Hierarchical architecture — 2 Edge × 3 clients

### Edge servers

```bash
python hierarchical/edge_server.py --edge-id 0 --num-edge-clients 3 --port 8090 --rounds 5
python hierarchical/edge_server.py --edge-id 1 --num-edge-clients 3 --port 8091 --rounds 5
```

### Edge 0 clients

```bash
python hierarchical/client_edge.py --edge-id 0 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090
python hierarchical/client_edge.py --edge-id 0 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090
python hierarchical/client_edge.py --edge-id 0 --local-id 2 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090
```

### Edge 1 clients

```bash
python hierarchical/client_edge.py --edge-id 1 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091
python hierarchical/client_edge.py --edge-id 1 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091
python hierarchical/client_edge.py --edge-id 1 --local-id 2 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091
```

At each round, every Edge produces the following files in `params/`:

- `edge_<id>_round<r>.npz`
- `edge_<id>_round<r>_keys.json`
- `edge_<id>_round<r>_meta.json`

The `meta.json` file contains, among other things, the number of examples actually aggregated, which the weighted Cloud aggregation needs.

## 6. Uniform then weighted Cloud aggregation

After round 5 of both Edge servers:

```bash
python hierarchical/cloud_server.py --num-edges 2 --edge-round 5 --weighting uniform
python hierarchical/cloud_server.py --num-edges 2 --edge-round 5 --weighting examples
```

## 7. Imbalance experiment without breaking the partitioning

We keep **3 clients per Edge**, but limit the local data volume for Edge 1:

Edge 0: `--max-train-samples 4000` for each of the three clients.

Edge 1: `--max-train-samples 1500` for each of the three clients.

The metadata files then automatically record the number of examples actually aggregated. The Cloud can compare the uniform average with the example-weighted average.

## 8. Analysis

Open:

```text
analysis/TP2_Analysis.ipynb
```

The notebook loads the CSV files present in `logs/` and generates the available curves.

## 9. Checks before the session

Before using the pack in the classroom, run:

```bash
python selfcheck.py
python -m pytest tests -q
```

Then prepare MNIST and run each planned scenario once to check ports, dependencies, and write permissions on `logs/` and `params/`.

## 10. Cleaning generated files (PowerShell)

```powershell
Remove-Item logs\* -Force
Remove-Item params\* -Force
```
