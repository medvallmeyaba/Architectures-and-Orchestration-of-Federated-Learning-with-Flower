@echo off
cd /d "%~dp0\.."
REM Edge 0: 4000 exemples/client ; Edge 1: 1500 exemples/client
start "E0-C0" cmd /k "python hierarchical\client_edge.py --edge-id 0 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090 --max-train-samples 4000"
start "E0-C1" cmd /k "python hierarchical\client_edge.py --edge-id 0 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090 --max-train-samples 4000"
start "E0-C2" cmd /k "python hierarchical\client_edge.py --edge-id 0 --local-id 2 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090 --max-train-samples 4000"
start "E1-C0" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091 --max-train-samples 1500"
start "E1-C1" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091 --max-train-samples 1500"
start "E1-C2" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 2 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091 --max-train-samples 1500"
