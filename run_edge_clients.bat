@echo off
cd /d "%~dp0\.."
start "E0-C0" cmd /k "python hierarchical\client_edge.py --edge-id 0 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090"
start "E0-C1" cmd /k "python hierarchical\client_edge.py --edge-id 0 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8090"
start "E1-C0" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 0 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091"
start "E1-C1" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 1 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091"
start "E1-C2" cmd /k "python hierarchical\client_edge.py --edge-id 1 --local-id 2 --num-edges 2 --clients-per-edge 3 --server 127.0.0.1:8091"
