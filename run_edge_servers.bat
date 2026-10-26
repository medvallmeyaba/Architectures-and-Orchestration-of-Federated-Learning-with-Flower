@echo off
cd /d "%~dp0\.."
start "Edge 0" cmd /k "python hierarchical\edge_server.py --edge-id 0 --num-edge-clients 3 --port 8090 --rounds 5"
start "Edge 1" cmd /k "python hierarchical\edge_server.py --edge-id 1 --num-edge-clients 3 --port 8091 --rounds 5"
