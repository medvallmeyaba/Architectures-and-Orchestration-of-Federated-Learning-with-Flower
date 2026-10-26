@echo off
cd /d "%~dp0\.."
python hierarchical\cloud_server.py --num-edges 2 --edge-round 5 --weighting uniform
python hierarchical\cloud_server.py --num-edges 2 --edge-round 5 --weighting examples
pause
