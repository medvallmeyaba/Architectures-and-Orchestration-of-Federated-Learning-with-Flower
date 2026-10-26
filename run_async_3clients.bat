@echo off
cd /d "%~dp0\.."
start "FL Async Server" cmd /k "python classical\server_async.py --num-clients 3 --max-updates 15 --port 8081 --log logs\async_homogeneous.csv"
timeout /t 2 >nul
start "Client 0" cmd /k "python classical\client.py --client-id 0 --num-clients 3 --server 127.0.0.1:8081 --delay 0"
start "Client 1" cmd /k "python classical\client.py --client-id 1 --num-clients 3 --server 127.0.0.1:8081 --delay 0"
start "Client 2" cmd /k "python classical\client.py --client-id 2 --num-clients 3 --server 127.0.0.1:8081 --delay 0"
