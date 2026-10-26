@echo off
cd /d "%~dp0\.."
start "FL Sync Server" cmd /k "python classical\server_sync.py --num-clients 3 --rounds 5 --port 8080 --log logs\sync_straggler.csv"
timeout /t 2 >nul
start "Client 0" cmd /k "python classical\client.py --client-id 0 --num-clients 3 --server 127.0.0.1:8080 --delay 0"
start "Client 1" cmd /k "python classical\client.py --client-id 1 --num-clients 3 --server 127.0.0.1:8080 --delay 1"
start "Client 2" cmd /k "python classical\client.py --client-id 2 --num-clients 3 --server 127.0.0.1:8080 --delay 3"
