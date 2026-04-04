@echo off
echo === Stopping Picgen ===
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *app.py*"
echo Picgen stopped.
pause
