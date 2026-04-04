@echo off
echo === Fixing Picgen Installation ===

echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

echo Reinstalling dependencies...
pip3 install -r requirements.txt --force-reinstall

echo === Fix Complete ===
echo Run: start.bat
pause
