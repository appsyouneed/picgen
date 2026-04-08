@echo off
echo === Picgen Image Editor Windows Setup ===

echo Creating cache directory...
if not exist "%USERPROFILE%\.cache\huggingface" mkdir "%USERPROFILE%\.cache\huggingface"

echo Installing PyTorch (will auto-detect CUDA 13.0)...
pip3 install torch torchvision --break-system-packages

echo Installing Python dependencies...
pip3 install -r requirements.txt

echo Installing Hugging Face CLI...
pip3 install huggingface_hub[cli]

echo Creating local model directories...
if not exist "models\Qwen-Image-Edit-2511" mkdir "models\Qwen-Image-Edit-2511"
if not exist "models\rapid-aio\v23" mkdir "models\rapid-aio\v23"

echo === Model Download ===
echo Models will be downloaded automatically on first run.
echo To pre-download models now, run:
echo   huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
echo   huggingface-cli download Phr00t/Qwen-Image-Edit-Rapid-AIO --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir models/rapid-aio

echo Verifying GPU accessibility...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB' if torch.cuda.is_available() else 'N/A')"

echo.
echo === Setup Complete ===
echo To start the app, run: start.bat
pause
