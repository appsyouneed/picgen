@echo off
echo =============================================
echo   Image Editor Setup Script (Windows)
echo =============================================
echo.

REM -----------------------------------------------------------
REM STEP 1: Check Python installation
REM -----------------------------------------------------------
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)
python --version

REM -----------------------------------------------------------
REM STEP 2: Check pip installation
REM -----------------------------------------------------------
echo.
echo [2/8] Checking pip installation...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip not found. Please install pip.
    pause
    exit /b 1
)
pip --version

REM -----------------------------------------------------------
REM STEP 3: Install Python dependencies
REM -----------------------------------------------------------
echo.
echo [3/8] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

REM -----------------------------------------------------------
REM STEP 4: Check Hugging Face CLI
REM -----------------------------------------------------------
echo.
echo [4/8] Checking Hugging Face CLI...
huggingface-cli --help >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Hugging Face CLI...
    pip install huggingface_hub[cli]
)

REM -----------------------------------------------------------
REM STEP 5: Create local model directories
REM -----------------------------------------------------------
echo.
echo [5/8] Creating local model directories...
if not exist "models" mkdir models
if not exist "models\Qwen-Image-Edit-2511" mkdir models\Qwen-Image-Edit-2511
if not exist "models\rapid-aio" mkdir models\rapid-aio
if not exist "models\rapid-aio\v23" mkdir models\rapid-aio\v23

REM -----------------------------------------------------------
REM STEP 6: Download models from Hugging Face
REM -----------------------------------------------------------
echo.
echo [6/8] Downloading models (this may take a while)...

echo   -^> Downloading Base Model...
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models\Qwen-Image-Edit-2511
if %errorlevel% neq 0 (
    echo ERROR: Failed to download base model.
    echo Trying alternative download method...
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-Edit-2511', local_dir='models/Qwen-Image-Edit-2511')"
    if %errorlevel% neq 0 (
        echo ERROR: Both download methods failed.
        pause
        exit /b 1
    )
)

echo   -^> Downloading NSFW Weights (v23)...
huggingface-cli download Phr00t/Qwen-Image-Edit-Rapid-AIO --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" --local-dir models\rapid-aio
if %errorlevel% neq 0 (
    echo ERROR: Failed to download NSFW weights.
    pause
    exit /b 1
)

REM -----------------------------------------------------------
REM STEP 7: Verify GPU accessibility
REM -----------------------------------------------------------
echo.
echo [7/8] Verifying GPU accessibility...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'GPU Device: None'); print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB' if torch.cuda.is_available() else '')"

REM Set memory optimization environment variable
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM -----------------------------------------------------------
REM STEP 8: Set environment variables
REM -----------------------------------------------------------
echo.
echo [8/8] Setting environment variables...
set BASE_MODEL_PATH=models\Qwen-Image-Edit-2511
set NSFW_WEIGHTS_PATH=models\rapid-aio\v23\Qwen-Rapid-AIO-NSFW-v23.safetensors

REM Persist environment variables (requires admin rights, so we'll skip for now)
REM setx BASE_MODEL_PATH "%cd%\models\Qwen-Image-Edit-2511"
REM setx NSFW_WEIGHTS_PATH "%cd%\models\rapid-aio\v23\Qwen-Rapid-AIO-NSFW-v23.safetensors"
REM setx PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

echo.
echo =============================================
echo   Setup Complete!
echo   BASE_MODEL_PATH     = %BASE_MODEL_PATH%
echo   NSFW_WEIGHTS_PATH   = %NSFW_WEIGHTS_PATH%
echo =============================================
echo.
echo   Press any key to start the server...
pause >nul

REM Source environment variables and start the app
python app.py --share
