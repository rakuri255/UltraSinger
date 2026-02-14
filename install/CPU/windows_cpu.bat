@echo off
setlocal enabledelayedexpansion

:: Set link mode to copy to avoid hardlink warnings on different filesystems
set UV_LINK_MODE=copy

:: Navigate to project root
pushd "%~dp0"
cd /d ..\..
echo Current directory: %cd%

:: Update PATH to include uv installation directory
set "PATH=%USERPROFILE%\.local\bin;!PATH!"

:: Remove old virtual environment if it exists to force recreation with correct Python version
::if exist .venv (
::    echo Removing old virtual environment...
::    rmdir /s /q .venv
::)

:: First, find Python using to get full path
set "PYTHON_EXE="

for %%V in (3.12) do (
    py -%%V --version >nul 2>&1
    if !errorlevel! equ 0 (
        :: Get the full path to the Python executable
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do (
            set "PYTHON_EXE=%%P"
        )
        goto :found_python
    )
)

:: Fallback to direct Python installations
for %%P in (python3.12 python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=%%P"
        goto :found_python
    )
)

:found_python
if "!PYTHON_EXE!"=="" (
    echo Error: No Python 3.12 installation found
    echo Please install Python 3.12 from python.org
    echo Note: Python 3.13 is not yet supported due to dependency constraints
    pause
    exit /b 1
)

echo Using Python: !PYTHON_EXE!
!PYTHON_EXE! --version

:: Install uv if not already installed
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Installing uv...
    powershell -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
)

:: Wait a moment for uv to be available
timeout /t 2 /nobreak >nul

:: Verify uv is available
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: uv could not be found or installed
    pause
    exit /b 1
)

echo uv is ready
uv --version

echo Syncing dependencies with uv...
uv sync --extra windows --python !PYTHON_EXE!
if !errorlevel! neq 0 (
    echo Error during uv sync
    pause
    exit /b 1
)

echo Installing PyTorch CPU version...
:: First remove any existing torch installation to avoid RECORD file issues
uv pip uninstall torch torchvision torchaudio -y 2>nul
:: Install PyTorch CPU version
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
if !errorlevel! neq 0 (
    echo Error during PyTorch installation
    pause
    exit /b 1
)

echo Installation completed successfully!
pause