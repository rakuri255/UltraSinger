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

:: Remove old virtual environment to ensure clean state (e.g., switching between CPU/CUDA)
if exist .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

:: First, find Python using to get full path
set "PYTHON_EXE="

for %%V in (3.13 3.12) do (
    py -%%V --version >nul 2>&1
    if !errorlevel! equ 0 (
        :: Get the full path to the Python executable
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)"') do (
            set "PYTHON_EXE=%%P"
        )
        goto :found_python
    )
)

:: Fallback to direct Python installations (verify version before accepting)
for %%P in (python3.13 python3.12 python3 python) do (
    where %%P >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_VER="
        for /f "delims=" %%V in ('%%P -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do (
            set "PY_VER=%%V"
        )
        if "!PY_VER!"=="3.13" (
            set "PYTHON_EXE=%%P"
            goto :found_python
        )
        if "!PY_VER!"=="3.12" (
            set "PYTHON_EXE=%%P"
            goto :found_python
        )
    )
)

:found_python
if "!PYTHON_EXE!"=="" (
    echo Error: No Python 3.12 or 3.13 installation found
    echo Please install Python 3.12 or 3.13 from python.org
    pause
    exit /b 1
)

echo Using Python: !PYTHON_EXE!
"!PYTHON_EXE!" --version

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

:: Clear skip-worktree flags if previously set by CUDA install script
:: (allows git to manage these files normally again since CPU is the default)
where git >nul 2>&1
if !errorlevel! equ 0 (
    git update-index --no-skip-worktree pyproject.toml 2>nul
    git update-index --no-skip-worktree uv.lock 2>nul
)

:: Ensure PyTorch index is set to CPU in pyproject.toml
:: (restores default if previously changed by CUDA install script)
:: Use .NET WriteAllText to avoid UTF-8 BOM that breaks TOML parsing
:: (PowerShell 5.x Set-Content -Encoding UTF8 adds a BOM)
powershell -NoProfile -Command "$c = [IO.File]::ReadAllText('pyproject.toml'); $c = $c -replace 'whl/cu\d+','whl/cpu'; [IO.File]::WriteAllText('pyproject.toml', $c)"

:: Regenerate lockfile with CPU PyTorch index
echo Resolving dependencies...
uv lock
if !errorlevel! neq 0 (
    echo Error during uv lock
    pause
    exit /b 1
)

echo Syncing dependencies...
uv sync --extra windows --python "!PYTHON_EXE!"
if !errorlevel! neq 0 (
    echo Error during uv sync
    pause
    exit /b 1
)

echo Installation completed successfully!
pause