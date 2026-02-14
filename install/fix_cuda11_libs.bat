@echo off
echo ========================================
echo CUDA 11.x Libraries Fix for WhisperX/Pyannote
echo ========================================
echo.
echo This script will download and install CUDA 11.x DLLs
echo that are required by pyannote.audio (used by WhisperX)
echo.
echo PyTorch 2.8.0 includes CUDA 12.x, but pyannote.audio
echo still requires CUDA 11.x DLLs to function properly.
echo.

cd /d "%~dp0\.."

echo Checking for virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found at .venv
    echo Please create the virtual environment first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Error activating virtual environment
    pause
    exit /b 1
)

echo Creating temporary directory...
if not exist temp_cuda11 mkdir temp_cuda11
cd temp_cuda11

echo.
echo Downloading CUDA 11.8 Toolkit Libraries (Windows)...
echo This may take a few minutes...
echo.
echo Downloading cuBLAS 11.x...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-11.11.3.6-archive.zip' -OutFile 'cublas.zip'}"

if %errorlevel% neq 0 (
    echo Error downloading cuBLAS 11.x
    echo.
    echo Please check your internet connection or download manually from:
    echo https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/
    echo.
    cd ..
    deactivate
    pause
    exit /b 1
)

echo.
echo Extracting CUDA 11.x Libraries...
if exist cublas.zip (
    powershell -Command "Expand-Archive -Path 'cublas.zip' -DestinationPath '.' -Force"
)

echo.
echo Copying CUDA 11.x DLLs to PyTorch installation...
set "COPIED=0"

if exist "libcublas-windows-x86_64-11.11.3.6-archive\bin\*.dll" (
    copy /Y "libcublas-windows-x86_64-11.11.3.6-archive\bin\*.dll" "%VIRTUAL_ENV%\Lib\site-packages\torch\lib\"
    set "COPIED=1"
)

if "%COPIED%"=="0" (
    echo Error: Could not find CUDA 11.x DLLs in extracted files
    echo Expected path: libcublas-windows-x86_64-11.11.3.6-archive\bin\
    cd ..
    deactivate
    pause
    exit /b 1
)

echo.
echo Successfully copied CUDA 11.x DLLs!
echo Listing copied cuBLAS files:
dir "%VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas*.dll" /b 2>nul

echo.
echo Cleaning up temporary files...
cd ..
rmdir /s /q temp_cuda11

echo Deactivating virtual environment...
deactivate

echo.
echo ========================================
echo CUDA 11.x Fix completed successfully!
echo ========================================
echo.
echo You can now run UltraSinger without CUDA 11.x DLL errors.
echo.
pause

