@echo off
echo ========================================
echo cuDNN 8.x Fix for WhisperX/Pyannote
echo ========================================
echo.
echo This script will download and install cuDNN 8.x DLLs
echo that are required by pyannote.audio (used by WhisperX)
echo.
echo PyTorch 2.8.0 includes cuDNN 9.x, but pyannote.audio
echo still requires cuDNN 8.x DLLs to function properly.
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
if not exist temp_cudnn mkdir temp_cudnn
cd temp_cudnn

echo.
echo Downloading cuDNN 8.9.7 for CUDA 11.x (Windows)...
echo This may take a few minutes...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip' -OutFile 'cudnn.zip'}"

if %errorlevel% neq 0 (
    echo Error downloading cuDNN 8.9.7
    echo.
    echo Alternative: Trying to extract from installed torch packages...
    echo Searching for existing PyTorch installations with cuDNN 8.x...

    REM Try to find cuDNN 8 DLLs in pip cache or other Python installations
    powershell -Command "& {$found = $false; Get-ChildItem -Path $env:LOCALAPPDATA\pip\cache -Recurse -Filter 'cudnn64_8.dll' -ErrorAction SilentlyContinue | ForEach-Object {echo $_.FullName; $found = $true}; if (-not $found) {exit 1}}"

    if %errorlevel% neq 0 (
        echo Error: Could not find or download cuDNN 8.x DLLs
        echo.
        echo Please manually download cuDNN 8.9.7 from NVIDIA:
        echo https://developer.nvidia.com/rdp/cudnn-archive
        echo.
        cd ..
        deactivate
        pause
        exit /b 1
    )
)

echo.
echo Extracting cuDNN 8.x DLLs...
if exist cudnn.zip (
    powershell -Command "Expand-Archive -Path 'cudnn.zip' -DestinationPath '.' -Force"
)

echo.
echo Copying cuDNN 8.x DLLs to PyTorch installation...
if exist "cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin\cudnn*.dll" (
    copy /Y "cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin\cudnn*.dll" "%VIRTUAL_ENV%\Lib\site-packages\torch\lib\"
    echo.
    echo Successfully copied cuDNN 8.x DLLs!
    echo Listing copied files:
    dir "%VIRTUAL_ENV%\Lib\site-packages\torch\lib\cudnn*.dll" /b
) else (
    echo Error: Could not find cuDNN DLLs in extracted files
    echo Expected path: cudnn-windows-x86_64-8.9.7.29_cuda11-archive\bin\
    cd ..
    deactivate
    pause
    exit /b 1
)

echo.
echo Cleaning up temporary files...
cd ..
rmdir /s /q temp_cudnn

echo Deactivating virtual environment...
deactivate

echo.
echo ========================================
echo cuDNN 8.x Fix completed successfully!
echo ========================================
echo.
echo You can now run UltraSinger without cuDNN errors.
echo.
pause

