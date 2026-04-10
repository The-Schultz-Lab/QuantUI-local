@echo off
echo QuantUI-local — Starting...
echo.

REM Check that the .sif exists before trying to launch
if not exist "%~dp0quantui-local.sif" (
    echo ERROR: quantui-local.sif not found.
    echo Build it first:  bash apptainer/build.sh
    echo Or download it from the GitHub Releases page.
    pause
    exit /b 1
)

REM Launch Voila in a new WSL window (stays open so you can see logs)
start "QuantUI-local" wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/schul/Documents/local-code-dir/repos-DEVS/QuantUI-local && apptainer run quantui-local.sif app"

REM Wait for Voila to start, then open the browser
echo Waiting for Voila to start...
timeout /t 6 /nobreak > nul
start http://localhost:8866

echo.
echo App is running at http://localhost:8866
echo Close the WSL window to stop the server.
