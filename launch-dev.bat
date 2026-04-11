@echo off
echo QuantUI-local DEV MODE — Using local notebook (no rebuild needed)
echo.

if not exist "%~dp0quantui-local.sif" (
    echo ERROR: quantui-local.sif not found.
    pause
    exit /b 1
)

REM Convert the Windows repo path to a WSL path for portability
for /f "delims=" %%i in ('wsl wslpath -a "%~dp0"') do set WSLPATH=%%i

REM Uses the local notebook on disk instead of the baked-in copy.
REM Edits to notebooks/ take effect immediately — no container rebuild needed.
start "QuantUI-local [dev]" wsl -d Ubuntu -- bash -c "cd '%WSLPATH%' && apptainer exec --cleanenv quantui-local.sif voila notebooks/molecule_computations.ipynb --no-browser --port=8866 --ServerApp.disable_check_xsrf=True"

echo Waiting for Voila to start...
timeout /t 6 /nobreak > nul
start http://localhost:8866

echo.
echo Dev server running at http://localhost:8866
echo Edit notebooks/ and refresh the browser — no rebuild needed.
echo Close the WSL window to stop.
