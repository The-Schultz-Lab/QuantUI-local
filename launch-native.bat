@echo off
echo QuantUI-local NATIVE MODE — Local conda env in WSL, no container
echo Use this when you have edited quantui/*.py and want to test immediately.
echo.

REM Convert the Windows repo path to a WSL path for portability
for /f "delims=" %%i in ('wsl wslpath -a "%~dp0"') do set WSLPATH=%%i

REM Runs Voila directly from the quantui-local conda env inside WSL.
REM pip install -e . ensures the local quantui/ package is importable and
REM up-to-date. All changes to quantui/*.py are live — no container rebuild needed.
REM Uses port 8867 to avoid conflict with container-based launchers on 8866.
start "QuantUI-local [native]" wsl -d Ubuntu -- bash -c "cd '%WSLPATH%' && source ~/miniconda3/etc/profile.d/conda.sh && conda activate quantui-local && pip install -e . -q && voila notebooks/molecule_computations.ipynb --no-browser --port=8867 --ServerApp.disable_check_xsrf=True"

echo Waiting for Voila to start...
timeout /t 6 /nobreak > nul
start http://localhost:8867

echo.
echo Native dev server running at http://localhost:8867
echo All local quantui/*.py changes are live — no rebuild needed.
echo Close the WSL window to stop.
