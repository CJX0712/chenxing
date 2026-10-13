@echo off
REM One-click reproducible verification for Chenxing (晨星).
REM Works with any Python that has numpy (no venv required).
cd /d "%~dp0.." || exit /b 1
python scripts\run_verification.py --report %*
