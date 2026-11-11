@echo off
chcp 65001 >nul
echo ================================================================
echo   KHOI DONG HE THONG AI CHAN DOAN BENH TOM (U2-Net + ConvNeXt)
echo ================================================================
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m streamlit run "%~dp0app.py"
) else (
    python -m streamlit run "%~dp0app.py"
)
pause
