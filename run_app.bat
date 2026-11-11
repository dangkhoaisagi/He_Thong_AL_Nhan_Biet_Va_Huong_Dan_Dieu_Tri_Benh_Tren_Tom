@echo off
chcp 65001 >nul
echo ================================================================
echo   KHOI DONG HE THONG AI CHAN DOAN BENH TOM (U2-Net + ConvNeXt)
echo ================================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
