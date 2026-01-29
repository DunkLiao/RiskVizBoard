@echo off
REM ========================================
REM Risk Indicator Dashboard - Startup Script
REM ========================================

echo.
echo [INFO] 正在關閉舊的 Streamlit 連線...
taskkill /IM streamlit.exe /F 2>nul
if %errorlevel% equ 0 (
    echo [OK] 已關閉舊連線
) else (
    echo [INFO] 未找到運行中的 Streamlit 進程
)

echo.
echo [INFO] 稍候 2 秒...
timeout /t 2 /nobreak

echo.
echo [INFO] 正在啟動 Risk Indicator Dashboard...
echo [URL] http://localhost:8501
echo.

REM 進入專案目錄
cd /d "%~dp0"

REM 啟動 Streamlit 應用
streamlit run main.py

pause
