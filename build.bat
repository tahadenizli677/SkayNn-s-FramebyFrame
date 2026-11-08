@echo off
setlocal
cd /d "%~dp0"

REM Determine Python executable
set "PY_CMD=python"
if exist "%LOCALAPPDATA%\Python\bin\python.exe" (
    set "PY_CMD=%LOCALAPPDATA%\Python\bin\python.exe"
)

echo Python kullaniliyor: %PY_CMD%

"%PY_CMD%" -m PyInstaller --noconfirm ArsenalAimbot.spec

if %ERRORLEVEL% neq 0 (
    echo [HATA] Derleme basarisiz oldu!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Derleme basariyla tamamlandi!
echo Cikti: dist\SkayNnsFramebyFrame.exe (Tek parca dosya, tum modeller ve ayarlar icine gomuldu)
echo.
pause
