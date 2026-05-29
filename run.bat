@echo off
:: ============================================================================
::  MindMeter - Depression Risk Screen
::  Habiba Farag Shehata
::  This launcher checks for python, syncs requirements, and starts the app.
:: ============================================================================
setlocal EnableExtensions
title MindMeter :: Depression Risk Screen
pushd "%~dp0"

echo.
echo  ----------------------------------------
echo   MindMeter - Depression Risk Screen
echo  ----------------------------------------
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [x] python is not on PATH. install Python 3.12 first, then re-run.
    popd
    pause
    exit /b 1
)

echo [+] step 1 of 2 :: syncing requirements
python -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo [x] pip install failed. see messages above.
    popd
    pause
    exit /b 1
)

echo [+] step 2 of 2 :: starting Flask on http://127.0.0.1:5000
echo.
python app.py

popd
endlocal
pause
