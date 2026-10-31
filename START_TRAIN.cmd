@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo The project Python environment is missing.
    echo Please follow the setup instructions in README.md first.
    pause
    exit /b 1
)
start "" /D "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" -m collm.launcher
endlocal
