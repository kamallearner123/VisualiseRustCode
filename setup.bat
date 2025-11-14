@echo off
REM Setup script for Rust Visual Memory Debugger (Windows)

echo Setting up Rust Visual Memory Debugger...

REM Check Python
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.8 or higher.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run migrations
echo Running database migrations...
python manage.py migrate

REM Create static directories
if not exist "static\css" mkdir static\css
if not exist "static\js" mkdir static\js

echo.
echo Setup complete!
echo.
echo To start the server:
echo   venv\Scripts\activate.bat
echo   python manage.py runserver
echo.
echo Then visit: http://localhost:8000

pause
