@echo off
REM Start script for Rust Visual Memory Debugger (Windows)

echo Starting Rust Visual Memory Debugger...

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found!
    echo Please run setup.bat first to set up the project.
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if database exists
if not exist "db.sqlite3" (
    echo Database not found. Running migrations...
    python manage.py migrate
)

REM Start the Django development server
echo Starting Django development server...
echo.
echo ================================================
echo   Rust Visual Memory Debugger
echo   Server: http://127.0.0.1:8000
echo   Press CTRL+C to stop the server
echo ================================================
echo.

python manage.py runserver
