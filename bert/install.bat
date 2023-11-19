@echo off
REM Create a Python virtual environment
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies from requirements.txt
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
REM Deactivate the virtual environment when done

REM Optional: Pause the script at the end to see the output
pause

REM End of the script
@echo on
