@echo off
setlocal

set "PDF=%~1"
if "%PDF%"=="" set "PDF=workScans\10MATD_combinedTEST.pdf"

if not exist "%PDF%" (
    echo PDF not found: %PDF%
    pause
    exit /b 1
)

python "%~dp0make_template.py" "%PDF%"
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
    echo.
    echo make_template.py exited with code %ERR%
    pause
)

endlocal & exit /b %ERR%
