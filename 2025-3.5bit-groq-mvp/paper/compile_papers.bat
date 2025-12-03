@echo off
REM Compile LaTeX Papers to PDF
REM Run this batch file after installing MiKTeX

echo ============================================================
echo Compiling Papers to PDF
echo ============================================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found!
    echo.
    echo Please ensure MiKTeX is installed and in your PATH.
    echo Try closing and reopening your terminal, or restart your computer.
    echo.
    pause
    exit /b 1
)

echo [OK] pdflatex found
pdflatex --version | findstr "MiKTeX"
echo.

REM Navigate to paper directory
cd /d "%~dp0"
echo Working directory: %CD%
echo.

echo ============================================================
echo Compiling Main Paper (paper.tex)
echo ============================================================
echo.

REM Compile main paper (run twice for cross-references)
echo [1/2] First pass...
pdflatex -interaction=nonstopmode paper.tex
if %errorlevel% neq 0 (
    echo.
    echo ERROR: First compilation of paper.tex failed!
    echo Check paper.log for details
    pause
    exit /b 1
)

echo [2/2] Second pass (for cross-references)...
pdflatex -interaction=nonstopmode paper.tex
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Second compilation of paper.tex failed!
    echo Check paper.log for details
    pause
    exit /b 1
)

echo.
echo [SUCCESS] paper.pdf created!
echo.

echo ============================================================
echo Compiling Supplementary Materials (supplementary.tex)
echo ============================================================
echo.

REM Compile supplementary (run twice for cross-references)
echo [1/2] First pass...
pdflatex -interaction=nonstopmode supplementary.tex
if %errorlevel% neq 0 (
    echo.
    echo ERROR: First compilation of supplementary.tex failed!
    echo Check supplementary.log for details
    pause
    exit /b 1
)

echo [2/2] Second pass (for cross-references)...
pdflatex -interaction=nonstopmode supplementary.tex
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Second compilation of supplementary.tex failed!
    echo Check supplementary.log for details
    pause
    exit /b 1
)

echo.
echo [SUCCESS] supplementary.pdf created!
echo.

echo ============================================================
echo Compilation Complete!
echo ============================================================
echo.

REM List generated PDFs with file sizes
echo Generated PDFs:
dir /b *.pdf 2>nul
if %errorlevel% equ 0 (
    echo.
    for %%f in (*.pdf) do (
        echo   %%f - %%~zf bytes
    )
) else (
    echo   (No PDFs found - check for errors above)
)

echo.
echo Files are ready in: %CD%
echo.

REM Clean up auxiliary files (optional)
choice /C YN /M "Clean up auxiliary files (.aux, .log, etc)?"
if %errorlevel% equ 1 (
    echo Cleaning up...
    del /q *.aux *.log *.out 2>nul
    echo Done!
)

echo.
echo ============================================================
echo Next Steps:
echo   1. Check paper.pdf and supplementary.pdf
echo   2. Ready for submission to ICML 2025 (Feb 1, 2025)
echo ============================================================
echo.

pause
