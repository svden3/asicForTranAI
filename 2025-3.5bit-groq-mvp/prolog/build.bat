@echo off
REM Build script for Prolog Inference Engine (Windows)
REM Requires: gfortran (Fortran 2023 support)

setlocal

echo =========================================================
echo Prolog Inference Engine Build (Fortran 2023)
echo Target: Groq LPU / Cerebras WSE via MLIR
echo =========================================================
echo.

REM Check if gfortran is installed
where gfortran >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: gfortran not found in PATH
    echo Please install MinGW-w64, TDM-GCC, or MSYS2
    exit /b 1
)

REM Check Fortran version
echo Checking gfortran version...
gfortran --version | findstr /C:"GNU Fortran"
echo.

REM Create output directories
if not exist bin mkdir bin

echo Compiling Prolog engine module...
gfortran -c prolog_engine.f90 -o bin\prolog_engine.o ^
    -std=f2023 ^
    -Wall -Wextra ^
    -O2 ^
    -g

if %errorlevel% neq 0 (
    echo ERROR: Failed to compile prolog_engine.f90
    exit /b 1
)

echo Compiling test program...
gfortran -c test_prolog_engine.f90 -o bin\test_prolog_engine.o ^
    -std=f2023 ^
    -Wall -Wextra ^
    -O2 ^
    -g

if %errorlevel% neq 0 (
    echo ERROR: Failed to compile test_prolog_engine.f90
    exit /b 1
)

echo Linking executable...
gfortran bin\prolog_engine.o bin\test_prolog_engine.o ^
    -o bin\test_prolog.exe ^
    -O2

if %errorlevel% neq 0 (
    echo ERROR: Failed to link executable
    exit /b 1
)

echo.
echo =========================================================
echo BUILD SUCCESSFUL
echo =========================================================
echo Executable: bin\test_prolog.exe
echo.
echo Run: bin\test_prolog
echo.

endlocal
