@echo off
REM Windows Build Script for Ada/SPARK Safety Layer
REM Requires: GNAT toolchain + gfortran in PATH

setlocal

echo ===============================================
echo Ada/SPARK Safety Layer Build (Windows)
echo ===============================================
echo.

REM Check if GNAT is installed
where gnatmake >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: gnatmake not found in PATH
    echo Please install GNAT from: https://www.adacore.com/download
    exit /b 1
)

REM Check if gfortran is installed
where gfortran >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: gfortran not found in PATH
    echo Please install MinGW-w64 or TDM-GCC
    exit /b 1
)

REM Create directories
if not exist obj mkdir obj
if not exist bin mkdir bin

echo Step 1: Compiling Fortran modules...
echo.

REM Compile matmul_int4_groq.f90
gfortran -c ..\matmul_int4_groq.f90 -o obj\matmul_int4_groq.o -Jobj -Wall -Wextra -std=f2023 -fcheck=all -O2 -g
if %errorlevel% neq 0 (
    echo ERROR: Failed to compile matmul_int4_groq.f90
    exit /b 1
)

REM Compile matmul_int4_ada_bridge.f90
gfortran -c matmul_int4_ada_bridge.f90 -o obj\matmul_int4_ada_bridge.o -Iobj -Jobj -Wall -Wextra -std=f2023 -fcheck=all -O2 -g
if %errorlevel% neq 0 (
    echo ERROR: Failed to compile matmul_int4_ada_bridge.f90
    exit /b 1
)

echo Fortran modules compiled successfully
echo.

echo Step 2: Building Ada safety layer...
echo.

REM Build Ada project
gnatmake -P ada_safety_layer.gpr -largs obj\matmul_int4_groq.o obj\matmul_int4_ada_bridge.o
if %errorlevel% neq 0 (
    echo ERROR: Failed to build Ada safety layer
    exit /b 1
)

echo.
echo ===============================================
echo BUILD SUCCESSFUL
echo ===============================================
echo Executable: bin\test_ada_safety.exe
echo.
echo Run tests: bin\test_ada_safety
echo Run SPARK verification: gnatprove -P ada_safety_layer.gpr
echo.

endlocal
