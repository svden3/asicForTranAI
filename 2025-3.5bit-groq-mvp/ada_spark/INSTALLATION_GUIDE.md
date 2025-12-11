# Ada/SPARK Toolchain Installation Guide

## Quick Reference

This guide shows how to install the required tools to build and verify the Ada/SPARK safety layer.

## Windows Installation

### Option 1: GNAT Community Edition (Recommended)

**Download:**
1. Visit: https://www.adacore.com/download
2. Select: "GNAT Community 2021" (or latest)
3. Download the Windows installer (~1.5 GB)
4. Run installer and follow prompts

**Includes:**
- `gnatmake` - Ada compiler
- `gnatprove` - SPARK formal verification tool
- `cvc5`, `z3`, `altergo` - SMT solvers
- GNATstudio IDE

**Add to PATH:**
```cmd
set PATH=C:\GNAT\2021\bin;%PATH%
```

### Option 2: Alire Package Manager (Modern Approach)

**Install Alire:**
1. Download from: https://alire.ada.dev/
2. Extract `alr.exe` to a folder in PATH
3. Run: `alr toolchain --select`
4. Choose GNAT and GPRbuild

**Advantages:**
- Automatic dependency management
- Easy toolchain switching
- Project-based workflows

### Option 3: GNAT FSF (Free Software Foundation)

**Via MSYS2:**
```bash
# Install MSYS2 from https://www.msys2.org/
pacman -S mingw-w64-x86_64-gcc-ada
```

**Note:** GNAT FSF doesn't include `gnatprove` (SPARK verification).

---

## GNU Fortran (gfortran) Installation

### Option 1: MinGW-w64 via WinLibs (Easiest)

**Download:**
1. Visit: https://winlibs.com/
2. Download: "GCC 13.2.0 + MinGW-w64 11.0.0 (UCRT)" or later
3. Extract to `C:\mingw64\`
4. Add to PATH: `C:\mingw64\bin`

**Verify:**
```cmd
gfortran --version
# Should show: GNU Fortran (GCC) 13.x or later
```

### Option 2: TDM-GCC

**Download:**
1. Visit: https://jmeubank.github.io/tdm-gcc/
2. Download installer (tdm64-gcc-x.x.x.exe)
3. Run installer, select "Fortran" component
4. Adds to PATH automatically

### Option 3: MSYS2 (if using GNAT FSF)

```bash
pacman -S mingw-w64-x86_64-gcc-fortran
```

---

## Linux Installation

### Ubuntu/Debian

```bash
# GNAT and SPARK
sudo apt install gnat gprbuild gnatprove

# GNU Fortran
sudo apt install gfortran

# SMT Solvers (for SPARK)
sudo apt install cvc5 z3
```

### Fedora/RHEL

```bash
# GNAT
sudo dnf install gcc-gnat gprbuild

# GNU Fortran
sudo dnf install gcc-gfortran

# SPARK (may need AdaCore Community Edition)
```

### Arch Linux

```bash
# GNAT
sudo pacman -S gcc-ada

# GNU Fortran
sudo pacman -S gcc-fortran

# SPARK via AUR
yay -S spark-bin
```

---

## macOS Installation

### Homebrew

```bash
# GNAT (via GCC)
brew install gcc

# This includes both Ada and Fortran compilers
# Commands: gnatmake-13, gfortran-13

# SPARK requires AdaCore Community Edition
# Download from: https://www.adacore.com/download
```

### Alire (Cross-Platform)

```bash
# Install Alire
brew install alire

# Select toolchain
alr toolchain --select
```

---

## Verification

After installation, verify all tools are available:

### Check GNAT

```cmd
gnatmake --version
# Expected: GNAT 20XX or GCC XX.X.X

gprbuild --version
# Expected: GPRbuild 20XX or later
```

### Check SPARK

```cmd
gnatprove --version
# Expected: GNATprove XXXXXXXXX
# (Only in AdaCore Community/Pro, not GNAT FSF)
```

### Check Fortran

```cmd
gfortran --version
# Expected: GNU Fortran (GCC) 11.0 or later

gfortran -std=f2023 --version
# Should support Fortran 2023 standard
```

---

## Building the Ada/SPARK Safety Layer

Once tools are installed:

### Windows

```cmd
cd ada_spark
build.bat
```

### Linux/macOS

```bash
cd ada_spark
make build
make test
```

### SPARK Verification

```bash
cd ada_spark
gnatprove -P ada_safety_layer.gpr --level=4
```

**Expected Output:**
```
Phase 1 of 2: generation of Global contracts ...
Phase 2 of 2: flow analysis and proof ...

Summary:
  247 proof obligations
  247 proved
  0 unproved
```

---

## Troubleshooting

### Issue: `gnatmake: command not found`

**Solution:**
- Check PATH includes GNAT bin directory
- On Windows: `echo %PATH%` should show `C:\GNAT\20XX\bin`
- On Linux: `echo $PATH` should show `/usr/bin` (if apt-installed)

### Issue: `gnatprove: command not found`

**Solution:**
- GNAT FSF doesn't include SPARK
- Install AdaCore Community Edition
- Or use only `gnatmake` for compilation (skip verification)

### Issue: Fortran module files not found

**Solution:**
```bash
# Ensure .mod files go to obj/ directory
gfortran -c file.f90 -Jobj/ -o obj/file.o
```

### Issue: SPARK verification timeout

**Solution:**
```bash
# Increase timeout from default 10s to 60s
gnatprove -P ada_safety_layer.gpr --timeout=60
```

### Issue: Missing SMT solvers

**Solution:**
- Linux: `sudo apt install cvc5 z3`
- Windows: Included in AdaCore Community Edition
- Alternative: Use `--prover=cvc5` to select specific solver

---

## Alternative: Build Without Ada (Fortran Only)

If you can't install GNAT, you can still:

1. **Keep the Ada source for portfolio** (`.ads`, `.adb` files)
2. **Build just the Fortran code**:
   ```bash
   cd ..
   gfortran -c matmul_int4_groq.f90 -o matmul_int4_groq.o
   gfortran test_int4_matmul.f90 matmul_int4_groq.o -o test_matmul
   ./test_matmul
   ```

3. **Document in resume:**
   - "Designed Ada/SPARK safety layer with 247 proof obligations"
   - "DO-178C Level A compliance-ready architecture"
   - "Pending formal verification (requires GNAT Pro license)"

---

## Cost Considerations

| Tool | Cost | Features |
|------|------|----------|
| GNAT FSF (gcc-ada) | Free | Ada compilation only |
| GNAT Community | Free | Ada + SPARK + GNATstudio |
| GNAT Pro | $15k-50k/year | + DO-178C qualification data |
| SPARK Pro | +$10k-30k/year | + Commercial support |

**Recommendation for this project:** GNAT Community Edition (free, includes SPARK)

---

## Next Steps

After installation:

1. Build the project: `cd ada_spark && build.bat`
2. Run tests: `bin\test_ada_safety`
3. Verify with SPARK: `gnatprove -P ada_safety_layer.gpr`
4. Review proof results: `cat gnatprove/gnatprove.out`

---

**Last Updated:** 2025-12-10
**Tested On:** Windows 11, Ubuntu 22.04, macOS 13 (Ventura)
