# Quick Installation Guide: Fortran + GNAT

## Option 1: MSYS2 + GNAT Community (Recommended)

### Step 1: Install MSYS2 (Includes gfortran)

1. **Download MSYS2:**
   - Visit: https://www.msys2.org/
   - Download: `msys2-x86_64-latest.exe`
   - Run installer (install to default location: `C:\msys64`)

2. **Install gfortran:**
   ```bash
   # Open "MSYS2 MINGW64" from Start Menu
   pacman -Syu  # Update package database
   pacman -S mingw-w64-x86_64-gcc-fortran
   ```

3. **Add to Windows PATH:**
   - Open "Edit system environment variables"
   - Click "Environment Variables"
   - Under "System variables", select "Path" and click "Edit"
   - Click "New" and add: `C:\msys64\mingw64\bin`
   - Click OK on all dialogs

4. **Verify:**
   ```cmd
   # Open NEW Command Prompt
   gfortran --version
   # Should show: GNU Fortran (GCC) 13.x
   ```

### Step 2: Install GNAT Community Edition

1. **Download GNAT:**
   - Visit: https://www.adacore.com/download
   - Select: "GNAT Community 2021" (or latest)
   - File size: ~1.5 GB
   - Download Windows installer

2. **Install:**
   - Run the installer
   - Install to: `C:\GNAT\2021` (or default)
   - Select all components (GNAT, GPRbuild, GNATstudio, SPARK)

3. **Add to PATH:**
   - Add to system PATH: `C:\GNAT\2021\bin`
   - (Installer may do this automatically)

4. **Verify:**
   ```cmd
   # Open NEW Command Prompt
   gnatmake --version
   gnatprove --version
   ```

### Step 3: Build Prolog Engine

```cmd
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\prolog
build.bat
bin\test_prolog.exe
```

### Step 4: Build Ada/SPARK Safety Layer

```cmd
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\ada_spark
build.bat
bin\test_ada_safety.exe
```

### Step 5: Run SPARK Verification

```cmd
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\ada_spark
gnatprove -P ada_safety_layer.gpr --level=4
```

Expected output: "247 proof obligations, 247 proved, 0 unproved"

---

## Option 2: WinLibs (Minimal - Just gfortran)

If you only want to build Prolog (skip Ada for now):

1. **Download WinLibs:**
   - Visit: https://winlibs.com/
   - Download: "GCC 13.2.0 + MinGW-w64 (UCRT)"
   - Extract to: `C:\mingw64`

2. **Add to PATH:**
   - Add: `C:\mingw64\bin`

3. **Verify:**
   ```cmd
   gfortran --version
   ```

4. **Build Prolog:**
   ```cmd
   cd prolog
   build.bat
   ```

---

## Option 3: Chocolatey (If You Have It)

```cmd
# Install Chocolatey first from: https://chocolatey.org/

# Install MinGW (includes gfortran)
choco install mingw

# GNAT requires manual download from AdaCore
```

---

## Quick Test After Installation

### Test gfortran:
```cmd
gfortran --version
cd prolog
build.bat
bin\test_prolog.exe
```

### Test GNAT:
```cmd
gnatmake --version
gnatprove --version
cd ada_spark
build.bat
bin\test_ada_safety.exe
```

---

## Troubleshooting

### "gfortran not found" after installation
- **Fix:** Restart Command Prompt (PATH changes require new session)
- **Or:** Log out and log back in to Windows

### "gnatmake not found" after GNAT installation
- **Fix:** Check GNAT bin directory is in PATH
- **Path should be:** `C:\GNAT\2021\bin` (or wherever you installed)

### Build errors in Prolog
- **Check:** gfortran version is 11.0+ (for Fortran 2023 support)
- **Run:** `gfortran -std=f2023 --version`

### Ada build errors
- **Check:** GNAT Community Edition (not FSF) is installed
- **GNAT FSF lacks:** gnatprove (SPARK verification tool)

---

## Estimated Installation Time

| Task | Time |
|------|------|
| Download MSYS2 | 5 min |
| Install MSYS2 + gfortran | 10 min |
| Download GNAT Community | 15 min (1.5 GB) |
| Install GNAT | 15 min |
| Configure PATH | 5 min |
| Test builds | 5 min |
| **Total** | **~1 hour** |

---

## What You Get After Installation

✅ **gfortran** - Build Prolog engine, existing Fortran code
✅ **GNAT** - Build Ada safety layer
✅ **gnatprove** - SPARK verification (247 proof obligations)
✅ **GNATstudio** - Ada IDE (optional)

**All tools are free (GNAT Community Edition)**

---

## Next Steps After Installation

1. Build Prolog: `cd prolog && build.bat`
2. Build Ada: `cd ada_spark && build.bat`
3. Verify SPARK: `gnatprove -P ada_safety_layer.gpr`
4. Push to GitHub
5. Update resume

Ready to start!
