# Install LaTeX Locally (Privacy-Focused)

## Method 1: Manual MiKTeX Installation (Recommended)

### Step 1: Download MiKTeX Installer

1. Download the installer directly:
   - Go to: https://miktex.org/download
   - Click "Download" for Windows
   - Or direct link: https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/basic-miktex-24.1-x64.exe

2. Save to your Downloads folder

### Step 2: Install MiKTeX

1. **Right-click** the downloaded `.exe` file
2. Select **"Run as administrator"**
3. Follow the installation wizard:
   - Installation scope: "Install for all users" (recommended)
   - Install directory: Default (`C:\Program Files\MiKTeX`)
   - Settings: Check "Install missing packages on-the-fly: Yes"
   - Click "Next" and "Install"

4. **Restart your terminal** after installation completes

### Step 3: Verify Installation

Open a new terminal and run:
```bash
pdflatex --version
```

You should see: `MiKTeX-pdfTeX 4.x...`

### Step 4: Compile Your Papers

```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper

# Compile main paper
pdflatex paper.tex
pdflatex paper.tex  # Run twice for cross-references

# Compile supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

**Output:**
- `paper.pdf`
- `supplementary.pdf`

---

## Method 2: Portable MiKTeX (No Installation)

If you don't want to install system-wide:

1. Download MiKTeX Portable:
   - https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/miktex-portable-24.1-x64.exe

2. Run the portable installer (no admin needed)
3. Choose installation folder (e.g., `C:\MiKTeX-Portable`)
4. Add to PATH temporarily:
   ```powershell
   $env:PATH = "C:\MiKTeX-Portable\texmfs\install\miktex\bin\x64;$env:PATH"
   ```

5. Compile as above

---

## Method 3: Use Chocolatey with Admin PowerShell

1. **Close this terminal**
2. **Open PowerShell as Administrator:**
   - Press Windows key
   - Type "PowerShell"
   - Right-click "Windows PowerShell"
   - Select "Run as Administrator"

3. **Install MiKTeX:**
   ```powershell
   choco install miktex -y
   ```

4. **Wait for installation** (5-10 minutes)

5. **Restart terminal** and verify:
   ```powershell
   pdflatex --version
   ```

6. **Navigate to paper folder:**
   ```powershell
   cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
   ```

7. **Compile:**
   ```powershell
   pdflatex paper.tex
   pdflatex paper.tex
   pdflatex supplementary.tex
   pdflatex supplementary.tex
   ```

---

## Troubleshooting

### Missing Packages Error

If you see "File 'xxx.sty' not found":

**MiKTeX Console (GUI):**
1. Open "MiKTeX Console" from Start Menu
2. Go to "Packages" tab
3. Search for the missing package
4. Click "Install"

**Command line:**
```bash
mpm --install=<package-name>
```

**Or enable auto-install:**
1. Open MiKTeX Console
2. Settings → "Install missing packages on-the-fly" → Yes

### Common Packages Needed

The paper uses these packages (should auto-install):
- times
- amsmath
- amssymb
- amsthm
- algorithm
- algorithmic
- graphicx
- booktabs
- url
- hyperref

---

## Privacy Notes

**Local compilation is completely private:**
- ✅ No files uploaded anywhere
- ✅ All processing on your machine
- ✅ PDFs created locally
- ✅ No internet connection needed (after MiKTeX installed)

**MiKTeX download:**
- Downloads LaTeX packages from CTAN mirrors
- Only downloads missing packages (if auto-install enabled)
- All standard, open-source LaTeX packages

---

## Expected Output

After successful compilation:

```
paper/
├── paper.pdf          ← Main paper (your submission)
├── supplementary.pdf  ← Supplementary materials
├── paper.tex
├── supplementary.tex
├── paper.aux
├── paper.log
└── figures/
```

**File sizes (approximate):**
- `paper.pdf`: ~500-800 KB (with embedded figures)
- `supplementary.pdf`: ~400-600 KB

---

## Quick Start Commands

**After installing MiKTeX:**

```bash
# Navigate to paper directory
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper

# Compile both papers
pdflatex paper.tex && pdflatex paper.tex
pdflatex supplementary.tex && pdflatex supplementary.tex

# Check output
ls -lh *.pdf
```

**Done!** You'll have:
- `paper.pdf` - Ready for submission
- `supplementary.pdf` - Extended materials

---

**Installation time:** 5-10 minutes
**Compilation time:** 1-2 minutes per paper
**Total time:** ~15 minutes from start to PDFs
