# LaTeX Compilation Guide

This guide provides instructions for compiling your paper into PDF.

## Option 1: Overleaf (Recommended - No Installation Required)

**Overleaf** is a free online LaTeX editor that requires no installation.

### Steps:

1. **Go to Overleaf:**
   - Visit: https://www.overleaf.com/
   - Sign up for a free account (or log in)

2. **Create New Project:**
   - Click "New Project" → "Upload Project"
   - Create a ZIP file with your paper files (see below)
   - Upload and extract

3. **Files to Upload:**
   ```
   paper/
   ├── paper.tex
   ├── supplementary.tex
   └── figures/
       ├── figure1_model_size.pdf
       ├── figure2_throughput.pdf
       ├── figure3_pareto.pdf
       ├── figure4_layer_breakdown.pdf
       ├── figure5_bit_packing.pdf
       ├── accuracy_vs_bitwidth.pdf
       ├── performance_comparison.pdf
       └── scalability.pdf
   ```

4. **Compile:**
   - Overleaf auto-compiles when you make changes
   - Or click the "Recompile" button
   - View the PDF in the right panel
   - Download PDF: Click "Download PDF" button

### Create ZIP for Overleaf:

Run this command to create a ZIP file:

```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
powershell Compress-Archive -Path paper.tex,supplementary.tex,figures -DestinationPath paper_submission.zip -Force
```

Then upload `paper_submission.zip` to Overleaf.

---

## Option 2: Local Installation (Requires Admin Rights)

### Install MiKTeX (Windows):

1. **Open PowerShell as Administrator:**
   - Press Windows key
   - Type "PowerShell"
   - Right-click "Windows PowerShell"
   - Select "Run as Administrator"

2. **Install MiKTeX:**
   ```powershell
   choco install miktex -y
   ```

   Or download manually from: https://miktex.org/download

3. **Restart your terminal** to refresh PATH

### Compile Locally:

```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper

# Compile main paper
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
bibtex paper        # If you have bibliography
pdflatex paper.tex  # Final compilation

# Compile supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

**Output:** `paper.pdf` and `supplementary.pdf`

---

## Option 3: Docker LaTeX (Advanced)

If you have Docker installed:

```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper

# Pull LaTeX Docker image
docker pull texlive/texlive:latest

# Compile paper
docker run --rm -v ${PWD}:/work -w /work texlive/texlive pdflatex paper.tex
docker run --rm -v ${PWD}:/work -w /work texlive/texlive pdflatex paper.tex

# Compile supplementary
docker run --rm -v ${PWD}:/work -w /work texlive/texlive pdflatex supplementary.tex
docker run --rm -v ${PWD}:/work -w /work texlive/texlive pdflatex supplementary.tex
```

---

## Troubleshooting

### Missing LaTeX Packages

If compilation fails with "package not found":

**MiKTeX (automatic):**
```bash
mpm --install=<package-name>
```

**Or:** Enable automatic package installation in MiKTeX Console

**Overleaf:** Packages are pre-installed, no action needed

### Common Errors

**Error: "File not found"**
- Check that all figure files are in `figures/` directory
- Ensure file names match exactly (case-sensitive)

**Error: "Undefined control sequence"**
- Check LaTeX syntax in `.tex` files
- Ensure all required packages are loaded

**Error: "Missing $ inserted"**
- Math expressions must be in `$...$` or `\[...\]`
- Check for unescaped special characters

---

## Quick Start: Overleaf Method

**Fastest way to get your PDFs (5 minutes):**

1. Create ZIP:
   ```bash
   cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
   powershell Compress-Archive -Path paper.tex,supplementary.tex,figures -DestinationPath paper_submission.zip -Force
   ```

2. Upload to Overleaf: https://www.overleaf.com/

3. Download compiled PDFs

**Done!** You now have:
- `paper.pdf` - Main paper (ready for submission)
- `supplementary.pdf` - Supplementary materials

---

## Current Status

✅ All LaTeX source files ready
✅ All 8 figures generated (PDF format)
✅ Ready for compilation

**Next Step:** Choose compilation method above (Overleaf recommended)

---

**Created:** 2025-12-02
**Location:** `C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper\`
