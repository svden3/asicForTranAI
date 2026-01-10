# ✅ ICML 2025 Conversion Checklist
**Target Deadline:** February 1, 2025 (44 days)
**Conversion Timeline:** January 1-15, 2025 (15 days)

---

## Overview

This checklist guides conversion of your paper from generic article format to ICML 2025 submission format.

**Current Format:** `\documentclass{article}` (generic)
**Target Format:** `\documentclass{icml2025}` (conference template)

---

## Phase 1: Download ICML Template (Jan 1-3)

### Step 1: Get Official Template

**Website:** https://icml.cc/Conferences/2025/StyleAuthorInstructions

**Download:**
- [ ] Visit ICML 2025 website
- [ ] Find "Author Kit" or "Style Files" section
- [ ] Download `icml2025.zip` or similar
- [ ] Extract to working directory

**Expected files in template:**
```
icml2025/
├── icml2025.sty          # Main style file
├── icml2025.bst          # Bibliography style
├── sample_paper.tex      # Example paper
├── README               # Instructions
└── fancyhdr.sty         # Header/footer styling (maybe)
```

### Step 2: Review Template Requirements

**Read carefully:**
- [ ] Page limit: 8 pages + unlimited references ✅
- [ ] Margin requirements
- [ ] Font requirements (usually Times)
- [ ] Anonymization rules (double-blind)
- [ ] Supplementary material guidelines

**ICML 2025 Specific Requirements:**
- **Format:** Two-column, 10pt font
- **Page limit:** 8 pages (excluding references)
- **Supplementary:** Unlimited pages, separate PDF
- **Anonymization:** Required (no author names/affiliations)
- **Code:** Encouraged but not required
- **Submission platform:** OpenReview

---

## Phase 2: Create ICML Version (Jan 4-8)

### Step 1: Setup New File

**Create `paper_icml.tex`:**

```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
cp paper.tex paper_icml.tex
```

**Or in ICML template folder:**
```bash
cd icml2025
cp sample_paper.tex ../paper/paper_icml.tex
# Then merge your content
```

### Step 2: Update Document Class

**Change line 1 of `paper_icml.tex`:**

❌ **Remove:**
```latex
\documentclass{article}
```

✅ **Replace with:**
```latex
\documentclass{icml2025}
```

**Copy style files to paper directory:**
```bash
cp icml2025/icml2025.sty paper/
cp icml2025/icml2025.bst paper/
```

### Step 3: Update Preamble

**ICML-specific commands:**

```latex
\documentclass{icml2025}

% Use ICML packages (check sample_paper.tex)
\usepackage{times}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{booktabs}
\usepackage{hyperref}

% ICML-specific settings
\icmltitlerunning{3.5-bit Quantization for LLM Inference}

\begin{document}

% Title (ICML format)
\icmltitle{3.5-bit Dynamic Asymmetric Quantization for \\
           Large Language Model Inference on ASIC Hardware}

% Authors (ANONYMIZED for submission)
% DO NOT include real names for initial submission
\icmlsetsymbol{equal}{*}

\begin{icmlauthorlist}
\icmlauthor{Anonymous Author 1}{equal,first}
\icmlauthor{Anonymous Author 2}{equal,second}
\end{icmlauthorlist}

\icmlaffiliation{first}{Anonymous Institution 1}
\icmlaffiliation{second}{Anonymous Institution 2}

\icmlcorrespondingauthor{Anonymous}{anonymous@email.com}

% Keywords
\icmlkeywords{Quantization, Large Language Models, LLM Compression, ASIC Hardware, Efficient Inference, Low-Bit Quantization}

% Abstract (ICML format)
\vskip 0.3in
\begin{abstract}
We present the first 3.5-bit quantization scheme for large language model (LLM)
inference, achieving 28.86\% speedup over INT4 quantization on Groq ASIC hardware
while improving quality by 10.6\%. Our approach combines asymmetric bit packing
(4-bit + 3-bit alternation) with dynamic per-column quantization, reducing model
size by 46\% compared to INT4. On LLaMA-70B, we achieve 14.94\% RMSE (vs 16.72\%
for INT4) with a model footprint of 32.6GB (vs 34.6GB for INT4). We provide a
pure Fortran 2023 implementation (78 lines) compiled via MLIR to Groq LPU,
achieving 6.995$\times$ CPU speedup with OpenMP+SIMD. All code and benchmarks
are publicly available.
\end{abstract}

% Rest of paper
\section{Introduction}
...
```

### Step 4: Content Mapping Checklist

**Map your sections to ICML format:**

- [ ] Title → `\icmltitle{}`
- [ ] Authors → `\icmlauthorlist{}` (anonymized)
- [ ] Abstract → `\begin{abstract}...\end{abstract}`
- [ ] Introduction → `\section{Introduction}`
- [ ] Related Work → `\section{Related Work}`
- [ ] Method → `\section{Method}` or `\section{3.5-bit Quantization}`
- [ ] Experiments → `\section{Experiments}`
- [ ] Results → `\section{Results}` (or merge with Experiments)
- [ ] Discussion → `\section{Discussion}` (optional)
- [ ] Conclusion → `\section{Conclusion}`
- [ ] References → `\bibliographystyle{icml2025}\bibliography{references}`
- [ ] Appendix → Supplementary material (separate file)

---

## Phase 3: Anonymization (Jan 4-8)

### Critical: Double-Blind Requirements

**REMOVE from paper:**
- [ ] ❌ Author names
- [ ] ❌ Author affiliations
- [ ] ❌ Author emails
- [ ] ❌ Acknowledgments section
- [ ] ❌ Funding information
- [ ] ❌ GitHub repository URL (or use anonymous)
- [ ] ❌ Any self-identifying information

**ANONYMIZE in text:**

❌ **Don't write:**
> "Our prior work [Xiao et al., 2024] showed..."

✅ **Instead write:**
> "Prior work [Anonymous, 2024] showed..."

❌ **Don't write:**
> "Code available at https://github.com/yourusername/repo"

✅ **Instead write:**
> "Code will be made publicly available upon acceptance."
>
> OR (if required):
> "Code available at anonymous repository: https://anonymous.4open.science/r/3p5bit-XXXX/"

**Self-citations:**
- [ ] Change self-citations to "[Anonymous, 2024]"
- [ ] Add to bibliography as:
  ```bibtex
  @article{anonymous2024,
    author = {Anonymous},
    title = {[Title of your prior work]},
    year = {2024},
    note = {Details omitted for anonymous review}
  }
  ```

### Create Anonymous GitHub (If Needed)

**Option 1: Anonymous4Science**
1. Go to: https://anonymous.4open.science
2. Upload code repository as ZIP
3. Get anonymous link: `https://anonymous.4open.science/r/3p5bit-XXXX/`
4. Link expires after review period
5. Use in paper: "Code: [anonymous link]"

**Option 2: Create Temporary Account**
1. Create new GitHub account: "icml2025-submission-1234"
2. Fork your repository
3. Remove identifying info from README
4. Set repository to public
5. Use in paper (will delete after review)

---

## Phase 4: Format Adjustments (Jan 9-12)

### Page Limit Compliance

**ICML limit:** 8 pages + unlimited references

**Check current page count:**
```bash
pdflatex paper_icml.tex
# Count pages in generated PDF (excluding references)
```

**If over 8 pages:**
- [ ] Move content to supplementary materials
- [ ] Shorten introduction/related work
- [ ] Make figures smaller
- [ ] Use more compact notation
- [ ] Reduce whitespace

**If under 8 pages:**
- [ ] Add more experimental details
- [ ] Expand related work
- [ ] Add additional figures
- [ ] Include more ablation studies

### Figure Formatting

**ICML figure requirements:**

- [ ] All figures referenced in text: `\ref{fig:model_size}`
- [ ] Figure captions below figures
- [ ] Clear labels and legends
- [ ] High resolution (300 DPI minimum)
- [ ] Readable in grayscale (some reviewers print B&W)

**Example:**
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\textwidth]{figures/figure1_model_size.pdf}
\caption{Model size comparison: Our 3.5-bit quantization achieves 46\%
         reduction vs INT4 (32.6GB vs 34.6GB for LLaMA-70B).}
\label{fig:model_size}
\end{figure}
```

### Table Formatting

**Use `booktabs` for professional tables:**

```latex
\begin{table}[t]
\centering
\caption{Performance comparison on LLaMA-70B.}
\label{tab:performance}
\begin{tabular}{lccc}
\toprule
Method & Size (GB) & RMSE (\%) & Time (ms) \\
\midrule
FP16 & 130.4 & 0.0 & 150.2 \\
INT8 & 65.2 & 5.3 & 120.5 \\
INT4 & 34.6 & 16.7 & 90.1 \\
\textbf{3.5-bit (Ours)} & \textbf{32.6} & \textbf{14.9} & \textbf{69.9} \\
\bottomrule
\end{tabular}
\end{table}
```

### Equation Formatting

**Number important equations:**

```latex
\begin{equation}
\label{eq:quantization}
\mathbf{w}_{quant} = \text{clip}\left(\left\lfloor \frac{\mathbf{w} - z}{s} \right\rfloor, q_{min}, q_{max}\right)
\end{equation}
```

**Reference in text:**
> "As shown in Equation~\ref{eq:quantization}, we quantize weights..."

---

## Phase 5: Bibliography (Jan 9-12)

### Use ICML Bibliography Style

**Update bibliography section:**

```latex
% At end of paper, before \end{document}

\bibliographystyle{icml2025}
\bibliography{references}
```

**Create `references.bib`:**

```bibtex
@article{gptq2023,
  author = {Frantar, Elias and Ashkboos, Saleh and others},
  title = {{GPTQ}: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  journal = {arXiv preprint arXiv:2210.17323},
  year = {2023}
}

@article{awq2023,
  author = {Lin, Ji and Tang, Jiaming and others},
  title = {{AWQ}: Activation-aware Weight Quantization for {LLM} Compression and Acceleration},
  journal = {arXiv preprint arXiv:2306.00978},
  year = {2023}
}

@article{smoothquant2023,
  author = {Xiao, Guangxuan and Lin, Ji and others},
  title = {{SmoothQuant}: Accurate and Efficient Post-Training Quantization for Large Language Models},
  journal = {ICML},
  year = {2023}
}

% Add 8-10 more key references
```

**Compile bibliography:**
```bash
pdflatex paper_icml.tex
bibtex paper_icml
pdflatex paper_icml.tex
pdflatex paper_icml.tex
```

---

## Phase 6: Supplementary Material (Jan 9-12)

### Convert to ICML Supplementary Format

**Create `supplementary_icml.tex`:**

```latex
\documentclass{article}  % Can use article class for supplementary
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{listings}

\title{\textbf{Supplementary Materials:}\\
3.5-bit Dynamic Asymmetric Quantization for LLM Inference}

\author{Anonymous}  % Keep anonymous
\date{}

\begin{document}
\maketitle

\section{Complete Algorithm Listings}
\subsection{Python Implementation}
\begin{lstlisting}[language=Python]
# Include full quantization code
\end{lstlisting}

\subsection{Fortran Implementation}
\begin{lstlisting}[language=Fortran]
! Include full Fortran code
\end{lstlisting}

\section{Extended Experimental Results}
\subsection{Ablation Studies}
% Additional tables and figures

\subsection{Layer-wise Analysis}
% Per-layer RMSE breakdown

\section{Reproducibility Guide}
\subsection{Hardware Requirements}
\subsection{Installation Instructions}
\subsection{Running Benchmarks}

\section{Formal Verification Details}
\subsection{Lean 4 Proofs}
% Include proof sketches

\end{document}
```

---

## Phase 7: Compilation & Testing (Jan 13-15)

### Compile ICML Version

**Full compilation:**
```bash
cd paper
pdflatex paper_icml.tex
bibtex paper_icml
pdflatex paper_icml.tex
pdflatex paper_icml.tex
```

**Expected output:**
- `paper_icml.pdf` (main paper, 8 pages + references)

**Compile supplementary:**
```bash
pdflatex supplementary_icml.tex
bibtex supplementary_icml
pdflatex supplementary_icml.tex
pdflatex supplementary_icml.tex
```

**Expected output:**
- `supplementary_icml.pdf` (10-20 pages)

### Verification Checklist

**Format compliance:**
- [ ] Uses `\documentclass{icml2025}` ✅
- [ ] Page count ≤ 8 (excluding references) ✅
- [ ] Two-column layout ✅
- [ ] 10pt font ✅
- [ ] All figures and tables fit within margins ✅

**Content compliance:**
- [ ] Anonymous (no author names) ✅
- [ ] No self-identifying information ✅
- [ ] All figures referenced in text ✅
- [ ] All tables numbered ✅
- [ ] All equations numbered (if referenced) ✅
- [ ] Bibliography formatted correctly ✅

**Readability:**
- [ ] PDF opens correctly ✅
- [ ] Figures clear and readable ✅
- [ ] No overlapping text ✅
- [ ] Hyperlinks work (if any) ✅
- [ ] No compilation warnings ✅

### Test on Different Viewers

- [ ] Adobe Reader
- [ ] Browser PDF viewer (Chrome, Firefox)
- [ ] OpenReview preview (if available)
- [ ] Print preview (check margins)

---

## Phase 8: Final Proofreading (Jan 13-15)

### Proofreading ICML Version

**Focus areas:**

1. **Abstract** (most critical):
   - [ ] Clearly states contribution
   - [ ] Mentions key results (28.86% speedup, 10.6% improvement)
   - [ ] Under 250 words
   - [ ] No typos

2. **Introduction:**
   - [ ] Motivation clear
   - [ ] 4 contributions numbered
   - [ ] Novelty emphasized ("first 3.5-bit")

3. **Results:**
   - [ ] Numbers match benchmark JSON files
   - [ ] Tables formatted consistently
   - [ ] Figure captions descriptive

4. **References:**
   - [ ] All citations complete
   - [ ] No "[?]" or missing refs
   - [ ] Consistent formatting

### Tools

- [ ] Grammarly (browser extension)
- [ ] LanguageTool (open-source)
- [ ] Manual read-through (out loud)
- [ ] Colleague review (optional)

---

## Common Conversion Issues

### Issue 1: Page Count Exceeds 8 Pages

**Solutions:**
1. Move content to supplementary
2. Shorten related work
3. Reduce figure sizes
4. Use `\vspace{-2mm}` between sections (sparingly)
5. Use `\small` for tables/figures

### Issue 2: Figures Don't Fit

**Solutions:**
1. Reduce figure width:
   ```latex
   \includegraphics[width=0.45\textwidth]{figure.pdf}
   ```
2. Use two-column figures:
   ```latex
   \begin{figure*}[t]  % Asterisk for two-column
   \includegraphics[width=0.95\textwidth]{figure.pdf}
   \end{figure*}
   ```
3. Move to supplementary

### Issue 3: Bibliography Too Long

**Solutions:**
1. Use abbreviated journal names
2. Remove URLs (if not essential)
3. Cite survey papers instead of individual papers
4. Move some citations to supplementary

### Issue 4: Compilation Errors

**Common errors:**

**Error:** `! Undefined control sequence \icmltitle`
**Fix:** Ensure `icml2025.sty` is in same directory as `.tex` file

**Error:** `! LaTeX Error: File 'icml2025.sty' not found`
**Fix:** Copy style file to paper directory

**Error:** Bibliography not appearing
**Fix:** Run bibtex after first pdflatex:
```bash
pdflatex paper_icml.tex
bibtex paper_icml
pdflatex paper_icml.tex
pdflatex paper_icml.tex
```

---

## Pre-Submission Checklist (Jan 15)

### Final Verification

**Files ready:**
- [ ] `paper_icml.pdf` (main paper, ≤8 pages + refs)
- [ ] `supplementary_icml.pdf` (supplementary, 10-20 pages)
- [ ] Both PDFs <50MB (OpenReview limit)
- [ ] Both PDFs open correctly

**Content verification:**
- [ ] Title correct
- [ ] Authors anonymous
- [ ] Abstract compelling
- [ ] All figures clear
- [ ] All tables complete
- [ ] References formatted
- [ ] No typos in abstract/intro/conclusion

**Technical verification:**
- [ ] Numbers match benchmarks
- [ ] Equations correct
- [ ] Algorithm listings accurate
- [ ] Code availability stated

**Compliance:**
- [ ] ICML format (two-column, 10pt)
- [ ] Page limit (≤8 pages main)
- [ ] Anonymized (no author info)
- [ ] Supplementary properly labeled

---

## Submission to OpenReview (Jan 28-31)

### Upload to ICML 2025 Portal

1. Go to: https://openreview.net
2. Find ICML 2025 submission track
3. Click "Submit Paper"

**Required fields:**
- **Title:** [Copy from paper]
- **Abstract:** [Copy from paper, max 250 words]
- **Keywords:** Quantization, LLM, ASIC, Compression, Inference
- **Main PDF:** Upload `paper_icml.pdf`
- **Supplementary PDF:** Upload `supplementary_icml.pdf`
- **Code:** [Anonymous GitHub link or "Upon acceptance"]
- **Conflicts:** [List conflicted reviewers]

**Verify before final submit:**
- [ ] PDFs uploaded correctly
- [ ] Abstract matches paper
- [ ] All metadata filled
- [ ] Conflicts declared
- [ ] Confirmation email received

---

## Post-Submission (Feb 1 - May)

### If Accepted (May 2025)

**Camera-ready preparation:**
- [ ] De-anonymize (add real author names)
- [ ] Add acknowledgments
- [ ] Add GitHub link (public)
- [ ] Incorporate reviewer feedback
- [ ] Recompile and submit camera-ready

### If Rejected (May 2025)

**Next steps:**
- [ ] Read reviews carefully
- [ ] Address concerns
- [ ] Revise paper
- [ ] Submit to NeurIPS 2025 (May 29 deadline)
- [ ] Or submit to MLSys 2026 (September deadline)

---

## Conversion Timeline Summary

| Date | Task | Duration |
|------|------|----------|
| **Jan 1-3** | Download ICML template, review requirements | 3 days |
| **Jan 4-8** | Convert to ICML format, anonymize | 5 days |
| **Jan 9-12** | Format adjustments, bibliography | 4 days |
| **Jan 13-15** | Compile, verify, proofread | 3 days |
| **Jan 16-27** | Buffer time, final checks | 12 days |
| **Jan 28** | Submit to OpenReview | 1 day |
| **Jan 29-31** | Buffer before deadline | 3 days |
| **Feb 1** | Submission deadline | — |

**Total time:** 15 days (with buffer)

---

**Created:** December 18, 2025
**For:** ICML 2025 Submission (February 1, 2025)
**Next Step:** Download ICML template on January 1, 2025
