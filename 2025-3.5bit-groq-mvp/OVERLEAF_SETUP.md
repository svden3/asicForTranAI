# 📝 Overleaf Setup Guide
**For ICML 2025 Submission**
**Date:** December 18, 2025

---

## Why Overleaf?

**Benefits:**
- ✅ No LaTeX installation needed (works in browser)
- ✅ Real-time collaboration (share with reviewers)
- ✅ Automatic compilation (no manual pdflatex commands)
- ✅ Version history (Git-based)
- ✅ Easy PDF download
- ✅ Free for academic use

**Perfect for:**
- Internal review (reviewers can comment inline)
- Collaborative writing
- Systems without LaTeX installed

---

## Quick Start (5 minutes)

### Step 1: Create Overleaf Account

1. Go to: https://www.overleaf.com
2. Click **"Register"** (top right)
3. Use institutional email (e.g., .edu) for free premium features
4. Or use Gmail/GitHub login

### Step 2: Create New Project

**Option A: Upload ZIP (Recommended)**

1. Create ZIP of paper files:
```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
zip -r paper_upload.zip paper.tex supplementary.tex figures/*.pdf
```

2. In Overleaf:
   - Click **"New Project"** → **"Upload Project"**
   - Select `paper_upload.zip`
   - Wait for upload (30 seconds)

**Option B: Manual Upload**

1. In Overleaf:
   - Click **"New Project"** → **"Blank Project"**
   - Name: "3.5bit-quantization-icml2025"

2. Upload files one by one:
   - Click **"Upload"** icon (top left)
   - Add `paper.tex`
   - Add `supplementary.tex`
   - Create `figures/` folder
   - Upload all 8 PDF figures

**Option C: Import from GitHub**

1. Push paper files to GitHub:
```bash
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp
git add paper/
git commit -m "Add paper files for ICML 2025"
git push
```

2. In Overleaf:
   - Click **"New Project"** → **"Import from GitHub"**
   - Select repository
   - Choose `paper/` subfolder

---

## Step 3: Compile Paper

1. Overleaf auto-compiles on file changes
2. Or click **"Recompile"** button (top right)
3. PDF appears in right pane
4. If errors: Check **"Logs and output files"** (bottom right)

**Expected result:**
- Paper compiles successfully ✅
- PDF shows title, abstract, introduction ✅
- Figures appear correctly ✅

---

## Step 4: Share with Reviewers

### For Internal Review

**Method 1: Share Link (Read-Only)**

1. Click **"Share"** button (top right)
2. Toggle **"Link Sharing"** to ON
3. Copy link (e.g., `https://www.overleaf.com/read/abcd1234`)
4. Send link to reviewers

**Reviewers can:**
- View compiled PDF
- Browse source files
- Download PDF
- Cannot edit (read-only)

**Method 2: Invite Collaborators (Read & Comment)**

1. Click **"Share"** → **"Invite Collaborators"**
2. Enter reviewer email addresses
3. Set permission to **"Can View"** or **"Can Edit"**
4. Click **"Share"**

**Reviewers can:**
- View PDF
- Add comments (if "Can Edit")
- Track changes
- Download PDF

---

## File Structure in Overleaf

```
project-root/
├── paper.tex                    # Main paper (8 pages)
├── supplementary.tex            # Supplementary (10 pages)
├── figures/
│   ├── figure1_model_size.pdf
│   ├── figure2_throughput.pdf
│   ├── figure3_pareto.pdf
│   ├── figure4_layer_breakdown.pdf
│   ├── figure5_bit_packing.pdf
│   ├── accuracy_vs_bitwidth.pdf
│   ├── performance_comparison.pdf
│   └── scalability.pdf
├── paper.bbl                    # Bibliography (after bibtex)
└── references.bib               # BibTeX file (if separate)
```

---

## Common Issues & Fixes

### Issue 1: Figures Not Showing

**Symptom:** Empty boxes where figures should be

**Fix:**
1. Check figure paths in paper.tex:
   ```latex
   \includegraphics{figures/figure1_model_size.pdf}
   ```
2. Ensure figures are in `figures/` folder
3. Recompile

### Issue 2: Bibliography Not Compiling

**Symptom:** [?] instead of citation numbers

**Fix:**
1. Click **"Logs and output files"**
2. Find "Run BibTeX" or similar
3. Overleaf auto-runs bibtex, but may need manual trigger
4. Or add references inline:
   ```latex
   \begin{thebibliography}{99}
   \bibitem{gptq} Frantar et al., GPTQ, 2023.
   \end{thebibliography}
   ```

### Issue 3: Compilation Timeout

**Symptom:** "Compilation timeout exceeded"

**Fix:**
1. Reduce figure sizes (compress PDFs)
2. Remove unnecessary packages
3. Split into smaller documents
4. Upgrade to premium (longer timeout)

### Issue 4: Package Not Found

**Symptom:** "! LaTeX Error: File 'packagename.sty' not found"

**Fix:**
1. Overleaf includes most packages by default
2. Check package name spelling
3. Use alternative package
4. Contact Overleaf support (usually auto-resolves)

---

## Compiling Both Documents

### Main Paper

1. Set main document: Right-click `paper.tex` → **"Set as Main File"**
2. Click **"Recompile"**
3. Download: **Menu** → **"Download PDF"** → Save as `paper.pdf`

### Supplementary Materials

1. Right-click `supplementary.tex` → **"Set as Main File"**
2. Click **"Recompile"**
3. Download: **Menu** → **"Download PDF"** → Save as `supplementary.pdf`

**Result:** Two separate PDFs ready for submission

---

## Version Control

### Track Changes

**Enable:**
1. Click **"Review"** (top menu)
2. Toggle **"Track Changes"** to ON
3. All edits show in colored annotations

**Review changes:**
1. Accept/reject changes individually
2. Or accept all: **"Review"** → **"Accept All Changes"**

### Version History

**Access:**
1. Click **"History"** (top menu)
2. See all previous versions
3. Click version to preview
4. **"Restore"** to revert

**Use cases:**
- Revert bad edits
- Compare reviewer feedback versions
- Track progress over time

---

## Collaboration Features

### Comments

**Add comment:**
1. Highlight text in source
2. Click **"Add Comment"** (right sidebar)
3. Type comment
4. Reviewers see in sidebar

**Reply to comment:**
1. Click comment thread
2. Type reply
3. Mark as **"Resolved"** when addressed

### Track Changes (For Reviewers)

**Enable:**
1. Share project with "Can Edit" permission
2. Ask reviewers to enable **"Track Changes"**
3. All edits show as suggestions
4. You approve/reject changes

---

## Download Options

### Download PDF

1. **Menu** → **"Download"** → **"PDF"**
2. Or click PDF pane → Download icon

### Download Source

1. **Menu** → **"Download"** → **"Source"**
2. Downloads ZIP of all .tex, .bib, and figures
3. Can compile locally later

### Download with Submission Files

1. **Menu** → **"Download"** → **"PDF with submission files"**
2. Includes PDF + LaTeX source in ZIP
3. Ready for conference submission

---

## Premium Features (Optional)

**Free Plan:**
- 1 collaborator
- 60-second compile timeout
- Basic features

**Premium ($12/month or $89/year):**
- Unlimited collaborators ✨
- 4-minute compile timeout
- Track changes
- Advanced history
- Dropbox/GitHub sync

**Student/Academic:**
- Often free with .edu email
- Check: https://www.overleaf.com/edu

---

## ICML 2025 Template Integration

### Step 1: Download ICML Template

1. Go to: https://icml.cc/Conferences/2025/StyleAuthorInstructions
2. Download `icml2025.zip`
3. Extract files

### Step 2: Upload to Overleaf

**Method 1: Upload ICML ZIP**

1. In Overleaf: **"New Project"** → **"Upload Project"**
2. Upload `icml2025.zip`
3. Overleaf creates project with template

**Method 2: Copy Style Files**

1. From `icml2025.zip`, upload:
   - `icml2025.sty` (style file)
   - `icml2025.bst` (bibliography style)
   - Any other .cls or .sty files

2. Update `paper.tex`:
   ```latex
   \documentclass{icml2025}
   % Rest of paper
   ```

### Step 3: Merge Your Content

1. Copy content from your `paper.tex` to ICML template
2. Adjust formatting:
   - Title: `\icmltitle{Your Title}`
   - Authors: `\icmlauthor{Anonymous}{Anonymous}` (for submission)
   - Abstract: `\icmlabstract{Your abstract}`

3. Recompile and verify

---

## Sharing for Internal Review

### Create Shareable Link

1. **Share** → **"Link Sharing"** → ON
2. Copy link: `https://www.overleaf.com/read/abcd1234efgh`
3. Send to reviewers with email template

### Email to Reviewers

```
Hi [Name],

I've uploaded the paper to Overleaf for easy review:

📄 Paper Link: https://www.overleaf.com/read/abcd1234efgh

You can:
• View the compiled PDF (right pane)
• Browse LaTeX source (left pane)
• Download PDF (Menu → Download)
• No Overleaf account needed for viewing

Please provide feedback by December 23.

Thanks!
```

### Track Who Viewed

1. Premium feature: See who opened the link
2. Free plan: Ask reviewers to confirm receipt

---

## Exporting for Submission

### Final Submission Package

1. **Menu** → **"Download"** → **"Source"**
2. Extract ZIP locally
3. Verify contains:
   - `paper.tex`
   - `supplementary.tex`
   - `figures/*.pdf`
   - `paper.bbl` (bibliography)
   - `icml2025.sty` (if using ICML template)

4. Test local compilation:
```bash
cd extracted_folder
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

5. If compiles: Ready for submission ✅

---

## Backup Strategy

### Overleaf Auto-Saves

- Every keystroke saved to cloud ✅
- Version history for 30 days (free) or forever (premium)

### Additional Backups

**Option 1: Download Weekly**
- Download source ZIP every week
- Save to local drive + cloud (Dropbox, Google Drive)

**Option 2: GitHub Sync**
- Premium feature: Auto-sync to GitHub
- Or manually push source files

**Option 3: Email to Self**
- Download PDF + source
- Email to yourself with date stamp

---

## Troubleshooting

### Can't Upload Figures

**Issue:** File size too large (>50MB project limit on free plan)

**Fix:**
1. Compress PDFs: Use `gs` or online tools
   ```bash
   gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 \
      -dPDFSETTINGS=/ebook -dNOPAUSE -dQUIET \
      -dBATCH -sOutputFile=compressed.pdf input.pdf
   ```
2. Or upgrade to premium (bigger limit)

### Collaborator Can't Edit

**Issue:** Invited collaborator sees read-only

**Fix:**
1. Check permission: **"Can Edit"** not "Can View"
2. Re-send invitation
3. Ask collaborator to log in (not anonymous)

### Compilation Errors

**Issue:** Red error messages

**Fix:**
1. Click **"Logs and output files"**
2. Scroll to first error
3. Common fixes:
   - Missing `\end{document}`
   - Typo in command: `\includ` → `\include`
   - Missing package: Add `\usepackage{packagename}`

### PDF Not Updating

**Issue:** Changes don't appear in PDF

**Fix:**
1. Manual recompile: Click **"Recompile"**
2. Clear cache: **"Logs"** → **"Clear cached files"**
3. Hard refresh browser: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

---

## Quick Reference

### Keyboard Shortcuts

| Action | Windows/Linux | Mac |
|--------|---------------|-----|
| **Recompile** | Ctrl+S | Cmd+S |
| **Find** | Ctrl+F | Cmd+F |
| **Find & Replace** | Ctrl+H | Cmd+H |
| **Comment Line** | Ctrl+/ | Cmd+/ |
| **Auto-complete** | Tab | Tab |

### Menu Locations

- **Share project:** Top right button
- **Download:** Menu (☰) → Download
- **History:** Top menu
- **Settings:** Menu (☰) → Settings
- **Logs:** Bottom right (expand)

---

## For Reviewers: How to Use

### Viewing the Paper

1. Click link: `https://www.overleaf.com/read/...`
2. PDF loads automatically (right pane)
3. Scroll through PDF
4. No account needed for viewing

### Adding Comments (If Invited)

1. Create free Overleaf account
2. Accept invitation email
3. Highlight text in source (left pane)
4. Click **"Add Comment"** (right sidebar)
5. Type feedback
6. Author sees comments and can reply

### Downloading PDF

1. Click PDF pane (right side)
2. Click download icon (top right of PDF)
3. Or: **Menu** → **"Download"** → **"PDF"**

---

## Timeline Integration

### Internal Review Phase (Dec 19-23)

**Dec 19:** Upload to Overleaf, share link with reviewers
**Dec 20-23:** Reviewers provide feedback (via comments or email)
**Dec 24-27:** Address feedback, update Overleaf project
**Dec 28:** Download revised version, proceed to ICML formatting

### Submission Phase (Jan 28-31)

**Jan 28:** Download final source from Overleaf
**Jan 28:** Upload to ICML OpenReview portal
**Jan 29-31:** Buffer time for any issues

---

## Alternative: Local LaTeX

If you prefer local compilation over Overleaf:

### Windows (MiKTeX)

```bash
# Install MiKTeX: https://miktex.org/download
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp\paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Linux

```bash
# Install TeX Live
sudo apt install texlive-full

cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Mac

```bash
# Install MacTeX: https://www.tug.org/mactex/
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## Summary Checklist

- [ ] Create Overleaf account
- [ ] Upload paper files (paper.tex, supplementary.tex, figures)
- [ ] Verify compilation succeeds
- [ ] Generate share link
- [ ] Send link to 2-3 reviewers
- [ ] Monitor comments and feedback
- [ ] Download final version for ICML submission

**Estimated setup time:** 10-15 minutes
**Benefit:** Easy collaboration, no LaTeX installation needed

---

**Created:** December 18, 2025
**For:** Internal review and ICML 2025 submission preparation
**Support:** https://www.overleaf.com/learn (tutorials and documentation)
