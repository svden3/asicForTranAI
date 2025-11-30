#!/bin/bash

# Generate PDF from ONE_PAGE_OVERVIEW.md for email attachments
# Usage: ./scripts/generate_pdf_overview.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT="$PROJECT_ROOT/docs/ONE_PAGE_OVERVIEW.md"
OUTPUT="$PROJECT_ROOT/docs/ONE_PAGE_OVERVIEW.pdf"

echo "📄 Generating PDF overview for Groq/Cerebras outreach..."
echo ""

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "❌ Error: pandoc is not installed"
    echo ""
    echo "Install options:"
    echo "  macOS:    brew install pandoc"
    echo "  Ubuntu:   sudo apt-get install pandoc texlive-latex-base"
    echo "  Windows:  choco install pandoc"
    echo ""
    echo "Alternative: Use online converter"
    echo "  1. Go to https://www.markdowntopdf.com/"
    echo "  2. Upload docs/ONE_PAGE_OVERVIEW.md"
    echo "  3. Download as PDF"
    exit 1
fi

# Check if LaTeX is installed (required for PDF generation)
if ! command -v pdflatex &> /dev/null && ! command -v xelatex &> /dev/null; then
    echo "⚠️  Warning: LaTeX not found. Installing BasicTeX..."
    echo ""
    echo "Install options:"
    echo "  macOS:    brew install --cask basictex"
    echo "  Ubuntu:   sudo apt-get install texlive-latex-base texlive-fonts-recommended"
    echo ""
    echo "Alternative: Use pandoc with HTML output"
    echo "  pandoc docs/ONE_PAGE_OVERVIEW.md -o docs/ONE_PAGE_OVERVIEW.html"
    echo "  Then print to PDF from browser"
    exit 1
fi

# Generate PDF
echo "🔧 Converting Markdown → PDF..."
pandoc "$INPUT" -o "$OUTPUT" \
    --pdf-engine=xelatex \
    -V geometry:margin=0.75in \
    -V fontsize=11pt \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    --metadata title="3.5-Bit Formally Verified LLM Inference" \
    --metadata author="Jim Xiao" \
    --metadata date="$(date +%Y-%m-%d)"

if [ -f "$OUTPUT" ]; then
    echo "✅ PDF generated successfully!"
    echo ""
    echo "📍 Location: $OUTPUT"
    echo "📊 Size: $(du -h "$OUTPUT" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the PDF: open docs/ONE_PAGE_OVERVIEW.pdf"
    echo "  2. Attach to Groq email: docs/GROQ_EMAIL_DRAFT.md"
    echo "  3. Attach to Cerebras email: docs/CEREBRAS_EMAIL_DRAFT.md"
else
    echo "❌ Error: PDF generation failed"
    exit 1
fi
