# Makefile for Paper 1: 3.5-bit Quantization (NeurIPS 2026)

# Main target
TARGET = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Main build target
all: $(TARGET).pdf

# Build PDF (with bibliography)
$(TARGET).pdf: $(TARGET).tex references.bib
	$(LATEX) $(LATEXFLAGS) $(TARGET).tex
	$(BIBTEX) $(TARGET)
	$(LATEX) $(LATEXFLAGS) $(TARGET).tex
	$(LATEX) $(LATEXFLAGS) $(TARGET).tex

# Quick build (no bibliography update)
quick: $(TARGET).tex
	$(LATEX) $(LATEXFLAGS) $(TARGET).tex

# Clean build artifacts
clean:
	rm -f $(TARGET).aux $(TARGET).log $(TARGET).bbl $(TARGET).blg \
	      $(TARGET).out $(TARGET).synctex.gz $(TARGET).fdb_latexmk \
	      $(TARGET).fls

# Clean everything including PDF
distclean: clean
	rm -f $(TARGET).pdf

# View PDF (macOS)
view: $(TARGET).pdf
	open $(TARGET).pdf

# Count words (approximate, excludes LaTeX commands)
wordcount:
	@detex $(TARGET).tex | wc -w

# Spell check
spell:
	@aspell check $(TARGET).tex

.PHONY: all quick clean distclean view wordcount spell
