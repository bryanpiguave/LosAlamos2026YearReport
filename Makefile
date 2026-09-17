# Build the technical report. bibtex + natbib, so the CRC TeX Live install
# (which lacks biber/biblatex) can compile it; Overleaf compiles it unchanged.
MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib generated/numbers.tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

figures:
	qsub scripts/qsub/make_figures.sh

# Everything Overleaf needs, in one archive (upload as "New Project > Upload").
overleaf: $(MAIN).pdf
	rm -f los_alamos_technical_report_overleaf.zip
	zip -r los_alamos_technical_report_overleaf.zip $(MAIN).tex references.bib generated/numbers.tex figures/*.png figures/*.pdf README.md

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc

.PHONY: all figures overleaf clean
