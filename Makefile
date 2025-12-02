# Simple build helper for the thesis

all: main.pdf

main.pdf: main.tex $(wildcard chapters/*.tex) $(wildcard appendices/*.tex) bibliography.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.pdf

.PHONY: all clean
