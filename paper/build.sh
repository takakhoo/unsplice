#!/bin/sh
# Build both papers. Needs TinyTeX (or TeX Live) with pdflatex and bibtex on PATH.
# The Interspeech class also needs lipsum, lineno, comment and courier (tlmgr install ...), or TEXMFHOME pointing at a tree that has them.
cd "$(dirname "$0")"
export PATH="$HOME/Library/TinyTeX/bin/universal-darwin:$PATH"
[ -d .texmf ] && export TEXMFHOME="$PWD/.texmf"
(cd .. && python3 experiments/make_numbers.py > /dev/null 2>&1)
run() { pdflatex -interaction=nonstopmode "$1" > /dev/null 2>&1; }
(cd esann && run unsplice_esann; bibtex unsplice_esann > /dev/null 2>&1; run unsplice_esann; run unsplice_esann; grep -E "Output written|Citation.*undefined|^!" unsplice_esann.log | sort -u)
(cd interspeech && P='\pdfmapfile{+ucr.map}\input{unsplice_interspeech}'; run "$P"; bibtex unsplice_interspeech > /dev/null 2>&1; run "$P"; run "$P"; grep -E "Output written|Citation.*undefined|^!" unsplice_interspeech.log | sort -u)
