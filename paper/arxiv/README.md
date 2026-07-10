# arXiv submission — OICC

This folder is assembled by `make_arxiv.py` into a self-contained arXiv upload.

## What arXiv needs (and what this provides)
arXiv compiles your source on its own TeXLive server. A valid submission is a
**single tarball** containing:
- the main `.tex` (self-contained; uses only standard TeXLive packages),
- all `figures/*.png` referenced by `\includegraphics`,
- (optional) a `00README.XXX` with processing hints.

We use a `thebibliography` block inside the `.tex` (no external `.bib`), so no
BibTeX pass is required and there is nothing else to upload.

## Build the tarball
```bash
python paper/arxiv/make_arxiv.py
# -> paper/arxiv/build/oicc_arxiv.tar.gz   (upload this to arxiv.org)
```
The script copies the current `paper/oicc_paper.tex` + `paper/figures/`, strips
any absolute paths, validates LaTeX balance and that every included figure
exists, and writes the tarball. If a local TeX engine is available it also does
a test compile.

## Submit (once you have the tarball)
1. Create/log in at https://arxiv.org and click **Submit**.
2. Primary category: **stat.ME** (Methodology). Cross-list: **stat.AP**
   (Applications), **cs.LG**, and optionally **stat.ML**.
3. Upload `oicc_arxiv.tar.gz`. arXiv will run pdflatex; fix any log errors it
   reports (rare with standard packages).
4. Paste the title + abstract from `metadata.txt`.
5. License: choose **CC BY 4.0** (recommended for reach) or arXiv's default
   non-exclusive license.

## Files
- `metadata.txt` — title, abstract, categories, comments, license (copy/paste).
- `make_arxiv.py` — assembles + validates + tars the submission.
- `build/` — generated; the uploadable tarball lands here.
