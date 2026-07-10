#!/usr/bin/env python
"""Assemble a self-contained arXiv submission tarball for the OICC paper.

Copies paper/oicc_paper.tex + paper/figures/ into a clean build dir, validates
LaTeX balance and figure presence, optionally test-compiles if a TeX engine is
available, and writes build/oicc_arxiv.tar.gz ready to upload to arxiv.org.

Usage:  python paper/arxiv/make_arxiv.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper/arxiv
PAPER = HERE.parent                             # paper
TEX = PAPER / "oicc_paper.tex"
FIGDIR = PAPER / "figures"
BUILD = HERE / "build"
STAGE = BUILD / "oicc_arxiv"


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def validate_tex(text: str) -> list[str]:
    """Return a list of problems (empty = OK)."""
    problems = []
    nb = len(re.findall(r"\\begin\{", text))
    ne = len(re.findall(r"\\end\{", text))
    if nb != ne:
        problems.append(f"\\begin ({nb}) != \\end ({ne})")
    if text.count("$") % 2 != 0:
        problems.append("odd number of $ (unbalanced inline math)")
    # every \includegraphics target must exist under figures/
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    for f in figs:
        name = Path(f).name
        if not (FIGDIR / name).exists():
            problems.append(f"missing figure: {f} (looked for figures/{name})")
    # every \cite key should have a \bibitem
    cited = set(re.findall(r"\\cite[tp]?\{([^}]+)\}", text))
    cited = {k.strip() for group in cited for k in group.split(",")}
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    missing = cited - defined
    if missing:
        problems.append(f"\\cite without \\bibitem: {sorted(missing)}")
    # absolute-path smell (real Windows drive path / home path -- NOT a LaTeX
    # line break "\\" which can follow a colon)
    if re.search(r"[A-Za-z]:\\Users|/Users/|/home/[a-z]|OneDrive", text):
        problems.append("absolute path found in .tex (arXiv builds in a sandbox)")
    return problems


def main() -> int:
    if not TEX.exists():
        fail(f"paper not found: {TEX}")
    text = TEX.read_text(encoding="utf-8")

    print("validating LaTeX ...")
    problems = validate_tex(text)
    if problems:
        for p in problems:
            print("  -", p)
        fail("fix the above before packaging")
    print("  OK: balanced, math even, all figures present, all cites resolved")

    # stage
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (STAGE / "figures").mkdir(parents=True)
    shutil.copy2(TEX, STAGE / "oicc_paper.tex")
    included = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    for f in included:
        name = Path(f).name
        shutil.copy2(FIGDIR / name, STAGE / "figures" / name)
    # arXiv processing hint (use pdflatex)
    (STAGE / "00README.XXX").write_text(
        "% arXiv build hints\n"
        "% main file: oicc_paper.tex ; engine: pdflatex ; no bibtex "
        "(uses thebibliography)\n",
        encoding="utf-8",
    )
    print(f"staged {1 + len(set(included))} source files in {STAGE}")

    # optional test compile
    engine = shutil.which("pdflatex") or shutil.which("xelatex")
    if engine:
        print(f"test-compiling with {Path(engine).name} ...")
        for _ in range(2):  # twice for refs
            r = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error",
                 "oicc_paper.tex"],
                cwd=STAGE, capture_output=True, text=True,
            )
        if (STAGE / "oicc_paper.pdf").exists():
            print("  compiled OK ->", STAGE / "oicc_paper.pdf")
        else:
            print("  WARNING: local compile failed; arXiv may still succeed. "
                  "Check the log:")
            print("\n".join(r.stdout.splitlines()[-15:]))
    else:
        print("no local TeX engine; skipping test compile "
              "(arXiv will compile it server-side).")

    # tar
    tar_path = BUILD / "oicc_arxiv.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file() and p.suffix not in {".aux", ".log", ".out", ".pdf"}:
                tf.add(p, arcname=str(p.relative_to(STAGE)))
    size_kb = tar_path.stat().st_size / 1024
    print(f"\nSUBMISSION TARBALL: {tar_path}  ({size_kb:.0f} KB)")
    print("Upload this to arxiv.org (primary category stat.ME). "
          "See paper/arxiv/metadata.txt for title/abstract/categories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
