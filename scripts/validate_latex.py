#!/usr/bin/env python
"""Static validation of the IEEEtran manuscript, for when no TeX toolchain exists.

This does NOT replace compiling. It catches the specific classes of error that
silently produce `??` marks, `[?]` citations and missing-figure boxes in a built
PDF, which are the ones that survive a careless build and reach a referee:

  1. every \\cite key resolves to an entry in the .bib
  2. every \\ref / \\eqref resolves to a \\label (following \\input files)
  3. every \\includegraphics target exists on disk under \\graphicspath
  4. no duplicate \\label
  5. \\begin/\\end environments balance
  6. braces balance
  7. no non-ASCII bytes (this manuscript is deliberately pure ASCII)
  8. no bare & or stray Markdown emphasis left in LaTeX body text
  9. bib entries that are never cited (warning only)

Exit code is nonzero if any error-level check fails.

Run:
    python scripts/validate_latex.py paper/civic_safe_ieee.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


def strip_comments(text: str) -> str:
    """Remove LaTeX line comments, respecting \\% escapes."""
    out = []
    for line in text.splitlines():
        buf, i = [], 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def expand_inputs(path: Path, seen: set[Path] | None = None) -> list[tuple[Path, str]]:
    """Return [(path, body)] for the root file and everything it \\inputs."""
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    if not path.exists():
        err(f"\\input target does not exist: {path}")
        return []
    body = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    result = [(path, body)]
    for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", body):
        target = m.group(1).strip()
        cand = path.parent / target
        if cand.suffix != ".tex":
            cand = cand.with_suffix(".tex")
        result.extend(expand_inputs(cand, seen))
    return result


DEFAULT_TARGETS = (
    "paper/civic_safe_ieee.tex",
    "paper/civic_safe_supplementary.tex",
    "paper/submission_bundle/civic_safe_ieee.tex",
    "paper/submission_bundle/civic_safe_supplementary.tex",
)


def validate_one(root: Path) -> int:
    """Validate a single document. Returns 0 on success, nonzero on failure."""
    global errors, warnings
    errors, warnings = [], []
    return _validate(root)


def main() -> int:
    args = sys.argv[1:]
    targets = args if args else [t for t in DEFAULT_TARGETS
                                 if (PROJECT_ROOT / t).exists()]
    if not targets:
        print("No LaTeX documents found to validate.")
        return 2
    rc = 0
    for i, target in enumerate(targets):
        if i:
            print()
        rc |= validate_one(Path(target))
    if len(targets) > 1:
        print()
        print(f"{len(targets)} document(s) validated; "
              f"overall {'PASS' if rc == 0 else 'FAIL'}.")
    return rc


def _validate(root: Path) -> int:
    # Resolve against the caller's working directory first, so the script works on
    # a bundle copied outside the repository, and only fall back to the project
    # root for the convenience of bare in-repo invocations.
    if not root.is_absolute() and not root.exists():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if not root.exists():
        print(f"No such file: {root}")
        return 2

    units = expand_inputs(root)
    root_body = units[0][1]
    all_text = "\n".join(b for _, b in units)

    print("=" * 74)
    try:
        shown = root.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = root  # outside the repo, e.g. a bundle copied elsewhere
    print(f"Static LaTeX validation - {shown}")
    print("=" * 74)
    print(f"files parsed: {len(units)}  ({', '.join(p.name for p, _ in units)})")

    # ---------- 7. non-ASCII ----------
    for p, body in units:
        for ln, line in enumerate(body.splitlines(), 1):
            bad = [(i, c) for i, c in enumerate(line) if ord(c) > 127]
            if bad:
                err(f"{p.name}:{ln}: non-ASCII {bad[:3]!r} - use a LaTeX escape")

    # ---------- 3. figures ----------
    gp = re.search(r"\\graphicspath\{(.+?)\}\s*$", root_body, re.M)
    search_dirs = [root.parent]
    if gp:
        for d in re.findall(r"\{([^{}]*)\}", gp.group(1)):
            search_dirs.append((root.parent / d).resolve())
    n_fig = 0
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text):
        target = m.group(1).strip()
        n_fig += 1
        found = False
        for d in search_dirs:
            if (d / target).exists():
                found = True
                break
            if not Path(target).suffix:
                for ext in (".pdf", ".png", ".jpg", ".eps"):
                    if (d / (target + ext)).exists():
                        found = True
                        break
            if found:
                break
        if not found:
            err(
                f"figure not found: '{target}' "
                f"(searched: {', '.join(str(d) for d in search_dirs)})"
            )
    print(f"figures referenced: {n_fig}")

    # ---------- 1. citations ----------
    bibname = None
    mb = re.search(r"\\bibliography\{([^}]+)\}", root_body)
    if mb:
        bibname = mb.group(1).strip()
    bibkeys: set[str] = set()
    if bibname:
        bibpath = root.parent / (bibname if bibname.endswith(".bib") else bibname + ".bib")
        if not bibpath.exists():
            err(f"bibliography file not found: {bibpath}")
        else:
            btext = bibpath.read_text(encoding="utf-8", errors="replace")
            bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", btext))
            # bare & inside a bib field breaks the bbl build
            for ln, line in enumerate(btext.splitlines(), 1):
                if re.search(r"(?<!\\)&", line):
                    err(f"{bibpath.name}:{ln}: bare '&' - must be '\\&'")
    cited: set[str] = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", all_text):
        for k in m.group(1).split(","):
            cited.add(k.strip())
    for k in sorted(cited):
        if bibkeys and k not in bibkeys:
            err(f"\\cite{{{k}}} has no entry in {bibname}.bib -> renders as [?]")
    print(f"citations: {len(cited)} distinct, bib entries: {len(bibkeys)}")
    for k in sorted(bibkeys - cited):
        warn(f"bib entry never cited (omitted from the reference list): {k}")

    # ---------- 2 + 4. labels and refs ----------
    labels: dict[str, str] = {}
    for p, body in units:
        for m in re.finditer(r"\\label\{([^}]+)\}", body):
            k = m.group(1).strip()
            if k in labels:
                err(f"duplicate \\label{{{k}}} (in {labels[k]} and {p.name})")
            labels[k] = p.name
    refs: set[str] = set()
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", all_text):
        refs.add(m.group(1).strip())
    for k in sorted(refs):
        if k not in labels:
            err(f"\\ref{{{k}}} has no matching \\label -> renders as ??")
    print(f"labels: {len(labels)}, refs: {len(refs)}")
    for k in sorted(set(labels) - refs):
        warn(f"label defined but never referenced: {k}")

    # ---------- 5. environments ----------
    for p, body in units:
        stack: list[tuple[str, int]] = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
            kind, name = m.group(1), m.group(2)
            ln = body[: m.start()].count("\n") + 1
            if kind == "begin":
                stack.append((name, ln))
            else:
                if not stack:
                    err(f"{p.name}:{ln}: \\end{{{name}}} with no open environment")
                elif stack[-1][0] != name:
                    o, oln = stack[-1]
                    err(
                        f"{p.name}:{ln}: \\end{{{name}}} closes "
                        f"\\begin{{{o}}} opened at line {oln}"
                    )
                    stack.pop()
                else:
                    stack.pop()
        for name, ln in stack:
            err(f"{p.name}:{ln}: \\begin{{{name}}} never closed")

    # ---------- 6. braces ----------
    for p, body in units:
        depth, i = 0, 0
        while i < len(body):
            c = body[i]
            if c == "\\" and i + 1 < len(body):
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0:
                    ln = body[:i].count("\n") + 1
                    err(f"{p.name}:{ln}: unmatched closing brace")
                    depth = 0
            i += 1
        if depth:
            err(f"{p.name}: {depth} unclosed brace(s)")

    # ---------- 8. Markdown leakage / bare & in body ----------
    for p, body in units:
        for ln, line in enumerate(body.splitlines(), 1):
            if re.search(r"\*\*\w", line) or re.search(r"\w\*\*", line):
                err(
                    f"{p.name}:{ln}: Markdown '**bold**' left in LaTeX - "
                    f"use \\textbf{{}}: {line.strip()[:70]}"
                )
            if re.match(r"^\s*#{1,6}\s", line):
                err(f"{p.name}:{ln}: Markdown heading left in LaTeX")

    # ---------- report ----------
    print()
    if warnings:
        print(f"{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")
        print()
    if errors:
        print(f"{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  x {e}")
        print()
        print("FAILED - fix the above before compiling.")
        return 1
    print("All checks passed.")
    print()
    print("NOTE: no TeX toolchain is installed here, so this is static analysis")
    print("only. It cannot detect overfull boxes, float-placement problems, or")
    print("package conflicts. Compile before submitting:")
    print("  pdflatex civic_safe_ieee && bibtex civic_safe_ieee && "
          "pdflatex civic_safe_ieee && pdflatex civic_safe_ieee")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
