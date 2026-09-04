#!/usr/bin/env python
"""Convert the generated LaTeX tables into IEEEtran two-column floats.

`scripts/ablation_study.py` writes single-column `table` floats to
outputs/tables/. IEEEtran's two-column journal layout needs wide tables to be
`table*` (spanning both columns, top of page), and the widest of them still
overflow \\textwidth at 10pt, so they get \\resizebox.

This keeps outputs/tables/ as the single source of truth -- it is never edited --
and writes converted copies under paper/tables/ that the manuscript \\inputs.
Re-run after regenerating tables and the manuscript picks the change up.

Run:
    python scripts/build_paper_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "outputs" / "tables"
DST = PROJECT_ROOT / "paper" / "tables"

# How each table is placed. `wide` -> table* spanning both columns (needed when
# a table has many columns or many rows). `small` -> \footnotesize, which shrinks
# the natural width so the shrink-to-fit guard has less work to do and the result
# stays legible; used only on the long tables.
#
# Every table gets the shrink-to-fit guard, which only ever scales DOWN. A bare
# \resizebox{\textwidth}{!}{...} would scale a narrow table UP to fill the page,
# which is why Table III (two data rows) must not be given one.
PLACEMENT = {
    "table1_main_results":       {"wide": True,  "small": True},
    "table2_conformal_fairness": {"wide": True,  "small": True},
    "table3_uncertainty":        {"wide": True,  "small": False},
    "table4_ablation":           {"wide": False, "small": True},
    "table5_loss_ablation":      {"wide": False, "small": True},
    "table6_ensemble":           {"wide": False, "small": False},
    "table7_policy_simulation":  {"wide": True,  "small": True},
}


def convert(text: str, *, wide: bool, small: bool, stem: str) -> str:
    """Rewrite one generated table into an IEEEtran float."""
    env = "table*" if wide else "table"
    box = r"\textwidth" if wide else r"\linewidth"

    # Float environment and placement specifier. A starred float cannot be
    # bottom- or here-placed, so [htbp] on a table* is silently downgraded and
    # tends to drift to the end of the document. [!t] is the correct specifier.
    text = re.sub(r"\\begin\{table\}\[[^\]]*\]", rf"\\begin{{{env}}}[!t]", text)
    text = re.sub(r"\\end\{table\}", rf"\\end{{{env}}}", text)

    if small:
        # After \caption, so the caption itself stays at normal size per IEEE style.
        text = re.sub(
            r"(\\label\{[^}]*\}[ \t]*\n)", r"\1  \\footnotesize\n", text, count=1
        )

    # Shrink-to-fit that never enlarges: inside \resizebox, \width is the natural
    # width of the box, so this picks min(natural, box). Inserted with
    # str.replace, so the backslashes are literal and must not be regex-escaped.
    guard = rf"\resizebox{{\ifdim\width>{box}{box}\else\width\fi}}{{!}}{{%"
    text = text.replace(r"\begin{tabular}", guard + "\n  \\begin{tabular}")
    text = text.replace(r"\end{tabular}", "\\end{tabular}%\n  }")

    header = (
        f"% AUTO-GENERATED from outputs/tables/{stem}.tex\n"
        f"% by scripts/build_paper_tables.py -- do not edit by hand.\n"
        f"% Regenerate tables with scripts/ablation_study.py, then re-run that script.\n"
    )
    return header + text.rstrip() + "\n"


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"No generated tables at {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in sorted(SRC.glob("table*.tex")):
        stem = path.stem
        cfg = PLACEMENT.get(stem)
        if cfg is None:
            print(f"  skip (no placement rule): {stem}")
            continue
        out = convert(
            path.read_text(encoding="utf-8"),
            wide=cfg["wide"],
            small=cfg["small"],
            stem=stem,
        )
        dest = DST / f"{stem}.tex"
        dest.write_text(out, encoding="utf-8")
        env = "table*" if cfg["wide"] else "table "
        print(f"  {stem:<28} -> {env}  footnotesize={str(cfg['small']):<5}")
        written += 1

    print(f"\n{written} tables written to {DST.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
