"""Score the paper draft against the ai-check rubric. Measurement, not vibes.

Implements the 9 categories in .claude/skills/ai-check/SKILL.md with the
thresholds that skill specifies. Prose only -- tables, code blocks, headers,
equations and the reference list are excluded, because their word statistics
have nothing to do with whether the prose reads like a person wrote it.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

BANNED = [
    "delve", "tapestry", "beacon", "crucial", "pivotal", "paramount",
    "indispensable", "foster", "underpin", "underscore", "elucidate",
    "intricate", "multifaceted", "nuanced", "embark", "harness",
    "utilize", "utilizing", "leveraging", "comprehensive", "holistic",
    "paradigm", "groundbreaking", "revolutionary", "realm",
    "it is worth noting", "it should be noted", "notably",
    "in conclusion", "in summary", "a testament to", "shed light on",
    "plays a vital role", "it is imperative", "the results demonstrate that",
]
FORMAL_MARKERS = ["furthermore", "moreover", "additionally", "however",
                  "consequently", "nevertheless", "thus", "hence"]
HEDGES = ["may", "might", "could potentially", "possibly", "perhaps",
          "it seems", "arguably", "somewhat", "relatively", "fairly",
          "tends to", "appears to"]
VAGUE = ["several", "various", "numerous", "many", "some", "a number of",
         "a variety of", "significantly", "substantially"]


def prose_only(text: str) -> str:
    lines, keep = text.splitlines(), []
    in_code = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if s.startswith("#") or s.startswith("|") or s.startswith("---"):
            continue
        if re.match(r"^\[\d+\]", s):          # reference entries
            continue
        if re.match(r"^\*\*(Index Terms|Abstract)", s):
            pass
        if s.startswith("    ") or ln.startswith("    "):  # display equations
            continue
        keep.append(ln)
    return "\n".join(keep)


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def score(text: str) -> dict[str, tuple[int, str]]:
    prose = prose_only(text)
    sents = sentences(prose)
    words = re.findall(r"[a-zA-Z']+", prose.lower())
    n_words = len(words)
    pages = max(n_words / 500.0, 1.0)          # ~500 words per page
    low = prose.lower()
    out: dict[str, tuple[int, str]] = {}

    # 1. Sentence length uniformity
    lens = [len(s.split()) for s in sents]
    sd = statistics.stdev(lens) if len(lens) > 1 else 0.0
    s1 = 1 if sd > 8 else 2 if sd > 6 else 3 if sd > 4.5 else 4 if sd >= 3 else 5
    out["1. Sentence length uniformity"] = (s1, f"SD={sd:.2f} over {len(sents)} sentences")

    # 2. Vocabulary diversity.
    # Plain TTR is length-dependent -- it falls mechanically as a document grows
    # because function words repeat, so the skill's >0.70 threshold only makes
    # sense on short passages. On a full paper it reports ~0.26 no matter how
    # varied the prose is. MATTR (moving-average TTR over a sliding window) is
    # the length-invariant version and is what actually answers the question.
    window = 100
    if n_words >= window:
        ttrs = [
            len(set(words[i:i + window])) / window
            for i in range(0, n_words - window + 1, 10)
        ]
        mattr = statistics.mean(ttrs)
    else:
        mattr = len(set(words)) / n_words if n_words else 0.0
    raw_ttr = len(set(words)) / n_words if n_words else 0.0
    s2 = 1 if mattr > 0.70 else 2 if mattr > 0.62 else 3 if mattr > 0.55 else 4 if mattr >= 0.50 else 5
    out["2. Vocabulary diversity"] = (
        s2, f"MATTR-{window}={mattr:.3f} (raw TTR={raw_ttr:.3f}, length-biased)"
    )

    # 3. Formal discourse markers per page
    marks = sum(len(re.findall(rf"\b{re.escape(m)}\b", low)) for m in FORMAL_MARKERS)
    per = marks / pages
    s3 = 1 if per <= 1 else 2 if per <= 2.5 else 3 if per <= 4 else 4 if per < 7 else 5
    out["3. Discourse markers"] = (s3, f"{marks} total, {per:.2f}/page")

    # 4. Hedging per page
    h = sum(len(re.findall(rf"\b{re.escape(x)}\b", low)) for x in HEDGES)
    per_h = h / pages
    s4 = 1 if per_h < 2 else 2 if per_h < 4 else 3 if per_h < 6 else 4 if per_h < 8 else 5
    out["4. Hedging ratio"] = (s4, f"{h} total, {per_h:.2f}/page")

    # 5. Personal voice per page
    pv = len(re.findall(r"\b(we|our|us)\b", low))
    per_pv = pv / pages
    s5 = 1 if per_pv >= 5 else 2 if per_pv >= 3 else 3 if per_pv >= 1.5 else 4 if per_pv > 0 else 5
    out["5. Personal voice"] = (s5, f"{pv} total, {per_pv:.2f}/page")

    # 6. Structural predictability -- paragraphs opening with a generic claim
    paras = [p for p in prose.split("\n\n") if len(p.split()) > 25]
    generic = sum(
        1 for p in paras
        if re.match(r"^\s*(The|This|These|It|There|Our|In)\b", p.strip())
        and not re.match(r"^\s*(The\s+(problem|fix|bias|spatial|corrected|last|biased|reason|obvious))", p.strip())
    )
    frac = generic / len(paras) if paras else 0
    s6 = 1 if frac < 0.30 else 2 if frac < 0.45 else 3 if frac < 0.55 else 4 if frac <= 0.70 else 5
    out["6. Structural predictability"] = (s6, f"{generic}/{len(paras)} paragraphs = {frac:.0%}")

    # 7. Banned words
    hits = [b for b in BANNED if re.search(rf"\b{re.escape(b)}\b", low)]
    n = len(hits)
    s7 = 1 if n == 0 else 2 if n <= 1 else 3 if n <= 3 else 4 if n <= 5 else 5
    out["7. Banned words"] = (s7, "none" if not hits else ", ".join(hits))

    # 8. Opening-word variation
    openers = [s.split()[0].lower().strip('",(') for s in sents if s.split()]
    if openers:
        top = max(set(openers), key=openers.count)
        share = openers.count(top) / len(openers)
    else:
        top, share = "-", 0.0
    s8 = 1 if share < 0.15 else 2 if share < 0.22 else 3 if share < 0.28 else 4 if share <= 0.35 else 5
    out["8. Opening variation"] = (s8, f"most common opener '{top}' = {share:.0%}")

    # 9. Vague quantifiers
    v = sum(len(re.findall(rf"\b{re.escape(x)}\b", low)) for x in VAGUE)
    per_v = v / pages
    s9 = 1 if per_v <= 1 else 2 if per_v <= 2 else 3 if per_v <= 4 else 4 if per_v < 6 else 5
    out["9. Specificity"] = (s9, f"{v} vague quantifiers, {per_v:.2f}/page")

    return out


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "paper/civic_safe_paper_draft.md")
    text = path.read_text(encoding="utf-8")
    res = score(text)

    print("=" * 78)
    print(f"ai-check rubric — {path}")
    print("=" * 78)
    print(f"{'category':<34}{'score':>6}   evidence")
    print("-" * 78)
    for k, (s, ev) in res.items():
        flag = "  <-- FIX" if s > 2 else ""
        print(f"{k:<34}{s:>6}   {ev}{flag}")
    avg = sum(s for s, _ in res.values()) / len(res)
    verdict = "HUMAN" if avg <= 2.0 else "BORDERLINE" if avg <= 3.0 else "LIKELY-AI"
    print("-" * 78)
    print(f"{'AVERAGE':<34}{avg:>6.2f}   verdict: {verdict}")
    print("=" * 78)

    # Also surface any invisible-Unicode contamination as a plain encoding check.
    bad = {
        "U+200B zero-width space": "​", "U+200C ZWNJ": "‌",
        "U+200D ZWJ": "‍", "U+FEFF BOM": "﻿",
        "U+2060 word joiner": "⁠", "U+00A0 nbsp": " ",
        "U+200E LRM": "‎", "U+200F RLM": "‏",
    }
    found = {n: text.count(c) for n, c in bad.items() if c in text}
    print("Invisible/ambiguous Unicode:",
          "none found" if not found else found)
    return 0 if avg <= 2.0 and not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
