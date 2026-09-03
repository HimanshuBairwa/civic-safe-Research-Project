---
name: paper-writer-ieee
description: Use when writing an IEEE-format research paper. Provides the complete IMRAD workflow with quality checkpoints. Invoke with "write the paper" or "draft section X".
---

# IEEE Research Paper Writer

## Workflow
1. Read ALL source code and results before writing anything
2. Write the analysis document first (paper/codebase_analysis.md)
3. Draft each section in this ORDER (not the paper order):
   - Methodology (III) — because you need to understand the system first
   - Experimental Setup (IV) — datasets, baselines, metrics
   - Results (V) — report all numbers honestly
   - Related Work (II) — position against prior art
   - Introduction (I) — now you know what the story is
   - Limitations (VI) — be brutally honest
   - Conclusion (VII) — summarize, do not introduce new info
   - Abstract — write LAST, after everything else
   - Title — confirm or revise

## Quality Checkpoints (Run After Each Section)
- Every number has a source file
- No banned AI phrases
- Sentence lengths vary (count words in 5 consecutive sentences — they should NOT all be similar)
- Section tells a clear story, not just lists features
- At least one concrete example or specific number per paragraph

## IEEE Formatting Rules
- Sections: I, II, III with subsections A, B, C
- Citations: [1], [2]-[4] in order of appearance
- Equations: numbered (1), (2), centered
- Tables: Roman numerals (TABLE I, TABLE II)
- Figures: Arabic numerals (Fig. 1, Fig. 2)
- Abstract: 200-250 words, no citations, no equations
