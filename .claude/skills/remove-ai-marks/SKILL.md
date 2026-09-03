---
name: remove-ai-marks
description: Remove AI provenance marks from text. Invisible Unicode (Layer A) and statistical patterns (Layer B). Use when asked to strip watermarks or clean AI output.
---
# Remove AI Marks (Text-Only Mode)
## Layer A: Unicode Cleaning
1. Run NFKC normalization on entire text
2. Strip all zero-width characters (U+200B, U+200C, U+200D, U+FEFF, U+2060-U+2063)
3. Replace homoglyphs (Cyrillic/Greek lookalikes) with ASCII
4. Remove directional markers (U+200E, U+200F, U+202A-U+202E)
## Layer B: Statistical Pattern Rewriting
1. Replace 15-20% of word choices with natural synonyms
2. Vary sentence openings within each paragraph
3. Break uniform sentence rhythms with fragments and questions
4. Replace AI collocations: "plays a crucial role" -> "matters", "it is important to note" -> delete
5. Add human texture: parenthetical asides, starting with "And"/"But"
6. Read aloud test: if it sounds robotic, rewrite
## Rules
- Never change technical terms, numbers, citations, equations
- Never change meaning of claims
- Only modify style and word choice
- Use as FINAL pass after all other editing
