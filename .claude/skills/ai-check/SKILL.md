---
name: ai-check
description: Score text across 9 AI detection categories. Reports verdict (HUMAN/BORDERLINE/LIKELY-AI) with flagged sentences. Use before submission.
---
# AI Detection Forensic Check
Score 1-5 per category (1=human, 5=AI):
1. Sentence Length Uniformity (SD of word counts per sentence: SD>8=1, SD<3=5)
2. Vocabulary Diversity (TTR: >0.70=1, <0.50=5)
3. Discourse Markers (formal markers per page: 0-1=1, 7+=5)
4. Hedging Ratio (hedge phrases per page: <2=1, 8+=5)
5. Personal Voice ("we"/"our"/observations per page: 5+=1, 0=5)
6. Structural Predictability (% following topic-evidence-analysis: <30%=1, >70%=5)
7. Banned Words (AI-smell words: 0=1, 6+=5)
8. Opening Variation (% same opening word: <15%=1, >35%=5)
9. Specificity (vague quantifiers per page: 0-1=1, 6+=5)

Overall: Avg 1-2=HUMAN, 2.1-3=BORDERLINE, 3.1+=LIKELY-AI
Output: Score table + flagged sentences + fix suggestions.
