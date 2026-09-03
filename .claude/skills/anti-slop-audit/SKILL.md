---
name: anti-slop-audit
description: Use after writing any text to audit it for AI-generated patterns and fix them. Invoke with "audit this for AI patterns" or "run anti-slop check".
---

# Anti-Slop Audit (AI Pattern Detector and Fixer)

## Detection Patterns to Fix

### 1. Uniformity (BIGGEST TELL)
AI text has suspiciously uniform sentence lengths. Human writing lurches — 5 words, then 25, then 12, then 8.
Test: Count words in 10 consecutive sentences. If the standard deviation is less than 5, the text is too uniform.
Fix: Split long sentences. Combine short ones. Add fragments. Vary deliberately.

### 2. Predictable Transitions
AI loves: "Moreover," then "Furthermore," then "Additionally," then "In conclusion,"
Humans use: "But" / "And" / "So" / "The problem is" / no transition at all
Fix: Delete 50% of transition words. Let ideas connect naturally.

### 3. The Rule of Three Structure
AI paragraphs follow: Topic sentence then Point 1 then Point 2 then Point 3 then Summary.
Humans do not write this way. Human paragraphs are messy, follow tangents, circle back.
Fix: Merge some points. Drop the summary sentence. Start mid-thought sometimes.

### 4. Hedging Overload
AI hedges everything: "may potentially contribute to possible improvements"
Humans commit: "This works." or "This does not."
Fix: If you have evidence, state the claim directly.

### 5. Vocabulary Tells (The AI Smell Words)
These words appear 10-50x more often in AI text than human text:
- "delve", "tapestry", "beacon", "landscape" (metaphorical)
- "crucial", "pivotal", "paramount", "indispensable"
- "foster", "underpin", "underscore", "elucidate"
- "intricate", "multifaceted", "nuanced"
- "embark", "navigate" (metaphorical), "harness"
Fix: Replace with plain English.

### 6. Perplexity and Burstiness
AI text has HIGH perplexity uniformity (every sentence is equally complex).
Human text has HIGH burstiness (some sentences are dead simple, some are dense).
Fix: After every dense technical sentence, add a short plain one.

## Output Format
After auditing, report:
1. Number of AI patterns found
2. Specific sentences flagged
3. The fixed version of each flagged sentence
4. Final cleaned text
