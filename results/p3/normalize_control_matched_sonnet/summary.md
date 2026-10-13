# Erase-handwriting control, sonnet set (82 pairs)

AUROC [95% CI] | catch at 2% [95% CI]

| condition | AUROC | catch at 2% |
|---|---|---|
| probe, original view | 0.91 [0.86, 0.94] | 50% [27, 73] |
| probe, normalized view | 0.86 [0.81, 0.91] | 39% [22, 63] |
| TA spoken score, original view | 0.54 [0.47, 0.60] | 5% [0, 11] |
| TA spoken score, normalized view | 0.56 [0.49, 0.63] | 4% [0, 11] |
| word counter, original code | 0.80 [0.74, 0.86] | 26% [13, 43] |
| word counter, normalized code | 0.81 [0.75, 0.86] | 29% [20, 40] |
| length ruler, original | 0.62 [0.57, 0.68] | 4% [0, 10] |
| length ruler, normalized | 0.63 [0.59, 0.68] | 4% [0, 12] |
| probe trained original, tested normalized | 0.85 [0.80, 0.90] | 34% [10, 56] |
| probe trained normalized, tested original | 0.86 [0.81, 0.90] | 43% [32, 60] |

- Layers chosen inside training: original [22, 20, 20, 20, 20], normalized [22, 21, 36, 19, 20]
