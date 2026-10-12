# Summarize control, Sonnet pairs (82)

AUROC [95% CI] | catch at 2% [95% CI]

| condition | AUROC | catch at 2% |
|---|---|---|
| probe on rate recording | 0.91 [0.86, 0.94] | 50% [27, 73] |
| probe on summarize recording | 0.86 [0.82, 0.90] | 33% [16, 61] |
| trained on rate, tested on summarize | 0.61 [0.58, 0.65] | 10% [2, 16] |
| trained on summarize, tested on rate | 0.59 [0.57, 0.63] | 7% [0, 12] |
| monitor's spoken score (rate prompt) | 0.54 [0.47, 0.60] | 5% [0, 11] |

- Layers chosen inside training: rate [22, 20, 20, 20, 20], summarize [29, 27, 16, 30, 19]
- Sanity, layer 0: rate 0.50, summarize 0.50
