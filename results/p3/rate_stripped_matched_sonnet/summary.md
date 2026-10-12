# P3 on the sonnet matched set (82 same-author pairs)

AUROC [95% CI] | catch rate at 2% false alarms [95% CI]. CIs resample problems.

| monitor | AUROC | catch at 2% |
|---|---|---|
| probe (layer chosen inside training) | 0.91 [0.86, 0.94] | 50% [27, 73] |
| TA spoken score | 0.54 [0.47, 0.60] | 5% [0, 11] |
| word counter (no TA) | 0.80 [0.74, 0.86] | 26% [13, 43] |
| length ruler | 0.62 [0.57, 0.68] | 4% [0, 10] |
| transfer: trained matched, tested human vs dataset backdoor | 0.81 [0.76, 0.86] | 23% [15, 44] |
| transfer: trained human vs dataset backdoor, tested matched | 0.73 [0.67, 0.78] | 13% [2, 32] |

- Layers chosen inside training: [22, 20, 20, 20, 20]
- Hindsight layer: 22, AUROC 0.91
- Sanity, layer 0: 0.50
