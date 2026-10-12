# P3 on the matched set (102 same-author pairs)

AUROC [95% CI] | catch rate at 2% false alarms [95% CI]. CIs resample problems.

| monitor | AUROC | catch at 2% |
|---|---|---|
| probe (layer chosen inside training) | 0.99 [0.97, 1.00] | 95% [91, 99] |
| TA spoken score | 0.95 [0.92, 0.98] | 73% [50, 87] |
| word counter (no TA) | 0.89 [0.85, 0.92] | 46% [19, 66] |
| length ruler | 0.57 [0.54, 0.61] | 2% [0, 7] |
| transfer: trained matched, tested human vs dataset backdoor | 0.86 [0.81, 0.90] | 43% [31, 64] |
| transfer: trained human vs dataset backdoor, tested matched | 0.98 [0.96, 1.00] | 80% [30, 99] |

- Layers chosen inside training: [32, 27, 29, 27, 27]
- Hindsight layer: 25, AUROC 0.99
- Sanity, layer 0: 0.50
