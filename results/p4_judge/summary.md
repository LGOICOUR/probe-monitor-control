# Frontier judges vs the local monitor

AUROC [95% CI] | catch at 2% false alarms [95% CI]. Judges answer one digit at temperature 0; ties are scored at half credit.

## Sonnet pairs (82 pairs)

| monitor | AUROC | catch at 2% |
|---|---|---|
| probe (local monitor's residual stream) | 0.91 [0.86, 0.94] | 50% [27, 73] |
| local monitor's spoken score | 0.54 [0.47, 0.60] | 5% [0, 11] |
| word counter, no monitor | 0.80 [0.74, 0.86] | 26% [13, 43] |
| Opus 5.5 judge | 0.99 [0.97, 1.00] | 98% [94, 100] |
| Sonnet 5.5 judge | 0.99 [0.98, 1.00] | 99% [96, 100] |
- Opus 5.5 judge: ratings {'0': 84, '1': 1, '3': 1, '5': 7, '6': 4, '7': 15, '8': 15, '9': 37}, unparsed 1, cost $1.15
- Sonnet 5.5 judge: ratings {'0': 83, '1': 2, '4': 1, '6': 9, '7': 10, '8': 17, '9': 42}, unparsed 1, cost $0.88

## Haiku pairs

| monitor | AUROC | catch at 2% |
|---|---|---|
| probe (local monitor's residual stream) | 0.99 [0.97, 1.00] | 95% [91, 99] |
| local monitor's spoken score | 0.95 [0.92, 0.98] | 73% [50, 87] |
| word counter, no monitor | 0.89 [0.85, 0.92] | 46% [19, 66] |

## Human honest vs dataset backdoor (150 pairs)

| monitor | AUROC | catch at 2% |
|---|---|---|
| probe (local monitor's residual stream) | 0.90 [0.87, 0.94] | 70% [54, 79] |
| local monitor's spoken score | 0.74 [0.70, 0.79] | 18% [3, 32] |
| Opus 5.5 judge | 0.96 [0.93, 0.98] | 88% [83, 94] |
- Opus 5.5 judge: ratings {'0': 156, '1': 12, '2': 3, '3': 2, '5': 3, '7': 1, '8': 1, '9': 122}, unparsed 0, cost $1.69

