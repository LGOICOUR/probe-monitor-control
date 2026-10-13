# Reasoning-first TA, sonnet set (82 pairs)

AUROC [95% CI] | catch at 2% [95% CI]

| condition | AUROC | catch at 2% |
|---|---|---|
| TA spoken score, immediate answer | 0.54 [0.47, 0.60] | 5% [0, 11] |
| TA spoken score after reasoning (digit probabilities) | 0.64 [0.58, 0.70] | 7% [2, 24] |
| TA spoken score after reasoning (written digit) | 0.60 [0.55, 0.65] | 0% [0, 0] |
| probe, immediate-answer recording | 0.91 [0.86, 0.94] | 50% [27, 73] |
| probe, post-rationale recording | 0.81 [0.75, 0.88] | 29% [16, 46] |
| probe trained immediate, tested post-rationale | 0.58 [0.53, 0.64] | 1% [0, 16] |
| probe trained post-rationale, tested immediate | 0.68 [0.63, 0.74] | 4% [1, 9] |
| word counter on the rationale text | 0.67 [0.60, 0.74] | 18% [4, 29] |

- Unparsed final digits: 2; written-digit histogram {'0': 130, '9': 34}; mean rationale length 150 tokens
- Layers chosen inside training: immediate [22, 20, 20, 20, 20], post-rationale [24, 34, 24, 24, 25]
