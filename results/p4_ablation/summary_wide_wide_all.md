# Probe-direction ablation, human honest vs dataset backdoor (150 pairs)

Spoken score = monitor's expected rating from its digit probabilities. Downstream check = a layer-36 probe trained on unablated training papers, applied to the ablated held-out papers.

| condition | spoken AUROC [95% CI] | catch at 2% |
|---|---|---|
| baseline, no ablation | 0.74 [0.70, 0.79] | 18% |
| wide (layers 15–36): probe direction removed | 0.72 [0.67, 0.77] | 7% |
| wide: random direction removed (control) | 0.74 [0.69, 0.79] | 19% |
| wide_all (layers 15–36, all positions): probe direction removed | 0.65 [0.60, 0.71] | 4% |
| wide_all: random direction removed (control) | 0.73 [0.69, 0.78] | 8% |

| condition | downstream layer-36 probe AUROC | catch |
|---|---|---|
| wide: after probe-direction ablation | 0.51 [0.46, 0.56] | 0% |
| wide: after random ablation | 0.90 [0.86, 0.93] | 68% |
| wide_all: after probe-direction ablation | 0.52 [0.46, 0.58] | 0% |
| wide_all: after random ablation | 0.82 [0.78, 0.85] | 34% |

- wide: mean rating shift after probe ablation, honest -0.02, backdoor -0.05; random: honest +0.02, backdoor -0.00
- wide_all: mean rating shift after probe ablation, honest +0.24, backdoor -0.07; random: honest -0.42, backdoor -0.11
