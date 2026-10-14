# Diff-in-means-direction ablation, human honest vs dataset backdoor (150 pairs)

Spoken score = monitor's expected rating from its digit probabilities. Downstream check = a layer-36 probe trained on unablated training papers, applied to the ablated held-out papers.

| condition | spoken AUROC [95% CI] | catch at 2% |
|---|---|---|
| baseline, no ablation | 0.74 [0.70, 0.79] | 18% |
| wide (layers 15–36): probe direction removed | 0.39 [0.33, 0.45] | 0% |
| wide: random direction removed (control) | 0.74 [0.69, 0.79] | 19% |
| wide_all (layers 15–36, all positions): probe direction removed | 0.23 [0.20, 0.27] | 0% |
| wide_all: random direction removed (control) | 0.73 [0.69, 0.78] | 8% |

| condition | downstream layer-36 probe AUROC | catch |
|---|---|---|
| wide: after probe-direction ablation | 0.62 [0.57, 0.67] | 5% |
| wide: after random ablation | 0.90 [0.86, 0.93] | 68% |
| wide_all: after probe-direction ablation | 0.77 [0.73, 0.81] | 25% |
| wide_all: after random ablation | 0.82 [0.78, 0.85] | 34% |

- wide: mean rating shift after probe ablation, honest +0.50, backdoor -0.31; random: honest +0.02, backdoor -0.00
- wide_all: mean rating shift after probe ablation, honest +1.22, backdoor +0.50; random: honest -0.42, backdoor -0.11
