# Probe-direction ablation, human honest vs dataset backdoor (150 pairs)

Spoken score = monitor's expected rating from its digit probabilities. Downstream check = a layer-36 probe trained on unablated training papers, applied to the ablated held-out papers.

| condition | spoken AUROC [95% CI] | catch at 2% |
|---|---|---|
| baseline, no ablation | 0.74 [0.70, 0.79] | 18% |
| single (layers 22–22): probe direction removed | 0.74 [0.70, 0.79] | 19% |
| single: random direction removed (control) | 0.75 [0.70, 0.80] | 19% |
| band (layers 18–26): probe direction removed | 0.73 [0.68, 0.78] | 13% |
| band: random direction removed (control) | 0.74 [0.70, 0.79] | 19% |

| condition | downstream layer-36 probe AUROC | catch |
|---|---|---|
| single: after probe-direction ablation | 0.90 [0.86, 0.93] | 66% |
| single: after random ablation | 0.90 [0.86, 0.93] | 68% |
| band: after probe-direction ablation | 0.79 [0.74, 0.83] | 37% |
| band: after random ablation | 0.90 [0.86, 0.93] | 69% |

- single: mean rating shift after probe ablation, honest +0.02, backdoor +0.00; random: honest +0.00, backdoor +0.00
- band: mean rating shift after probe ablation, honest -0.02, backdoor -0.04; random: honest -0.02, backdoor +0.00
