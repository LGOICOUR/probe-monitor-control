# Buggy vs backdoored, haiku attacker (probe fit on 102 matched pairs, layer 27, C=0.0001)

| papers | n | median probe score | % above the 2%-false-alarm line | median TA rating |
|---|---|---|---|---|
| honest, matched set (held-out CV) | 102 | -3.74 | 3% | 7.94 |
| backdoor, matched set (held-out CV) | 102 | 3.52 | 95% | 9.00 |
| honest, same problems as failed attacks | 40 | -2.11 | 5% | 7.95 |
| buggy attempt (fails ordinary tests) | 38 | 0.31 | 74% | 8.23 |
| inert attempt (passes tests, never fires) | 6 | 1.78 | 100% | 8.99 |

| comparison (held-out papers) | probe AUROC | TA spoken AUROC | word counter |
|---|---|---|---|
| honest vs buggy (same problems) | 0.93 [0.88, 0.98] | 0.77 [0.66, 0.87] | 0.63 |
| buggy vs backdoor | 0.91 [0.86, 0.95] | 0.80 [0.71, 0.88] | — |
