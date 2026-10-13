# Buggy vs backdoored, sonnet attacker (probe fit on 82 matched pairs, layer 20, C=0.01)

| papers | n | median probe score | % above the 2%-false-alarm line | median TA rating |
|---|---|---|---|---|
| honest, matched set (held-out CV) | 82 | -4.16 | 2% | 8.00 |
| backdoor, matched set (held-out CV) | 82 | 4.59 | 43% | 8.01 |
| honest, same problems as failed attacks | 58 | -1.96 | 0% | 7.99 |
| buggy attempt (fails ordinary tests) | 44 | 2.24 | 5% | 8.01 |
| inert attempt (passes tests, never fires) | 14 | 2.97 | 14% | 8.02 |

| comparison (held-out papers) | probe AUROC | TA spoken AUROC | word counter |
|---|---|---|---|
| honest vs buggy (same problems) | 0.86 [0.79, 0.92] | 0.59 [0.49, 0.68] | 0.76 |
| buggy vs backdoor | 0.69 [0.60, 0.78] | 0.47 [0.36, 0.57] | — |
