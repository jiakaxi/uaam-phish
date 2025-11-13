# S0 / S1 Model Performance Summary

**Metrics: Mean ± 95% CI (per model)**

| Model | AUROC | ECE_pre | ECE_post | Brier_post |
|--|--|--|--|--|
| corrupt | 0.9995 ± 0.0000 | 0.0242 ± 0.0017 | 0.0242 ± 0.0017 | 0.0159 ± 0.0017 |
| s0_earlyconcat | 0.9791 ± 0.0309 | 0.0271 ± 0.0205 | 0.0271 ± 0.0205 | 0.0253 ± 0.0201 |
| s0_lateavg | 0.9776 ± 0.0247 | 0.0423 ± 0.0072 | 0.0423 ± 0.0072 | 0.0287 ± 0.0187 |
| s1_lateavg | 1.0000 ± 0.0000 | 0.0480 ± 0.0344 | 0.0411 ± 0.0464 | 0.0112 ± 0.0085 |

**Entries per model:** corrupt (n=9), s0_earlyconcat (n=6), s0_lateavg (n=6), s1_lateavg (n=3)
