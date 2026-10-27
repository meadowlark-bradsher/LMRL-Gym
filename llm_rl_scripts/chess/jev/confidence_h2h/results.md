# Confidence head-to-head — results

Grader hash `a9d18ee1fc0c7dc7` · 200 positions · B=5000 · near-tie (≥2 acceptable) 130/200


## Event: acceptable (cp_lost <=50)

| condition | n_triples | n_pos | ECE | Brier | AUROC | quintile lift | sel@80 |
|---|---|---|---|---|---|---|---|
| j_noul | 6098 | 200 | 0.174 | 0.188 | 0.660 | 0.394 | 0.675 |
| l_logprob | 6098 | 200 | 0.177 | 0.202 | 0.588 | 0.250 | 0.500 |
| l_verbal_gpt | 190 | 190 | 0.657 | 0.647 | 0.532 | 0.724 | 0.347 |
| l_verbal_claude | 200 | 200 | 0.162 | 0.266 | 0.579 | 0.225 | 0.775 |

**Paired j_noul − baseline (95% CI); ECE neg = Noul better; AUROC/sel pos = Noul better:**

| vs | ΔECE | ΔAUROC | Δsel@80 |
|---|---|---|---|
| l_logprob | -0.0027 [-0.0110, +0.0075] | +0.0715 [-0.0253, +0.1674] | +0.1750 [-0.0500, +0.3500] |
| l_verbal_gpt | -0.4817 [-0.5614, -0.3977] | +0.1341 [+0.0139, +0.2509] | +0.3110 [+0.1404, +0.4915] |
| l_verbal_claude | +0.0129 [-0.0681, +0.0548] | +0.0808 [-0.0253, +0.1917] | -0.1000 [-0.2843, +0.0925] |

## Event: best (cp_lost ==0)

| condition | n_triples | n_pos | ECE | Brier | AUROC | quintile lift | sel@80 |
|---|---|---|---|---|---|---|---|
| j_noul | 6098 | 200 | 0.018 | 0.038 | 0.725 | 0.369 | 0.525 |
| l_logprob | 6098 | 200 | 0.044 | 0.050 | 0.667 | 0.181 | 0.300 |
| l_verbal_gpt | 190 | 190 | 0.799 | 0.781 | 0.577 | 0.474 | 0.214 |
| l_verbal_claude | 200 | 200 | 0.157 | 0.200 | 0.746 | 0.500 | 0.750 |

**Paired j_noul − baseline (95% CI); ECE neg = Noul better; AUROC/sel pos = Noul better:**

| vs | ΔECE | ΔAUROC | Δsel@80 |
|---|---|---|---|
| l_logprob | -0.0259 [-0.0312, -0.0169] | +0.0584 [-0.0753, +0.1926] | +0.2250 [+0.0114, +0.4000] |
| l_verbal_gpt | -0.7811 [-0.8321, -0.7222] | +0.1576 [+0.0144, +0.2962] | +0.3383 [+0.1402, +0.4965] |
| l_verbal_claude | -0.1387 [-0.2011, -0.0952] | -0.0206 [-0.1233, +0.0798] | -0.2250 [-0.3811, +0.0333] |

## Cost / latency / coverage

| condition | mean latency s | mean cost $ | total $ |
|---|---|---|---|
| j_noul | 0.1299 | - | - |
| l_logprob | 0.7227 | 0.00066 | 0.1323 |
| l_verbal_gpt | 0.8763 | 0.00082 | 0.1631 |
| l_verbal_claude | 4.4114 | 0.00675 | 1.3506 |

l_logprob mean coverage: 0.666  ·  Jev within-position Noul SD (mean): 0.0027

