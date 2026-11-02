# Table 2. Fidelity (v2, continuous filtering)

| Band   |   Envelope shape r |   Log-power r |   Log-power MAE (log10) |   Relative power MAE |   |d_orig - d_sym| (band mean) |
|:-------|-------------------:|--------------:|------------------------:|---------------------:|-------------------------------:|
| Delta  |              0.873 |        0.9144 |                  0.0222 |              0.01396 |                          0.059 |
| Theta  |              0.814 |        0.8696 |                  0.0314 |              0.00607 |                          0.237 |
| Alpha  |              0.801 |        0.9612 |                  0.0468 |              0.00722 |                          0.035 |
| Beta   |              0.697 |        0.9253 |                  0.0364 |              0.00323 |                          0.103 |
| Gamma  |              0.697 |        0.9389 |                  0.0538 |              0.0024  |                          0.104 |

- r_d: 0.9781
- r_d_heldout_participants: 0.9868
- r_d_out_of_fold: 0.9738
- spearman: 0.9817
- top10: 6
- top20: 18
- top30: 25
- mean_abs_d_orig: 0.4541
- mean_abs_d_sym: 0.5047

Band-mean Cohen's d (original): delta +0.26, theta +0.61, alpha -0.66, beta -0.37, gamma +0.25
