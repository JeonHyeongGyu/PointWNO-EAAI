# Canonical PointWNO V2 vs PointFNO — validation benchmark report (2026-11-13T09:57:22)

Validation only. n = 3 seeds per arm; the verdict rule is descriptive (no p-values). Datasets are never pooled.

Missing models (3): rae/V3/s0, rae/V3/s1, rae/V3/s2

## naca: NACA 4-digit, 2-D, M_inf 0.65, alpha 3 deg (transonic, variable geometry)

**Claim (frozen rule):** V2 shock-capturing advantage established on this dataset (frozen rule; n = 3, descriptive)

| pair | metric | n | mean a-b | 95 % CI | a lower (seeds) | verdict | failure-confounded |
|---|---|---|---|---|---|---|---|
| NACA-MAIN (V2J2-FNOM) | P1_shock_p_nrmse | 3 | -0.006579 | [-0.01065, -0.002513] | 3 | V2J2 consistently lower | False |
| NACA-MAIN (V2J2-FNOM) | P2_thickness_abs_log | 3 | -0.1039 | [-0.1541, -0.05369] | 3 | V2J2 consistently lower | False |
| NACA-MAIN (V2J2-FNOM) | val_loss | 3 | -0.001651 | [-0.002197, -0.001105] | 3 | V2J2 consistently lower | False |
| NACA-MAIN (V2J2-FNOM) | global_nrmse_mean | 3 | -0.01163 | [-0.01482, -0.008445] | 3 | V2J2 consistently lower | False |
| NACA-CTX-F1 (V2J2-F1) | P1_shock_p_nrmse | 3 | 0.0008633 | [-0.00764, 0.009367] | 1 | no detectable difference at n = 3 | False |
| NACA-CTX-F1 (V2J2-F1) | P2_thickness_abs_log | 3 | -0.02977 | [-0.2243, 0.1648] | 2 | no detectable difference at n = 3 | False |
| NACA-CTX-F1 (V2J2-F1) | val_loss | 3 | -0.0009189 | [-0.001528, -0.0003095] | 3 | V2J2 consistently lower | False |
| NACA-CTX-F1 (V2J2-F1) | global_nrmse_mean | 3 | -0.007175 | [-0.01291, -0.001441] | 3 | V2J2 consistently lower | False |
| NACA-CTX-V3 (V2J2-C3) | P1_shock_p_nrmse | 3 | 0.001075 | [-0.007048, 0.009199] | 1 | no detectable difference at n = 3 | False |
| NACA-CTX-V3 (V2J2-C3) | P2_thickness_abs_log | 3 | 0.01316 | [-0.02328, 0.04961] | 0 | no detectable difference at n = 3 | False |
| NACA-CTX-V3 (V2J2-C3) | val_loss | 3 | -0.001041 | [-0.001749, -0.0003339] | 3 | V2J2 consistently lower | False |
| NACA-CTX-V3 (V2J2-C3) | global_nrmse_mean | 3 | -0.007494 | [-0.01238, -0.002607] | 3 | V2J2 consistently lower | False |
| NACA-CTX-V3FNO (C3-F1) | P1_shock_p_nrmse | 3 | -0.0002122 | [-0.002243, 0.001818] | 2 | no detectable difference at n = 3 | False |
| NACA-CTX-V3FNO (C3-F1) | P2_thickness_abs_log | 3 | -0.04294 | [-0.2363, 0.1504] | 2 | no detectable difference at n = 3 | False |
| NACA-CTX-V3FNO (C3-F1) | val_loss | 3 | 0.0001225 | [-0.00119, 0.001435] | 2 | no detectable difference at n = 3 | False |
| NACA-CTX-V3FNO (C3-F1) | global_nrmse_mean | 3 | 0.000319 | [-0.009843, 0.01048] | 2 | no detectable difference at n = 3 | False |

Failure counts (count/denominator per seed model):

- BM_NACA_REUSE_C3_s0 false_surface_shock: 1/5 ()
- BM_NACA_REUSE_C3_s0 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_C3_s0 pred_shock_not_detected: 2/7 (74;78)
- BM_NACA_REUSE_C3_s1 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_C3_s1 pred_shock_not_detected: 2/7 (74;78)
- BM_NACA_REUSE_C3_s2 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_C3_s2 pred_shock_not_detected: 1/7 (74)
- BM_NACA_REUSE_F1_s0 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_F1_s0 pred_shock_not_detected: 1/7 (74)
- BM_NACA_REUSE_F1_s1 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_F1_s1 pred_shock_not_detected: 1/7 (78)
- BM_NACA_REUSE_F1_s2 false_surface_shock: 1/5 ()
- BM_NACA_REUSE_F1_s2 gt_shock_present: 7/12 ()
- BM_NACA_REUSE_F1_s2 pred_shock_not_detected: 1/7 (78)
- BM_NACA_FNOM_s0 false_surface_shock: 1/5 ()
- BM_NACA_FNOM_s0 gt_shock_present: 7/12 ()
- BM_NACA_FNOM_s0 pred_shock_not_detected: 2/7 (74;78)
- BM_NACA_FNOM_s1 gt_shock_present: 7/12 ()
- BM_NACA_FNOM_s2 false_surface_shock: 1/5 ()
- BM_NACA_FNOM_s2 gt_shock_present: 7/12 ()
- BM_NACA_FNOM_s2 pred_shock_not_detected: 2/7 (74;78)
- BM_NACA_V2J2_s0 false_surface_shock: 1/5 ()
- BM_NACA_V2J2_s0 gt_shock_present: 7/12 ()
- BM_NACA_V2J2_s0 pred_shock_not_detected: 1/7 (74)
- BM_NACA_V2J2_s1 false_surface_shock: 2/5 ()
- BM_NACA_V2J2_s1 gt_shock_present: 7/12 ()
- BM_NACA_V2J2_s1 pred_shock_not_detected: 1/7 (74)
- BM_NACA_V2J2_s2 false_surface_shock: 2/5 ()
- BM_NACA_V2J2_s2 gt_shock_present: 7/12 ()

## rae: RAE2822, 2-D fixed geometry, (M_inf, alpha) varied

**Claim (frozen rule):** no shock-capturing difference established on this dataset

| pair | metric | n | mean a-b | 95 % CI | a lower (seeds) | verdict | failure-confounded |
|---|---|---|---|---|---|---|---|
| RAE-MAIN (V2J2-FNO34) | P1_shock_p_nrmse | 3 | 0.0002994 | [-0.001471, 0.00207] | 1 | no detectable difference at n = 3 | False |
| RAE-MAIN (V2J2-FNO34) | P2_thickness_abs_log | 3 | 0.009296 | [-0.06866, 0.08725] | 1 | no detectable difference at n = 3 | False |
| RAE-MAIN (V2J2-FNO34) | val_loss | 3 | -1.202e-07 | [-1.127e-06, 8.864e-07] | 2 | no detectable difference at n = 3 | False |
| RAE-MAIN (V2J2-FNO34) | global_nrmse_mean | 3 | -7.301e-05 | [-0.0002495, 0.0001035] | 3 | no detectable difference at n = 3 | False |
| RAE-CTX-V3 (V2J2-V3) | P1_shock_p_nrmse | 0 | nan | [nan, nan] | 0 | incomplete (n = 0) | False |
| RAE-CTX-V3 (V2J2-V3) | P2_thickness_abs_log | 0 | nan | [nan, nan] | 0 | incomplete (n = 0) | False |
| RAE-CTX-V3 (V2J2-V3) | val_loss | 0 | nan | [nan, nan] | 0 | incomplete (n = 0) | False |
| RAE-CTX-V3 (V2J2-V3) | global_nrmse_mean | 0 | nan | [nan, nan] | 0 | incomplete (n = 0) | False |
| RAE-CTX-500 (V3_500-FNO_500) | P1_shock_p_nrmse | 3 | -0.0003225 | [-0.003283, 0.002638] | 1 | no detectable difference at n = 3 | False |
| RAE-CTX-500 (V3_500-FNO_500) | P2_thickness_abs_log | 3 | -0.01981 | [-0.1727, 0.1331] | 2 | no detectable difference at n = 3 | False |
| RAE-CTX-500 (V3_500-FNO_500) | val_loss | 3 | 1.569e-07 | [-1.824e-06, 2.138e-06] | 1 | no detectable difference at n = 3 | False |
| RAE-CTX-500 (V3_500-FNO_500) | global_nrmse_mean | 3 | 1.651e-05 | [-0.0003952, 0.0004282] | 1 | no detectable difference at n = 3 | False |

Failure counts (count/denominator per seed model):

- BM_RAE_FNO34_s0 false_surface_shock: 2/228 ()
- BM_RAE_FNO34_s0 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_FNO34_s0 gt_shock_present: 138/366 ()
- BM_RAE_FNO34_s0 physically_invalid_pred: 38/183 (49;65;136;137;172;176;185;189;221;295;330;335;336;364;375;397;399;431;439;478;481;593;601;844;882;931;946;983;990;997;1042;1052;1055;1057;1163;1191;1215;1216)
- BM_RAE_FNO34_s0 pred_shock_not_detected: 3/138 (814up;927lo;1062up)
- BM_RAE_FNO34_s1 false_surface_shock: 3/228 ()
- BM_RAE_FNO34_s1 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_FNO34_s1 gt_shock_present: 138/366 ()
- BM_RAE_FNO34_s1 physically_invalid_pred: 17/183 (127;330;481;492;553;581;683;688;718;913;927;1052;1055;1057;1139;1163;1191)
- BM_RAE_FNO34_s1 pred_shock_not_detected: 2/138 (716up;927lo)
- BM_RAE_FNO34_s2 false_surface_shock: 5/228 ()
- BM_RAE_FNO34_s2 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_FNO34_s2 gt_shock_present: 138/366 ()
- BM_RAE_FNO34_s2 physically_invalid_pred: 37/183 (19;36;53;70;172;176;221;222;233;267;335;366;375;376;405;464;481;553;581;646;668;678;688;718;720;838;913;927;949;983;990;1006;1052;1057;1163;1191;1215)
- BM_RAE_FNO34_s2 pred_shock_not_detected: 3/138 (97lo;814up;1062up)
- BM_RAE_REUSE_FNO_500_s0 false_surface_shock: 3/228 ()
- BM_RAE_REUSE_FNO_500_s0 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_FNO_500_s0 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_FNO_500_s0 physically_invalid_pred: 85/183 (61;65;80;95;97;99;136;137;160;176;181;185;189;221;222;233;253;267;269;289;322;330;335;336;355;357;364;366;375;376;377;397;399;405;429;431;437;464;478;481;553;556;572;580;581;601;625;673;676;678;688;698;710;718;732;793;838;844;874;882;893;905;909;913;931;946;983;990;997;1006;1009;1017;1052;1055;1057;1063;1093;1113;1139;1140;1162;1163;1191;1215;1216)
- BM_RAE_REUSE_FNO_500_s0 pred_shock_not_detected: 7/138 (80lo;97lo;172up;716up;814up;927lo;1062up)
- BM_RAE_REUSE_FNO_500_s1 false_surface_shock: 1/228 ()
- BM_RAE_REUSE_FNO_500_s1 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_FNO_500_s1 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_FNO_500_s1 physically_invalid_pred: 50/183 (19;36;53;65;70;160;176;181;189;221;222;233;267;269;330;355;375;377;399;405;438;464;481;494;553;581;646;668;676;678;683;688;718;868;874;882;905;913;927;983;990;1006;1052;1055;1057;1113;1149;1163;1191;1215)
- BM_RAE_REUSE_FNO_500_s1 pred_shock_not_detected: 6/138 (80lo;97lo;716up;814up;927lo;1062up)
- BM_RAE_REUSE_FNO_500_s2 false_surface_shock: 5/228 ()
- BM_RAE_REUSE_FNO_500_s2 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_FNO_500_s2 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_FNO_500_s2 physically_invalid_pred: 49/183 (127;137;160;172;176;185;221;222;233;267;269;275;335;336;375;376;405;431;464;481;553;556;561;581;593;601;625;668;678;688;718;874;905;913;927;946;983;990;997;1006;1052;1054;1057;1062;1063;1113;1163;1191;1216)
- BM_RAE_REUSE_FNO_500_s2 pred_shock_not_detected: 3/138 (97lo;814up;1062up)
- BM_RAE_V2J2_s0 false_surface_shock: 2/228 ()
- BM_RAE_V2J2_s0 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_V2J2_s0 gt_shock_present: 138/366 ()
- BM_RAE_V2J2_s0 physically_invalid_pred: 20/183 (21;135;160;176;189;221;222;267;481;553;581;868;874;913;931;983;1042;1113;1163;1191)
- BM_RAE_V2J2_s0 pred_shock_not_detected: 2/138 (814up;1062up)
- BM_RAE_V2J2_s1 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_V2J2_s1 gt_shock_present: 138/366 ()
- BM_RAE_V2J2_s1 physically_invalid_pred: 4/183 (221;222;376;983)
- BM_RAE_V2J2_s1 pred_shock_not_detected: 2/138 (814up;1062up)
- BM_RAE_V2J2_s2 false_surface_shock: 4/228 ()
- BM_RAE_V2J2_s2 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_V2J2_s2 gt_shock_present: 138/366 ()
- BM_RAE_V2J2_s2 physically_invalid_pred: 3/183 (189;431;1215)
- BM_RAE_V2J2_s2 pred_shock_not_detected: 1/138 (814up)
- BM_RAE_REUSE_V3_500_s0 false_surface_shock: 5/228 ()
- BM_RAE_REUSE_V3_500_s0 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_V3_500_s0 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_V3_500_s0 physically_invalid_pred: 50/183 (127;137;160;176;185;222;233;267;269;335;336;355;376;380;395;431;459;464;478;481;553;581;593;601;625;629;668;676;678;688;698;718;720;732;754;793;838;874;913;946;949;997;1052;1055;1057;1113;1140;1163;1191;1216)
- BM_RAE_REUSE_V3_500_s0 pred_shock_not_detected: 4/138 (172up;927lo;1062up;1215up)
- BM_RAE_REUSE_V3_500_s1 false_surface_shock: 2/228 ()
- BM_RAE_REUSE_V3_500_s1 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_V3_500_s1 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_V3_500_s1 physically_invalid_pred: 7/183 (160;176;330;397;439;882;1215)
- BM_RAE_REUSE_V3_500_s1 pred_shock_not_detected: 2/138 (814up;1062up)
- BM_RAE_REUSE_V3_500_s2 false_surface_shock: 2/228 ()
- BM_RAE_REUSE_V3_500_s2 gt_shock_band_empty: 6/102 (172;525;814;833;1062;1215)
- BM_RAE_REUSE_V3_500_s2 gt_shock_present: 138/366 ()
- BM_RAE_REUSE_V3_500_s2 physically_invalid_pred: 54/183 (19;36;49;53;65;70;137;160;172;176;185;189;222;233;267;269;335;336;375;376;405;431;437;464;481;553;581;601;625;646;668;678;683;688;718;732;874;882;913;946;983;990;997;1006;1055;1057;1063;1066;1113;1140;1163;1191;1215;1216)
- BM_RAE_REUSE_V3_500_s2 pred_shock_not_detected: 4/138 (172up;814up;927lo;1062up)

## nozzle: Axisymmetric nozzle plume, M_inf family, split A

**Claim (frozen rule):** no shock-capturing difference established on this dataset

| pair | metric | n | mean a-b | 95 % CI | a lower (seeds) | verdict | failure-confounded |
|---|---|---|---|---|---|---|---|
| NOZ-MAIN-1 (V2C-F00) | P1_shock_p_nrmse | 3 | 0.0012 | [-0.0105, 0.0129] | 2 | no detectable difference at n = 3 | False |
| NOZ-MAIN-1 (V2C-F00) | P2_thickness_abs_log | 3 | 0.2457 | [-0.3215, 0.813] | 0 | no detectable difference at n = 3 | False |
| NOZ-MAIN-1 (V2C-F00) | val_loss | 3 | -3.187e-05 | [-0.0001401, 7.631e-05] | 3 | no detectable difference at n = 3 | False |
| NOZ-MAIN-1 (V2C-F00) | global_nrmse_mean | 3 | -0.0009028 | [-0.003288, 0.001482] | 3 | no detectable difference at n = 3 | False |
| NOZ-MAIN-2 (V2S-F10) | P1_shock_p_nrmse | 3 | -0.001247 | [-0.005628, 0.003134] | 2 | no detectable difference at n = 3 | False |
| NOZ-MAIN-2 (V2S-F10) | P2_thickness_abs_log | 3 | 0.08768 | [-0.5317, 0.7071] | 1 | no detectable difference at n = 3 | False |
| NOZ-MAIN-2 (V2S-F10) | val_loss | 3 | -5.949e-05 | [-0.0001929, 7.396e-05] | 2 | no detectable difference at n = 3 | False |
| NOZ-MAIN-2 (V2S-F10) | global_nrmse_mean | 3 | -0.001622 | [-0.004089, 0.000845] | 3 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-S (V2S-F11) | P1_shock_p_nrmse | 3 | 0.0005775 | [-0.006244, 0.007399] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-S (V2S-F11) | P2_thickness_abs_log | 3 | -0.06549 | [-0.6695, 0.5385] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-S (V2S-F11) | val_loss | 3 | -4.299e-05 | [-0.0001447, 5.874e-05] | 3 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-S (V2S-F11) | global_nrmse_mean | 3 | -0.001339 | [-0.003903, 0.001225] | 3 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-C (V2C-F11) | P1_shock_p_nrmse | 3 | 0.004662 | [-0.006817, 0.01614] | 0 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-C (V2C-F11) | P2_thickness_abs_log | 3 | 0.04818 | [-0.5834, 0.6798] | 1 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-C (V2C-F11) | val_loss | 3 | 7.63e-05 | [-4.191e-05, 0.0001945] | 0 | no detectable difference at n = 3 | False |
| NOZ-CTX-F11-C (V2C-F11) | global_nrmse_mean | 3 | 0.001271 | [-0.001414, 0.003955] | 0 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-C (V2C-W00) | P1_shock_p_nrmse | 3 | -0.001316 | [-0.00909, 0.006457] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-C (V2C-W00) | P2_thickness_abs_log | 3 | -0.06103 | [-0.9427, 0.8206] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-C (V2C-W00) | val_loss | 3 | -0.0001252 | [-0.0002227, -2.767e-05] | 3 | V2C consistently lower | False |
| NOZ-CTX-V3-C (V2C-W00) | global_nrmse_mean | 3 | -0.003028 | [-0.00496, -0.001096] | 3 | V2C consistently lower | False |
| NOZ-CTX-V3-S (V2S-W11) | P1_shock_p_nrmse | 3 | -0.001167 | [-0.00486, 0.002525] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-S (V2S-W11) | P2_thickness_abs_log | 3 | -0.077 | [-0.642, 0.488] | 2 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-S (V2S-W11) | val_loss | 3 | -5.336e-05 | [-0.0001846, 7.79e-05] | 3 | no detectable difference at n = 3 | False |
| NOZ-CTX-V3-S (V2S-W11) | global_nrmse_mean | 3 | -0.001711 | [-0.004143, 0.0007202] | 3 | no detectable difference at n = 3 | False |

Failure counts (count/denominator per seed model):

- ABL_REF_FNO_s0 gt_shock_present: 6/6 ()
- ABL_REF_FNO_s0 physically_invalid_pred: 2/6 (f107;f125)
- ABL_REF_FNO_s0 pred_shock_not_detected: 1/6 (f053)
- ABL_REF_FNO_s0 pressure_ratio_not_computable: 2/6 (f107;f125)
- ABL_REF_FNO_s0 upstream_nonpositive_p: 2/6 (f107;f125)
- ABL_REF_FNO_s1 gt_shock_present: 6/6 ()
- ABL_REF_FNO_s1 physically_invalid_pred: 2/6 (f107;f125)
- ABL_REF_FNO_s1 pred_shock_not_detected: 1/6 (f053)
- ABL_REF_FNO_s1 pressure_ratio_not_computable: 2/6 (f107;f125)
- ABL_REF_FNO_s1 upstream_nonpositive_p: 2/6 (f107;f125)
- ABL_REF_FNO_s2 gt_shock_present: 6/6 ()
- ABL_REF_FNO_s2 physically_invalid_pred: 3/6 (f093;f107;f125)
- ABL_REF_FNO_s2 pred_shock_not_detected: 1/6 (f053)
- ABL_REF_FNO_s2 pressure_ratio_not_computable: 3/6 (f093;f107;f125)
- ABL_REF_FNO_s2 upstream_nonpositive_p: 3/6 (f093;f107;f125)
- NOZ_HPO_A_F10_s0 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F10_s0 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_F10_s0 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_F10_s0 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_F10_s0 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_F10_s1 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F10_s1 physically_invalid_pred: 2/6 (f107;f125)
- NOZ_HPO_A_F10_s1 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_F10_s1 pressure_ratio_not_computable: 2/6 (f107;f125)
- NOZ_HPO_A_F10_s1 upstream_nonpositive_p: 2/6 (f107;f125)
- NOZ_HPO_A_F10_s2 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F10_s2 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_F10_s2 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_F10_s2 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_F10_s2 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_F11_s0 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F11_s0 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_F11_s0 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_F11_s0 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_F11_s0 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_F11_s1 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F11_s1 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_F11_s1 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_F11_s1 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_F11_s1 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_F11_s2 gt_shock_present: 6/6 ()
- NOZ_HPO_A_F11_s2 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_F11_s2 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2C_s0 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2C_s0 physically_invalid_pred: 1/6 (f125)
- NOZ_WV_A_V2C_s0 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2C_s0 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_WV_A_V2C_s0 upstream_nonpositive_p: 1/6 (f125)
- NOZ_WV_A_V2C_s1 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2C_s1 physically_invalid_pred: 1/6 (f125)
- NOZ_WV_A_V2C_s1 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2C_s2 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2C_s2 physically_invalid_pred: 1/6 (f125)
- NOZ_WV_A_V2C_s2 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2S_s0 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2S_s0 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2S_s1 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2S_s1 physically_invalid_pred: 1/6 (f125)
- NOZ_WV_A_V2S_s1 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2S_s1 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_WV_A_V2S_s1 upstream_nonpositive_p: 1/6 (f125)
- NOZ_WV_A_V2S_s2 gt_shock_present: 6/6 ()
- NOZ_WV_A_V2S_s2 physically_invalid_pred: 1/6 (f125)
- NOZ_WV_A_V2S_s2 pred_shock_not_detected: 1/6 (f053)
- NOZ_WV_A_V2S_s2 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_WV_A_V2S_s2 upstream_nonpositive_p: 1/6 (f125)
- ABL_REF_WNO_s0 gt_shock_present: 6/6 ()
- ABL_REF_WNO_s0 physically_invalid_pred: 1/6 (f125)
- ABL_REF_WNO_s0 pred_shock_not_detected: 1/6 (f053)
- ABL_REF_WNO_s0 pressure_ratio_not_computable: 1/6 (f125)
- ABL_REF_WNO_s0 upstream_nonpositive_p: 1/6 (f125)
- ABL_REF_WNO_s1 gt_shock_present: 6/6 ()
- ABL_REF_WNO_s1 physically_invalid_pred: 1/6 (f125)
- ABL_REF_WNO_s1 pred_shock_not_detected: 2/6 (f053;f068)
- ABL_REF_WNO_s1 pressure_ratio_not_computable: 1/6 (f125)
- ABL_REF_WNO_s1 upstream_nonpositive_p: 1/6 (f125)
- ABL_REF_WNO_s2 gt_shock_present: 6/6 ()
- ABL_REF_WNO_s2 physically_invalid_pred: 2/6 (f107;f125)
- ABL_REF_WNO_s2 pred_shock_not_detected: 1/6 (f053)
- ABL_REF_WNO_s2 pressure_ratio_not_computable: 1/6 (f125)
- ABL_REF_WNO_s2 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_W11_s0 gt_shock_present: 6/6 ()
- NOZ_HPO_A_W11_s0 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_W11_s0 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_W11_s0 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_W11_s0 upstream_nonpositive_p: 1/6 (f125)
- NOZ_HPO_A_W11_s1 gt_shock_present: 6/6 ()
- NOZ_HPO_A_W11_s1 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_W11_s1 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_W11_s2 gt_shock_present: 6/6 ()
- NOZ_HPO_A_W11_s2 physically_invalid_pred: 1/6 (f125)
- NOZ_HPO_A_W11_s2 pred_shock_not_detected: 1/6 (f053)
- NOZ_HPO_A_W11_s2 pressure_ratio_not_computable: 1/6 (f125)
- NOZ_HPO_A_W11_s2 upstream_nonpositive_p: 1/6 (f125)

