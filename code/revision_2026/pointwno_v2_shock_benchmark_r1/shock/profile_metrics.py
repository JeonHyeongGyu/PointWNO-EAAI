"""Frozen 1-D shock-profile metrics for airfoil surfaces (NACA upper surface, RAE2822 upper / lower surface).

Profile: pressure coefficient q(s) along the surface, s = x/c ordered leading edge -> trailing edge (a compression shock
is a RISE of Cp). Anchor s_a: the GT shock position fixed BEFORE any model output exists (configs/gt_shock_cases.json).
Window: |s - s_a| <= HALF (0.06 c, the rae_shock.detect pre/post window). All quantities of a prediction are computed
in the SAME window as its GT, on the SAME surface samples.

  rise(q):      q1 = min q on the upstream half [s_a - HALF, s_a], q2 = max q on the downstream half [s_a, s_a + HALF],
                jump = q2 - q1 (> 0 required); s_q1 / s_q2 their positions (s_q1 < s_q2 required)
  10-90 width:  on [s_q1, s_q2] the first s where q >= q1 + 0.1 jump and the first s after it where q >= q1 + 0.9 jump
                (linear interpolation between samples); width = s90 - s10 (> 0)
  location:     s of max dq/ds on [s_q1, s_q2]
Per (case, side) metrics of a prediction P against the GT profile G:
  thickness_ratio = width_P / width_G ; thickness_abs_log = |ln ratio|            (PRIMARY 2)
  location_abs_err = |s_loc,P - s_loc,G|
  jump_abs_rel_err = |jump_P / jump_G - 1|
  grad_nrmse = RMS(dq/ds_P - dq/ds_G) / max|dq/ds_G| over the window samples
  post_overshoot = max(0, max q_P[s > s_a] - max q_G[s > s_a]) / jump_G
  pre_undershoot = max(0, min q_G[s <= s_a] - min q_P[s <= s_a]) / jump_G
  tv_abs_rel_err = |TV(q_P) / TV(q_G) - 1| on the window
  spurious_osc   = max(0, XS(q_P) - XS(q_G)) / jump_G, XS(q) = TV(q) - (max q - min q) on the window (0 for a
                   monotone profile; the excess variation is oscillation)
Failure codes (returned, never hidden): 'window' (< MIN_SAMPLES samples on either side), 'no_rise' (jump <= 0 or
s_q1 >= s_q2), 'no_crossing' (10 / 90 % level not crossed), 'nonfinite'. A failed GT profile marks the case
'gt_profile_failed' (reported; the case is then not a valid thickness case for ANY model). A failed prediction profile
leaves its thickness / location / jump metrics NaN and is counted in the failure rate of that model.
"""
from __future__ import annotations

import math

import numpy as np

HALF = 0.06
MIN_SAMPLES = 4
PROFILE_VERSION = "bm_surface_profile_v1"   # repeated abscissae: first sample kept (fixed before any benchmark model exists)


def _cross(s, q, level, start):
    for i in range(start, len(q) - 1):
        if q[i] < level <= q[i + 1]:
            t = (level - q[i]) / (q[i + 1] - q[i])
            return s[i] + t * (s[i + 1] - s[i]), i
        if q[i] >= level:
            return s[i], i
    if len(q) and q[-1] >= level:
        return s[-1], len(q) - 1
    return float("nan"), -1


def _sorted_unique(s, *qs):
    """Sort by s (stable) and keep the FIRST sample of every repeated s (surface point lists can repeat a coordinate,
    e.g. RAE2822 airfoil.npy has 8 repeated x; a repeated abscissa would make dq/ds infinite)."""
    s = np.asarray(s, float)
    o = np.argsort(s, kind="stable")
    s = s[o]
    keep = np.concatenate(([True], np.diff(s) > 0))
    return (s[keep],) + tuple(np.asarray(q, float)[o][keep] for q in qs)


def rise(s, q, s_a, half=HALF):
    s, q = _sorted_unique(s, q)
    w = np.abs(s - s_a) <= half
    up = w & (s <= s_a)
    dn = w & (s >= s_a)
    if up.sum() < MIN_SAMPLES or dn.sum() < MIN_SAMPLES:
        return {"fail": "window"}
    if not np.all(np.isfinite(q[w])):
        return {"fail": "nonfinite"}
    iu, idn = np.where(up)[0], np.where(dn)[0]
    i1 = iu[int(np.argmin(q[iu]))]
    i2 = idn[int(np.argmax(q[idn]))]
    q1, q2 = float(q[i1]), float(q[i2])
    jump = q2 - q1
    if not (jump > 0) or not (i1 < i2):
        return {"fail": "no_rise", "q1": q1, "q2": q2}
    seg_s, seg_q = s[i1:i2 + 1], q[i1:i2 + 1]
    s10, k10 = _cross(seg_s, seg_q, q1 + 0.1 * jump, 0)
    s90, _ = _cross(seg_s, seg_q, q1 + 0.9 * jump, max(k10, 0))
    if not (math.isfinite(s10) and math.isfinite(s90)) or not (s90 > s10):
        return {"fail": "no_crossing", "q1": q1, "q2": q2}
    g = np.gradient(seg_q, seg_s) if len(seg_s) > 1 else np.array([0.0])
    return {"fail": None, "q1": q1, "q2": q2, "jump": jump, "s10": float(s10), "s90": float(s90),
            "width": float(s90 - s10), "loc": float(seg_s[int(np.argmax(g))]), "max_grad": float(np.max(g))}


def _tv(q):
    return float(np.abs(np.diff(q)).sum())


def compare(s, q_pred, q_gt, s_a, half=HALF):
    """All per-(case, side) metrics of q_pred against q_gt (same samples s). Returns (metrics, gt_rise, pred_rise)."""
    s, qp, qg = _sorted_unique(s, q_pred, q_gt)
    rg, rp = rise(s, qg, s_a, half), rise(s, qp, s_a, half)
    nan = float("nan")
    m = {k: nan for k in ("thickness_ratio", "thickness_abs_log", "location_abs_err", "jump_abs_rel_err", "grad_nrmse",
                          "post_overshoot", "pre_undershoot", "tv_abs_rel_err", "spurious_osc")}
    m["gt_profile_failed"] = rg["fail"] is not None
    m["pred_profile_fail"] = rp["fail"]
    if rg["fail"] is not None:
        return m, rg, rp
    w = np.abs(s - s_a) <= half
    jg = rg["jump"]
    if np.all(np.isfinite(qp[w])):
        gp, gg = np.gradient(qp[w], s[w]), np.gradient(qg[w], s[w])
        m["grad_nrmse"] = float(np.sqrt(np.mean((gp - gg) ** 2)) / max(np.max(np.abs(gg)), 1e-30))
        post, pre = w & (s > s_a), w & (s <= s_a)
        m["post_overshoot"] = float(max(0.0, qp[post].max() - qg[post].max()) / jg)
        m["pre_undershoot"] = float(max(0.0, qg[pre].min() - qp[pre].min()) / jg)
        tvg = _tv(qg[w])
        m["tv_abs_rel_err"] = float(abs(_tv(qp[w]) / max(tvg, 1e-30) - 1.0))
        xs_p = _tv(qp[w]) - (qp[w].max() - qp[w].min())
        xs_g = tvg - (qg[w].max() - qg[w].min())
        m["spurious_osc"] = float(max(0.0, xs_p - xs_g) / jg)
    if rp["fail"] is None:
        r = rp["width"] / rg["width"]
        m["thickness_ratio"] = float(r)
        m["thickness_abs_log"] = float(abs(math.log(r)))
        m["location_abs_err"] = float(abs(rp["loc"] - rg["loc"]))
        m["jump_abs_rel_err"] = float(abs(rp["jump"] / jg - 1.0))
    return m, rg, rp
