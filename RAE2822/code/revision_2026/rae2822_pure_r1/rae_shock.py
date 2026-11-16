"""Surface shock detector (fixed before any model exists; tuned only on CFD ground truth, see evaluation_metrics.md).

On one surface (LE->TE ordered x, Cp) with free-stream Mach M:
  p/p_inf = 1 + gamma/2 M^2 Cp,  isentropic Mach M_is from p0/p (p0/p_inf = (1 + 0.2 M^2)^3.5)
  R(x) = p(x + L/2) / p(x - L/2) over a window L = 0.04 c, evaluated on 2000 x-stations in [0.02 + L/2, 0.98 - L/2],
         and set to 0 where M_is(x - L/2) < 1 (compression must start from supersonic flow)
  shock candidate = argmax R; position x_s = max dp/dx within +-L of it; strength p2/p1 = max p in [x_s, x_s+0.06]
         over min p in [x_s-0.06, x_s]; M1 = max M_is in [x_s-0.06, x_s]
Classes: subcritical (max M_is < 1); supercritical_noshock (R < 1.05 or no supersonic start);
  weak_ambiguous (1.05 <= R < 1.10); LE_shock (R >= 1.10, x_s < 0.10); midchord_shock (R >= 1.10, x_s >= 0.10).
A smooth isentropic recompression gives R close to 1 over 0.04 c; R = 1.10 is the normal-shock pressure ratio
for M1 ~ 1.04.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
L_WIN = 0.04
R_WEAK = 1.05
R_SHOCK = 1.10
X_LE = 0.10
X_RANGE = (0.01, 0.99)


def isentropic_mach(cp, Minf):
    p = 1 + 0.5 * GAMMA * Minf ** 2 * cp
    p0 = (1 + 0.2 * Minf ** 2) ** 3.5
    return p, np.sqrt(np.maximum(5 * ((p0 / np.maximum(p, 1e-9)) ** (1 / 3.5) - 1), 0))


def detect(x, cp, Minf):
    x = np.asarray(x, float)
    cp = np.asarray(cp, float)
    p, mis = isentropic_mach(cp, Minf)
    keep = (x >= X_RANGE[0]) & (x <= X_RANGE[1]) & np.isfinite(cp)
    x, p, mis = x[keep], p[keep], mis[keep]
    out = dict(maxMis=float(mis.max()) if len(mis) else np.nan, R=np.nan, xs=np.nan, M1=np.nan, p21=np.nan)
    if len(x) < 10:
        out["cls"] = "invalid"
        return out
    xc = np.linspace(0.02 + L_WIN / 2, 0.98 - L_WIN / 2, 2000)
    pa = np.interp(xc - L_WIN / 2, x, p)
    pb = np.interp(xc + L_WIN / 2, x, p)
    ma = np.interp(xc - L_WIN / 2, x, mis)
    R = np.where(ma >= 1.0, pb / pa, 0.0)
    k = int(np.argmax(R))
    if R[k] > 0:
        ww = (x >= xc[k] - L_WIN) & (x <= xc[k] + L_WIN)
        dpdx = np.gradient(p, x)
        xs = float(x[ww][np.argmax(dpdx[ww])])
        w1 = (x >= xs - 0.06) & (x <= xs)
        w2 = (x >= xs) & (x <= xs + 0.06)
        out.update(R=float(R[k]), xs=xs, M1=float(mis[w1].max()), p21=float(p[w2].max() / p[w1].min()))
    out["cls"] = classify(out["R"], out["xs"], out["maxMis"])
    return out


def classify(R, xs, maxMis):
    if not maxMis >= 1:
        return "subcritical"
    if not R >= R_WEAK:
        return "supercritical_noshock"
    if R < R_SHOCK:
        return "weak_ambiguous"
    return "LE_shock" if xs < X_LE else "midchord_shock"
