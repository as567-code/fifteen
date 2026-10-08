"""Independent checks on the fast engine.

`lp_month` solves the monthly demand-charge problem as a linear program (HiGHS) with no
shared code from `fifteen.shave`. If the greedy/bisection oracle and the LP disagree, one of
them is wrong. `check_dispatch` verifies a dispatch trace obeys physics.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .battery import Battery


def lp_month(load, masks, rates, bat: Battery, s0: float, dt: float = 0.25, weights=None):
    """min sum_k r_k T_k  s.t. battery dynamics, 0<=d,c<=P, 0<=s<=E, d<=L, L-d+c <= w_kt T_k."""
    load = np.asarray(load, float)
    if weights is None:
        weights = np.where(np.asarray(masks, bool), 1.0, np.inf)
    weights = np.atleast_2d(np.asarray(weights, float))
    rates = np.asarray(rates, float)
    n, K = load.size, weights.shape[0]
    P, E, ec, ed = bat.power_kw, bat.energy_kwh, bat.eta_charge, bat.eta_discharge
    # variable layout: d[0:n], c[n:2n], s[2n:3n+1], T[3n+1:3n+1+K]
    nv = 3 * n + 1 + K
    iD, iC, iS, iT = 0, n, 2 * n, 3 * n + 1
    cost = np.zeros(nv)
    cost[iT:] = rates
    cost[iD:iD + n] = 1e-7  # tie-breaker: no pointless cycling
    cost[iC:iC + n] = 1e-7

    # equality: s[t+1] - s[t] - ec*dt*c[t] + dt/ed*d[t] = 0 ; s[0] = s0
    rows = np.arange(n)
    Aeq = sp.lil_matrix((n + 1, nv))
    Aeq[rows, iS + rows + 1] = 1.0
    Aeq[rows, iS + rows] = -1.0
    Aeq[rows, iC + rows] = -ec * dt
    Aeq[rows, iD + rows] = dt / ed
    Aeq[n, iS] = 1.0
    beq = np.zeros(n + 1)
    beq[n] = s0

    # inequality: -d[t] + c[t] - w_kt T_k <= -L[t]  for every t with finite weight
    blocks, b_ub = [], []
    for k in range(K):
        idx = np.flatnonzero(np.isfinite(weights[k]) & (weights[k] < 1e17))
        m = idx.size
        A = sp.lil_matrix((m, nv))
        A[np.arange(m), iD + idx] = -1.0
        A[np.arange(m), iC + idx] = 1.0
        A[np.arange(m), iT + k] = -weights[k][idx]
        blocks.append(A.tocsr())
        b_ub.append(-load[idx])
    Aub = sp.vstack(blocks).tocsr()
    bub = np.concatenate(b_ub)

    bounds = (
        [(0.0, min(P, max(Lt, 0.0))) for Lt in load]  # no export
        + [(0.0, P)] * n
        + [(0.0, E)] * (n + 1)
        + [(0.0, None)] * K
    )
    res = linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    T = res.x[iT:]
    return {"thresholds": T, "cost": float(rates @ T), "d": res.x[iD:iD + n], "c": res.x[iC:iC + n], "s": res.x[iS:iS + n + 1]}


def check_dispatch(load, net, soc, bat: Battery, s0: float, dt: float = 0.25, tol: float = 1e-6) -> list[str]:
    """Return a list of violated invariants (empty list = clean)."""
    load, net, soc = (np.asarray(a, float) for a in (load, net, soc))
    problems = []
    flow = net - load  # +charge / -discharge (AC, kW)
    if (np.abs(flow) > bat.power_kw + tol).any():
        problems.append("power limit exceeded")
    if (soc < -tol).any() or (soc > bat.energy_kwh + tol).any():
        problems.append("state of charge out of bounds")
    if (net < -tol).any():
        problems.append("exports to grid (net load < 0)")
    prev = np.concatenate([[s0], soc[:-1]])
    expected = prev + np.where(flow > 0, flow * bat.eta_charge * dt, flow * dt / bat.eta_discharge)
    if np.abs(expected - soc).max() > 1e-6 * max(1.0, bat.energy_kwh):
        problems.append("energy not conserved between intervals")
    return problems
