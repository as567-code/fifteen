"""Value stacking: demand-charge shaving + Mass Save ConnectedSolutions Daily Dispatch.

ConnectedSolutions pays $/kW-summer on the battery's AVERAGE discharge across all of the
season's events (pay-for-performance; batteries are metered at the battery, no baseline).
Events are announced day-ahead, so their timing is known; the member's own monthly peak is
not. Both streams draw on the same 470 kWh, and an event that lands on the member's peak day
can leave the battery empty when the building needs it.

`lp_stack_month` finds the best split for one month with perfect foresight of the load:
    minimise  rate * billing_demand  -  sum_events  v_e * mean_{t in e}(discharge_t - charge_t)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .battery import Battery


def lp_stack_month(load, w, rate, events, v_event, bat: Battery, s0: float, dt: float = 0.25):
    """events: list of index arrays (intervals of each event inside this month).
    v_event: $ per kW of average event performance for ONE event (= season $/kW / n_events).
    w: billing-demand weights (1 counted, >=1e17 ignored, 1/f discounted)."""
    load = np.asarray(load, float)
    w = np.asarray(w, float)
    n = load.size
    P, E, ec, ed = bat.power_kw, bat.energy_kwh, bat.eta_charge, bat.eta_discharge
    nv = 3 * n + 2
    iD, iC, iS, iT = 0, n, 2 * n, 3 * n + 1
    cost = np.zeros(nv)
    cost[iT] = rate
    cost[iD:iD + n] = 1e-7
    cost[iC:iC + n] = 1e-7
    for ev in events:
        a = v_event / len(ev)
        cost[iD + ev] -= a
        cost[iC + ev] += a

    rows = np.arange(n)
    Aeq = sp.lil_matrix((n + 1, nv))
    Aeq[rows, iS + rows + 1] = 1.0
    Aeq[rows, iS + rows] = -1.0
    Aeq[rows, iC + rows] = -ec * dt
    Aeq[rows, iD + rows] = dt / ed
    Aeq[n, iS] = 1.0
    beq = np.zeros(n + 1)
    beq[n] = s0

    idx = np.flatnonzero(w < 1e17)
    m = idx.size
    A = sp.lil_matrix((m, nv))
    A[np.arange(m), iD + idx] = -1.0
    A[np.arange(m), iC + idx] = 1.0
    A[np.arange(m), iT] = -w[idx]
    bounds = ([(0.0, min(P, max(L, 0.0))) for L in load] + [(0.0, P)] * n + [(0.0, E)] * (n + 1) + [(0.0, None)])
    res = linprog(cost, A_ub=A.tocsr(), b_ub=-load[idx], A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    d, c, s = res.x[iD:iD + n], res.x[iC:iC + n], res.x[iS:iS + n + 1]
    perf = [float((d[ev] - c[ev]).mean()) for ev in events]
    net = load - d + c
    billed = float((net[idx] / w[idx]).max()) if m else 0.0
    return {"net": net, "soc_end": float(s[-1]), "billed_kw": billed, "event_kw": perf}
