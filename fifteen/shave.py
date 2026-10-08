"""Peak shaving: dispatch simulation and the perfect-foresight optimum.

The key structural fact this module relies on:

    For "keep net load <= tau_t", the greedy rule
        discharge exactly what is needed when load > tau_t,
        charge as much as allowed (without pushing net load above tau_t) otherwise
    keeps the state of charge as high as any feasible policy at every step.
    So if greedy cannot hold tau, nothing can.

That turns "minimise the monthly demand charge" into a 1-D (or nested 2-D) search over
thresholds, each candidate checked by a 15-minute simulation compiled with numba.
`fifteen.audit.lp_month` solves the same problem as a linear program, and the test suite
asserts that both give the same answer.

Billing demand is generalised with per-interval weights w_t:
    billing kW = max_t net_t / w_t
w_t = 1 counts an interval fully, w_t = inf ignores it, and w_t = 1/f models tariffs such as
Eversource G-2 where "demand recorded during off-peak hours will be reduced by 55 percent"
(f = 0.45). Holding billing demand <= T then means net_t <= T * w_t.

Units: load and net load in kW (15-minute average), energy in kWh, dt in hours.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .battery import Battery

DT = 0.25
INF = 1e18


@njit(cache=True)
def _hold(load, tau, p, e, ec, ed, s0, dt, out_net, out_soc):
    """Greedy dispatch against per-interval thresholds. Returns (max violation, end soc)."""
    s = s0
    worst = 0.0
    for t in range(load.shape[0]):
        L = load[t]
        th = tau[t]
        if L > th:
            need = L - th
            dmax = min(p, s * ed / dt, L)
            d = need if need < dmax else dmax
            net = L - d
            s -= d * dt / ed
            v = net - th
            if v > worst:
                worst = v
        else:
            room = th - L
            if room > p:
                room = p
            cmax = (e - s) / (ec * dt)
            c = room if room < cmax else cmax
            if c < 0.0:
                c = 0.0
            net = L + c
            s += c * ec * dt
        if s < 0.0:
            s = 0.0
        if s > e:
            s = e
        out_net[t] = net
        out_soc[t] = s
    return worst, s


@njit(cache=True)
def _feasible(load, w0, T0, w1, T1, p, e, ec, ed, s0, dt):
    """Can net_t <= min(T0*w0_t, T1*w1_t) be held for every t?"""
    s = s0
    for t in range(load.shape[0]):
        L = load[t]
        th = INF if w0[t] >= INF else T0 * w0[t]
        if w1[t] < INF and T1 * w1[t] < th:
            th = T1 * w1[t]
        if L > th:
            need = L - th
            dmax = min(p, s * ed / dt, L)
            if need > dmax + 1e-9:
                return False
            s -= need * dt / ed
        else:
            room = th - L
            if room > p:
                room = p
            cmax = (e - s) / (ec * dt)
            c = room if room < cmax else cmax
            if c > 0.0:
                s += c * ec * dt
        if s < 0.0:
            s = 0.0
        if s > e:
            s = e
    return True


@njit(cache=True)
def _min_T(load, w, wo, To, p, e, ec, ed, s0, dt, iters):
    """Smallest T with net_t <= T*w_t (and <= To*wo_t) for all t. INF if impossible."""
    hi = 0.0
    lo = 0.0
    for t in range(load.shape[0]):
        if w[t] < INF:
            v = load[t] / w[t]
            if v > hi:
                hi = v
            u = (load[t] - p) / w[t]
            if u > lo:
                lo = u
    if hi == 0.0:
        return 0.0
    if not _feasible(load, w, hi, wo, To, p, e, ec, ed, s0, dt):
        return INF
    if _feasible(load, w, lo, wo, To, p, e, ec, ed, s0, dt):
        return lo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _feasible(load, w, mid, wo, To, p, e, ec, ed, s0, dt):
            hi = mid
        else:
            lo = mid
    return hi


def hold(load: np.ndarray, tau: np.ndarray | float, bat: Battery, s0: float | None = None, dt: float = DT):
    """Simulate greedy dispatch holding net load under `tau` (scalar or per-interval)."""
    load = np.ascontiguousarray(load, dtype=np.float64)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=np.float64), load.shape).copy()
    net = np.empty_like(load)
    soc = np.empty_like(load)
    s0 = bat.energy_kwh if s0 is None else float(s0)
    worst, s_end = _hold(load, tau_arr, bat.power_kw, bat.energy_kwh, bat.eta_charge, bat.eta_discharge, s0, dt, net, soc)
    return net, soc, float(worst), float(s_end)


def masks_to_weights(masks: np.ndarray) -> np.ndarray:
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    return np.where(masks, 1.0, INF)


@dataclass
class MonthPlan:
    thresholds: np.ndarray  # billing demand per component, kW
    tau: np.ndarray  # per-interval net-load limit actually enforced
    cost: float  # sum_k rate_k * threshold_k


def plan_month(
    load: np.ndarray,
    masks: np.ndarray | None,
    rates: np.ndarray,
    bat: Battery,
    s0: float,
    dt: float = DT,
    iters: int = 50,
    weights: np.ndarray | None = None,
) -> MonthPlan:
    """Perfect-foresight billing demands minimising sum_k rate_k * max_t(net_t / w_kt).

    Pass either boolean `masks` (K, T) or float `weights` (K, T). K = 1 is solved by bisection.
    K = 2 is solved by golden-section over component 0 with an inner bisection on component 1;
    the cost is convex in the outer threshold because the feasible set of thresholds is the
    projection of a polyhedron.
    """
    load = np.ascontiguousarray(load, dtype=np.float64)
    W = np.ascontiguousarray(masks_to_weights(masks) if weights is None else np.atleast_2d(weights), dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    p, e, ec, ed = bat.power_kw, bat.energy_kwh, bat.eta_charge, bat.eta_discharge
    K = W.shape[0]
    none = np.full(load.shape, INF)
    if K == 1:
        thresholds = np.array([_min_T(load, W[0], none, INF, p, e, ec, ed, s0, dt, iters)])
    elif K == 2:
        def inner(T0: float) -> float:
            return _min_T(load, W[1], W[0], T0, p, e, ec, ed, s0, dt, iters)

        a = _min_T(load, W[0], none, INF, p, e, ec, ed, s0, dt, iters)
        b = float(np.max(np.where(W[0] < INF, load / W[0], 0.0)))
        gr = (math.sqrt(5) - 1) / 2
        f = lambda T0: rates[0] * T0 + rates[1] * inner(T0)  # noqa: E731
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(60):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = f(d)
        T0 = min([a, b, c, d], key=f)
        thresholds = np.array([T0, inner(T0)])
    else:
        raise NotImplementedError("use fifteen.audit.lp_month for >2 demand components")

    tau = np.min(np.where(W >= INF, INF, thresholds[:, None] * W), axis=0)
    return MonthPlan(thresholds=thresholds, tau=tau, cost=float(rates @ thresholds))
