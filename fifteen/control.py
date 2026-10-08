"""Non-anticipative peak-shaving controllers and the backtest that scores them.

A real controller sees the building's load as it happens (it can respond inside the
15-minute interval) but never the future. Every policy here is a rule for choosing the
threshold tau_t it defends; the battery action is then the same greedy rule used by the
oracle. Policies differ only in how much they know and how they set tau.

    timer     fixed daily discharge window learned from history (what simple systems do)
    ratchet   start each month at (last month's peak - beta*P), raise tau when overrun
    mpc       re-plan every 15 min against a day-ahead quantile forecast, corrected by the
              load seen so far today; never defend below the month's already-sunk peak

The "sunk peak" rule is the important bit of economics: a demand charge is set by the single
highest interval of the month, so once the month's peak is X, shaving anything below X is
worth exactly zero and only drains the battery before the day that matters.
"""

from __future__ import annotations

import numpy as np
from numba import njit

SLOTS = 96


@njit(cache=True)
def _step(L, th, s, p, e, ec, ed, dt):
    """Greedy action at one interval. Returns (net, new_soc)."""
    if L > th:
        need = L - th
        dmax = min(p, s * ed / dt, L)
        d = need if need < dmax else dmax
        s -= d * dt / ed
        net = L - d
    else:
        room = th - L
        if room > p:
            room = p
        cmax = (e - s) / (ec * dt)
        c = room if room < cmax else cmax
        if c < 0.0:
            c = 0.0
        s += c * ec * dt
        net = L + c
    if s < 0.0:
        s = 0.0
    if s > e:
        s = e
    return net, s


@njit(cache=True)
def _step2(L, th_dis, th_chg, s, p, e, ec, ed, dt):
    """Like _step, but recharging may not push net load above th_chg (<= th_dis)."""
    if L > th_dis:
        return _step(L, th_dis, s, p, e, ec, ed, dt)
    if L >= th_chg:
        return L, s
    return _step(L, th_chg, s, p, e, ec, ed, dt)


@njit(cache=True)
def _feasible_path(f, w, T, s, p, e, ec, ed, dt):
    for h in range(f.shape[0]):
        L = f[h]
        th = 1e18 if w[h] >= 1e17 else T * w[h]
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
        if s > e:
            s = e
    return True


@njit(cache=True)
def _min_T_path(f, w, s, p, e, ec, ed, dt):
    """Smallest billing demand T holdable over forecast path f (net_h <= T * w_h)."""
    hi = 0.0
    lo = 0.0
    for h in range(f.shape[0]):
        if w[h] < 1e17:
            v = f[h] / w[h]
            if v > hi:
                hi = v
            u = (f[h] - p) / w[h]
            if u > lo:
                lo = u
    if _feasible_path(f, w, lo, s, p, e, ec, ed, dt):
        return lo
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _feasible_path(f, w, mid, s, p, e, ec, ed, dt):
            hi = mid
        else:
            lo = mid
    return hi


@njit(cache=True)
def run_mpc(load, fc, month, w, p, e, ec, ed, s0, dt, margin, alpha, k_obs, decay):
    """Forecast-driven receding-horizon control.

    load  (T,) actual kW;  fc (T,) day-ahead forecast path (kW) for the same intervals
    month (T,) month id;   w (T,) billing-demand weights (1 = counts, 1e18 = ignored)
    margin: kW added to the planned billing demand; alpha/k_obs/decay: intraday correction.
    Returns net (T,), soc (T,).
    """
    n = load.shape[0]
    net = np.empty(n)
    soc = np.empty(n)
    s = s0
    mtd = 0.0  # billing demand already set this month (sunk)
    buf = np.empty(SLOTS)
    wbuf = np.empty(SLOTS)
    for t in range(n):
        if t == 0 or month[t] != month[t - 1]:
            mtd = 0.0
        slot = t % SLOTS
        r = 1.0
        k = k_obs if slot >= k_obs else slot
        if k > 0:
            a = 0.0
            b = 0.0
            for j in range(t - k, t):
                a += load[j]
                b += fc[j]
            if b > 1e-9:
                r = a / b
                if r > 3.0:
                    r = 3.0
                if r < 0.33:
                    r = 0.33
        H = SLOTS - slot
        for h in range(H):
            if h == 0:
                buf[0] = load[t]  # the controller measures the current interval in real time
            else:
                buf[h] = fc[t + h] * (1.0 + alpha * (r - 1.0) * decay**h)
            wbuf[h] = w[t + h]
        if w[t] < 1e17:
            Th = _min_T_path(buf[:H], wbuf[:H], s, p, e, ec, ed, dt) + margin
            if Th < mtd:
                Th = mtd
            th = Th * w[t]
            # recharge only under the month's already-sunk peak: free, and never creates the bill
            nt, s = _step2(load[t], th, mtd * w[t], s, p, e, ec, ed, dt)
        else:
            nt, s = _step(load[t], 1e18, s, p, e, ec, ed, dt)
        net[t] = nt
        soc[t] = s
        if w[t] < 1e17 and nt / w[t] > mtd:
            mtd = nt / w[t]
    return net, soc


@njit(cache=True)
def run_ratchet(load, month, w, start_T, p, e, ec, ed, s0, dt):
    """Billing-demand target starts at start_T[t at month start] and only rises to the realised value."""
    n = load.shape[0]
    net = np.empty(n)
    soc = np.empty(n)
    s = s0
    th = 0.0
    for t in range(n):
        if t == 0 or month[t] != month[t - 1]:
            th = start_T[t]
        if w[t] < 1e17:
            nt, s = _step(load[t], th * w[t], s, p, e, ec, ed, dt)
            if nt / w[t] > th:
                th = nt / w[t]
        else:
            nt, s = _step(load[t], 1e18, s, p, e, ec, ed, dt)
        net[t] = nt
        soc[t] = s
    return net, soc


@njit(cache=True)
def run_timer(load, slot_of_day, weekday, start_slot, n_slots, charge_start, charge_slots, p, e, ec, ed, s0, dt):
    """Discharge flat-out across a fixed weekday window; recharge at a gentle rate overnight."""
    n = load.shape[0]
    net = np.empty(n)
    soc = np.empty(n)
    s = s0
    dis_rate = min(p, e * ed / (n_slots * dt))
    chg_rate = min(p, e / (ec * charge_slots * dt))
    for t in range(n):
        sl = slot_of_day[t]
        L = load[t]
        if weekday[t] and start_slot <= sl < start_slot + n_slots:
            d = min(dis_rate, s * ed / dt, L)
            s -= d * dt / ed
            nt = L - d
        elif charge_start <= sl < charge_start + charge_slots:
            c = min(chg_rate, (e - s) / (ec * dt))
            s += c * ec * dt
            nt = L + c
        else:
            nt = L
        if s < 0.0:
            s = 0.0
        if s > e:
            s = e
        net[t] = nt
        soc[t] = s
    return net, soc


def monthly_peaks(x: np.ndarray, month: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Billing demand per month: max(x_t / w_t) over counted intervals (w may be a bool mask)."""
    w = np.asarray(w)
    if w.dtype == bool:
        w = np.where(w, 1.0, 1e18)
    ids = np.unique(month)
    out = []
    for m in ids:
        sel = (month == m) & (w < 1e17)
        out.append(float((x[sel] / w[sel]).max()) if sel.any() else 0.0)
    return np.array(out)
