"""Evaluate: what is one Powerblock worth at each of the 995 library sites?

For every site and every modeled tariff: monthly baseline bill, the perfect-foresight
(oracle) dispatch, the shaved bill, and the savings. Oracle = the ceiling on value; the
operate backtest says how much of it a real controller keeps.

Also writes the per-site monthly features used by the bill-only screener.

Run with:  .venv/bin/python -I scripts/02_value_sites.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fifteen.audit import check_dispatch  # noqa: E402
from fifteen.battery import POWERBLOCK  # noqa: E402
from fifteen.shave import hold, plan_month  # noqa: E402
from fifteen.tariff import Calendar, demand_weights, load_tariffs, monthly_bills  # noqa: E402

OUT = os.path.join(ROOT, "data", "results")
os.makedirs(OUT, exist_ok=True)


def eval_year_slice(kw, start, year):
    q = pd.Timedelta("15min")
    y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
    off = int((y0 - pd.Timestamp(start)) / q)
    return kw[off:off + int((y1 - y0) / q)].astype(np.float64)


def oracle_dispatch(load, cal, tariff, bat):
    """Perfect-foresight dispatch month by month, chaining state of charge, audited."""
    W = np.vstack([demand_weights(d, cal) for d in tariff["demand"]])
    rates = np.array([d["usd_per_kw"] for d in tariff["demand"]])
    net = np.empty_like(load)
    soc = np.empty_like(load)
    s = s_start = bat.energy_kwh
    for _, a, b in cal.months():
        plan = plan_month(load[a:b], None, rates, bat, s0=s, weights=W[:, a:b])
        n_, so, worst, s = hold(load[a:b], plan.tau, bat, s0=s)
        if worst > 1e-6:
            raise AssertionError("oracle plan not held by greedy dispatch")
        net[a:b], soc[a:b] = n_, so
    problems = check_dispatch(load, net, soc, bat, s0=s_start)
    if problems:
        raise AssertionError(problems)
    return net


def main():
    t0 = time.time()
    tariffs = load_tariffs(os.path.join(ROOT, "site", "data", "tariffs.json"))
    meta = pd.read_csv(os.path.join(ROOT, "data", "processed", "library_sites.csv"))
    lib = np.load(os.path.join(ROOT, "data", "processed", "library.npz"))
    bat = POWERBLOCK
    rows, monthly = [], []
    for i, r in enumerate(meta.itertuples()):
        load = eval_year_slice(lib[r.site_id], r.start, r.year)
        cal = Calendar(pd.Timestamp(f"{r.year}-01-01"), load.size)
        site_month = pd.DataFrame({"month": range(1, 13)})
        site_month["peak_kw"] = [load[cal.month == m].max() for m in range(1, 13)]
        site_month["kwh"] = [load[cal.month == m].sum() * 0.25 for m in range(1, 13)]
        rec = {"site_id": r.site_id}
        for t in tariffs:
            net = oracle_dispatch(load, cal, t, bat)
            b0 = monthly_bills(load, t, cal)
            b1 = monthly_bills(net, t, cal)
            sav = b0.total_usd - b1.total_usd
            rec[f"{t['id']}:bill"] = b0.total_usd.sum()
            rec[f"{t['id']}:demand_bill"] = b0.demand_usd.sum()
            rec[f"{t['id']}:savings"] = sav.sum()
            rec[f"{t['id']}:demand_savings"] = (b0.demand_usd - b1.demand_usd).sum()
            rec[f"{t['id']}:loss_cost"] = (b1.energy_usd - b0.energy_usd).sum()
            rec[f"{t['id']}:months_off"] = sav.sum() / (b0.total_usd.sum() / 12)
            site_month[f"{t['id']}:bill"] = b0.total_usd.to_numpy()
            site_month[f"{t['id']}:savings"] = sav.to_numpy()
            site_month[f"{t['id']}:kw:base"] = b0["kw0"].to_numpy()
            site_month[f"{t['id']}:kw:net"] = b1["kw0"].to_numpy()
        site_month.insert(0, "site_id", r.site_id)
        monthly.append(site_month)
        rows.append(rec)
        if i % 100 == 0:
            print(f"{i}/{len(meta)} sites  {time.time() - t0:.0f}s", flush=True)
    res = meta.merge(pd.DataFrame(rows), on="site_id")
    res.to_csv(os.path.join(OUT, "value_sites.csv"), index=False)
    pd.concat(monthly).to_csv(os.path.join(OUT, "value_monthly.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
