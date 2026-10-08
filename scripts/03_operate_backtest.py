"""Operate: how much of a Powerblock's perfect-foresight value can a real controller capture?

Test bed: the 154 real metered UCI customers (2012-2014). Protocol, no peeking:
  stage 1  train the forecaster on 2012, tune every policy's knobs on 2013
  stage 2  retrain the forecaster on 2012-2013, run the tuned policies once on 2014

Score = billed-demand reduction a policy achieves / the oracle's reduction, summed over all
site-months (kW-months; for a single-rate tariff this equals the share of demand dollars).
Tuning uses the tariff-agnostic "all hours" rule; the test reports all hours plus the two
real Massachusetts billing rules (National Grid G-3 peak window, Eversource G-2 off-peak
discount) so the knobs are not fitted to the tariff they are scored on.

Run with:  .venv/bin/python -I -u scripts/03_operate_backtest.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fifteen.battery import POWERBLOCK  # noqa: E402
from fifteen.control import monthly_peaks, run_mpc, run_ratchet, run_timer  # noqa: E402
from fifteen.forecast import (SLOTS, WARMUP_DAYS, QuantileForecaster, assert_no_lookahead,  # noqa: E402
                              day_features, pinball, seasonal_naive, to_days)
from fifteen.shave import hold, plan_month  # noqa: E402
from fifteen.tariff import Calendar, demand_weights, load_tariffs  # noqa: E402

OUT = os.path.join(ROOT, "data", "results")
os.makedirs(OUT, exist_ok=True)
BAT = POWERBLOCK
P, E, EC, ED, DT = BAT.power_kw, BAT.energy_kwh, BAT.eta_charge, BAT.eta_discharge, 0.25
COUNTRY = "PT"
ALL_HOURS = {"id": "all-hours", "demand": [{"usd_per_kw": 1.0, "peak": []}]}


def load_sites():
    meta = pd.read_csv(os.path.join(ROOT, "data", "processed", "library_sites.csv"))
    meta = meta[meta.source.str.startswith("UCI")].reset_index(drop=True)
    lib = np.load(os.path.join(ROOT, "data", "processed", "library.npz"))
    sites = []
    for r in meta.itertuples():
        X, dates = to_days(lib[r.site_id], r.start)
        sites.append({"id": r.site_id, "X": X, "dates": dates, "peak": r.peak_kw})
    return sites


def year_slice(dates, year):
    ix = np.flatnonzero(dates.year == year)
    return ix[0], ix[-1] + 1


def build_rows(sites, years, frac, seed):
    """Pooled, subsampled training rows (scale-free) from the given years."""
    rng = np.random.default_rng(seed)
    Fs, ys = [], []
    for st in sites:
        F, scale = day_features(st["X"], st["dates"], COUNTRY)
        days = np.arange(WARMUP_DAYS, st["X"].shape[0])
        keep_day = np.isin(st["dates"][days].year, years)
        y = (st["X"][days] / scale[days, None]).ravel()
        mask = np.repeat(keep_day, SLOTS) & (rng.random(y.size) < frac)
        Fs.append(F[mask].astype(np.float32))
        ys.append(y[mask].astype(np.float32))
    return np.vstack(Fs), np.concatenate(ys)


def forecast_year(model, st, year):
    """Day-ahead quantile paths (kW) for every interval of `year`, plus a simple benchmark."""
    F, scale = day_features(st["X"], st["dates"], COUNTRY)
    a, b = year_slice(st["dates"], year)
    rows = slice((a - WARMUP_DAYS) * SLOTS, (b - WARMUP_DAYS) * SLOTS)
    pred = model.predict(F[rows])
    out = {q: (v.reshape(b - a, SLOTS) * scale[a:b, None]).ravel() for q, v in pred.items()}
    # benchmark forecaster: same slot, same weekday, mean of the previous four weeks
    X = st["X"]
    out["seasonal"] = np.mean([X[a - 7 * k:b - 7 * k] for k in range(1, 5)], axis=0).ravel()
    return out


def oracle(load, month, w):
    """Perfect-foresight monthly billing demand, chaining state of charge across months."""
    s = E
    net = np.empty_like(load)
    for m in np.unique(month):
        ix = np.flatnonzero(month == m)
        plan = plan_month(load[ix], None, np.array([1.0]), BAT, s0=s, weights=w[ix][None, :])
        n_, _, _, s = hold(load[ix], plan.tau, BAT, s0=s)
        net[ix] = n_
    return net


def timer_window(X_hist, dates_hist):
    """2-hour weekday window that most often contains the daily peak in the training history."""
    wk = np.asarray(dates_hist.dayofweek < 5)
    peak_slot = X_hist[wk].argmax(1)
    counts = np.bincount(peak_slot, minlength=SLOTS)
    roll = np.array([counts[s:s + 8].sum() for s in range(SLOTS - 7)])
    return int(roll.argmax())


def evaluate(st, year, fc, params, tariff, keep_nets=False):
    a, b = year_slice(st["dates"], year)
    load = st["X"][a:b].ravel()
    dates = st["dates"][a:b]
    month = np.repeat(np.asarray(dates.month), SLOTS).astype(np.int64)
    w = demand_weights(tariff["demand"][0], Calendar(dates[0], load.size))
    res = {"base": monthly_peaks(load, month, w)}
    nets = {"oracle": oracle(load, month, w)}
    prev = st["X"][a - 31:a].ravel()
    w_prev = demand_weights(tariff["demand"][0], Calendar(st["dates"][a - 31], prev.size))
    prev_peak = np.concatenate([monthly_peaks(prev, np.zeros(prev.size, np.int64), w_prev), res["base"][:-1]])
    counts = np.bincount(month)[np.unique(month)]
    for beta in params["ratchet_betas"]:
        start_T = np.repeat(np.maximum(prev_peak - beta * P, 0.0), counts).astype(np.float64)
        nets[f"ratchet_{beta}"], _ = run_ratchet(load, month, w, start_T, P, E, EC, ED, E, DT)
    start = timer_window(st["X"][:a], st["dates"][:a])
    slot = np.tile(np.arange(SLOTS), b - a)
    wkday = np.repeat(np.asarray(dates.dayofweek < 5), SLOTS)
    nets["timer"], _ = run_timer(load, slot, wkday, start, 8, 0, 24, P, E, EC, ED, E, DT)
    for q, margin, alpha in params["mpc_grid"]:
        nets[f"mpc_{q}_{margin}_{alpha}"], _ = run_mpc(load, fc[q], month, w, P, E, EC, ED, E, DT, margin, alpha, 4, 0.9)
    for margin, alpha in params["seasonal_grid"]:
        nets[f"seasonal_{margin}_{alpha}"], _ = run_mpc(load, fc["seasonal"], month, w, P, E, EC, ED, E, DT, margin, alpha, 4, 0.9)
    for k, v in nets.items():
        res[k] = monthly_peaks(v, month, w)
    if keep_nets:
        res["_nets"], res["_month"], res["_w"] = nets, month, w
    return res


def score(results, key):
    num = sum((r["base"] - r[key]).sum() for r in results)
    den = sum((r["base"] - r["oracle"]).sum() for r in results)
    return float(num / den)


def find_miss(st, r, mpc_key, rat_key):
    """The month where the ratchet lost the most against the forecast controller, and the day it happened."""
    loss = r[rat_key] - r[mpc_key]
    m = int(np.argmax(loss))
    month, w = r["_month"], r["_w"]
    sel = np.flatnonzero(month == np.unique(month)[m])
    billed = r["_nets"][rat_key][sel] / np.where(w[sel] < 1e17, w[sel], np.inf)
    day = int(np.argmax(billed) // SLOTS) + 1
    return {"site_id": st["id"], "month": int(np.unique(month)[m]), "day": max(1, day - 1), "days": 3,
            "loss_kw": float(loss[m]), "peak_kw": float(st["peak"])}


def main():
    t0 = time.time()
    sites = load_sites()
    tariffs = load_tariffs(os.path.join(ROOT, "site", "data", "tariffs.json"))
    print(f"{len(sites)} metered sites")
    assert_no_lookahead(sites[0]["X"], sites[0]["dates"], COUNTRY, d=400)
    print("look-ahead check passed")

    params = {
        "ratchet_betas": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 100.0],
        "mpc_grid": list(itertools.product([0.5, 0.8, 0.95], [0.0, 10.0, 25.0], [0.0, 0.5, 1.0])),
        "seasonal_grid": list(itertools.product([0.0, 10.0, 25.0, 50.0], [0.5, 1.0])),
    }
    report = {"n_sites": len(sites), "battery": {"power_kw": P, "usable_kwh": E, "round_trip": BAT.round_trip}}

    # ---------- stage 1: train on 2012, tune on 2013 (all-hours rule)
    F, y = build_rows(sites, [2012], frac=0.25, seed=1)
    print("stage-1 rows", F.shape, f"{time.time() - t0:.0f}s", flush=True)
    m1 = QuantileForecaster().fit(F, y)
    val = []
    for i, st in enumerate(sites):
        val.append(evaluate(st, 2013, forecast_year(m1, st, 2013), params, ALL_HOURS))
        if i % 25 == 0:
            print(f"  validation {i}/{len(sites)} {time.time() - t0:.0f}s", flush=True)
    keys = [k for k in val[0] if k != "base"]
    val_scores = {k: score(val, k) for k in keys}
    best = {fam: max((k for k in keys if k.startswith(fam)), key=val_scores.get) for fam in ("ratchet", "mpc", "seasonal")}
    print("validation 2013 best:", {k: (v, round(val_scores[v], 3)) for k, v in best.items()}, "timer", round(val_scores["timer"], 3), flush=True)
    report["validation_2013"] = val_scores
    report["chosen"] = best

    # ---------- stage 2: retrain on 2012-2013, test on 2014 with the tuned knobs only
    F, y = build_rows(sites, [2012, 2013], frac=0.2, seed=2)
    print("stage-2 rows", F.shape, f"{time.time() - t0:.0f}s", flush=True)
    m2 = QuantileForecaster().fit(F, y)
    beta = float(best["ratchet"].split("_")[1])
    _, q, margin, alpha = best["mpc"].split("_")
    _, s_margin, s_alpha = best["seasonal"].split("_")
    test_params = {"ratchet_betas": [beta], "mpc_grid": [(float(q), float(margin), float(alpha))],
                   "seasonal_grid": [(float(s_margin), float(s_alpha))]}
    report["params"] = {"q": float(q), "margin": float(margin), "alpha": float(alpha), "beta": beta,
                        "margin_seasonal": float(s_margin), "alpha_seasonal": float(s_alpha)}
    keymap = {"oracle": "oracle", "mpc": f"mpc_{float(q)}_{float(margin)}_{float(alpha)}",
              "seasonal": f"seasonal_{float(s_margin)}_{float(s_alpha)}", "ratchet": f"ratchet_{beta}", "timer": "timer"}
    fq = {"pinball_p50": [], "pinball_p50_same_slot_last_week": [], "pinball_p50_same_weekday_4wk": [],
          "coverage_p80": [], "coverage_p95": []}
    test = {t: [] for t in ["all-hours"] + [x["id"] for x in tariffs]}
    per_site, forecasts = [], {}
    for i, st in enumerate(sites):
        fc = forecast_year(m2, st, 2014)
        forecasts[st["id"]] = np.asarray(fc[float(q)], np.float32)
        for t in [ALL_HOURS] + tariffs:
            test[t["id"]].append(evaluate(st, 2014, fc, test_params, t, keep_nets=(t["id"] == "es-g2-boston")))
        a, b = year_slice(st["dates"], 2014)
        act = st["X"][a:b].ravel()
        sc = act.mean()
        fq["pinball_p50"].append(pinball(act / sc, fc[0.5] / sc, 0.5))
        fq["pinball_p50_same_slot_last_week"].append(pinball(act / sc, seasonal_naive(st["X"])[a:b].ravel() / sc, 0.5))
        fq["pinball_p50_same_weekday_4wk"].append(pinball(act / sc, fc["seasonal"] / sc, 0.5))
        fq["coverage_p80"].append(float((act <= fc[0.8]).mean()))
        fq["coverage_p95"].append(float((act <= fc[0.95]).mean()))
        rows = {"site_id": st["id"], "peak_kw": float(st["peak"])}
        for tid in test:
            rr = test[tid][-1]
            rows.update({f"{tid}:{name}": float((rr["base"] - rr[k]).sum()) for name, k in keymap.items()})
        per_site.append(rows)
        if i % 25 == 0:
            print(f"  test {i}/{len(sites)} {time.time() - t0:.0f}s", flush=True)
    report["test_2014"] = {tid: {name: score(rs, k) for name, k in keymap.items()} for tid, rs in test.items()}
    report["test_2014_kw_months"] = {tid: {name: float(sum((r["base"] - r[k]).sum() for r in rs)) for name, k in keymap.items()} for tid, rs in test.items()}
    report["forecast_quality_2014"] = {k: float(np.mean(v)) for k, v in fq.items()}
    report["months_worse_than_no_battery"] = {tid: {name: int(sum(((r[k] - r["base"]) > 1e-6).sum() for r in rs)) for name, k in keymap.items()} for tid, rs in test.items()}
    report["site_months"] = int(sum(r["base"].size for r in test["all-hours"]))
    misses = [find_miss(st, r, keymap["mpc"], keymap["ratchet"]) for st, r in zip(sites, test["es-g2-boston"])]
    good = [m for m in misses if 400 <= m["peak_kw"] <= 1500]
    report["miss_example"] = max(good or misses, key=lambda m: m["loss_kw"])
    pd.DataFrame(per_site).to_csv(os.path.join(OUT, "operate_per_site_2014.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "forecasts_2014.npz"), **forecasts)
    with open(os.path.join(OUT, "operate_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("TEST 2014:", json.dumps(report["test_2014"], indent=1))
    print("forecast:", report["forecast_quality_2014"])
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
