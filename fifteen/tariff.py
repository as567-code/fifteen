"""Tariffs and monthly bills.

A tariff is plain JSON (site/data/tariffs.json) shared by the Python and browser engines.
Each demand component has peak sub-windows and an `offpeak_factor`:
    billed kW = max(peak-window max, offpeak_factor x off-peak max)
(offpeak_factor 0 = off-peak demand is free; a component with no peak windows bills all hours).
Window weekdays use JavaScript numbering (0 = Sunday) so one spec drives both engines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

DT = 0.25
INF = 1e18


@dataclass
class Calendar:
    start: pd.Timestamp
    n: int

    def __post_init__(self):
        self.ts = pd.date_range(self.start, periods=self.n, freq="15min")
        self.month = np.asarray(self.ts.month)
        self.js_dow = (np.asarray(self.ts.dayofweek) + 1) % 7  # Mon=0 -> 1 ... Sun=6 -> 0
        self.hour = np.asarray(self.ts.hour) + np.asarray(self.ts.minute) / 60.0

    def months(self):
        out = []
        for m in np.unique(self.month):
            ix = np.flatnonzero(self.month == m)
            out.append((int(m), ix[0], ix[-1] + 1))
        return out


def window_mask(subwindows: list[dict], cal: Calendar) -> np.ndarray:
    m = np.zeros(cal.n, dtype=np.bool_)
    for w in subwindows:
        sel = np.isin(cal.js_dow, w["weekdays"]) & (cal.hour >= w["start"]) & (cal.hour < w["end"])
        if w.get("months"):
            sel &= np.isin(cal.month, w["months"])
        m |= sel
    return m


def demand_weights(dc: dict, cal: Calendar) -> np.ndarray:
    if not dc.get("peak"):
        return np.ones(cal.n)
    on = window_mask(dc["peak"], cal)
    f = dc.get("offpeak_factor", 0.0)
    return np.where(on, 1.0, (1.0 / f) if f > 0 else INF)


def load_tariffs(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["tariffs"]


def customer_charge(tariff: dict, billed_kw_year_max: float) -> float:
    for cap, usd in tariff.get("customer_charge_tiers", []):
        if billed_kw_year_max <= cap:
            return usd
    return tariff["customer_charge"]


def monthly_bills(x: np.ndarray, tariff: dict, cal: Calendar) -> pd.DataFrame:
    W = [demand_weights(d, cal) for d in tariff["demand"]]
    en = tariff["energy"]
    on = window_mask(en.get("peak", []), cal)
    sup = en.get("supply_cents", 0.0)
    rows = []
    for m, a, b in cal.months():
        r = {"month": m}
        dem = 0.0
        for k, (d, w) in enumerate(zip(tariff["demand"], W)):
            sel = w[a:b] < INF
            kw = float((x[a:b][sel] / w[a:b][sel]).max()) if sel.any() else 0.0
            r[f"kw{k}"] = kw
            dem += kw * d["usd_per_kw"]
        kwh_on = float(x[a:b][on[a:b]].sum() * DT)
        kwh_off = float(x[a:b][~on[a:b]].sum() * DT)
        r["demand_usd"] = dem
        r["kwh"] = kwh_on + kwh_off
        r["energy_usd"] = (kwh_on * (en["peak_cents"] + sup) + kwh_off * (en["offpeak_cents"] + sup)) / 100
        rows.append(r)
    df = pd.DataFrame(rows)
    df["customer"] = customer_charge(tariff, df["kw0"].max())
    df["total_usd"] = df.customer + df.demand_usd + df.energy_usd
    return df
