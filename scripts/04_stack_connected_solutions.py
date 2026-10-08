"""Stack: what happens when a Powerblock also earns ConnectedSolutions Daily Dispatch revenue?

Event days for summer 2018 are built from real ISO-NE data the way the program calls them:
the 42 highest-load days of June-September (Eversource's real 2025 calendar had 42 Daily
Dispatch events, 17 of them 3-hour and 25 of them 2-hour, weekends included). The 17 biggest
days get 3-hour events, the rest 2-hour, each placed on ISO-NE's peak hour and kept inside the
program's 3-8 pm window. Incentive: $200 per kW of average event performance per summer.

Buildings: the 791 ComStock Massachusetts buildings (same 2018 weather as the ISO-NE data).
Every summer month is solved as an LP that co-optimises the member's demand charge with event
performance (perfect foresight of load; events are announced day-ahead in reality).

Run with:  .venv/bin/python -I scripts/04_stack_connected_solutions.py
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

from fifteen.battery import POWERBLOCK  # noqa: E402
from fifteen.stack import lp_stack_month  # noqa: E402
from fifteen.tariff import Calendar, demand_weights, load_tariffs  # noqa: E402

CS_USD_PER_KW = 200.0
N_EVENTS, N_THREE_HOUR = 42, 17
OUT = os.path.join(ROOT, "data", "results")


def build_events():
    iso = pd.read_excel(os.path.join(ROOT, "data", "raw", "isone", "2018_smd_hourly.xlsx"), sheet_name="ISO NE CA")
    iso["hour_start"] = iso.Hr_End.astype(int) - 1
    iso["date"] = pd.to_datetime(iso.Date)
    summer = iso[iso.date.dt.month.isin([6, 7, 8, 9])]
    daily = summer.loc[summer.groupby("date").System_Load.idxmax(), ["date", "hour_start", "System_Load"]]
    daily = daily.sort_values("System_Load", ascending=False).head(N_EVENTS).reset_index(drop=True)
    events = []
    for i, r in daily.iterrows():
        dur = 3 if i < N_THREE_HOUR else 2
        start = int(np.clip(r.hour_start - 1, 15, 20 - dur))
        events.append({"date": r.date.strftime("%Y-%m-%d"), "start_hour": start, "hours": dur,
                       "iso_peak_mw": float(r.System_Load), "iso_peak_hour": int(r.hour_start)})
    return sorted(events, key=lambda e: e["date"])


def main():
    t0 = time.time()
    events = build_events()
    with open(os.path.join(OUT, "cs_events_2018.json"), "w") as f:
        json.dump(events, f, indent=1)
    print(f"{len(events)} events, {sum(e['hours'] for e in events)} event-hours")
    tariffs = load_tariffs(os.path.join(ROOT, "site", "data", "tariffs.json"))
    meta = pd.read_csv(os.path.join(ROOT, "data", "processed", "library_sites.csv"))
    meta = meta[meta.site_id.str.startswith("cs-")]
    lib = np.load(os.path.join(ROOT, "data", "processed", "library.npz"))
    monthly = pd.read_csv(os.path.join(OUT, "value_monthly.csv")).set_index(["site_id", "month"])
    cal = Calendar(pd.Timestamp("2018-01-01"), 35040)
    ev_idx = {}
    for e in events:
        d = pd.Timestamp(e["date"])
        i0 = int((d - cal.start) / pd.Timedelta("15min")) + e["start_hour"] * 4
        ev_idx.setdefault(d.month, []).append(np.arange(i0, i0 + e["hours"] * 4))
    W = {t["id"]: demand_weights(t["demand"][0], cal) for t in tariffs}
    rows = []
    bat = POWERBLOCK
    v = CS_USD_PER_KW / len(events)
    for k, r in enumerate(meta.itertuples()):
        load = lib[r.site_id].astype(float)
        for t in tariffs:
            rate = t["demand"][0]["usd_per_kw"]
            s = bat.energy_kwh
            perf, dem_stack, dem_alone = [], 0.0, 0.0
            for m, a, b in cal.months():
                if m not in (6, 7, 8, 9):
                    continue
                evs = [ix - a for ix in ev_idx.get(m, [])]
                res = lp_stack_month(load[a:b], W[t["id"]][a:b], rate, evs, v, bat, s0=s)
                s = res["soc_end"]
                perf += res["event_kw"]
                base_kw = monthly.loc[(r.site_id, m), f"{t['id']}:kw:base"]
                alone_kw = monthly.loc[(r.site_id, m), f"{t['id']}:kw:net"]
                dem_stack += rate * (base_kw - res["billed_kw"])
                dem_alone += rate * (base_kw - alone_kw)
            rows.append({"site_id": r.site_id, "tariff": t["id"], "cs_avg_kw": float(np.mean(perf)),
                         "cs_revenue": CS_USD_PER_KW * float(np.mean(perf)),
                         "summer_demand_savings_stacked": dem_stack, "summer_demand_savings_alone": dem_alone})
        if k % 50 == 0:
            print(f"{k}/{len(meta)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stack_sites.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
