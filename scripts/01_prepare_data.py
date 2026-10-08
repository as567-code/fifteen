"""Turn raw downloads into one compact, regular 15-minute load library.

Sources (both public):
  * KIT "Load profile data of 50 industrial plants in Germany for one year" (Zenodo 3899018,
    CC-BY-4.0). Real metered 15-min average kW, 20 plants in 2016 + 30 different plants in 2017.
  * NREL ComStock 2025 release 3 (AMY2018), Massachusetts individual-building 15-min timeseries,
    buildings with annual peak 300 kW - 3 MW (the range where one Powerblock matters).
  * UCI "ElectricityLoadDiagrams20112014" (Trindade, 2015; CC-BY-4.0): real metered 15-min kW of
    370 customers of a Portuguese utility. We keep 2012-2014 for customers with a 2014 peak of
    300 kW - 5 MW and >95% non-zero data in 2013 and 2014. Three years per site lets the
    dispatch backtest train on 2012-2013 and test on an untouched 2014.

Output: data/processed/library.npz + data/processed/library_sites.csv
All timestamps are interval-START, naive local time.

Run with:  .venv/bin/python -I scripts/01_prepare_data.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


def regular_year(series: pd.Series, year: int) -> tuple[np.ndarray, dict]:
    """Reindex a period-start series onto the full 15-min grid of `year`; report repairs."""
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
    s = series[~series.index.duplicated(keep="first")].reindex(idx)
    missing = int(s.isna().sum())
    s = s.interpolate(limit=8, limit_direction="both")
    still = int(s.isna().sum())
    s = s.fillna(s.median())
    return s.to_numpy(np.float32), {"missing_intervals": missing, "filled_by_median": still}


def kit() -> list[dict]:
    sites = []
    for fname, year in (("LoadProfile_20IPs_2016.csv", 2016), ("LoadProfile_30IPs_2017.csv", 2017)):
        df = pd.read_csv(os.path.join(RAW, "kit_industrial", fname), sep=";", skiprows=1)
        ts = df.pop("Time stamp").astype(str)
        dst_repeat = ts.str.endswith(" b")  # repeated hour when German DST ends; drop duplicates
        df, ts = df[~dst_repeat], ts[~dst_repeat]
        end = pd.to_datetime(ts, format="%d.%m.%Y %H:%M:%S")
        df.index = end - pd.Timedelta(minutes=15)  # period-ending -> period-start
        for col in df.columns:
            kw, rep = regular_year(pd.to_numeric(df[col], errors="coerce"), year)
            sites.append({
                "site_id": f"kit{year % 100}-{col.replace('LG ', '').strip().zfill(2)}",
                "source": "KIT industrial (metered)",
                "label": f"German industrial plant {col.replace('LG ', '#')} ({year})",
                "btype": "Industrial (metered)",
                "utility": "",
                "year": year,
                "kw": kw,
                "repairs": rep | {"dst_rows_dropped": int(dst_repeat.sum())},
            })
    return sites


def uci() -> list[dict]:
    df = pd.read_parquet(os.path.join(RAW, "uci_ld", "ld.parquet"))
    df.index = df.index - pd.Timedelta(minutes=15)  # labels are period-ending
    # Documented DST artefacts: on the March change day the 4 intervals labelled 01:00-01:45
    # (period-end) are zero; on the October change day they hold two hours of energy.
    repairs = {"dst_march_interpolated": 0, "dst_october_halved": 0}
    for y in (2012, 2013, 2014):
        for month, kind in ((3, "march"), (10, "october")):
            last_sun = max(d for d in pd.date_range(f"{y}-{month:02d}-01", periods=31, freq="D") if d.month == month and d.weekday() == 6)
            ix = pd.date_range(last_sun + pd.Timedelta("00:45:00"), periods=4, freq="15min")
            if kind == "october":
                df.loc[ix] = df.loc[ix] / 2.0
                repairs["dst_october_halved"] += 4
            else:
                df.loc[ix] = np.nan
                repairs["dst_march_interpolated"] += 4
    df = df.interpolate(limit=8)
    sites = []
    d13, d14 = df.loc["2013"], df.loc["2014"]
    keep = [c for c in df.columns
            if 300 <= d14[c].max() <= 5000 and (d13[c] > 0).mean() > 0.95 and (d14[c] > 0).mean() > 0.95]
    for c in keep:
        s = df.loc["2012":"2014", c]
        idx = pd.date_range("2012-01-01", "2015-01-01", freq="15min", inclusive="left")
        s = s.reindex(idx)
        missing = int(s.isna().sum())
        s = s.interpolate(limit_direction="both")
        sites.append({
            "site_id": f"pt-{c.replace('MT_', '')}",
            "source": "UCI Portugal utility (metered)",
            "label": f"Portuguese utility customer {c}",
            "btype": "C&I customer (metered)",
            "utility": "",
            "year": 2014,
            "start": "2012-01-01",
            "kw": s.to_numpy(np.float32),
            "repairs": repairs | {"missing_after_reindex": missing},
        })
    return sites


def comstock() -> list[dict]:
    meta = pd.read_csv(os.path.join(OUT, "comstock_library_meta.csv")).set_index("bldg_id")
    sites = []
    files = sorted(glob.glob(os.path.join(RAW, "comstock_ts", "*-0.parquet")))
    for f in files:
        bid = int(os.path.basename(f).split("-")[0])
        if bid not in meta.index:
            continue
        df = pd.read_parquet(f, columns=["timestamp", "out.electricity.total.energy_consumption"])
        start = pd.to_datetime(df["timestamp"]) - pd.Timedelta(minutes=15)
        # ComStock timestamps are local STANDARD time all year. Tariff windows, ISO-NE hours and
        # ConnectedSolutions events are clock time, so convert EST -> America/New_York (EDT in summer).
        clock = start.dt.tz_localize("Etc/GMT+5").dt.tz_convert("America/New_York").dt.tz_localize(None)
        kw = pd.Series(df["out.electricity.total.energy_consumption"].to_numpy() * 4.0, index=clock)  # kWh/15min -> kW
        kw, rep = regular_year(kw, 2018)
        rep["timezone"] = "EST -> America/New_York clock time"
        m = meta.loc[bid]
        sites.append({
            "site_id": f"cs-{bid}",
            "source": "NREL ComStock MA (simulated, AMY2018)",
            "label": f"{m.btype} · {m.county.replace('MA, ', '')}",
            "btype": m.btype,
            "utility": m.utility,
            "year": 2018,
            "kw": kw,
            "repairs": rep | {"sqft": float(m.sqft), "weekday_hours": float(m.wk_hours)},
        })
    return sites


def main():
    sites = kit() + comstock() + uci()
    for st in sites:
        st.setdefault("start", f"{st['year']}-01-01")
    years = sorted({s["year"] for s in sites})
    # store each site as its own row; years differ in length (2016 is a leap year)
    arrays = {s["site_id"]: s["kw"] for s in sites}
    np.savez_compressed(os.path.join(OUT, "library.npz"), **arrays)
    rows = []
    q = pd.Timedelta("15min")
    for s in sites:
        kw = s["kw"].astype(float)
        # headline stats always describe the evaluation year
        y0, y1 = pd.Timestamp(f"{s['year']}-01-01"), pd.Timestamp(f"{s['year'] + 1}-01-01")
        off = int((y0 - pd.Timestamp(s["start"])) / q)
        ky = kw[off:off + int((y1 - y0) / q)]
        rows.append({
            "site_id": s["site_id"], "source": s["source"], "label": s["label"], "btype": s["btype"],
            "utility": s["utility"], "year": s["year"], "start": s["start"], "n": kw.size,
            "peak_kw": ky.max(), "mean_kw": ky.mean(), "annual_mwh": ky.sum() * 0.25 / 1000,
            "load_factor": ky.mean() / ky.max() if ky.max() > 0 else np.nan,
            "repairs": json.dumps(s["repairs"]),
        })
    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(OUT, "library_sites.csv"), index=False)
    print(meta.groupby("source").agg(n=("site_id", "size"), peak_med=("peak_kw", "median"), lf=("load_factor", "median")))
    print("years:", years)


if __name__ == "__main__":
    sys.exit(main())
