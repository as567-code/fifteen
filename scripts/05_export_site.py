"""Export every result the web page shows into site/data/. Every number on the page comes from here.

Run with:  .venv/bin/python -I scripts/05_export_site.py
"""

from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fifteen.battery import POWERBLOCK  # noqa: E402

RES = os.path.join(ROOT, "data", "results")
PROC = os.path.join(ROOT, "data", "processed")
SITE = os.path.join(ROOT, "site", "data")
os.makedirs(os.path.join(SITE, "sites"), exist_ok=True)
REPO = os.environ.get("FIFTEEN_REPO", "")

TYPE_NAMES = {
    "Warehouse": "Warehouses", "Grocery": "Grocery stores", "RetailStandalone": "Big-box retail",
    "RetailStripmall": "Strip-mall retail", "LargeOffice": "Large offices", "MediumOffice": "Small and medium offices",
    "SmallOffice": "Small and medium offices", "Hospital": "Hospitals", "Outpatient": "Outpatient clinics",
    "PrimarySchool": "Primary schools", "SecondarySchool": "Secondary schools", "LargeHotel": "Hotels",
    "SmallHotel": "Hotels", "FullServiceRestaurant": "Restaurants",
}
TARIFFS = ["ngrid-g3", "es-g2-boston"]
RATES = {"ngrid-g3": 10.61, "es-g2-boston": 33.95}


def usd(v):
    return f"${v:,.0f}"


def usdk(v):
    return f"${v / 1000:,.0f}k" if abs(v) >= 10000 else f"${v / 1000:,.1f}k"


def pct(v, d=0):
    return f"{100 * v:.{d}f}%"


def enc(arr, scale):
    q = np.clip(np.round(np.asarray(arr, float) / scale), 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode()


def eval_slice(kw, start, year):
    q = pd.Timedelta("15min")
    y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
    off = int((y0 - pd.Timestamp(start)) / q)
    return kw[off:off + int((y1 - y0) / q)].astype(float)


def main():
    meta = pd.read_csv(os.path.join(PROC, "library_sites.csv"))
    val = pd.read_csv(os.path.join(RES, "value_sites.csv"))
    mon = pd.read_csv(os.path.join(RES, "value_monthly.csv"))
    op = json.load(open(os.path.join(RES, "operate_report.json")))
    lib = np.load(os.path.join(PROC, "library.npz"))
    fcs = np.load(os.path.join(RES, "forecasts_2014.npz"))
    stack = pd.read_csv(os.path.join(RES, "stack_sites.csv")) if os.path.exists(os.path.join(RES, "stack_sites.csv")) else None

    # ---------------------------------------------------------------- library (sites a Powerblock fits)
    dup = []
    arrs = {}
    for r in meta.itertuples():
        arrs[r.site_id] = eval_slice(lib[r.site_id], r.start, r.year)
    seen = {}
    for sid, a in arrs.items():
        h = hash(a.tobytes())
        if h in seen:
            dup.append((seen[h], sid))
        else:
            seen[h] = sid
    drop = {b for _, b in dup}
    v = val[(val.peak_kw >= 250) & ~val.site_id.isin(drop)].copy()

    def group(r):
        if r.source.startswith("KIT"):
            return "Manufacturers, metered (Germany)"
        if r.source.startswith("UCI"):
            return "C&I customers, metered (Portugal)"
        return TYPE_NAMES.get(r.btype, r.btype)

    v["group"] = v.apply(group, axis=1)

    def label(r):
        if r.source.startswith("KIT"):
            return f"German manufacturer {r.site_id.split('-')[1]} ({r.year}, metered, {r.peak_kw:,.0f} kW peak)"
        if r.source.startswith("UCI"):
            return f"Portuguese C&I customer {r.site_id.split('-')[1]} (2014, metered, {r.peak_kw:,.0f} kW peak)"
        county = r.label.split("·")[-1].strip()
        return f"{TYPE_NAMES.get(r.btype, r.btype)[:-1] if TYPE_NAMES.get(r.btype, '').endswith('s') else TYPE_NAMES.get(r.btype, r.btype)}, {county} (2018, simulated, {r.peak_kw:,.0f} kW peak)"

    v["nice"] = v.apply(label, axis=1)

    def top_slice(x, year):
        """Per month: the most energy any single day holds above (monthly peak - 250 kW). Median over months.
        A battery that recharges overnight can take the full 250 kW off only if this fits in the battery."""
        days = x[: x.size // 96 * 96].reshape(-1, 96)
        d0 = pd.Timestamp(f"{year}-01-01")
        out = []
        for m in range(12):
            idx = [i for i in range(days.shape[0]) if (d0 + pd.Timedelta(days=i)).month == m + 1]
            blk = days[idx]
            line = blk.max() - POWERBLOCK.power_kw
            out.append(np.clip(blk - line, 0, None).sum(1).max() * 0.25)
        return float(np.median(out))

    v["slice"] = [top_slice(arrs[sid], yr) for sid, yr in zip(v.site_id, v.year)]
    monthly = mon.set_index(["site_id", "month"])
    rows, fields = [], ["id", "label", "group", "source", "utility", "peak", "lf", "mpk", "mkwh", "slice"]
    for t in TARIFFS:
        fields += [f"{t}:save", f"{t}:months", f"{t}:kwcut"]
    for _, r in v.iterrows():
        m = monthly.loc[r.site_id]
        row = [r.site_id, r.nice, r.group, r.source.split(" ")[0], r.utility if isinstance(r.utility, str) else "",
               round(r.peak_kw, 1), round(r.load_factor, 4),
               [round(x, 1) for x in m.peak_kw], [round(x) for x in m.kwh], round(r.slice, 1)]
        for t in TARIFFS:
            row += [round(r[f"{t}:savings"]), round(r[f"{t}:months_off"], 3), round(r[f"{t}:demand_savings"] / RATES[t] / 12, 1)]
        rows.append(row)
    json.dump({"fields": fields, "rows": rows}, open(os.path.join(SITE, "library.json"), "w"), separators=(",", ":"))

    # ---------------------------------------------------------------- featured sites with full 15-minute data
    feat = []
    uci = v[v.source.str.startswith("UCI")].sort_values("peak_kw")
    miss = op["miss_example"]
    test_ok = pd.read_csv(os.path.join(RES, "operate_per_site_2014.csv")).set_index("site_id")
    cap = (test_ok["es-g2-boston:mpc"] / test_ok["es-g2-boston:oracle"]).rename("cap")
    hero_pool = uci[(uci.peak_kw.between(450, 1200))].join(cap, on="site_id")
    hero_pool = hero_pool[hero_pool.cap.between(0.85, 1.0)].sort_values("es-g2-boston:savings", ascending=False)
    hero_id = hero_pool.site_id.iloc[0]
    pick_uci = list(dict.fromkeys([hero_id, miss["site_id"]] + list(uci.site_id.iloc[np.linspace(0, len(uci) - 1, 18).astype(int)])))
    pick_kit = list(v[v.source.str.startswith("KIT")].sort_values("peak_kw").site_id)
    pick_cs = []
    for g, d in v[v.source.str.startswith("NREL")].groupby("group"):
        d = d.sort_values("es-g2-boston:savings")
        pick_cs += list(d.site_id.iloc[[int(len(d) * 0.25), int(len(d) * 0.5), int(len(d) * 0.75)]].unique())
    groups = {"UCI": "Metered C&I customers, Portugal (2014)", "KIT": "Metered manufacturers, Germany (2016)", "NREL": "Simulated Massachusetts buildings (2018)"}
    for sid in pick_uci + pick_kit + pick_cs:
        r = v[v.site_id == sid].iloc[0]
        load = arrs[sid]
        fc = fcs[sid].astype(float) if sid in fcs.files else None
        scale = max(load.max(), fc.max() if fc is not None else 0) / 65000
        src = r.source.split(" ")[0]
        note = {"UCI": "Real meter data from a Portuguese utility customer; the forecast controller uses the machine-learning forecast trained only on 2012-2013.",
                "KIT": "Real meter data from a German small/mid-size manufacturer (KIT dataset). The forecast controller here uses a simple same-weekday average.",
                "NREL": "NREL ComStock simulation of a real Massachusetts building type with actual 2018 weather. The forecast controller here uses a simple same-weekday average."}[src]
        out = {"id": sid, "label": r.nice, "year": int(r.year), "scale": scale, "load": enc(load, scale), "source": r.source, "btype": r.btype, "note": note}
        if fc is not None:
            out["fc"] = enc(fc[: load.size], scale)
        json.dump(out, open(os.path.join(SITE, "sites", f"{sid}.json"), "w"), separators=(",", ":"))
        feat.append({"id": sid, "label": r.nice, "group": groups[src]})

    # hero month: biggest controller cut on the hero site under Eversource
    hm = monthly.loc[hero_id]
    hero_month = int((hm["es-g2-boston:kw:base"] - hm["es-g2-boston:kw:net"]).idxmax())
    ok_months = [mo for mo in range(1, 13) if abs(hm.loc[mo, "es-g2-boston:kw:base"] - hm.loc[mo, "peak_kw"]) < 1e-6]
    if ok_months:
        hero_month = max(ok_months, key=lambda mo: hm.loc[mo, "es-g2-boston:kw:base"] - hm.loc[mo, "es-g2-boston:kw:net"])

    # ---------------------------------------------------------------- headline numbers
    cs = v[v.source.str.startswith("NREL")]
    med = {t: float(cs[f"{t}:savings"].median()) for t in TARIFFS}
    med_m = {t: float(cs[f"{t}:months_off"].median()) for t in TARIFFS}
    by_group = cs.groupby("group")["es-g2-boston:months_off"].median().sort_values(ascending=False)
    from scipy.stats import spearmanr
    cut_es = v["es-g2-boston:demand_savings"] / 33.95 / 12
    rho_lf = float(spearmanr(v.load_factor, cut_es).statistic)
    rho_slice = float(spearmanr(v.slice, cut_es).statistic)
    knee = POWERBLOCK.energy_kwh * POWERBLOCK.eta_discharge
    below = v.slice <= knee
    cut_below, cut_above = float(cut_es[below].median()), float(cut_es[~below].median())
    share_below = float(below.mean())
    by_src = v.groupby(v.source.str.split(" ").str[0])["es-g2-boston:savings"].median()
    metered = v[~v.source.str.startswith("NREL")]
    T = op["test_2014"]
    ah, es, ng = T["all-hours"], T["es-g2-boston"], T["ngrid-g3"]
    fq = op["forecast_quality_2014"]
    p = op["params"]
    kwm = op["test_2014_kw_months"]["es-g2-boston"]
    usd_gap = (kwm["mpc"] - kwm["ratchet"]) * 33.95 / op["n_sites"]

    # ---------------------------------------------------------------- screener: leave-one-out kNN accuracy
    hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

    def feats(pk, kwh):
        pk, kwh = np.asarray(pk, float), np.asarray(kwh, float)
        lf = kwh / (np.maximum(pk, 1e-6) * np.array(hours))
        mp = pk.mean()
        return np.array([np.log(mp), lf.mean(), lf.std(), pk.max() / mp, pk.min() / mp])

    Fm = np.array([feats(r[7], r[8]) for r in rows])
    mu, sd = Fm.mean(0), Fm.std(0)
    Z = (Fm - mu) / sd
    K = 15
    screen = {}
    for t in TARIFFS:
        y = np.array([r[fields.index(f"{t}:kwcut")] for r in rows])
        D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, 1)[:, :K]
        pred = np.median(y[nn], 1)
        lo, hi = np.quantile(y[nn], 0.1, axis=1), np.quantile(y[nn], 0.9, axis=1)
        err = np.abs(pred - y) / np.maximum(y, 1)
        base_err = np.abs(np.median(y) - y) / np.maximum(y, 1)
        met = np.array([r[3] != "NREL" for r in rows])
        screen[t] = {"median_abs_pct_err": float(np.median(err)), "within_25pct": float((err <= 0.25).mean()),
                     "metered_median_abs_pct_err": float(np.median(err[met])), "metered_baseline": float(np.median(base_err[met])),
                     "baseline_median_abs_pct_err": float(np.median(base_err)), "interval_coverage": float(((y >= lo) & (y <= hi)).mean())}
    ex = v[v.site_id == "kit16-01"]
    ex_m = monthly.loc["kit16-01"] if len(ex) else monthly.loc[cs.site_id.iloc[0]]
    example = [[round(a), round(b)] for a, b in zip(ex_m.peak_kw, ex_m.kwh)]

    # ---------------------------------------------------------------- stacking
    stack_block = None
    if stack is not None:
        st = stack.merge(val[["site_id", "ngrid-g3:demand_savings", "es-g2-boston:demand_savings"]], on="site_id")
        st = st[st.site_id.isin(cs.site_id)]
        srows, totals = [], {}
        for t in TARIFFS:
            d = st[st.tariff == t]
            alone = d[f"{t}:demand_savings"].mean()
            lost = (d.summer_demand_savings_alone - d.summer_demand_savings_stacked).mean()
            csr = d.cs_revenue.mean()
            name = "National Grid G-3" if t == "ngrid-g3" else "Eversource G-2, Boston"
            srows.append({"label": f"{name}: demand only", "total": alone, "segs": [{"v": alone, "kind": "demand"}]})
            srows.append({"label": f"{name}: demand + events", "total": alone - lost + csr,
                          "segs": [{"v": alone - lost, "kind": "demand"}, {"v": csr, "kind": "cs"}]})
            totals[t] = {"alone": alone, "lost": lost, "cs": csr, "stacked": alone - lost + csr,
                         "cs_kw": float(d.cs_avg_kw.mean()), "lost_share": float(lost / alone) if alone else 0.0,
                         "share_sites_lose_over_10pct": float(((d.summer_demand_savings_alone - d.summer_demand_savings_stacked) > 0.1 * d[f"{t}:demand_savings"]).mean())}
        events = json.load(open(os.path.join(RES, "cs_events_2018.json")))
        stack_block = {
            "title": "Average Massachusetts building: one Powerblock, with and without ConnectedSolutions",
            "sub": f"{len(cs)} ComStock buildings, 2018 weather. {len(events)} Daily Dispatch events placed on ISO-NE's {len(events)} highest-load summer days of 2018; $200 per kW of average event performance. Perfect foresight of the building's load.",
            "xmax": float(max(r["total"] for r in srows) * 1.15), "rows": srows, "totals": totals,
        }

    # ---------------------------------------------------------------- prose
    ng_t, es_t = stack_block["totals"]["ngrid-g3"], stack_block["totals"]["es-g2-boston"] if stack_block else (None, None)
    find_kicker = (f"Same battery, same building, very different money: the median Massachusetts building in the library saves "
                   f"{usd(med['es-g2-boston'])} a year on Eversource's Greater Boston rate and {usd(med['ngrid-g3'])} on National Grid's G-3. "
                   f"Which utility a prospect pays matters more than what they make.")
    find_prose = (f"<p>The two tariffs price the same kilowatt very differently. Eversource G-2 charges $33.95 per kW of billed demand because it folds transmission into the demand charge. "
                  f"National Grid G-3 charges $10.61 and bills transmission per kWh, where a battery cannot touch it. A Powerblock that cuts {round(float((cs['es-g2-boston:demand_savings'] / 33.95 / 12).median()))} kW off a typical building's monthly peak "
                  f"is therefore worth {med_m['es-g2-boston']:.1f} months of the bill in Eversource territory and {med_m['ngrid-g3']:.1f} in National Grid's.</p>"
                  f"<p>Within a territory, the shape of the load decides the rest, and not in the way the usual metric suggests. Load factor, the number most screening starts with, barely predicts a Powerblock's value (rank correlation {rho_lf:.2f} with kW cut). "
                  f"What does predict it is how much energy sits in the top 250 kW of the month's load curve on the hardest day (rank correlation {rho_slice:.2f}). A full battery can deliver about {knee:.0f} kWh. "
                  f"Below that line a site's typical cut is {cut_below:.0f} kW of the 250; above it, {cut_above:.0f} kW. That top slice is invisible on a bill and obvious in 15-minute data.</p>"
                  f"<p>The metered sites are worth the most: a median {usdk(by_src['KIT'])} a year for the German manufacturers and {usdk(by_src['UCI'])} for the Portuguese C&amp;I customers, against {usdk(by_src['NREL'])} for simulated Massachusetts buildings. Real factories are spikier than simulations, which is good news for a company that targets manufacturers.</p>")
    find_prose2 = ""
    if stack_block:
        find_prose2 = (f"<h3>The other half of the money: grid events</h3><p>Powertown's own FAQ says members get demand-response revenue without curtailing their operations. "
                       f"Mass Save's ConnectedSolutions Daily Dispatch pays $200 per kW of average battery discharge across the summer's events, usually 30 to 60 events of two or three hours between 3 and 8 pm. "
                       f"A 2-hour, 470 kWh battery can average about {ng_t['cs_kw']:.0f} kW across a realistic event mix, worth roughly {usd(ng_t['cs'])} a summer. In National Grid territory that is "
                       f"{ng_t['cs'] / max(ng_t['alone'], 1):.1f}× the member's demand savings. In that territory the battery's main job is the grid; the member's peak is the side dish.</p>"
                       f"<p>The two streams compete for the same 470 kWh, so I co-optimized them month by month with a linear program. Stacking pays: the average building's total rises to {usd(es_t['stacked'])} a year on the Eversource rate and {usd(ng_t['stacked'])} on National Grid's. "
                       f"The cost is the demand savings given up when an event drains the battery before the building's own peak, or when energy is held back for an event instead of a peak: {pct(es_t['lost_share'])} of demand savings under Eversource and {pct(ng_t['lost_share'])} under National Grid. "
                       f"It is cheap where the building peaks on hot afternoons, when one discharge earns both, and expensive where it does not. Secondary schools give up almost nothing; warehouses give up the most. "
                       f"{pct(es_t['share_sites_lose_over_10pct'])} of buildings lose more than a tenth of their Eversource demand savings. Those are the sites where event participation has to be decided day by day.</p>")
    find_prose3 = ("<h3>Screening before there is interval data</h3><p>Most prospects can hand over a year of bills long before anyone pulls 15-minute data from the utility. "
                   f"The screener below finds the {K} buildings in the library whose twelve bills look most like the prospect's (size, load factor, seasonality) and reports what a Powerblock did for them. "
                   f"Tested by leaving each site out in turn, its median error on the real, metered sites is {pct(screen['es-g2-boston']['metered_median_abs_pct_err'])} (Eversource) and {pct(screen['ngrid-g3']['metered_median_abs_pct_err'])} (National Grid), "
                   f"against {pct(screen['es-g2-boston']['metered_baseline'])} for guessing the library median. On simulated buildings it looks better ({pct(screen['es-g2-boston']['median_abs_pct_err'])}), because simulations have look-alikes; the metered number is the honest one. "
                   f"Good enough to decide which sites deserve a data request; not good enough to price a contract, because bills cannot see the top slice.</p>")
    op_kicker = (f"Perfect foresight is a ceiling, not a plan. On {op['n_sites']} real metered customers and a year none of the models had seen, a forecast-driven controller kept "
                 f"{pct(es['mpc'])} of the ceiling's value. A tuned rule-of-thumb kept {pct(es['ratchet'])}, and a fixed daily schedule {pct(es['timer'])}.")
    op_prose = (f"<p>The demand charge is set by the single worst interval of the month, so the controller's real job is to know, every fifteen minutes, whether today is the day that matters. "
                f"Discharging on an ordinary afternoon wastes energy that a harder afternoon later in the month will need. Holding back on the hard day lets the peak through, and Powertown's own FAQ says that miss is Powertown's to absorb.</p>"
                f"<p>The forecast controller re-plans every 15 minutes. A gradient-boosted quantile model, trained across all sites on scale-free load, issues a day-ahead forecast of the building's load; "
                f"tuning on 2013 chose its {'median' if p['q'] == 0.5 else str(int(p['q'] * 100)) + 'th percentile'} plus a {p['margin']:g} kW safety margin. "
                f"The controller corrects that forecast with what the meter has shown so far today, then computes the lowest billed demand the battery can still defend for the rest of the day. "
                f"It never defends below the month's already-sunk peak: once the month has set a peak of X kW, shaving below X is worth exactly nothing and only drains the battery before the day that counts. "
                f"For the same reason it never recharges above X, so it cannot create a peak of its own.</p>"
                f"<p>The protocol is the important part. The model was trained on 2012, every controller's knobs were tuned on 2013 (the baselines were tuned too), and 2014 was run once. "
                f"The knobs were tuned on a plain all-hours demand rule and only then scored on the two Massachusetts billing rules, so they are not fitted to the tariff they are judged on.</p>")
    op_prose2 = (f"<p>Two findings matter for Powertown. First, the forecast is worth real money: on the Eversource rule, the gap between the forecast controller and the tuned ratchet is about {usd(usd_gap)} per Powerblock per year. "
                 f"Second, a simple forecast gets most of the way. The same controller fed a same-weekday average instead of the ML model kept {pct(es['seasonal'])}. "
                 f"The ML forecast cut median-forecast error by {pct(1 - fq['pinball_p50'] / fq['pinball_p50_same_weekday_4wk'])} against that average, and its 95th-percentile band covered {pct(fq['coverage_p95'], 1)} of actual intervals. "
                 f"It is calibrated, but control quality rises more slowly than forecast quality. Better forecasts help most on the few days that set the month's peak, which is where the next model should focus.</p>")
    op_prose3 = ("<p>The worst failure mode is quiet. A controller that dumps its energy early on the month's hardest day looks fine for weeks, then produces one interval that costs a month of savings. "
                 "The two panels above show a real case. Use the Evaluate instrument to replay any site with any controller.</p>")
    rate_es = 33.95
    miss_txt = f"{miss['site_id'].replace('pt-', 'Customer ')}, {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][miss['month'] - 1]} 2014"
    summary = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "repo": REPO,
        "battery": {"power_kw": POWERBLOCK.power_kw, "nameplate_kwh": POWERBLOCK.nameplate_kwh, "usable_kwh": POWERBLOCK.energy_kwh, "round_trip": POWERBLOCK.round_trip},
        "hero": {"site_id": hero_id, "month": hero_month, "tariff": "es-g2-boston"},
        "evaluate_default": hero_id,
        "evaluate_kicker": (f"Under the two big Massachusetts rates, one Powerblock at a typical library building removes {med_m['ngrid-g3']:.1f} to {med_m['es-g2-boston']:.1f} months of the electric bill with perfect timing. "
                            f"Powertown advertises one to three. The data says that is right for the right sites in the right territory, and it is the controller that has to earn it."),
        "featured": feat,
        "find": {"kicker": find_kicker, "prose": find_prose, "prose2": find_prose2, "prose3": find_prose3, "default_tariff": "es-g2-boston",
                 "slice_title": "What actually predicts the value: the energy in the top 250 kW",
                 "slice_sub": f"Each dot is a site ({len(v)}). x: on the hardest day of a typical month, the energy above (monthly peak − 250 kW). y: billed kW the Powerblock removes per month on the Eversource rule, perfect foresight. {pct(share_below)} of sites sit left of the line.",
                 "slice_knee": knee, "slice_tariff": "es-g2-boston", "slice_rho": rho_slice, "lf_rho": rho_lf,
                 "strip_sub": f"Each dot is one building with a peak of at least 250 kW ({len(v)} sites). Perfect-foresight value of one 250 kW / 522 kWh Powerblock under the selected 2026 tariff. Click a dot with a full profile to open it above."},
        "stack": stack_block,
        "screener": {"k": K, "mu": mu.tolist(), "sigma": sd.tolist(), "hours": hours, "example": example, "accuracy": screen,
                     "sub": f"Twelve months of peak kW and kWh, the two numbers on every commercial bill. Leave-one-out median error on the metered sites: {pct(screen['es-g2-boston']['metered_median_abs_pct_err'])} (Eversource), {pct(screen['ngrid-g3']['metered_median_abs_pct_err'])} (National Grid)."},
        "operate": {
            "kicker": op_kicker, "prose": op_prose, "prose2": op_prose2, "prose3": op_prose3,
            "params": {"q": p["q"], "margin": p["margin"], "alpha": p["alpha"], "beta": p["beta"], "margin_seasonal": p["margin_seasonal"], "alpha_seasonal": p["alpha_seasonal"]},
            "chart_title": "Share of the perfect-foresight value each controller kept",
            "chart_sub": f"{op['n_sites']} metered customers, 2014 (never seen in training or tuning), Eversource G-2 Greater Boston billing rule. Bars are total billed-kW reduction across {op['site_months']:,} site-months, relative to the oracle.",
            "bars": [
                {"label": "Perfect foresight (ceiling)", "v": es["oracle"], "kind": "oracle"},
                {"label": "Forecast controller, ML forecast", "v": es["mpc"], "kind": "ours"},
                {"label": "Forecast controller, same-weekday average", "v": es["seasonal"], "kind": "base"},
                {"label": "Ratchet rule (tuned)", "v": es["ratchet"], "kind": "base"},
                {"label": "Fixed daily discharge window", "v": es["timer"], "kind": "base"},
            ],
            "by_tariff": T,
            "miss": {**miss, "tariff": "es-g2-boston",
                     "title": f"Anatomy of a miss: {miss_txt}",
                     "sub": f"Top: the tuned ratchet rule. Bottom: the forecast controller. Same building, same battery, same three days. The month's billed demand differs by {miss['loss_kw']:.0f} kW, about {usd(miss['loss_kw'] * rate_es)} on the Eversource rule."},
        },
        "audit": audit_items(op, dup, v, screen, stack_block),
        "next": next_html(stack_block),
    }
    json.dump(summary, open(os.path.join(SITE, "summary.json"), "w"), indent=1)
    print("rho lf", rho_lf, "rho slice", rho_slice, "below", share_below, cut_below, cut_above, by_src.to_dict())
    print("hero", hero_id, hero_month, "featured", len(feat), "library", len(rows), "dups", dup)
    print(json.dumps({k: summary["operate"]["bars"][i]["v"] for i, k in enumerate(["oracle", "mpc", "seasonal", "ratchet", "timer"])}, indent=0))


def audit_items(op, dup, v, screen, stack_block):
    mw = op["months_worse_than_no_battery"]["es-g2-boston"]
    items = [
        ("pass", "The fast optimizer agrees with an independent linear program",
         "Perfect-foresight dispatch uses a greedy rule plus bisection, which is fast enough to run in a browser. A separate LP (HiGHS) solves the same month from scratch. Tests on single-window, two-window and off-peak-discount billing rules agree to within 0.05 kW."),
        ("pass", "Every dispatch obeys the battery's physics",
         "On every one of the 2 × 995 perfect-foresight runs, the code checks that power never exceeds 250 kW, charge stays within 0 to 470 kWh, energy is conserved every 15 minutes with round-trip losses, and nothing is exported to the grid. Any violation stops the pipeline."),
        ("pass", "The browser engine reproduces the Python engine",
         "The JavaScript that values your uploaded data is a port of the Python engine. Tests run both on the same inputs and require identical controller traces (to 10⁻⁶ kW) and identical bills (to the cent) on both tariffs."),
        ("pass", "The forecaster cannot see the future",
         "A test scrambles every day from d onward and asserts that day d's features do not change. Training (2012), tuning (2013) and testing (2014) never overlap, and the tuning used a different billing rule from the one the headline is scored on."),
        ("pass", "An AI agent gathered the tariffs; a test checks its work",
         "A research agent pulled 31 official National Grid, Eversource and Mass Save documents. A test re-reads the hash-pinned PDFs and requires every modeled number and billing rule to appear verbatim in the document it is attributed to."),
        ("pass", "The forecast controller never bills more than no battery would",
         f"Across {op['site_months']:,} test site-months on the Eversource rule, months whose billed demand ended above the no-battery bill: forecast controller {mw['mpc']}, ratchet {mw['ratchet']}, fixed daily window {mw['timer']}. "
         "For the forecast controller this is now guaranteed by construction and checked by a property test that feeds it deliberately terrible forecasts."),
        ("found", "The first controller could make a month worse",
         "An earlier version recharged up to the threshold it had planned from the forecast. When the forecast ran high, that recharge became the month's new peak: 5 of 1,848 test months billed more than with no battery at all. "
         "The fix uses the same sunk-peak logic. Recharging below the month's already-set peak is free; above it, the recharge can create the bill, so the controller never does that."),
        ("found", "Daylight-saving artefacts in the Portuguese meter data",
         "On the March clock change the 1 am hour reads zero, and in October it holds two hours of energy. The October interval would show up as a fake monthly peak. Those intervals are interpolated and halved respectively, 24 per site."),
        ("found", "The simulated Massachusetts buildings were an hour off all summer",
         "NREL ComStock timestamps are standard time year-round, while tariff windows and ISO-NE hours follow the clock. Converting to Eastern clock time moved the median valuation by under 2%, and it decides which hour an event or a peak window lands on."),
        ("found", "Two of the German plants are the same meter",
         f"{', '.join(a + ' and ' + b for a, b in dup)} are byte-identical. One was dropped, because a duplicate would let the screener's leave-one-out test find its own twin. The 2017 KIT plants also peak at 8 to 170 kW, too small for a 250 kW battery, so they are excluded."),
        ("found", "The baseline's best setting sat on the edge of its search grid",
         f"In the first two runs the tuned ratchet rule kept choosing the most aggressive start it was offered. The grid was widened until the choice was interior or hit the natural floor of zero (final: start each month {op['params']['beta']:g} × 250 kW below last month's peak), so the comparison is not won by handicapping the baseline."),
        ("assume", "Usable energy and efficiency are not published",
         f"I assume 90% usable depth of discharge (470 of 522 kWh) and {POWERBLOCK.round_trip:.1%} AC round-trip efficiency (97.5% PCS each way times 95% DC). Both are parameters, and every result can be re-run."),
        ("assume", "Real load shapes, Massachusetts bills",
         "The metered sites are Portuguese and German. Their load shapes are real; the bills are what those shapes would pay under 2026 Massachusetts tariffs. It is a stand-in until the same pipeline runs on Powertown members' Green Button data."),
        ("assume", "Simulated buildings are smoother than real ones",
         "ComStock has no compressor starts or forklift chargers, so it likely understates spiky sites and overstates how forecastable a building is. That is why every control claim on this page uses metered data only."),
        ("assume", "Small tariff simplifications",
         "Holidays are left inside peak windows, kVA billing and power factor are ignored, and Eversource's 12-month customer-charge lookback uses the evaluation year. None of these is quantified here; each is a short change to the tariff JSON."),
    ]
    if stack_block:
        items.append(("assume", "ConnectedSolutions event days are reconstructed",
                      "2018 events are placed on the 42 highest-load ISO-NE summer days, with Eversource's real 2025 mix of 2- and 3-hour events inside the 3-8 pm window. Performance is metered at the battery with no export."))
    return [{"status": s, "what": w, "why": y} for s, w, y in items]


def next_html(stack_block):
    return (
        "<p>Four things, in the order I would want them to exist:</p>"
        "<ol>"
        "<li><strong>Instant, audited value on every prospect.</strong> Wire this engine into the site-prospecting flow, so a member's Green Button export or twelve bills produce a value range, the months-off-bill number, and the assumptions behind it before anyone visits the site. It should improve with every installed Powerblock's real savings.</li>"
        "<li><strong>A fleet controller that treats events and peaks as one problem.</strong> The stacking result says the conflict is small on average and large at specific sites on specific days. The controller should decide each event day per site, using ISO-NE's day-ahead forecast and each member's own risk of a new monthly peak.</li>"
        "<li><strong>The transmission coincident-peak option, priced honestly.</strong> National Grid G-3 lets a customer swap the 3.455¢/kWh transmission charge for $21.91/kW measured at ISO-NE's monthly peak hour. Under that option every kW removed at ISO-NE's peak hour is worth $21.91 a month, so 250 kW delivered at the right hour every month is worth up to about $65,000 a year. Doing it reliably is a forecasting problem, and it is the kind of rate switch a member would never attempt without a battery and a model.</li>"
        "<li><strong>A tariff watcher with an audit harness.</strong> National Grid has issued nine Massachusetts rate summaries so far in 2026 (M.D.P.U. 1-26-A through 1-26-I). An agent should read each DPU filing, propose the new tariff JSON, and only merge it once a check like the one in Audit shows every number in the source PDF, then re-price the fleet.</li>"
        "</ol>"
    )


if __name__ == "__main__":
    main()
