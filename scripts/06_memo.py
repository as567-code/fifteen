"""Two-page PDF memo for the application's "share your best work" upload. Numbers come from
site/data/summary.json (written by 05_export_site.py); rendered with headless Chrome.

Run with:  .venv/bin/python -I scripts/06_memo.py https://<live-page-url>
"""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def pct(v, d=0):
    return f"{100 * v:.{d}f}%"


def usd(v):
    return f"${v:,.0f}"


def bars_svg(bars):
    w, rowh, left = 640, 26, 250
    h = rowh * len(bars) + 10
    out = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="Archivo, Helvetica, Arial" font-size="12">']
    for i, b in enumerate(bars):
        y = 5 + i * rowh
        bw = (w - left - 60) * max(0.0, b["v"])
        color = {"ours": "#1d5fd0", "oracle": "#c9d1ce", "base": "#9aa6ab"}[b["kind"]]
        out.append(f'<text x="{left - 10}" y="{y + 18}" text-anchor="end" fill="#1f2b33">{html.escape(b["label"])}</text>')
        out.append(f'<rect x="{left}" y="{y + 5}" width="{bw:.1f}" height="18" rx="2" fill="{color}"/>')
        out.append(f'<text x="{left + bw + 8:.1f}" y="{y + 18}" fill="#1f2b33" font-weight="600">{pct(b["v"], 1)}</text>')
    out.append("</svg>")
    return "".join(out)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "(link)"
    S = json.load(open(os.path.join(ROOT, "site", "data", "summary.json")))
    op = json.load(open(os.path.join(ROOT, "data", "results", "operate_report.json")))
    O, K = S["operate"], S["stack"]["totals"]
    es = O["by_tariff"]["es-g2-boston"]
    acc = S["screener"]["accuracy"]
    lib_n = len(json.load(open(os.path.join(ROOT, "site", "data", "library.json")))["rows"])
    med = {}
    import pandas as pd
    val = pd.read_csv(os.path.join(ROOT, "data", "results", "value_sites.csv"))
    cs = val[val.source.str.startswith("NREL")]
    for t in ("ngrid-g3", "es-g2-boston"):
        med[t] = (cs[f"{t}:savings"].median(), cs[f"{t}:months_off"].median())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fifteen Minutes: memo</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
@page {{ size: letter; margin: 0.5in 0.62in; }}
body {{ font-family: "Source Serif 4", Georgia, serif; color: #1f2b33; font-size: 10pt; line-height: 1.42; margin: 0; }}
h1 {{ font-family: Archivo, Helvetica, sans-serif; font-stretch: 72%; font-weight: 800; font-size: 30pt; line-height: .95; margin: 0 0 6pt; letter-spacing: -.01em; }}
h2 {{ font-family: Archivo, Helvetica, sans-serif; font-weight: 650; font-size: 12pt; margin: 11pt 0 4pt; }}
.keep {{ break-inside: avoid; }}
.ui, .by, td, th, .link {{ font-family: Archivo, Helvetica, sans-serif; }}
.by {{ color: #66747b; font-size: 9.5pt; margin-bottom: 10pt; }}
.link {{ display: block; margin: 10pt 0 12pt; padding: 9pt 12pt; border: 1.5pt solid #1f2b33; border-radius: 6pt; font-size: 12pt; font-weight: 650; color: #1f2b33; text-decoration: none; }}
.link small {{ display: block; font-weight: 400; color: #66747b; font-size: 9pt; margin-top: 2pt; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8pt 14pt; margin-top: 6pt; }}
.f {{ border-left: 3pt solid #1d5fd0; padding-left: 8pt; }}
.f.r {{ border-color: #cf1f2e; }}
.f b {{ font-family: Archivo, Helvetica, sans-serif; font-size: 14pt; display: block; line-height: 1.1; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.5pt; }}
td, th {{ text-align: left; padding: 3pt 6pt 3pt 0; border-bottom: .5pt solid #dde3e0; vertical-align: top; }}
.pb {{ page-break-before: always; }}
p {{ margin: 0 0 6pt; }}
ul {{ margin: 0 0 6pt 14pt; padding: 0; }} li {{ margin-bottom: 3pt; }}
.fig {{ margin: 6pt 0 2pt; }}
.cap {{ font-family: Archivo, Helvetica, sans-serif; font-size: 8.5pt; color: #66747b; }}
</style></head><body>
<h1>The most expensive fifteen minutes in a factory</h1>
<div class="by">Aditya Swaroop, for Powertown's Machine Learning Engineer, Intern (Term-Time) application. October 2026.</div>
<p>A Powerblock earns its keep in one 15-minute interval a month, plus a few dozen summer afternoons when the grid calls. I built an open instrument around that fact. It finds the Massachusetts buildings where a 250 kW / 522 kWh battery pays, prices each one under the real 2026 National Grid and Eversource tariffs, and runs the battery without seeing the future. Every result is reproducible and audited.</p>
<a class="link" href="{html.escape(url)}">{html.escape(url)}<small>Interactive page. Value your own Green Button data in the browser, replay any site with any controller, and read the audit trail.</small></a>
<h2>Four findings</h2>
<div class="grid">
<div class="f"><b>{pct(es['mpc'])} of the ceiling</b>A forecast-driven controller kept this share of the perfect-foresight value on {op['n_sites']} real metered customers in a year no model had seen. A tuned rule-of-thumb kept {pct(es['ratchet'])}; a fixed daily window kept {pct(es['timer'])}.</div>
<div class="f r"><b>{med['es-g2-boston'][1]:.1f} vs {med['ngrid-g3'][1]:.1f} months off the bill</b>Median library building under Eversource G-2 Boston vs National Grid G-3 ({usd(med['es-g2-boston'][0])} vs {usd(med['ngrid-g3'][0])} a year). National Grid bills transmission per kWh, which shaving cannot touch, so territory matters more than building type.</div>
<div class="f"><b>{usd(K['ngrid-g3']['stacked'])} vs {usd(K['ngrid-g3']['alone'])}</b>Average National Grid building with ConnectedSolutions Daily Dispatch stacked on demand savings, vs demand savings alone. Co-optimizing the two gives up only {pct(K['ngrid-g3']['lost_share'])} of demand savings ({pct(K['es-g2-boston']['lost_share'])} under Eversource).</div>
<div class="f"><b>Load factor predicts almost nothing</b>Across {lib_n} sites, load factor barely ranks a Powerblock's value. The energy in the top 250 kW of the hardest day's load does (rank correlation {S['find']['slice_rho']:.2f}). A bill cannot show it; 15-minute data can. A twelve-bill screener still triages prospects with {pct(acc['es-g2-boston']['metered_median_abs_pct_err'])} median error on metered sites.</div>
</div>
<div class="keep"><h2>Controller performance, 2014 test year (Eversource G-2 billing rule)</h2>
<div class="fig">{bars_svg(O['bars'])}</div>
<div class="cap">Share of the oracle's billed-kW reduction kept, summed over {op['site_months']:,} site-months. Model trained on 2012, every controller tuned on 2013 (baselines too), 2014 run once. The forecast controller never billed a month above the no-battery bill.</div></div>

<div class="pb"></div>
<h2>How it works</h2>
<ul>
<li><b>Exact perfect-foresight dispatch.</b> A greedy dispatch rule plus bisection finds the lowest achievable billed demand each month. It handles National Grid's peak window and Eversource's 55% off-peak discount, and it is fast enough to run in a browser. An independent HiGHS linear program confirms it to within 0.05 kW.</li>
<li><b>Probabilistic forecasting.</b> One gradient-boosted quantile model is trained across sites on scale-free load, so a new site with four weeks of data has a forecast on day one. Its 95% band covered {pct(op['forecast_quality_2014']['coverage_p95'], 1)} of actual 2014 intervals.</li>
<li><b>Sunk-peak control.</b> The controller re-plans every 15 minutes against today's forecast, corrected by the meter. It never defends below the month's already-set peak, since shaving below it is worth nothing and drains energy the hardest day will need, and it never recharges above it, so it cannot create a peak of its own.</li>
<li><b>Value stacking.</b> A month-by-month LP co-optimizes demand charges with $200/kW ConnectedSolutions Daily Dispatch. Events go on ISO-NE's 42 highest-load summer days of 2018.</li>
</ul>
<h2>Audit trail (use AI aggressively, then audit it ruthlessly)</h2>
<table>
<tr><th>Check</th><th>Result</th></tr>
<tr><td>Fast optimizer vs independent LP; battery physics on every run</td><td>Agree; zero violations across 1,990 annual runs</td></tr>
<tr><td>Browser engine vs Python engine</td><td>Identical traces (1e-6 kW) and bills (to the cent)</td></tr>
<tr><td>Forecaster look-ahead; train/tune/test separation</td><td>Perturbation test passes; 2012 / 2013 / 2014</td></tr>
<tr><td>AI-gathered tariffs vs the official PDFs (hash-pinned)</td><td>Every modeled number appears verbatim in its source</td></tr>
<tr><td>Data defects found</td><td>Daylight-saving spikes (UCI), standard-time clock (ComStock), a duplicated meter (KIT): fixed and documented</td></tr>
</table>
<h2>What I would build first at Powertown</h2>
<ul>
<li>Instant, audited value on every prospect, from Green Button data or twelve bills, inside the site-prospecting flow.</li>
<li>A fleet controller that decides ConnectedSolutions participation per site, per day, from ISO-NE forecasts and each member's peak risk.</li>
<li>Pricing National Grid's transmission coincident-peak option ($21.91/kW at ISO-NE's monthly peak hour), a rate switch that only makes sense with a battery and a forecast.</li>
<li>A tariff-watching agent whose output only merges when a check like the one above finds every number in the source PDF.</li>
</ul>
<h2>About me</h2>
<p>M.S. Computer Science, UT Dallas (Dec 2026). Previously: GPU and distributed systems at Princeton Research Computing, fault-tolerant production services, and evaluation harnesses that grade hundreds of submissions. I built this with AI agents in the loop and my own judgment on what to trust; code, data sources and tests are in the repository linked from the page.</p>
<p class="ui" style="font-size:9pt;color:#66747b">adityaswaroop23.rrps@gmail.com, github.com/as567-code. Independent project, not affiliated with Powertown; Powerblock parameters from powertownusa.com.</p>
</body></html>"""
    out_html = os.path.join(ROOT, "site", "memo.html")
    open(out_html, "w").write(page)
    os.makedirs(os.path.join(ROOT, "deliverables"), exist_ok=True)
    pdf = os.path.join(ROOT, "site", "memo.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "--virtual-time-budget=8000", "file://" + out_html], check=True, capture_output=True)
    shutil.copy(pdf, os.path.join(ROOT, "deliverables", "Aditya_Swaroop_Powertown_Fifteen_Minutes.pdf"))
    print("wrote", pdf)


if __name__ == "__main__":
    main()
