"""The browser engine (site/engine.js) must reproduce the Python engine."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from fifteen.battery import POWERBLOCK
from fifteen.control import run_mpc, run_ratchet
from fifteen.shave import plan_month

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def run_js(snippet: str, payload: dict):
    code = (
        f"const F = require({json.dumps(os.path.join(ROOT, 'site', 'engine.js'))});"
        "const P = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
        f"{snippet}"
    )
    out = subprocess.run([NODE, "-e", code], input=json.dumps(payload), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def load_for(seed, days=10):
    rng = np.random.default_rng(seed)
    n = 96 * days
    tod = (np.arange(n) % 96) / 96
    x = 500 + 300 * np.clip(np.sin(np.pi * (tod - 0.3) / 0.45), 0, None) + rng.normal(0, 30, n)
    for _ in range(8):
        i = rng.integers(0, n - 10)
        x[i:i + rng.integers(1, 10)] += rng.uniform(80, 400)
    return np.clip(x, 0, None)


@pytest.mark.parametrize("seed", range(5))
def test_plan_month_parity(seed):
    x = load_for(seed)
    n = x.size
    on = ((np.arange(n) % 96) >= 32) & ((np.arange(n) % 96) < 84)
    for masks, rates in (([np.ones(n, bool)], [1.0]), ([np.ones(n, bool), on], [9.0, 15.0])):
        py = plan_month(x, np.vstack(masks), np.array(rates), POWERBLOCK, s0=POWERBLOCK.energy_kwh * 0.7)
        js = run_js(
            "const b = F.battery(1); const W = P.masks.map(a => Float64Array.from(a, v => v ? 1 : 1e18));"
            "console.log(JSON.stringify(F.planMonth(P.x, 0, P.x.length, W, P.rates, b.e * 0.7, b)));",
            {"x": x.tolist(), "masks": [m.astype(int).tolist() for m in masks], "rates": rates},
        )
        cost_py = float(np.dot(rates, py.thresholds))
        cost_js = float(np.dot(rates, js))
        assert cost_js == pytest.approx(cost_py, rel=1e-5, abs=1e-3)


@pytest.mark.parametrize("seed", range(3))
def test_controllers_parity(seed):
    x = load_for(seed + 7, days=20)
    n = x.size
    rng = np.random.default_rng(seed)
    fc = x * (1 + rng.normal(0, 0.08, n))
    month = (np.arange(n) // (96 * 10)).astype(np.int64)
    tod = np.arange(n) % 96
    w = np.where((tod >= 32) & (tod < 84), 1.0, 1 / 0.45)  # Eversource-style off-peak discount
    b = POWERBLOCK
    net_py, _ = run_mpc(x, fc, month, w, b.power_kw, b.energy_kwh, b.eta_charge, b.eta_discharge, b.energy_kwh, 0.25, 10.0, 0.5, 4, 0.9)
    start = np.array([x[:960].max() - 120.0, x[960:].max() - 120.0])
    net_r, _ = run_ratchet(x, month, w, np.repeat(start, 960), b.power_kw, b.energy_kwh, b.eta_charge, b.eta_discharge, b.energy_kwh, 0.25)
    js = run_js(
        "const b = F.battery(1); const w = Float64Array.from(P.w);"
        "const a = F.runMPC(P.x, P.fc, P.month, w, b, {margin: 10, alpha: 0.5, kObs: 4, decay: 0.9});"
        "const r = F.runRatchet(P.x, P.month, w, P.start, b);"
        "console.log(JSON.stringify({mpc: Array.from(a.net), rat: Array.from(r.net)}));",
        {"x": x.tolist(), "fc": fc.tolist(), "month": month.tolist(), "start": start.tolist(), "w": w.tolist()},
    )
    assert np.allclose(js["mpc"], net_py, atol=1e-6)
    assert np.allclose(js["rat"], net_r, atol=1e-6)


def test_bills_parity():
    """Same tariff JSON, same load -> same monthly bills in both engines."""
    import pandas as pd
    from fifteen.tariff import Calendar, load_tariffs, monthly_bills
    tariffs = load_tariffs(os.path.join(ROOT, "site", "data", "tariffs.json"))
    x = load_for(3, days=365)
    cal = Calendar(pd.Timestamp("2018-01-01"), x.size)
    for t in tariffs:
        py = monthly_bills(x, t, cal)
        js = run_js(
            "const cal = F.calendar(2018, P.x.length); F.prepareTariff(P.t, cal, P.x.length);"
            "console.log(JSON.stringify(F.bills(P.x, P.t, cal).map(o => o.total)));",
            {"x": x.tolist(), "t": t},
        )
        assert np.allclose(js, py.total_usd.to_numpy(), rtol=1e-9, atol=1e-6), t["id"]
