// Fifteen — browser engine. A line-for-line port of fifteen/shave.py and fifteen/control.py so a
// visitor can value their own interval data without a server. tests/test_engine_parity.py runs
// this file under node and checks it against the Python engine.
(function (root) {
  "use strict";
  const DT = 0.25, SLOTS = 96, INF = 1e18;

  function battery(n = 1, o = {}) {
    const power = 250 * n, nameplate = 522 * n;
    const usable = o.usable ?? 0.9, pcs = o.pcs ?? 0.975, dc = o.dc ?? 0.95;
    const rte = pcs * pcs * dc, eta = Math.sqrt(rte);
    return { p: power, e: nameplate * usable, ec: eta, ed: eta, rte, nameplate };
  }

  // greedy step: defend threshold th; returns [net, soc]
  function step(L, th, s, b) {
    let net;
    if (L > th) {
      const need = L - th;
      const dmax = Math.min(b.p, (s * b.ed) / DT, L);
      const d = need < dmax ? need : dmax;
      s -= (d * DT) / b.ed;
      net = L - d;
    } else {
      let room = th - L; if (room > b.p) room = b.p;
      const cmax = (b.e - s) / (b.ec * DT);
      let c = room < cmax ? room : cmax; if (c < 0) c = 0;
      s += c * b.ec * DT;
      net = L + c;
    }
    if (s < 0) s = 0; if (s > b.e) s = b.e;
    return [net, s];
  }

  const thAt = (w, T, i) => (w[i] >= INF ? INF : T * w[i]);

  // can net_i <= min(T*w_i, To*wo_i) be held for every i in [i0, i1)?
  function feasible(load, i0, i1, w, T, wo, To, s, b) {
    for (let i = i0; i < i1; i++) {
      const L = load[i];
      let th = thAt(w, T, i);
      if (wo) { const t2 = thAt(wo, To, i); if (t2 < th) th = t2; }
      if (L > th) {
        const need = L - th, dmax = Math.min(b.p, (s * b.ed) / DT, L);
        if (need > dmax + 1e-9) return false;
        s -= (need * DT) / b.ed;
      } else {
        let room = th - L; if (room > b.p) room = b.p;
        const cmax = (b.e - s) / (b.ec * DT);
        const c = room < cmax ? room : cmax;
        if (c > 0) s += c * b.ec * DT;
      }
      if (s < 0) s = 0; if (s > b.e) s = b.e;
    }
    return true;
  }

  // smallest billing demand T with net_i <= T*w_i on [i0, i1) (optionally also <= To*wo_i)
  function minThreshold(load, i0, i1, w, s0, b, wo, To) {
    let hi = 0, lo = 0;
    for (let i = i0; i < i1; i++) if (w[i] < INF) {
      const v = load[i] / w[i]; if (v > hi) hi = v;
      const u = (load[i] - b.p) / w[i]; if (u > lo) lo = u;
    }
    if (hi === 0) return 0;
    if (!feasible(load, i0, i1, w, hi, wo, To, s0, b)) return INF;
    if (feasible(load, i0, i1, w, lo, wo, To, s0, b)) return lo;
    for (let k = 0; k < 50; k++) {
      const mid = 0.5 * (lo + hi);
      if (feasible(load, i0, i1, w, mid, wo, To, s0, b)) hi = mid; else lo = mid;
    }
    return hi;
  }

  // Perfect-foresight plan for one month with 1 or 2 demand components (weights W[k]).
  function planMonth(load, i0, i1, W, rates, s0, b) {
    if (W.length === 1) return [minThreshold(load, i0, i1, W[0], s0, b)];
    const inner = (T0) => minThreshold(load, i0, i1, W[1], s0, b, W[0], T0);
    let a = minThreshold(load, i0, i1, W[0], s0, b), c2 = 0;
    for (let i = i0; i < i1; i++) if (W[0][i] < INF) c2 = Math.max(c2, load[i] / W[0][i]);
    const f = (T0) => rates[0] * T0 + rates[1] * inner(T0);
    const gr = (Math.sqrt(5) - 1) / 2;
    let c = c2 - gr * (c2 - a), d = a + gr * (c2 - a), fc = f(c), fd = f(d);
    for (let k = 0; k < 60; k++) {
      if (fc <= fd) { c2 = d; d = c; fd = fc; c = c2 - gr * (c2 - a); fc = f(c); }
      else { a = c; c = d; fc = fd; d = a + gr * (c2 - a); fd = f(d); }
    }
    let best = a, bf = f(a);
    for (const x of [c2, c, d]) { const v = f(x); if (v < bf) { bf = v; best = x; } }
    return [best, inner(best)];
  }

  // Simulate a year (array of month ranges) under the oracle; returns net load and per-month plans
  function oracleYear(load, months, W, rates, b) {
    const net = new Float64Array(load.length), soc = new Float64Array(load.length);
    let s = b.e;
    const plans = [];
    for (const [i0, i1] of months) {
      const T = planMonth(load, i0, i1, W, rates, s, b);
      plans.push(T);
      for (let i = i0; i < i1; i++) {
        let th = INF;
        for (let k = 0; k < W.length; k++) { const t2 = thAt(W[k], T[k], i); if (t2 < th) th = t2; }
        const r = step(load[i], th, s, b); net[i] = r[0]; s = r[1]; soc[i] = s;
      }
    }
    return { net, soc, plans };
  }

  // ---------- non-anticipative controllers (see fifteen/control.py)
  function runMPC(load, fc, monthId, w, b, opt = {}) {
    const margin = opt.margin ?? 0, alpha = opt.alpha ?? 0.5, kObs = opt.kObs ?? 4, decay = opt.decay ?? 0.9;
    const n = load.length, net = new Float64Array(n), soc = new Float64Array(n), thr = new Float64Array(n);
    let s = b.e, mtd = 0;
    const buf = new Float64Array(SLOTS), wb = new Float64Array(SLOTS);
    for (let t = 0; t < n; t++) {
      if (t === 0 || monthId[t] !== monthId[t - 1]) mtd = 0;
      const slot = t % SLOTS;
      let r = 1;
      const k = slot >= kObs ? kObs : slot;
      if (k > 0) {
        let a = 0, bb = 0;
        for (let j = t - k; j < t; j++) { a += load[j]; bb += fc[j]; }
        if (bb > 1e-9) r = Math.min(3, Math.max(0.33, a / bb));
      }
      const H = SLOTS - slot;
      for (let h = 0; h < H; h++) {
        buf[h] = h === 0 ? load[t] : fc[t + h] * (1 + alpha * (r - 1) * Math.pow(decay, h));
        wb[h] = w[t + h];
      }
      let th = INF, res;
      if (w[t] < INF) {
        let Th = minThresholdPath(buf, H, wb, s, b) + margin;
        if (Th < mtd) Th = mtd;
        th = Th * w[t];
        // recharge only under the month's already-sunk peak: free, and never creates the bill
        const thc = mtd * w[t];
        if (load[t] > th) res = step(load[t], th, s, b);
        else if (load[t] >= thc) res = [load[t], s];
        else res = step(load[t], thc, s, b);
      } else res = step(load[t], INF, s, b);
      net[t] = res[0]; s = res[1]; soc[t] = s; thr[t] = th;
      if (w[t] < INF && res[0] / w[t] > mtd) mtd = res[0] / w[t];
    }
    return { net, soc, thr };
  }

  // planning bisection used by the controller: 30 iterations, no infeasibility pre-check
  // (mirrors fifteen/control.py::_min_T_path exactly so traces match the Python backtest)
  function minThresholdPath(f, H, w, s, b) {
    let hi = 0, lo = 0;
    for (let h = 0; h < H; h++) if (w[h] < INF) {
      const v = f[h] / w[h]; if (v > hi) hi = v;
      const u = (f[h] - b.p) / w[h]; if (u > lo) lo = u;
    }
    if (feasiblePath(f, H, w, lo, s, b)) return lo;
    for (let k = 0; k < 30; k++) { const mid = 0.5 * (lo + hi); if (feasiblePath(f, H, w, mid, s, b)) hi = mid; else lo = mid; }
    return hi;
  }
  function feasiblePath(f, H, w, T, s, b) {
    for (let h = 0; h < H; h++) {
      const L = f[h], th = w[h] >= INF ? INF : T * w[h];
      if (L > th) {
        const need = L - th, dmax = Math.min(b.p, (s * b.ed) / DT, L);
        if (need > dmax + 1e-9) return false;
        s -= (need * DT) / b.ed;
      } else {
        let room = th - L; if (room > b.p) room = b.p;
        const cmax = (b.e - s) / (b.ec * DT), c = room < cmax ? room : cmax;
        if (c > 0) s += c * b.ec * DT;
      }
      if (s > b.e) s = b.e;
    }
    return true;
  }

  function runRatchet(load, monthId, w, startT, b) {
    const n = load.length, net = new Float64Array(n), soc = new Float64Array(n), thr = new Float64Array(n);
    let s = b.e, th = 0;
    for (let t = 0; t < n; t++) {
      if (t === 0 || monthId[t] !== monthId[t - 1]) th = startT[monthId[t]];
      let res;
      if (w[t] < INF) { res = step(load[t], th * w[t], s, b); if (res[0] / w[t] > th) th = res[0] / w[t]; }
      else res = step(load[t], INF, s, b);
      net[t] = res[0]; s = res[1]; soc[t] = s; thr[t] = w[t] < INF ? th * w[t] : NaN;
    }
    return { net, soc, thr };
  }

  function runTimer(load, weekday, startSlot, nSlots, b) {
    const n = load.length, net = new Float64Array(n), soc = new Float64Array(n);
    let s = b.e;
    const dis = Math.min(b.p, (b.e * b.ed) / (nSlots * DT)), chg = Math.min(b.p, b.e / (b.ec * 24 * DT));
    for (let t = 0; t < n; t++) {
      const sl = t % SLOTS, L = load[t];
      let nt = L;
      if (weekday[Math.floor(t / SLOTS)] && sl >= startSlot && sl < startSlot + nSlots) {
        const d = Math.min(dis, (s * b.ed) / DT, L); s -= (d * DT) / b.ed; nt = L - d;
      } else if (sl < 24) {
        const c = Math.min(chg, (b.e - s) / (b.ec * DT)); s += c * b.ec * DT; nt = L + c;
      }
      if (s < 0) s = 0; if (s > b.e) s = b.e;
      net[t] = nt; soc[t] = s;
    }
    return { net, soc };
  }

  // ---------- calendar + bills
  // intervals are interval-START, local time, 15-min, starting at Jan 1 of `year`
  function calendar(year, n) {
    const days = Math.floor(n / SLOTS);
    const month = new Uint8Array(n), dow = new Uint8Array(days), monthOfDay = new Uint8Array(days);
    const months = [];
    let cur = -1, start = 0;
    for (let d = 0; d < days; d++) {
      const dt = new Date(Date.UTC(year, 0, 1 + d));
      dow[d] = dt.getUTCDay(); // 0 = Sunday
      monthOfDay[d] = dt.getUTCMonth();
      if (dt.getUTCMonth() !== cur) { if (cur >= 0) months.push([start, d * SLOTS]); cur = dt.getUTCMonth(); start = d * SLOTS; }
      for (let j = 0; j < SLOTS; j++) month[d * SLOTS + j] = dt.getUTCMonth();
    }
    months.push([start, days * SLOTS]);
    return { days, month, dow, monthOfDay, months };
  }

  // union of peak sub-windows: [{months?, weekdays, start, end}]
  function windowMask(subs, cal, n) {
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const d = Math.floor(i / SLOTS), hour = (i % SLOTS) / 4, mo = cal.month[i] + 1;
      for (const w of subs) {
        if (w.months && !w.months.includes(mo)) continue;
        if (!w.weekdays.includes(cal.dow[d])) continue;
        if (hour >= w.start && hour < w.end) { m[i] = 1; break; }
      }
    }
    return m;
  }

  // billing-demand weights: 1 in peak windows, 1/f off-peak (f = offpeak_factor), INF if off-peak is free
  function demandWeights(dc, cal, n) {
    const w = new Float64Array(n);
    if (!dc.peak || !dc.peak.length) { w.fill(1); return w; }
    const on = windowMask(dc.peak, cal, n), f = dc.offpeak_factor || 0;
    for (let i = 0; i < n; i++) w[i] = on[i] ? 1 : f > 0 ? 1 / f : INF;
    return w;
  }

  function prepareTariff(tariff, cal, n) {
    tariff._W = tariff.demand.map((dc) => demandWeights(dc, cal, n));
    tariff._onpeak = windowMask(tariff.energy.peak || [], cal, n);
    tariff._rates = tariff.demand.map((d) => d.usd_per_kw);
    return tariff._W;
  }

  function customerCharge(tariff, maxBilledKW) {
    for (const [cap, usd] of tariff.customer_charge_tiers || []) if (maxBilledKW <= cap) return usd;
    return tariff.customer_charge;
  }

  // Monthly bills for a net-load series (call prepareTariff first)
  function bills(x, tariff, cal) {
    const out = [];
    const W = tariff._W, en = tariff.energy, sup = en.supply_cents || 0;
    for (let mi = 0; mi < cal.months.length; mi++) {
      const [i0, i1] = cal.months[mi];
      const demand = tariff.demand.map((dc, k) => {
        let mx = 0; for (let i = i0; i < i1; i++) if (W[k][i] < INF) { const v = x[i] / W[k][i]; if (v > mx) mx = v; }
        return { name: dc.name, kw: mx, usd: mx * dc.usd_per_kw };
      });
      let kwhOn = 0, kwhOff = 0;
      for (let i = i0; i < i1; i++) { if (tariff._onpeak[i]) kwhOn += x[i] * DT; else kwhOff += x[i] * DT; }
      const energyUsd = (kwhOn * (en.peak_cents + sup) + kwhOff * (en.offpeak_cents + sup)) / 100;
      const demandUsd = demand.reduce((a, d) => a + d.usd, 0);
      out.push({ month: mi, demand, demandUsd, kwh: kwhOn + kwhOff, energyUsd });
    }
    const maxKW = Math.max(...out.map((o) => o.demand[0].kw));
    const cust = customerCharge(tariff, maxKW);
    for (const o of out) { o.customer = cust; o.total = cust + o.demandUsd + o.energyUsd; }
    return out;
  }

  // ---------- CSV ingestion (Green Button-style exports vary; be forgiving, report what we did)
  function parseIntervalCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
    const rows = [];
    const notes = [];
    for (const line of lines) {
      const cells = line.split(/[,;\t]/).map((c) => c.trim().replace(/^"|"$/g, ""));
      let ts = null, val = null;
      for (const c of cells) {
        if (ts === null) { const t = Date.parse(c.replace(" ", "T")); if (!isNaN(t) && /\d{4}|\d{1,2}\/\d{1,2}/.test(c)) { ts = t; continue; } }
        if (ts !== null && val === null) { const v = parseFloat(c); if (isFinite(v)) val = v; }
      }
      if (ts !== null && val !== null) rows.push([ts, val]);
    }
    if (rows.length < 96 * 28) throw new Error("Need at least 4 weeks of 15-minute (or hourly) interval data; found " + rows.length + " usable rows.");
    rows.sort((a, b) => a[0] - b[0]);
    const step0 = medianStep(rows);
    let factor = 1;
    if (Math.abs(step0 - 3600e3) < 60e3) notes.push("Hourly data detected: each hour is spread over four 15-minute intervals (this understates spiky peaks).");
    else if (Math.abs(step0 - 900e3) > 60e3) throw new Error("Interval length " + step0 / 60e3 + " min not supported (need 15 or 60).");
    // values: kWh per interval or kW? Heuristic: header text, else assume kWh per interval
    const head = lines.slice(0, 3).join(" ").toLowerCase();
    const isKW = /\bkw\b(?!h)/.test(head) && !/kwh/.test(head);
    const hours = step0 / 3600e3;
    if (!isKW) { factor = 1 / hours; notes.push("Values read as kWh per interval and converted to average kW."); }
    else notes.push("Values read as average kW.");
    const t0 = new Date(rows[0][0]);
    const year = t0.getFullYear();
    const start = Date.UTC(year, t0.getMonth(), t0.getDate());
    const n = Math.floor((rows[rows.length - 1][0] - rows[0][0]) / 900e3) + (hours === 1 ? 4 : 1);
    const kw = new Float64Array(Math.min(n, 366 * SLOTS)).fill(NaN);
    for (const [t, v] of rows) {
      const local = new Date(t);
      const idx = Math.round((Date.UTC(local.getFullYear(), local.getMonth(), local.getDate(), local.getHours(), local.getMinutes()) - start) / 900e3);
      for (let k = 0; k < (hours === 1 ? 4 : 1); k++) if (idx + k >= 0 && idx + k < kw.length) kw[idx + k] = v * factor;
    }
    let gaps = 0;
    for (let i = 0; i < kw.length; i++) if (isNaN(kw[i])) { gaps++; kw[i] = i > 0 ? kw[i - 1] : 0; }
    if (gaps) notes.push(gaps + " missing intervals filled with the previous value.");
    const startDay = Math.floor((start - Date.UTC(year, 0, 1)) / 864e5);
    return { kw, year, startDay, notes };
  }
  function medianStep(rows) {
    const d = [];
    for (let i = 1; i < Math.min(rows.length, 2000); i++) d.push(rows[i][0] - rows[i - 1][0]);
    d.sort((a, b) => a - b);
    return d[Math.floor(d.length / 2)];
  }

  root.Fifteen = { DT, SLOTS, INF, battery, step, minThreshold, planMonth, oracleYear, runMPC, runRatchet, runTimer, calendar, windowMask, demandWeights, bills, prepareTariff, customerCharge, parseIntervalCSV };
  if (typeof module !== "undefined") module.exports = root.Fifteen;
})(typeof window !== "undefined" ? window : globalThis);
