// Fifteen — page logic. Data contract is written by scripts/05_export_site.py.
(function () {
  "use strict";
  const F = window.Fifteen, C = window.Charts;
  const $ = (id) => document.getElementById(id);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const usd = (v) => (v < 0 ? "−$" : "$") + Math.round(Math.abs(v)).toLocaleString("en-US");
  const usdK = (v) => (Math.abs(v) >= 1000 ? (v < 0 ? "−$" : "$") + (Math.abs(v) / 1000).toFixed(Math.abs(v) >= 1e5 ? 0 : 1) + "k" : usd(v));
  const kw = (v) => Math.round(v).toLocaleString("en-US") + " kW";
  const pct = (v, d = 0) => (100 * v).toFixed(d) + "%";
  const state = { tariffs: [], summary: null, lib: null, site: null, tariffId: null, policy: "mpc", month: 6, cache: {} };

  // ---------------------------------------------------------------- theme
  (function theme() {
    const btn = $("theme"), root = document.documentElement;
    let saved = null; try { saved = localStorage.getItem("fifteen-theme"); } catch (e) { /* storage blocked */ }
    const q = new URLSearchParams(location.search).get("theme");
    if (q === "light" || q === "dark") saved = q;
    if (saved) root.setAttribute("data-theme", saved);
    const isDark = () => root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    const label = () => { btn.textContent = isDark() ? "Light" : "Dark"; };
    label();
    btn.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("fifteen-theme", next); } catch (e) { /* storage blocked */ }
      label(); redrawAll();
    });
  })();

  // ---------------------------------------------------------------- data
  async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }
  function decodeU16(b64, scale) {
    const bin = atob(b64), n = bin.length >> 1, out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = (bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)) * scale;
    return out;
  }
  async function loadSite(id) {
    if (state.cache[id]) return state.cache[id];
    const j = await getJSON("data/sites/" + id + ".json");
    const site = { id, label: j.label, note: j.note, year: j.year, load: decodeU16(j.load, j.scale), fc: j.fc ? decodeU16(j.fc, j.scale) : null, fcKind: j.fc ? "ml" : "seasonal", source: j.source, btype: j.btype };
    state.cache[id] = site;
    return site;
  }

  // ---------------------------------------------------------------- simple forecaster for sites without the ML model
  // Same slot, same weekday, mean of the previous four weeks (fewer early in the year). Day-ahead, no look-ahead.
  function seasonalForecast(load) {
    const S = F.SLOTS, days = Math.floor(load.length / S), fc = new Float64Array(load.length);
    for (let d = 0; d < days; d++) {
      let k = 0;
      for (let j = 0; j < S; j++) fc[d * S + j] = 0;
      for (let w = 1; w <= 4; w++) {
        const dd = d - 7 * w; if (dd < 0) break;
        for (let j = 0; j < S; j++) fc[d * S + j] += load[dd * S + j];
        k++;
      }
      if (k) for (let j = 0; j < S; j++) fc[d * S + j] /= k;
      else { const prev = d > 0 ? d - 1 : 0; for (let j = 0; j < S; j++) fc[d * S + j] = d > 0 ? load[prev * S + j] : load[0]; }
    }
    return fc;
  }

  // ---------------------------------------------------------------- run every policy for one site + tariff
  function analyze(site, tariff) {
    const key = site.id + "|" + tariff.id;
    if (state.cache[key]) return state.cache[key];
    const n = Math.floor(site.load.length / F.SLOTS) * F.SLOTS, load = site.load.subarray(0, n);
    const cal = F.calendar(site.year, n), b = F.battery(1);
    const t = JSON.parse(JSON.stringify(tariff));
    F.prepareTariff(t, cal, n);
    const W = t._W, rates = t._rates, w = W[0];
    const monthId = cal.month;
    const P = state.summary.operate.params;
    const res = { cal, tariff: t, load, base: F.bills(load, t, cal) };
    const o = F.oracleYear(load, cal.months, W, rates, b);
    res.oracle = { net: o.net, soc: o.soc };
    const fc = site.fc ? site.fc.subarray(0, n) : seasonalForecast(load);
    const m = F.runMPC(load, fc, monthId, w, b, { margin: site.fc ? P.margin : P.margin_seasonal, alpha: site.fc ? P.alpha : P.alpha_seasonal, kObs: 4, decay: 0.9 });
    res.mpc = { net: m.net, soc: m.soc };
    // ratchet: start each month at the previous month's billed demand minus beta * P
    const startT = new Float64Array(12);
    for (let mi = 0; mi < 12; mi++) {
      const prev = mi === 0 ? res.base[0].demand[0].kw : res.base[mi - 1].demand[0].kw;
      startT[mi] = Math.max(0, prev - P.beta * b.p);
    }
    const r = F.runRatchet(load, monthId, w, startT, b);
    res.ratchet = { net: r.net, soc: r.soc };
    // timer: the 2-hour weekday window that most often holds the daily peak
    const S = F.SLOTS, cnt = new Float64Array(S), wk = new Uint8Array(cal.days);
    for (let d = 0; d < cal.days; d++) {
      wk[d] = cal.dow[d] >= 1 && cal.dow[d] <= 5 ? 1 : 0;
      if (!wk[d]) continue;
      let bi = 0; for (let j = 1; j < S; j++) if (load[d * S + j] > load[d * S + bi]) bi = j;
      cnt[bi]++;
    }
    let bs = 0, bv = -1;
    for (let s0 = 0; s0 <= S - 8; s0++) { let v = 0; for (let j = 0; j < 8; j++) v += cnt[s0 + j]; if (v > bv) { bv = v; bs = s0; } }
    const tm = F.runTimer(load, wk, bs, 8, b);
    res.timer = { net: tm.net, soc: tm.soc, window: bs };
    for (const p of ["oracle", "mpc", "ratchet", "timer"]) {
      res[p].bills = F.bills(res[p].net, t, cal);
      res[p].savings = res.base.reduce((a, x, i) => a + x.total - res[p].bills[i].total, 0);
      res[p].kwCut = res.base.map((x, i) => x.demand[0].kw - res[p].bills[i].demand[0].kw);
    }
    res.annualBill = res.base.reduce((a, x) => a + x.total, 0);
    state.cache[key] = res;
    return res;
  }

  // ---------------------------------------------------------------- hero meter
  const meter = { svg: null, anim: null };
  function drawDial(maxKW) {
    const s = $("dial"); s.innerHTML = "<title id='dialTitle'>Demand meter</title>";
    const NS = "http://www.w3.org/2000/svg", cx = 200, cy = 205, R = 165;
    const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); s.appendChild(e); return e; };
    mk("path", { d: `M ${cx - R - 14} ${cy} A ${R + 14} ${R + 14} 0 0 1 ${cx + R + 14} ${cy} Z`, class: "face" });
    const ang = (v) => Math.PI * (1 - v / maxKW);
    const step = C.niceTicks(0, maxKW, 6);
    const minor = (step[1] - step[0]) / 5;
    for (let v = 0; v <= maxKW + 1e-6; v += minor) {
      const a = ang(v), major = Math.abs(v / (step[1] - step[0]) - Math.round(v / (step[1] - step[0]))) < 1e-6;
      const r0 = major ? R - 14 : R - 7;
      mk("line", { x1: cx + r0 * Math.cos(a), y1: cy - r0 * Math.sin(a), x2: cx + R * Math.cos(a), y2: cy - R * Math.sin(a), class: "tick" + (major ? "" : " minor"), "stroke-width": major ? 1.6 : 1 });
      if (major) { const t = mk("text", { x: cx + (R - 30) * Math.cos(a), y: cy - (R - 30) * Math.sin(a) + 4, "text-anchor": "middle", class: "ticklabel" }); t.textContent = C.fmtK(v); }
    }
    const u = mk("text", { x: cx, y: cy + 26, "text-anchor": "middle", class: "unit" }); u.textContent = "kW, 15-minute average";
    meter.clock = mk("text", { x: cx, y: cy + 45, "text-anchor": "middle", class: "clock" });
    meter.drag = mk("line", { x1: cx, y1: cy, x2: cx - R + 8, y2: cy, class: "drag" });
    meter.needle = mk("line", { x1: cx, y1: cy, x2: cx - R + 20, y2: cy, class: "needle" });
    mk("circle", { cx, cy, r: 7, class: "hub" });
    meter.set = (needleKW, dragKW) => {
      const a1 = ang(Math.min(needleKW, maxKW)), a2 = ang(Math.min(dragKW, maxKW));
      meter.needle.setAttribute("x2", cx + (R - 22) * Math.cos(a1)); meter.needle.setAttribute("y2", cy - (R - 22) * Math.sin(a1));
      meter.drag.setAttribute("x2", cx + (R - 6) * Math.cos(a2)); meter.drag.setAttribute("y2", cy - (R - 6) * Math.sin(a2));
    };
  }

  async function initHero() {
    const h = state.summary.hero;
    const site = await loadSite(h.site_id);
    const tariff = state.tariffs.find((t) => t.id === h.tariff);
    const res = analyze(site, tariff);
    const [i0, i1] = res.cal.months[h.month - 1];
    const base = res.load.subarray(i0, i1), net = res.mpc.net.subarray(i0, i1);
    const w = res.tariff._W[0].subarray(i0, i1);
    let maxKW = 0; for (const v of base) maxKW = Math.max(maxKW, v);
    maxKW = C.niceTicks(0, maxKW * 1.05, 6).slice(-1)[0];
    if (maxKW < Math.max(...base)) maxKW *= 1.2;
    drawDial(maxKW);
    const rate = tariff.demand[0].usd_per_kw;
    const billedBase = res.base[h.month - 1].demand[0].kw, billedNet = res.mpc.bills[h.month - 1].demand[0].kw;
    const offF = tariff.demand[0].offpeak_factor;
    $("meterCaption").textContent = `${site.label}, ${MONTHS[h.month - 1]} ${site.year}, billed on ${tariff.label} at $${rate}/kW. The thin needle is the building's 15-minute demand. The red drag pointer only moves up, and only during peak hours${offF ? " (off-peak demand counts at " + Math.round(offF * 100) + "%)" : ""}: it is the number the utility bills.`;
    const finish = (withBlock) => {
      const billed = withBlock ? billedNet : billedBase;
      $("peakValue").textContent = kw(billed);
      if (withBlock) { $("savedLabel").textContent = "Saved this month"; $("savedValue").textContent = usd((billedBase - billedNet) * rate); }
      else { $("savedLabel").textContent = "Demand charge"; $("savedValue").textContent = usd(billedBase * rate); }
    };
    const play = (withBlock) => {
      cancelAnimationFrame(meter.anim);
      const series = withBlock ? net : base, n = series.length;
      const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
      let drag = 0, i = 0;
      const perFrame = Math.max(1, Math.round(n / 540));
      $("peakLabel").textContent = withBlock ? "Billed demand with a Powerblock" : "Billed demand this month";
      if (reduce) { for (let k = 0; k < n; k++) if (w[k] < F.INF) drag = Math.max(drag, series[k] / w[k]); meter.set(series[n - 1], drag); finish(withBlock); return; }
      // the needle shows raw demand; the drag pointer shows the billed figure so far (off-peak scaled by the tariff)
      const frame = () => {
        let shown = 0;
        for (let k = 0; k < perFrame && i < n; k++, i++) { shown = Math.max(shown, series[i]); if (w[i] < F.INF) drag = Math.max(drag, series[i] / w[i]); }
        meter.set(shown, drag);
        const day = Math.floor(i / 96) + 1, hr = Math.floor((i % 96) / 4);
        meter.clock.textContent = `${MONTHS[h.month - 1]} ${day}, ${((hr + 11) % 12) + 1}:00 ${hr < 12 ? "am" : "pm"}`;
        $("peakValue").textContent = kw(drag);
        if (i < n) meter.anim = requestAnimationFrame(frame); else finish(withBlock);
      };
      frame();
    };
    $("replayBase").onclick = () => play(false);
    $("replayBlock").onclick = () => play(true);
    const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { play(false); io.disconnect(); } }, { threshold: 0.4 });
    io.observe($("meter"));
  }

  // ---------------------------------------------------------------- evaluate
  const POLICIES = [["oracle", "Perfect foresight"], ["mpc", "Forecast controller"], ["ratchet", "Ratchet"], ["timer", "Fixed timer"]];
  function segmented(container, items, current, onPick) {
    container.innerHTML = "";
    for (const [id, label] of items) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.setAttribute("aria-pressed", id === current ? "true" : "false");
      b.addEventListener("click", () => { for (const x of container.children) x.setAttribute("aria-pressed", "false"); b.setAttribute("aria-pressed", "true"); onPick(id); });
      container.appendChild(b);
    }
  }

  function initEvaluate() {
    const pick = $("sitePick"), groups = {};
    for (const f of state.summary.featured) (groups[f.group] = groups[f.group] || []).push(f);
    for (const g in groups) {
      const og = document.createElement("optgroup"); og.label = g;
      for (const f of groups[g]) { const o = document.createElement("option"); o.value = f.id; o.textContent = f.label; og.appendChild(o); }
      pick.appendChild(og);
    }
    pick.value = state.summary.evaluate_default;
    pick.addEventListener("change", () => selectSite(pick.value));
    state.tariffId = state.summary.hero.tariff;
    segmented($("tariffPick"), state.tariffs.map((t) => [t.id, t.label]), state.tariffId, (id) => { state.tariffId = id; renderEvaluate(); });
    segmented($("policyPick"), POLICIES, state.policy, (id) => { state.policy = id; renderEvaluate(); });
    const mp = $("monthPick");
    MONTHS.forEach((m, i) => { const o = document.createElement("option"); o.value = i; o.textContent = m; mp.appendChild(o); });
    mp.addEventListener("change", () => { state.month = +mp.value; renderEvaluate(); });
    // upload
    const drop = $("drop"), input = $("fileIn");
    const handle = (file) => {
      const msg = $("fileMsg"); msg.className = "msg"; msg.textContent = "Reading " + file.name + "…";
      const rd = new FileReader();
      rd.onload = () => {
        try {
          const p = F.parseIntervalCSV(String(rd.result));
          let load = p.kw;
          if (p.startDay > 0 || load.length < 365 * 96) {
            // align to a calendar year: pad the front with the first week repeated, cut to one year
            const full = new Float64Array(365 * 96);
            for (let i = 0; i < full.length; i++) { const k = i - p.startDay * 96; full[i] = k >= 0 && k < load.length ? load[k] : load[((i % (7 * 96)) + 7 * 96) % Math.min(load.length, 7 * 96)]; }
            load = full;
            p.notes.push("Data did not cover a full calendar year; missing days were filled by repeating a week, so treat the annual total as indicative.");
          }
          const id = "upload-" + Date.now();
          state.cache[id] = { id, label: "Your data", note: "", year: p.year, load, fc: null, fcKind: "seasonal" };
          const o = document.createElement("option"); o.value = id; o.textContent = "Your data (" + file.name.slice(0, 40) + ")";
          pick.insertBefore(o, pick.firstChild); pick.value = id;
          msg.textContent = p.notes.join(" ");
          selectSite(id);
        } catch (e) { msg.className = "msg err"; msg.textContent = e.message; }
      };
      rd.readAsText(file);
    };
    input.addEventListener("change", () => input.files[0] && handle(input.files[0]));
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag"); if (e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]); });
    selectSite(pick.value);
  }

  async function selectSite(id) {
    state.site = await loadSite(id);
    const res = analyze(state.site, state.tariffs.find((t) => t.id === state.tariffId));
    // open on the month where the battery matters most
    let best = 0, bv = -1;
    res.oracle.kwCut.forEach((v, i) => { const d = res.base[i].demand[0].kw - res.mpc.bills[i].demand[0].kw; if (v - d > bv && i > 0) { bv = v - d; best = i; } });
    state.month = best; $("monthPick").value = best;
    renderEvaluate();
  }

  function renderEvaluate() {
    if (!state.site) return;
    const tariff = state.tariffs.find((t) => t.id === state.tariffId);
    const res = analyze(state.site, tariff), p = res[state.policy];
    const rate = tariff.demand[0].usd_per_kw;
    $("evalTitle").textContent = state.site.label;
    $("evalSub").textContent = (state.site.note ? state.site.note + " " : "") + `${state.site.year} load, billed on ${tariff.label}: ${tariff.demand[0].rule}`;
    const avgCut = p.kwCut.reduce((a, b) => a + b, 0) / 12;
    const months = p.savings / (res.annualBill / 12);
    const capture = res.oracle.savings > 0 ? p.savings / res.oracle.savings : 0;
    const stats = [
      ["Annual bill without battery", usd(res.annualBill), ""],
      ["Annual savings", usd(p.savings), "battery"],
      ["Months of bill removed", months.toFixed(2), "battery"],
      ["Average billed kW cut", kw(avgCut) + `<small> of 250</small>`, ""],
      [state.policy === "oracle" ? "Ceiling (perfect foresight)" : "Share of the ceiling kept", state.policy === "oracle" ? "100%" : pct(capture), state.policy === "oracle" ? "" : "demand"],
    ];
    $("evalStats").innerHTML = stats.map(([k, v, c]) => `<div class="stat ${c}"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
    // month view
    const [i0, i1] = res.cal.months[state.month];
    const load = res.load.subarray(i0, i1), net = p.net.subarray(i0, i1), soc = p.soc.subarray(i0, i1), w = res.tariff._W[0].subarray(i0, i1);
    const counted = new Uint8Array(load.length); for (let i = 0; i < load.length; i++) counted[i] = w[i] === 1 ? 1 : 0;
    const argmaxW = (x) => { let bi = -1, bv = -1; for (let i = 0; i < x.length; i++) if (w[i] < F.INF && x[i] / w[i] > bv) { bv = x[i] / w[i]; bi = i; } return bi; };
    const pb = argmaxW(load), pn = argmaxW(net);
    const bBase = res.base[state.month].demand[0].kw, bNet = p.bills[state.month].demand[0].kw;
    const y = state.site.year, mo = state.month;
    const timeLabel = (i) => { const d = new Date(Date.UTC(y, mo, 1 + Math.floor(i / 96))); const h = Math.floor((i % 96) / 4), mm = (i % 4) * 15; return `${MONTHS[mo]} ${d.getUTCDate()}, ${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}`; };
    C.monthChart($("monthChart"), {
      load, net, soc, socMax: F.battery(1).e, counted,
      peakBase: pb, peakNet: pn, billedKW: [bBase, bNet],
      offCeiling: tariff.demand[0].offpeak_factor ? [[bBase / tariff.demand[0].offpeak_factor, false], [bNet / tariff.demand[0].offpeak_factor, true]] : null,
      peakBaseLabel: `Sets the bill without battery: ${kw(bBase)}${w[pb] > 1 ? " (off-peak, billed at " + Math.round(100 / w[pb]) + "%)" : ""}`,
      peakNetLabel: `with: ${kw(bNet)}${w[pn] > 1 ? " (off-peak)" : ""}`,
      dayLabel: (dd) => `${MONTHS[mo]} ${dd + 1}`, timeLabel,
    });
    const offF = tariff.demand[0].offpeak_factor;
    $("monthLegend").innerHTML = `<span><i style="background:var(--load)"></i>Building load</span><span><i style="background:var(--battery)"></i>Grid draw with the Powerblock</span><span><i style="background:var(--demand)"></i>Billed demand: dashed without, solid with the battery${offF ? "; dotted lines are the off-peak equivalents" : ""}</span><span><i style="background:var(--band);height:10px;border:1px solid var(--rule)"></i>${offF ? "Peak hours (off-peak demand counts at " + Math.round(offF * 100) + "%)" : "Peak hours, the only intervals that count"}</span>`;
    $("monthNote").textContent = `${MONTHS[mo]}: ${usd((bBase - bNet) * rate)} off the demand charge, ${kw(bBase - bNet)} lower billed demand.`;
    C.yearBars($("yearChart"), {
      base: res.base.map((x) => x.demand[0].kw), net: p.bills.map((x) => x.demand[0].kw), labels: MONTHS,
      save: res.base.map((x, i) => usdK(x.total - p.bills[i].total)),
      onPick: (i) => { state.month = i; $("monthPick").value = i; renderEvaluate(); },
    });
    const rows = res.base.map((x, i) => `<tr><td>${MONTHS[i]}</td><td>${Math.round(x.demand[0].kw)}</td><td>${Math.round(p.bills[i].demand[0].kw)}</td><td>${usd(x.total)}</td><td>${usd(p.bills[i].total)}</td><td>${usd(x.total - p.bills[i].total)}</td></tr>`).join("");
    $("billTable").innerHTML = `<table><thead><tr><th>Month</th><th>Billed kW</th><th>With battery</th><th>Bill</th><th>With battery</th><th>Saved</th></tr></thead><tbody>${rows}</tbody></table><p class="note">Bills include customer, demand, delivery and supply energy charges (supply at 14.3¢/kWh); round-trip losses are charged as extra energy.</p>`;
  }

  // ---------------------------------------------------------------- find
  function libRows() {
    const L = state.lib, idx = {}; L.fields.forEach((f, i) => (idx[f] = i));
    return L.rows.map((r) => { const o = {}; for (const f in idx) o[f] = r[idx[f]]; return o; });
  }
  function initFind() {
    const S = state.summary.find, rows = libRows();
    $("findKicker").textContent = S.kicker;
    $("findProse").innerHTML = S.prose;
    $("findProse2").innerHTML = S.prose2;
    $("sliceTitle").textContent = S.slice_title; $("sliceSub").textContent = S.slice_sub;
    const drawSlice = () => C.scatter($("sliceChart"), { points: rows.map((r) => ({ x: r.slice, y: r[S.slice_tariff + ":kwcut"], metered: r.source !== "NREL", label: r.label })),
      xmin: 40, xmax: 12000, ymax: 260, xticks: [50, 100, 300, 1000, 3000, 10000], xLabel: "Energy in the top 250 kW of a typical month's load curve (kWh, log scale)", yLabel: "Billed kW cut per month",
      vline: { x: S.slice_knee, label: `${Math.round(S.slice_knee)} kWh: what a full battery can deliver` } });
    drawSlice(); state.redrawSlice = drawSlice;
    $("findProse3").innerHTML = S.prose3;
    $("stripSub").textContent = S.strip_sub;
    let tid = state.summary.find.default_tariff;
    const draw = () => {
      const groups = {};
      for (const r of rows) { (groups[r.group] = groups[r.group] || { v: [], ids: [] }); groups[r.group].v.push(r[tid + ":save"]); groups[r.group].ids.push(r.id); }
      const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
      const list = Object.entries(groups).map(([k, g]) => ({ label: `${k} (${g.v.length})`, values: g.v, ids: g.ids, median: med(g.v) }));
      list.sort((a, b) => b.median - a.median);
      const xmax = tid === "ngrid-g3" ? 40000 : 120000;
      C.stripPlot($("stripChart"), { rows: list, xmax, fmt: usdK, xLabel: "Savings per year, one Powerblock (perfect foresight)",
        onPick: (id) => { if (state.summary.featured.some((f) => f.id === id)) { $("sitePick").value = id; selectSite(id); $("evaluate").scrollIntoView({ behavior: "smooth" }); } } });
    };
    segmented($("stripTariff"), state.tariffs.map((t) => [t.id, t.label]), tid, (id) => { tid = id; draw(); });
    draw();
    state.redrawFind = draw;
    // stacking
    const K = state.summary.stack;
    if (K) {
      $("stackTitle").textContent = K.title; $("stackSub").textContent = K.sub;
      const drawStack = () => C.hbars($("stackChart"), { xmax: K.xmax, fmt: usdK, left: 250, rows: K.rows.map((r) => ({ label: r.label, text: usdK(r.total), segs: r.segs.map((s) => ({ v: s.v, color: s.kind === "cs" ? "--ink-2" : "--battery", opacity: s.kind === "lost" ? 0.25 : 1 })) })) });
      drawStack(); state.redrawStack = drawStack;
      $("stackLegend").innerHTML = `<span><i style="background:var(--battery)"></i>Member's demand-charge savings</span><span><i style="background:var(--ink-2)"></i>ConnectedSolutions Daily Dispatch revenue</span>`;
    } else $("stackInstrument").hidden = true;
    initScreener(rows);
  }

  // ---------------------------------------------------------------- bill-only screener (k nearest neighbours; same features as scripts/05_export_site.py)
  function billFeatures(peaks, kwhs, hours) {
    const lf = peaks.map((p, i) => kwhs[i] / (Math.max(p, 1e-6) * hours[i]));
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((x) => (x - m) ** 2))); };
    const mp = mean(peaks);
    return [Math.log(mp), mean(lf), sd(lf), Math.max(...peaks) / mp, Math.min(...peaks) / mp];
  }
  function initScreener(rows) {
    const S = state.summary.screener;
    $("screenSub").textContent = S.sub;
    const mu = S.mu, sigma = S.sigma, k = S.k;
    const lib = rows.map((r) => ({ r, f: billFeatures(r.mpk, r.mkwh, S.hours).map((v, i) => (v - mu[i]) / sigma[i]) }));
    let tid = state.summary.find.default_tariff;
    segmented($("screenTariff"), state.tariffs.map((t) => [t.id, t.label]), tid, (id) => { tid = id; });
    $("screenExample").onclick = () => { $("billPaste").value = S.example.map((x) => x.join(", ")).join("\n"); };
    $("screenRun").onclick = () => {
      const msg = $("screenMsg"); msg.className = "msg"; msg.textContent = "";
      const lines = $("billPaste").value.split(/\n/).map((l) => l.trim()).filter(Boolean);
      const vals = lines.map((l) => l.split(/[,\s;]+/).map(Number)).filter((a) => a.length >= 2 && a.every(isFinite));
      if (vals.length !== 12) { msg.className = "msg err"; msg.textContent = `Need 12 lines of "peak kW, kWh"; found ${vals.length}.`; return; }
      const peaks = vals.map((v) => v[0]), kwhs = vals.map((v) => v[1]);
      if (peaks.some((p, i) => kwhs[i] > p * S.hours[i])) { msg.className = "msg err"; msg.textContent = "A month's kWh is larger than its peak kW running every hour, which is impossible. Check the column order (peak kW first)."; return; }
      const f = billFeatures(peaks, kwhs, S.hours).map((v, i) => (v - mu[i]) / sigma[i]);
      // a site's own bills are not evidence about itself: drop exact matches (the built-in example is a library site)
      const nn = lib.map((x) => ({ r: x.r, d: Math.hypot(...x.f.map((v, i) => v - f[i])) })).filter((x) => x.d > 1e-6).sort((a, b) => a.d - b.d).slice(0, k);
      const cut = nn.map((x) => x.r[tid + ":kwcut"]).sort((a, b) => a - b);
      const q = (p) => cut[Math.min(cut.length - 1, Math.floor(p * cut.length))];
      const rate = state.tariffs.find((t) => t.id === tid).demand[0].usd_per_kw;
      const bill = S.bill_estimate[tid](peaks, kwhs);
      $("screenStats").innerHTML = [
        ["Likely billed-kW cut per month", `${Math.round(q(0.5))} kW`, "battery"],
        ["Range across look-alike sites", `${Math.round(q(0.1))}–${Math.round(q(0.9))} kW`, ""],
        ["Demand savings per year", usd(q(0.5) * rate * 12), "battery"],
        ["Months of bill removed", ((q(0.5) * rate * 12) / (bill / 12)).toFixed(2), ""],
      ].map(([a, b, c]) => `<div class="stat ${c}"><div class="k">${a}</div><div class="v">${b}</div></div>`).join("");
      $("screenTable").innerHTML = `<table><thead><tr><th>Closest sites in the library</th><th>Avg peak</th><th>Load factor</th><th>kW cut</th></tr></thead><tbody>${nn.slice(0, 8).map((x) => `<tr><td>${x.r.label}</td><td>${kw(x.r.mpk.reduce((a, b) => a + b, 0) / 12)}</td><td>${(x.r.lf * 100).toFixed(0)}%</td><td>${Math.round(x.r[tid + ":kwcut"])}</td></tr>`).join("")}</tbody></table>`;
    };
    // tariff-specific bill estimate from bills alone (customer + demand + energy)
    S.bill_estimate = {};
    for (const t of state.tariffs) {
      S.bill_estimate[t.id] = (peaks, kwhs) => {
        const e = t.energy, cents = (e.peak_cents * 0.55 + e.offpeak_cents * 0.45) + (e.supply_cents || 0);
        return peaks.reduce((a, p, i) => a + t.customer_charge + p * t.demand[0].usd_per_kw + (kwhs[i] * cents) / 100, 0);
      };
    }
  }

  // ---------------------------------------------------------------- operate
  function initOperate() {
    const O = state.summary.operate;
    $("opKicker").textContent = O.kicker;
    $("opProse").innerHTML = O.prose;
    $("opProse2").innerHTML = O.prose2;
    $("opProse3").innerHTML = O.prose3;
    $("opTitle").textContent = O.chart_title; $("opSub").textContent = O.chart_sub;
    const draw = () => C.hbars($("opChart"), { xmax: 1, fmt: (v) => pct(v), left: 290, rows: O.bars.map((b) => ({ label: b.label, text: pct(b.v, 1), segs: [{ v: Math.max(0, b.v), color: b.kind === "ours" ? "--battery" : b.kind === "oracle" ? "--ink" : "--load", opacity: b.kind === "oracle" ? 0.25 : 1 }] })) });
    draw(); state.redrawOp = draw;
    // a real miss: one month, two controllers
    $("opSiteTitle").textContent = O.miss.title; $("opSiteSub").textContent = O.miss.sub;
    drawMiss();
  }
  async function drawMiss() {
    const O = state.summary.operate.miss;
    const site = await loadSite(O.site_id), tariff = state.tariffs.find((t) => t.id === O.tariff);
    const res = analyze(site, tariff);
    const [i0, i1] = res.cal.months[O.month - 1];
    const d0 = (O.day - 1) * 96, d1 = d0 + O.days * 96;
    const sl = (a) => a.subarray(i0 + d0, i0 + d1);
    const load = sl(res.load), a = sl(res.mpc.net), b = sl(res.ratchet.net), w = sl(res.tariff._W[0]);
    const counted = new Uint8Array(load.length); for (let i = 0; i < load.length; i++) counted[i] = w[i] === 1 ? 1 : 0;
    const host = $("opSiteChart"); host.innerHTML = "";
    const top = document.createElement("div"), bot = document.createElement("div"); host.append(top, bot);
    const argmaxW = (x) => { let bi = 0, bv = -1; for (let i = 0; i < x.length; i++) if (w[i] < F.INF && x[i] / w[i] > bv) { bv = x[i] / w[i]; bi = i; } return bi; };
    const billed = (x) => { const i = argmaxW(x); return x[i] / w[i]; };
    const label = (i) => { const d = new Date(Date.UTC(site.year, O.month - 1, O.day + Math.floor(i / 96))); const h = Math.floor((i % 96) / 4), mm = (i % 4) * 15; return `${MONTHS[O.month - 1]} ${d.getUTCDate()}, ${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}`; };
    const dayLabel = (dd) => `${MONTHS[O.month - 1]} ${O.day + dd}`;
    const f = tariff.demand[0].offpeak_factor;
    const common = { load, counted, socMax: F.battery(1).e, peakBase: null, dayLabel, timeLabel: label };
    C.monthChart(top, { ...common, net: b, soc: sl(res.ratchet.soc), peakNet: argmaxW(b), billedKW: [billed(load), billed(b)], offCeiling: f ? [[billed(b) / f, true]] : null, peakBaseLabel: "", peakNetLabel: "Ratchet rule bills " + kw(billed(b)) });
    C.monthChart(bot, { ...common, net: a, soc: sl(res.mpc.soc), peakNet: argmaxW(a), billedKW: [billed(load), billed(a)], offCeiling: f ? [[billed(a) / f, true]] : null, peakBaseLabel: "", peakNetLabel: "Forecast controller bills " + kw(billed(a)) });
    $("opSiteLegend").innerHTML = `<span><i style="background:var(--load)"></i>Building load</span><span><i style="background:var(--battery)"></i>Grid draw with the Powerblock</span><span><i style="background:var(--demand)"></i>Billed demand in these three days</span><span><i style="background:var(--band);height:10px;border:1px solid var(--rule)"></i>Peak hours</span>`;
  }

  // ---------------------------------------------------------------- audit + next
  function initAudit() {
    $("ledger").innerHTML = state.summary.audit.map((a) => `<li><div class="status ${a.status}">${{ pass: "Passed", found: "Found and fixed", assume: "Assumption" }[a.status]}</div><div><div class="what">${a.what}</div><div class="why">${a.why}</div></div></li>`).join("");
    $("nextBody").innerHTML = state.summary.next;
    $("repoLink").innerHTML = (state.summary.repo ? `<a href="${state.summary.repo}">${state.summary.repo.replace("https://", "")}</a><br>` : "") + `<a href="memo.pdf">Two-page summary (PDF)</a>`;
    $("evalKicker").textContent = state.summary.evaluate_kicker;
  }

  function redrawAll() {
    if (state.site) renderEvaluate();
    state.redrawFind && state.redrawFind();
    state.redrawSlice && state.redrawSlice();
    state.redrawStack && state.redrawStack();
    state.redrawOp && state.redrawOp();
    if (state.summary) drawMiss();
  }
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(C.hideTip, 100); });

  // ---------------------------------------------------------------- boot
  (async function boot() {
    try {
      const [t, s, l] = await Promise.all([getJSON("data/tariffs.json"), getJSON("data/summary.json"), getJSON("data/library.json")]);
      state.tariffs = t.tariffs; state.summary = s; state.lib = l;
      state.summary.operate.params = s.operate.params;
      initAudit();
      await initHero();
      initEvaluate();
      initFind();
      initOperate();
    } catch (e) {
      console.error(e);
      const p = document.createElement("p");
      p.className = "frame msg err";
      p.textContent = `Could not load the data files (${e.message}). If you opened index.html from disk, serve the folder instead: python3 -m http.server`;
      document.querySelector("main").prepend(p);
    }
  })();
})();
