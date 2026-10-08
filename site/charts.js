// Small hand-rolled SVG charts. No library: the page's color rules (red = billed peak,
// blue = battery) are enforced here in one place.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgIn(container, w, h) {
    container.innerHTML = "";
    const s = el("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "xMidYMid meet" });
    container.appendChild(s);
    return s;
  }
  const fmtK = (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(v >= 1e4 ? 0 : 1) + "k" : Math.round(v).toString());
  function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const tip = () => document.getElementById("tip");
  function showTip(container, x, y, html) {
    const t = tip(); const r = container.getBoundingClientRect();
    t.innerHTML = html; t.style.left = r.left + window.scrollX + x + "px"; t.style.top = r.top + window.scrollY + y + "px"; t.classList.add("on");
  }
  function hideTip() { tip().classList.remove("on"); }

  // ---------- one month at 15-minute resolution: load, net, threshold, billed peaks, state of charge
  function monthChart(container, d) {
    const W = 960, H = 340, m = { l: 48, r: 12, t: 26, b: 86 }, socH = 34;
    const s = svgIn(container, W, H);
    const n = d.load.length, iw = W - m.l - m.r, ih = H - m.t - m.b;
    let ymax = 0; for (let i = 0; i < n; i++) ymax = Math.max(ymax, d.load[i], d.net ? d.net[i] : 0);
    ymax *= 1.06;
    const X = (i) => m.l + (i / (n - 1)) * iw, Y = (v) => m.t + ih - (v / ymax) * ih;
    // counted (peak-window) shading
    if (d.counted) {
      let i = 0;
      while (i < n) {
        if (d.counted[i]) { let j = i; while (j < n && d.counted[j]) j++; el("rect", { x: X(i), y: m.t, width: Math.max(1, X(j) - X(i)), height: ih, fill: css("--band") }, s); i = j; } else i++;
      }
    }
    const g = el("g", { class: "grid" }, s);
    for (const v of niceTicks(0, ymax, 5)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--rule-soft") }, g);
      const t = el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, s); t.textContent = fmtK(v);
    }
    const yl = el("text", { x: 4, y: 12 }, s); yl.textContent = "kW";
    // day ticks
    const days = Math.round(n / 96), stepD = days <= 7 ? 1 : 7;
    for (let dd = 0; dd < days; dd += stepD) {
      const t = el("text", { x: X(Math.min(n - 1, dd * 96)), y: m.t + ih + 16, "text-anchor": "middle" }, s);
      t.textContent = d.dayLabel ? d.dayLabel(dd) : "Day " + (dd + 1);
    }
    const path = (arr) => { let p = ""; for (let i = 0; i < n; i++) p += (i ? "L" : "M") + X(i).toFixed(1) + "," + Y(arr[i]).toFixed(1); return p; };
    if (d.net) {
      // shaved area between load and net where battery discharges
      let p = "";
      for (let i = 0; i < n; i++) p += (i ? "L" : "M") + X(i).toFixed(1) + "," + Y(Math.max(d.load[i], d.net[i])).toFixed(1);
      for (let i = n - 1; i >= 0; i--) p += "L" + X(i).toFixed(1) + "," + Y(Math.min(d.load[i], d.net[i])).toFixed(1);
      el("path", { d: p + "Z", fill: css("--battery-soft"), stroke: "none" }, s);
    }
    el("path", { d: path(d.load), fill: "none", stroke: css("--load"), "stroke-width": 1, opacity: d.net ? 0.75 : 1 }, s);
    if (d.net) el("path", { d: path(d.net), fill: "none", stroke: css("--battery"), "stroke-width": 1.25 }, s);
    // billed demand: dashed = without battery, solid = with; markers sit on the interval that sets each bill
    const hline = (kw, solid) => el("line", { x1: m.l, x2: W - m.r, y1: Y(kw), y2: Y(kw), stroke: css("--demand"), "stroke-width": 1, "stroke-dasharray": solid ? "none" : "5 4", opacity: 0.75 }, s);
    if (d.offCeiling) {
      for (const [kw, solid] of d.offCeiling) if (kw < ymax) {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(kw), y2: Y(kw), stroke: css("--demand"), "stroke-width": 1, "stroke-dasharray": solid ? "1 3" : "1 6", opacity: 0.6 }, s);
      }
    }
    const mark = (idx, filled, label, kw) => {
      if (idx == null || idx < 0) return;
      const series = filled ? d.net : d.load, v = series[idx];
      hline(kw, filled);
      el("circle", { cx: X(idx), cy: Y(v), r: 5, fill: filled ? css("--demand") : css("--panel"), stroke: css("--demand"), "stroke-width": 2 }, s);
      const right = X(idx) > W - 220;
      const t = el("text", { x: X(idx) + (right ? -9 : 9), y: Y(v) - 9, "text-anchor": right ? "end" : "start", class: "halo" }, s);
      t.style.fill = css("--demand"); t.style.fontWeight = 600; t.textContent = label;
    };
    mark(d.peakBase, false, d.peakBaseLabel, d.billedKW ? d.billedKW[0] : d.load[d.peakBase]);
    if (d.net) mark(d.peakNet, true, d.peakNetLabel, d.billedKW ? d.billedKW[1] : d.net[d.peakNet]);
    // state of charge strip
    if (d.soc) {
      const y0 = H - socH - 18, sh = socH;
      el("rect", { x: m.l, y: y0, width: iw, height: sh, fill: css("--paper-2") }, s);
      let p = "M" + m.l + "," + (y0 + sh);
      for (let i = 0; i < n; i++) p += "L" + X(i).toFixed(1) + "," + (y0 + sh - (d.soc[i] / d.socMax) * sh).toFixed(1);
      p += "L" + (W - m.r) + "," + (y0 + sh) + "Z";
      el("path", { d: p, fill: css("--battery"), opacity: 0.55 }, s);
      const t = el("text", { x: m.l - 6, y: y0 + sh / 2 + 4, "text-anchor": "end" }, s); t.textContent = "Charge";
    }
    // hover
    const ov = el("rect", { x: m.l, y: m.t, width: iw, height: H - m.t - 18, fill: "transparent" }, s);
    const cursor = el("line", { y1: m.t, y2: H - 18, stroke: css("--ink"), "stroke-width": 1, opacity: 0 }, s);
    ov.addEventListener("pointermove", (ev) => {
      const r = s.getBoundingClientRect(), sx = ((ev.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(n - 1, Math.round(((sx - m.l) / iw) * (n - 1))));
      cursor.setAttribute("x1", X(i)); cursor.setAttribute("x2", X(i)); cursor.setAttribute("opacity", 0.35);
      const html = (d.timeLabel ? d.timeLabel(i) : "Interval " + i) + "<br>Building " + Math.round(d.load[i]) + " kW" + (d.net ? "<br>Grid with battery " + Math.round(d.net[i]) + " kW" : "") + (d.soc ? "<br>Battery " + Math.round((100 * d.soc[i]) / d.socMax) + "%" : "");
      showTip(container, (X(i) / W) * r.width, (Y(d.load[i]) / H) * r.height, html);
    });
    ov.addEventListener("pointerleave", () => { cursor.setAttribute("opacity", 0); hideTip(); });
  }

  // ---------- twelve months: billed demand without and with the battery
  function yearBars(container, d) {
    const W = 960, H = 210, m = { l: 48, r: 12, t: 22, b: 26 };
    const s = svgIn(container, W, H), iw = W - m.l - m.r, ih = H - m.t - m.b;
    const n = d.base.length, ymax = Math.max(...d.base) * 1.08;
    const bw = iw / n, Y = (v) => m.t + ih - (v / ymax) * ih;
    for (const v of niceTicks(0, ymax, 4)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--rule-soft") }, s);
      const t = el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, s); t.textContent = fmtK(v);
    }
    const t0 = el("text", { x: 4, y: 12 }, s); t0.textContent = "Billed kW";
    for (let i = 0; i < n; i++) {
      const x = m.l + i * bw;
      el("rect", { x: x + bw * 0.14, y: Y(d.base[i]), width: bw * 0.34, height: ih - (Y(d.base[i]) - m.t), fill: css("--load"), opacity: 0.45, rx: 2 }, s);
      el("rect", { x: x + bw * 0.52, y: Y(d.net[i]), width: bw * 0.34, height: ih - (Y(d.net[i]) - m.t), fill: css("--battery"), rx: 2 }, s);
      const lab = el("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle" }, s); lab.textContent = d.labels[i];
      if (d.save) { const tt = el("text", { x: x + bw / 2, y: Math.min(Y(d.base[i]), Y(d.net[i])) - 5, "text-anchor": "middle" }, s); tt.textContent = d.save[i]; tt.style.fill = css("--ink-2"); }
      const hit = el("rect", { x, y: m.t, width: bw, height: ih, fill: "transparent", style: "cursor:pointer" }, s);
      hit.addEventListener("click", () => d.onPick && d.onPick(i));
      hit.addEventListener("pointermove", () => showTip(container, ((x + bw / 2) / W) * container.clientWidth, (Y(d.base[i]) / H) * container.clientWidth * (H / W), `${d.labels[i]}: ${Math.round(d.base[i])} kW billed without, ${Math.round(d.net[i])} kW with`));
      hit.addEventListener("pointerleave", hideTip);
    }
  }

  // ---------- strip plot: one row per group, one dot per site
  function stripPlot(container, d) {
    const rowH = 30, m = { l: 236, r: 70, t: 26, b: 30 }, W = 960, H = m.t + m.b + rowH * d.rows.length;
    const s = svgIn(container, W, H), iw = W - m.l - m.r;
    const X = (v) => m.l + (Math.max(0, Math.min(v, d.xmax)) / d.xmax) * iw;
    for (const v of niceTicks(0, d.xmax, 6)) {
      el("line", { x1: X(v), x2: X(v), y1: m.t - 6, y2: H - m.b, stroke: css("--rule-soft") }, s);
      const t = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, s); t.textContent = d.fmt(v);
    }
    const xl = el("text", { x: W - m.r, y: m.t - 10, "text-anchor": "end" }, s); xl.textContent = d.xLabel;
    const med = el("text", { x: W - 4, y: m.t - 10, "text-anchor": "end" }, s); med.textContent = "Median";
    d.rows.forEach((row, r) => {
      const cy = m.t + r * rowH + rowH / 2;
      const lab = el("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end" }, s); lab.textContent = row.label; lab.style.fill = css("--ink"); lab.style.fontSize = "12.5px";
      row.values.forEach((v, k) => {
        const jitter = ((k * 7919) % 17) / 17 - 0.5;
        const c = el("circle", { cx: X(v), cy: cy + jitter * (rowH * 0.55), r: 3.1, fill: css("--ink"), opacity: 0.32 }, s);
        if (row.ids) { c.style.cursor = "pointer"; c.addEventListener("click", () => d.onPick && d.onPick(row.ids[k])); }
      });
      const mv = row.median;
      el("line", { x1: X(mv), x2: X(mv), y1: cy - rowH * 0.42, y2: cy + rowH * 0.42, stroke: css("--ink"), "stroke-width": 2.5 }, s);
      const t = el("text", { x: W - 4, y: cy + 4, "text-anchor": "end" }, s); t.textContent = d.fmt(mv); t.style.fill = css("--ink"); t.style.fontWeight = 600;
    });
  }

  // ---------- horizontal bars with optional stacked segments
  function hbars(container, d) {
    const rowH = d.rowH || 40, m = { l: d.left || 210, r: 80, t: 10, b: 28 }, W = 960, H = m.t + m.b + rowH * d.rows.length;
    const s = svgIn(container, W, H), iw = W - m.l - m.r;
    const X = (v) => m.l + (v / d.xmax) * iw;
    for (const v of niceTicks(0, d.xmax, 5)) {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: css("--rule-soft") }, s);
      const t = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, s); t.textContent = d.fmt(v);
    }
    d.rows.forEach((row, r) => {
      const y = m.t + r * rowH + rowH * 0.18, h = rowH * 0.6;
      const lab = el("text", { x: m.l - 12, y: y + h / 2 + 4, "text-anchor": "end" }, s); lab.textContent = row.label; lab.style.fill = css("--ink"); lab.style.fontSize = "12.5px";
      let x0 = 0;
      row.segs.forEach((sg) => {
        el("rect", { x: X(x0), y, width: Math.max(0, X(x0 + sg.v) - X(x0)), height: h, fill: css(sg.color), opacity: sg.opacity ?? 1, rx: 2 }, s);
        x0 += sg.v;
      });
      const t = el("text", { x: X(Math.max(x0, 0)) + 8, y: y + h / 2 + 4 }, s); t.textContent = row.text; t.style.fill = css("--ink"); t.style.fontWeight = 600;
    });
  }

  // ---------- scatter with a log x axis and an optional vertical reference line
  function scatter(container, d) {
    const W = 960, H = 380, m = { l: 56, r: 16, t: 16, b: 46 };
    const s = svgIn(container, W, H), iw = W - m.l - m.r, ih = H - m.t - m.b;
    const lx0 = Math.log10(d.xmin), lx1 = Math.log10(d.xmax);
    const X = (v) => m.l + ((Math.log10(Math.max(d.xmin, Math.min(d.xmax, v))) - lx0) / (lx1 - lx0)) * iw;
    const Y = (v) => m.t + ih - (v / d.ymax) * ih;
    for (const v of niceTicks(0, d.ymax, 5)) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css("--rule-soft") }, s);
      const t = el("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }, s); t.textContent = Math.round(v);
    }
    for (const v of d.xticks) {
      el("line", { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ih, stroke: css("--rule-soft") }, s);
      const t = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, s); t.textContent = v.toLocaleString("en-US");
    }
    const xl = el("text", { x: m.l + iw / 2, y: H - 8, "text-anchor": "middle" }, s); xl.textContent = d.xLabel;
    const yl = el("text", { x: 4, y: m.t + 4 }, s); yl.textContent = d.yLabel;
    if (d.vline) {
      el("line", { x1: X(d.vline.x), x2: X(d.vline.x), y1: m.t, y2: m.t + ih, stroke: css("--ink"), "stroke-width": 1.5, "stroke-dasharray": "4 3" }, s);
      const t = el("text", { x: X(d.vline.x) + 8, y: m.t + 14, class: "halo" }, s); t.textContent = d.vline.label; t.style.fill = css("--ink");
    }
    d.points.forEach((p) => {
      const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: p.metered ? 3.6 : 2.8, fill: p.metered ? "none" : css("--ink"), stroke: css("--ink"), "stroke-width": p.metered ? 1.4 : 0, opacity: p.metered ? 0.8 : 0.28 }, s);
      c.addEventListener("pointermove", () => { const r = container.getBoundingClientRect(); showTip(container, (X(p.x) / W) * r.width, (Y(p.y) / H) * r.height, `${p.label}<br>${Math.round(p.x)} kWh in the top 250 kW, ${Math.round(p.y)} kW cut`); });
      c.addEventListener("pointerleave", hideTip);
    });
  }

  window.Charts = { monthChart, yearBars, stripPlot, hbars, scatter, niceTicks, fmtK, css, hideTip };
})();
