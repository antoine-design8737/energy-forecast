/* ============================================================
   Electricity Demand Forecast — front-end logic
   Loads data.json (baked by export_site_data.py) and renders
   every chart with ECharts. No backend at runtime.
   ============================================================ */
(function () {
  "use strict";

  const C = {
    ink: "#0F172A", muted: "#64748B", border: "#E2E8F0",
    blue: "#2563EB", indigo: "#4F46E5", violet: "#7C3AED",
    amber: "#F59E0B", green: "#10B981", red: "#EF4444", grid: "rgba(15,23,42,0.08)",
  };
  const FONT = "Inter, system-ui, sans-serif";
  const MINUS = "−";

  // ---- formatting helpers ------------------------------------------------
  const parisHour = (ms) => new Date(ms).toLocaleTimeString("en-GB",
    { timeZone: "Europe/Paris", hour: "2-digit", minute: "2-digit", hour12: false });
  const parisDay = (ms) => new Date(ms).toLocaleDateString("en-GB",
    { timeZone: "Europe/Paris", weekday: "short", day: "2-digit", month: "short" });
  const monthYear = (iso) => new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB",
    { timeZone: "UTC", month: "short", year: "numeric" });
  const setText = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };

  // ---- shared ECharts scaffolding ---------------------------------------
  function baseOption(extra) {
    return Object.assign({
      textStyle: { fontFamily: FONT, color: C.ink },
      grid: { left: 52, right: 18, top: 38, bottom: 42, containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#fff", borderColor: C.border, borderWidth: 1,
        textStyle: { color: C.ink, fontFamily: FONT },
        extraCssText: "box-shadow:0 8px 24px -12px rgba(15,23,42,.4); border-radius:10px;",
      },
      legend: { top: 4, right: 6, icon: "roundRect", itemWidth: 14, itemHeight: 8,
        textStyle: { color: C.muted, fontFamily: FONT } },
    }, extra);
  }
  const axisLine = { lineStyle: { color: C.border } };
  const splitLine = { lineStyle: { color: C.grid } };

  // ---- chart registry (for resize) --------------------------------------
  const charts = {};
  function mount(id) {
    const el = document.getElementById(id);
    const inst = echarts.init(el, null, { renderer: "canvas" });
    charts[id] = inst;
    return inst;
  }

  let DATA = null;
  let dayIndex = {};   // date -> day object
  let dayList = [];    // ordered day objects
  let cursor = 0;      // current day position

  fetch("data.json")
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => { DATA = d; init(); })
    .catch(() => document.getElementById("load-error").classList.remove("is-hidden"));

  // =======================================================================
  function init() {
    fillText();
    drawBenchmark();
    drawResiduals();
    drawErrorByHour();
    drawFeatureImportance();
    setupForecast();
    setupTabs();
    setupReveal();
    window.addEventListener("resize", () => {
      Object.values(charts).forEach((c) => c.resize());
    });
  }

  // ---- text bindings -----------------------------------------------------
  function fillText() {
    const h = DATA.headline, m = DATA.meta;
    const startY = m.span_start.slice(0, 4), endY = m.span_end.slice(0, 4);
    setText("nav-mape", h.mape + "%");
    setText("hero-mape", h.mape + "%");
    setText("chip-mape", h.mape + "%");
    setText("chip-imp", MINUS + h.improvement_pct + "%");
    setText("chip-mae", h.mae_gw + " GW");
    setText("chip-rows", m.rows.toLocaleString("en-US"));
    setText("chip-span", "Hours · " + startY + "–" + endY);

    setText("kpi-mape", h.mape + "%");
    setText("kpi-mae", h.mae_gw + " GW");
    setText("kpi-mae-note", "≈ " + (h.mae_gw / h.avg_gw * 100).toFixed(1) +
      "% of the " + Math.round(h.avg_gw) + " GW average load");
    setText("kpi-rmse", h.rmse_gw + " GW");
    setText("kpi-imp", MINUS + h.improvement_pct + "%");
    setText("kpi-imp-note", "≈ " + h.fold + "× lower error than last-week-same-hour");

    setText("tile-span", monthYear(m.span_start) + " – " + monthYear(m.span_end) + " · France");
    setText("tile-feats", m.n_features + " predictive features");
    setText("foot-gen", "Updated " + m.generated);
  }

  // ---- benchmark ---------------------------------------------------------
  function drawBenchmark() {
    const b = DATA.benchmark;
    const colors = ["#CBD5E1", "#94A3B8", C.green];
    mount("chart-benchmark").setOption(baseOption({
      legend: { show: false },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        backgroundColor: "#fff", borderColor: C.border, textStyle: { color: C.ink },
        valueFormatter: (v) => v + "%" },
      xAxis: { type: "category", data: b.map((x) => x.name),
        axisLine, axisTick: { show: false }, axisLabel: { color: C.muted, fontFamily: FONT, lineHeight: 15 } },
      yAxis: { type: "value", name: "MAPE (%)  ·  lower is better", nameTextStyle: { color: C.muted },
        max: Math.ceil(Math.max.apply(null, b.map((x) => x.mape)) * 1.2),
        axisLine: { show: false }, splitLine, axisLabel: { color: C.muted, fontFamily: FONT } },
      series: [{
        type: "bar", barWidth: "46%",
        data: b.map((x, i) => ({ value: x.mape, itemStyle: { color: colors[i], borderRadius: [8, 8, 0, 0] } })),
        label: { show: true, position: "top", formatter: "{c}%", fontWeight: 700,
          color: C.ink, fontFamily: FONT, fontSize: 14 },
      }],
    }));
  }

  // ---- residual histogram ------------------------------------------------
  function drawResiduals() {
    const rh = DATA.diagnostics.residual_hist;
    let zero = 0, best = Infinity;
    rh.centers.forEach((c, i) => { if (Math.abs(c) < best) { best = Math.abs(c); zero = i; } });
    mount("chart-resid").setOption(baseOption({
      legend: { show: false },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        backgroundColor: "#fff", borderColor: C.border, textStyle: { color: C.ink } },
      xAxis: { type: "category", data: rh.centers.map((c) => c.toFixed(1)), name: "Forecast error (GW)",
        nameLocation: "middle", nameGap: 30, nameTextStyle: { color: C.muted },
        axisLine, axisTick: { show: false }, axisLabel: { color: C.muted, fontFamily: FONT, interval: 6 } },
      yAxis: { type: "value", name: "Count", nameTextStyle: { color: C.muted },
        axisLine: { show: false }, splitLine, axisLabel: { color: C.muted, fontFamily: FONT } },
      series: [{
        type: "bar", data: rh.counts, barWidth: "92%",
        itemStyle: { color: C.blue, borderRadius: [3, 3, 0, 0] },
        markLine: { silent: true, symbol: "none", lineStyle: { color: C.ink, type: "dashed" },
          data: [{ xAxis: zero }], label: { show: false } },
      }],
    }));
  }

  // ---- error by hour -----------------------------------------------------
  function drawErrorByHour() {
    const eh = DATA.diagnostics.error_by_hour;
    mount("chart-hour").setOption(baseOption({
      legend: { show: false },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        backgroundColor: "#fff", borderColor: C.border, textStyle: { color: C.ink },
        valueFormatter: (v) => (+v).toFixed(2) + "%" },
      xAxis: { type: "category", data: eh.map((x) => x.hour), name: "Hour of day (local)",
        nameLocation: "middle", nameGap: 30, nameTextStyle: { color: C.muted },
        axisLine, axisTick: { show: false }, axisLabel: { color: C.muted, fontFamily: FONT, interval: 1 } },
      yAxis: { type: "value", name: "MAPE (%)", nameTextStyle: { color: C.muted },
        axisLine: { show: false }, splitLine, axisLabel: { color: C.muted, fontFamily: FONT } },
      series: [{ type: "bar", data: eh.map((x) => x.mape), barWidth: "60%",
        itemStyle: { color: C.indigo, borderRadius: [4, 4, 0, 0] } }],
    }));
  }

  // ---- feature importance ------------------------------------------------
  function drawFeatureImportance() {
    const fi = DATA.diagnostics.feature_importance.slice().reverse(); // smallest -> top
    const gcol = { load: C.amber, weather: C.red, calendar: C.blue };
    mount("chart-feat").setOption(baseOption({
      legend: { show: false },
      grid: { left: 8, right: 30, top: 16, bottom: 30, containLabel: true },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
        backgroundColor: "#fff", borderColor: C.border, textStyle: { color: C.ink } },
      xAxis: { type: "value", name: "Importance (gain)", nameLocation: "middle", nameGap: 28,
        nameTextStyle: { color: C.muted }, axisLine: { show: false }, splitLine,
        axisLabel: { color: C.muted, fontFamily: FONT } },
      yAxis: { type: "category", data: fi.map((x) => x.name),
        axisLine, axisTick: { show: false }, axisLabel: { color: C.ink, fontFamily: FONT, fontSize: 11 } },
      series: [{ type: "bar", barWidth: "62%",
        data: fi.map((x) => ({ value: x.gain, itemStyle: { color: gcol[x.group], borderRadius: [0, 5, 5, 0] } })) }],
    }));
  }

  // ---- interactive forecast ---------------------------------------------
  function setupForecast() {
    dayList = DATA.days;
    dayList.forEach((d) => (dayIndex[d.date] = d));
    cursor = dayList.length - 1;

    const picker = document.getElementById("day-picker");
    picker.min = dayList[0].date;
    picker.max = dayList[dayList.length - 1].date;
    picker.value = dayList[cursor].date;

    picker.addEventListener("change", () => {
      const d = dayIndex[picker.value];
      if (d) { cursor = dayList.indexOf(d); renderDay(); }
      else { picker.value = dayList[cursor].date; }   // snap back if out of range
    });
    document.getElementById("prev-day").addEventListener("click", () => {
      cursor = Math.max(0, cursor - 1); syncPicker(); renderDay();
    });
    document.getElementById("next-day").addEventListener("click", () => {
      cursor = Math.min(dayList.length - 1, cursor + 1); syncPicker(); renderDay();
    });

    mount("chart-24h");
    mount("chart-week");
    renderDay();
  }
  function syncPicker() { document.getElementById("day-picker").value = dayList[cursor].date; }

  function renderDay() {
    const day = dayList[cursor];
    const s = DATA.series, a = s.actual_gw, p = s.pred_gw, t = s.t;
    const start = day.start, end = start + 24;

    // -- day stat chips --
    setText("d-mape", day.mape + "%");
    setText("d-mae", day.mae_gw + " GW");
    setText("d-rmse", day.rmse_gw + " GW");
    setText("d-vs", day.vs_naive == null ? "—" : (day.vs_naive >= 0 ? "+" : MINUS) + Math.abs(day.vs_naive) + "%");

    // -- 24h chart --
    const cats = [], act = [], fc = [];
    for (let i = start; i < end; i++) { cats.push(parisHour(t[i] * 1000)); act.push(a[i]); fc.push(p[i]); }
    charts["chart-24h"].setOption(baseOption({
      tooltip: { trigger: "axis", backgroundColor: "#fff", borderColor: C.border,
        textStyle: { color: C.ink }, valueFormatter: (v) => (v == null ? "—" : (+v).toFixed(2) + " GW") },
      xAxis: { type: "category", boundaryGap: false, data: cats,
        axisLine, axisTick: { show: false }, axisLabel: { color: C.muted, fontFamily: FONT, interval: 1 } },
      yAxis: { type: "value", scale: true, name: "Load (GW)", nameTextStyle: { color: C.muted },
        axisLine: { show: false }, splitLine, axisLabel: { color: C.muted, fontFamily: FONT } },
      series: [
        { name: "Actual", type: "line", data: act, smooth: true, symbol: "none",
          lineStyle: { color: C.blue, width: 3 },
          areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1,
            [{ offset: 0, color: "rgba(37,99,235,0.18)" }, { offset: 1, color: "rgba(37,99,235,0)" }]) } },
        { name: "Forecast", type: "line", data: fc, smooth: true, symbol: "none",
          lineStyle: { color: C.amber, width: 3, type: "dashed" } },
      ],
    }), true);

    // -- week context (±3 days) --
    const lo = Math.max(0, start - 72), hi = Math.min(t.length, end + 72);
    const wc = [], wa = [], wf = [];
    for (let i = lo; i < hi; i++) {
      wc.push(parisDay(t[i] * 1000));
      wa.push(a[i]);
      wf.push(i >= start && i < end ? p[i] : null);
    }
    charts["chart-week"].setOption(baseOption({
      tooltip: { trigger: "axis", backgroundColor: "#fff", borderColor: C.border,
        textStyle: { color: C.ink }, valueFormatter: (v) => (v == null ? "—" : (+v).toFixed(2) + " GW") },
      xAxis: { type: "category", boundaryGap: false, data: wc,
        axisLine, axisTick: { show: false },
        axisLabel: { color: C.muted, fontFamily: FONT, interval: 23, hideOverlap: true } },
      yAxis: { type: "value", scale: true, name: "Load (GW)", nameTextStyle: { color: C.muted },
        axisLine: { show: false }, splitLine, axisLabel: { color: C.muted, fontFamily: FONT } },
      series: [
        { name: "Actual load", type: "line", data: wa, smooth: true, symbol: "none",
          lineStyle: { color: C.blue, width: 2 },
          markArea: { silent: true, itemStyle: { color: "rgba(245,158,11,0.10)" },
            data: [[{ xAxis: start - lo }, { xAxis: end - lo - 1 }]] } },
        { name: "Forecast (selected day)", type: "line", data: wf, smooth: true, symbol: "none",
          connectNulls: false, lineStyle: { color: C.amber, width: 3, type: "dashed" } },
      ],
    }), true);
  }

  // ---- diagnostics tabs --------------------------------------------------
  function setupTabs() {
    const tabs = document.querySelectorAll(".tab");
    const map = { resid: "chart-resid", hour: "chart-hour", feat: "chart-feat" };
    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const key = tab.dataset.tab;
      document.querySelectorAll(".tab-panel").forEach((p) =>
        p.classList.toggle("is-hidden", p.dataset.panel !== key));
      const cap = {
        resid: "Forecast errors over the full 90-day hold-out. Centred on zero ⇒ no systematic bias.",
        hour: "Average error for each hour of the day. The model is reliable around the clock.",
        feat: "Top features by gain — amber = load history, red = weather, blue = calendar. Physics, as expected.",
      };
      setText("cap-resid", cap[key]);
      const c = charts[map[key]];
      if (c) requestAnimationFrame(() => c.resize());   // hidden->shown needs a resize
    }));
  }

  // ---- reveal on scroll --------------------------------------------------
  function setupReveal() {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); } });
    }, { threshold: 0.08 });
    document.querySelectorAll(".reveal").forEach((el) => obs.observe(el));
  }
})();
