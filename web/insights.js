"use strict";

// Pick two series, a period and a method.
//
// The one rule everything follows: correlation and regression run on RETURNS,
// never on prices. Two trending series show a correlation near 1 with no
// relationship at all (spurious regression, Granger & Newbold 1974). Only the
// methods flagged inLevels (z-score, percentile, cointegration, drawdown)
// work on levels, and for those that's the correct thing to do.

const Insights = (() => {
let D, BYKEY, TSi;
let seriesA = "btc", seriesB = "dxy", method = "rolling_corr";
let startIdx = 0, endIdx = 0;
let win = 90, horizon = 90, nBins = 5, maxLag = 60;

const METHODS = [
  {
    id: "rolling_corr", name: "Rolling correlation (Pearson)", series: 2, win: true,
    technical: "Pearson correlation between the daily log returns of A and B over a rolling window. Goes from −1 to +1 and measures short-term linear association. Runs on returns, not prices, to avoid spurious regression between trending series.",
    plain: "Whether the two tend to go up and down together day to day. Near +1 they move together, near −1 in opposite directions, near 0 there's no relationship. The line shows how that changed over time, because the relationship between assets isn't fixed.",
  },
  {
    id: "rolling_spearman", name: "Rolling correlation (Spearman)", series: 2, win: true,
    technical: "Same as above, on the ranks of the returns instead of their values. Picks up any monotonic relationship, not only linear, and resists outliers, which matters for BTC and its heavy tails.",
    plain: "Same idea as the usual correlation, but it doesn't get thrown by crazy days. A single 30% crash skews the regular one; this one holds up because it looks at the order of the moves, not their size.",
  },
  {
    id: "lead_lag", name: "Lead/lag (who moves first)", series: 2, lags: true,
    technical: "Cross-correlation between A's returns at t and B's at t+k, sweeping k. A peak at k>0 suggests A leads B; at k<0, the other way around. It's lagged association, not causality: no Granger test and no control for confounders.",
    plain: "Checks whether one of the two moves first and the other only reacts days later. The tallest bar is the most likely lag. Careful: even when a pattern shows up it doesn't prove one causes the other, it can be chance or both reacting to something else.",
  },
  {
    id: "beta", name: "Beta and R²", series: 2,
    technical: "OLS regression of B's returns on A's. Beta is the sensitivity (B moves beta% when A moves 1%), R² is the share of B's variance explained by A, and the beta's t-stat gives statistical significance.",
    plain: "Beta answers: when A goes up 1%, how much does B usually move? R² answers: how much of that A actually explains, from 0% to 100%. Low R² with a high beta means the relationship exists but it's noisy and not very reliable.",
  },
  {
    id: "cointegration", name: "Cointegration (Engle-Granger)", series: 2, inLevels: true,
    technical: "Two steps: ADF in levels to confirm both are I(1); then OLS between them and ADF on the residuals, against MacKinnon's critical values. If the residuals are stationary there's a long-run equilibrium and gaps tend to close.",
    plain: "Correlation asks whether they move together day to day; cointegration asks whether they're tied together in the long run, like a dog on a leash: it wanders off but always comes back. If it passes, big gaps between the two tend to close. It's a demanding test and most pairs fail it.",
  },
  {
    id: "conditional", name: "Conditional forward returns", series: 2, horizon: true, inLevels: true,
    technical: "Splits metric A into quantile bins and measures the distribution of B's returns over the next N days after each observation. Shows median, quartiles and hit rate per bin. Samples overlap, so the ranges are optimistic and the number of independent cycles is small.",
    plain: "This is the 'does the pattern repeat' one. It splits history into bins (say low, mid and high MVRV) and shows what Bitcoin did in the following months in each. Today's bin is highlighted. It's not a forecast: there are few cycles and the past may simply not repeat.",
  },
  {
    id: "zscore", name: "Z-score", series: 1, inLevels: true,
    technical: "Standardizes the series by the period's mean and standard deviation: z = (x − mean) ÷ sd. Tells how many deviations the current value is from normal. Assumes a fairly stable distribution, which metrics with a structural trend violate.",
    plain: "Answers 'is this weird?'. Zero is the period's normal, +2 well above it, −2 well below. Useful to see whether a metric is at a historical extreme.",
  },
  {
    id: "percentile", name: "Historical percentile", series: 1, inLevels: true,
    technical: "Position of each value in the period's empirical cumulative distribution. Doesn't assume normality, which makes it a better fit than the z-score for skewed series like MVRV and volatility.",
    plain: "Where today's value sits in history, from 0 to 100. The 95th percentile means only 5% of days were higher than today. More honest than the z-score when the metric is lopsided.",
  },
  {
    id: "volatility", name: "Realized volatility", series: 1, win: true,
    technical: "Standard deviation of log returns over the rolling window, annualized by √252. Measures realized risk, looking back: not implied volatility and not a forecast.",
    plain: "How much the price swung. High means strong swings, low means calm. Calm stretches often come before big moves, but they don't say which way.",
  },
  {
    id: "drawdown", name: "Drawdown", series: 1, inLevels: true,
    technical: "Percentage drop from the running maximum up to each date. Shows the depth and length of the declines within the selected period.",
    plain: "How far the asset is below its own record. −50% means it halved since the top. The most direct measure of the pain of holding it.",
  },
];
const byId = id => METHODS.find(m => m.id === id);
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const el = id => document.getElementById(id);
const fmtD = iso => { const [y, m, d] = iso.split("-"); return `${MONTHS[m - 1]} ${+d}, ${y}`; };
const num = (v, c = 2) => (v == null || !isFinite(v)) ? "—"
  : v.toLocaleString("en-US", {minimumFractionDigits: c, maximumFractionDigits: c});

const LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: {color: "#8b949e", size: 11},
  margin: {l: 56, r: 16, t: 10, b: 34},
  hovermode: "x unified",
  hoverlabel: {bgcolor: "#161b22", bordercolor: "#30363d", font: {color: "#e6edf3", size: 11}},
  showlegend: false,
  xaxis: {gridcolor: "#1c2128", linecolor: "#30363d", zeroline: false},
  yaxis: {gridcolor: "#1c2128", linecolor: "#30363d", zeroline: true, zerolinecolor: "#30363d"},
};
const CFG = {responsive: true, displaylogo: false,
             modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]};

const slicedDates = () => D.dates.slice(startIdx, endIdx + 1);
const clip = arr => arr.slice(startIdx, endIdx + 1);

function compute() {
  const m = byId(method);
  const A = BYKEY[seriesA], B = BYKEY[seriesB];
  const rA = Stats.returns(A.values).values;
  const rB = B ? Stats.returns(B.values).values : null;
  const gr = el("insChart"), nm = el("insNumbers");
  nm.innerHTML = "";

  const warn = t => `<div class="warn">${t}</div>`;
  const row = (r, k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const table = ls => `<table class="tab">${ls.join("")}</table>`;

  // under ~2 months of business days every number below is noise
  if (endIdx - startIdx < 40) {
    Plotly.purge(gr);
    gr.innerHTML = "";
    nm.innerHTML = warn("Period too short for reliable statistics. Pick a longer window.");
    return;
  }

  switch (method) {
  case "rolling_corr":
  case "rolling_spearman": {
    const fn = method === "rolling_corr" ? Stats.pearson : Stats.spearman;
    const s = Stats.rollingCorr(rA, rB, win, startIdx, endIdx, fn);
    const y = clip(s);
    const val = y.filter(v => v != null);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: slicedDates(), y,
      line: {color: "#F7931A", width: 1.6}, connectgaps: false,
      hovertemplate: "correlation %{y:.3f}<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: [-1, 1],
         title: {text: `corr. ${win}d`, font: {size: 10}}}}, CFG);
    const p = Stats.validPairs(rA, rB, startIdx, endIdx);
    nm.innerHTML = table([
      row(0, "Correlation over the whole period", num(fn(p.x, p.y), 3)),
      row(0, "Mean of the rolling window", num(Stats.mean(val), 3)),
      row(0, "Min / Max", `${num(Math.min(...val), 3)} / ${num(Math.max(...val), 3)}`),
      row(0, "Latest value", num(val[val.length - 1], 3)),
      row(0, "Days used", p.x.length.toLocaleString("en-US")),
    ]);
    break;
  }

  case "lead_lag": {
    const res = Stats.leadLag(rA, rB, maxLag, startIdx, endIdx);
    const valid = res.filter(r => r.r != null);
    const peak = valid.reduce((a, b) => Math.abs(b.r) > Math.abs(a.r) ? b : a, valid[0]);
    Plotly.react(gr, [{
      type: "bar", x: valid.map(r => r.k), y: valid.map(r => r.r),
      marker: {color: valid.map(r => r.k === peak.k ? "#F7931A" : "#30475e")},
      hovertemplate: "lag %{x}d<br>correlation %{y:.3f}<extra></extra>",
    }], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `← ${B.label} leads   |   ${A.label} leads →`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: "correlation", font: {size: 10}}}}, CFG);
    const zero = valid.find(r => r.k === 0);
    nm.innerHTML = table([
      row(0, "Peak correlation", num(peak.r, 3)),
      row(0, "Lag at the peak", `${peak.k} days`),
      row(0, "Reading", peak.k === 0 ? "They move at the same time"
        : peak.k > 0 ? `${A.label} tends to move ${peak.k}d earlier`
                     : `${B.label} tends to move ${-peak.k}d earlier`),
      row(0, "Correlation at lag 0", num(zero ? zero.r : null, 3)),
    ]) + warn("Lagged association isn't causality. Test enough lags and some peak shows up by chance.");
    break;
  }

  case "beta": {
    const r = Stats.betaOLS(rA, rB, startIdx, endIdx);
    if (!r) { nm.innerHTML = warn("Not enough data."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const p = Stats.validPairs(rA, rB, startIdx, endIdx);
    const xs = p.x, ys = p.y;
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    Plotly.react(gr, [
      {type: "scattergl", mode: "markers", x: xs.map(v => v * 100), y: ys.map(v => v * 100),
       marker: {color: "#F7931A", size: 3, opacity: .35},
       hovertemplate: `${A.label} %{x:.2f}%<br>${B.label} %{y:.2f}%<extra></extra>`},
      {type: "scatter", mode: "lines",
       x: [minX * 100, maxX * 100],
       y: [(r.alpha + r.beta * minX) * 100, (r.alpha + r.beta * maxX) * 100],
       line: {color: "#e6edf3", width: 1.5}, hoverinfo: "skip"},
    ], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `${A.label} daily return (%)`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: `${B.label} daily return (%)`, font: {size: 10}}}}, CFG);
    // |t| > 1.96 ~ 5% two-sided, normal approximation is fine at these n
    const signif = Math.abs(r.tBeta) > 1.96;
    nm.innerHTML = table([
      row(0, "Beta", num(r.beta, 3)),
      row(0, "Std. error / t", `${num(r.seBeta, 3)} / ${num(r.tBeta, 2)}`),
      row(0, "Significant at 5%", signif ? "yes" : "no"),
      row(0, "R²", `${num(r.r2 * 100, 1)}%`),
      row(0, "Correlation", num(r.corr, 3)),
      row(0, "Observations", r.n.toLocaleString("en-US")),
    ]) + warn(`Reading: when ${A.label} moves 1%, ${B.label} moves ${num(r.beta, 2)}% on average. `
      + `An R² of ${num(r.r2 * 100, 1)}% means ${num(100 - r.r2 * 100, 1)}% of the move comes from other things.`);
    break;
  }

  case "cointegration": {
    const p = Stats.validPairs(A.values, B.values, startIdx, endIdx);
    const c = Stats.cointegration(p.x, p.y);
    if (!c) { nm.innerHTML = warn("Not enough data."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const dts = p.idx.map(i => D.dates[i]);
    const sd = Stats.stdev(c.residuals);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: dts, y: c.residuals.map(v => v / sd),
      line: {color: "#F7931A", width: 1.4},
      hovertemplate: "gap %{y:.2f}σ<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis,
         title: {text: "residual (std devs)", font: {size: 10}}}}, CFG);
    const ls = [
      row(0, `ADF in levels, ${A.label}`, `${num(c.adfX, 2)} (${c.stationaryX ? "stationary" : "non-stationary"})`),
      row(0, `ADF in levels, ${B.label}`, `${num(c.adfY, 2)} (${c.stationaryY ? "stationary" : "non-stationary"})`),
      row(0, "Estimated relationship", `${B.label} = ${num(c.alpha, 2)} + ${num(c.beta, 4)} × ${A.label}`),
      row(0, "Regression R²", `${num(c.r2 * 100, 1)}%`),
      row(0, "ADF on residuals", `${num(c.tResid, 3)} (${c.lags} lags)`),
      row(0, "Critical values", `1%: ${Stats.CV_EG[1]} · 5%: ${Stats.CV_EG[5]} · 10%: ${Stats.CV_EG[10]}`),
      row(0, "Current gap", `${num(c.residuals[c.residuals.length - 1] / sd, 2)}σ`),
    ];
    let verdict;
    if (!c.applicable) {
      verdict = warn("Test does NOT apply: cointegration assumes both series are non-stationary in levels. "
        + "At least one is already stationary, so the result below has no valid reading.");
    } else if (c.level) {
      verdict = `<div class="ok">Cointegrated at the ${c.level} level. There's a long-run equilibrium: `
        + `big gaps tended to close within this period.</div>`;
    } else {
      verdict = warn("Not cointegrated. Can't reject that the gaps are permanent: "
        + "the two may move together at times, but there's no long-run anchor.");
    }
    nm.innerHTML = table(ls) + verdict;
    break;
  }

  case "conditional": {
    const r = Stats.conditionalReturns(A.values, B.values, horizon, nBins, startIdx, endIdx);
    if (!r) { nm.innerHTML = warn("Not enough sample for that horizon."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const labels = r.bins.map((f, i) =>
      `${A.prefix}${num(f.from, A.decimals)}–${A.prefix}${num(f.to, A.decimals)}`);
    Plotly.react(gr, [
      {type: "bar", name: "median", x: labels, y: r.bins.map(f => f.median),
       marker: {color: r.bins.map((_, i) => i === r.currentBin ? "#F7931A" : "#30475e")},
       error_y: {type: "data", symmetric: false,
                 array: r.bins.map(f => f.p75 - f.median),
                 arrayminus: r.bins.map(f => f.median - f.p25),
                 color: "#8b949e", thickness: 1, width: 4},
       hovertemplate: "%{x}<br>median %{y:.1f}%<extra></extra>"},
    ], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `${A.label} bin`, font: {size: 10}}, type: "category"},
      yaxis: {...LAYOUT.yaxis, title: {text: `${B.label} return over ${horizon}d (%)`, font: {size: 10}}}}, CFG);
    const ls = [`<tr><th>${A.label} bin</th><th>n</th><th>Median</th><th>Mean</th><th>% positive</th></tr>`];
    r.bins.forEach((f, i) => {
      ls.push(`<tr class="${i === r.currentBin ? "current" : ""}">`
        + `<td>${labels[i]}${i === r.currentBin ? " <b>← today</b>" : ""}</td>`
        + `<td>${f.n}</td><td>${num(f.median, 1)}%</td><td>${num(f.mean, 1)}%</td>`
        + `<td>${num(f.pctPositive, 0)}%</td></tr>`);
    });
    nm.innerHTML = table(ls)
      + warn(`${A.label} is at ${A.prefix}${num(r.currentValue, A.decimals)}${A.suffix} today. `
        + `The ${horizon}-day windows overlap, so the observations aren't independent: `
        + `the ${r.n} points are far fewer than ${r.n} distinct cases. Treat it as historical context, not a forecast.`);
    break;
  }

  case "zscore":
  case "percentile": {
    const z = method === "zscore" ? Stats.zscore(A.values, startIdx, endIdx)
                                  : Stats.percentile(A.values, startIdx, endIdx);
    const y = clip(z.series);
    const val = y.filter(v => v != null);
    const zs = method === "zscore";
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: slicedDates(), y,
      line: {color: A.color, width: 1.5}, connectgaps: false,
      hovertemplate: (zs ? "%{y:.2f}σ" : "percentile %{y:.0f}") + "<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: zs ? null : [0, 100],
         title: {text: zs ? "std devs" : "percentile", font: {size: 10}}}}, CFG);
    const current = val[val.length - 1];
    nm.innerHTML = table([
      row(0, "Current value", `${A.prefix}${num(A.values[endIdx], A.decimals)}${A.suffix}`),
      row(0, zs ? "Current z-score" : "Current percentile", zs ? `${num(current, 2)}σ` : `${num(current, 0)}`),
      ...(zs ? [row(0, "Period mean", num(z.mean, A.decimals)),
                row(0, "Standard deviation", num(z.stdev, A.decimals))]
             : [row(0, "Period median", num(Stats.quantile(z.sorted, .5), A.decimals)),
                row(0, "p5–p95 range", `${num(Stats.quantile(z.sorted, .05), A.decimals)} – ${num(Stats.quantile(z.sorted, .95), A.decimals)}`)]),
      row(0, "Reading", zs
        ? (Math.abs(current) > 2 ? "Extreme (>2σ)" : Math.abs(current) > 1 ? "Out of the ordinary" : "Within normal")
        : (current > 95 ? "Top of the period's range" : current < 5 ? "Bottom of the period's range" : "Middle of the range")),
    ]);
    break;
  }

  case "volatility": {
    const s = Stats.volatility(rA, win, startIdx, endIdx);
    const y = clip(s), val = y.filter(v => v != null);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: slicedDates(), y,
      line: {color: "#FB923C", width: 1.5}, connectgaps: false,
      hovertemplate: "%{y:.1f}% annualized<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: `${win}d vol, annualized (%)`, font: {size: 10}}}}, CFG);
    const ord = [...val].sort((a, b) => a - b);
    nm.innerHTML = table([
      row(0, "Current volatility", `${num(val[val.length - 1], 1)}% annualized`),
      row(0, "Period median", `${num(Stats.quantile(ord, .5), 1)}% annualized`),
      row(0, "Min / Max", `${num(ord[0], 1)}% / ${num(ord[ord.length - 1], 1)}%`),
      row(0, "Current percentile", `${num(ord.filter(v => v < val[val.length - 1]).length / ord.length * 100, 0)}`),
    ]);
    break;
  }

  case "drawdown": {
    const d = Stats.drawdown(A.values, startIdx, endIdx);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: slicedDates(), y: clip(d.series),
      fill: "tozeroy", fillcolor: "rgba(248,81,73,.15)",
      line: {color: "#f85149", width: 1.3}, connectgaps: false,
      hovertemplate: "%{y:.1f}%<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: "drop from the top (%)", font: {size: 10}}}}, CFG);
    nm.innerHTML = table([
      row(0, "Current drawdown", `${num(d.current, 1)}%`),
      row(0, "Worst drawdown in the period", `${num(d.maxDD, 1)}%`),
      row(0, "Date of the worst", d.iMax != null ? fmtD(D.dates[d.iMax]) : "—"),
    ]);
    break;
  }
  }

  el("insExplain").innerHTML =
    `<div class="exp"><div class="exp-title">How it works</div><p>${m.technical}</p></div>` +
    `<div class="exp"><div class="exp-title">In plain words</div><p>${m.plain}</p></div>`;
}

function buildSelects() {
  for (const sid of ["selA", "selB"]) {
    const s = el(sid);
    s.innerHTML = "";
    for (const g of D.groups) {
      const og = document.createElement("optgroup");
      og.label = g.name;
      for (const x of D.series.filter(v => v.group === g.id)) {
        const o = document.createElement("option");
        o.value = x.key; o.textContent = x.label;
        og.appendChild(o);
      }
      s.appendChild(og);
    }
  }
  el("selA").value = seriesA;
  el("selB").value = seriesB;
}

function buildMethods() {
  const c = el("insMethods");
  c.innerHTML = "";
  for (const m of METHODS) {
    const b = document.createElement("button");
    b.className = "method"; b.dataset.id = m.id; b.textContent = m.name;
    b.addEventListener("click", () => { method = m.id; syncControls(); compute(); });
    c.appendChild(b);
  }
}

function syncControls() {
  const m = byId(method);
  for (const b of document.querySelectorAll(".method"))
    b.setAttribute("aria-pressed", String(b.dataset.id === method));
  el("rowB").style.display = m.series === 2 ? "" : "none";
  el("labelB").textContent = method === "conditional" ? "Target (what happened next)" : "Series B";
  el("labelA").textContent = method === "conditional" ? "Condition (reference metric)" : "Series A";
  el("rowWindow").style.display = m.win ? "" : "none";
  el("rowHorizon").style.display = m.horizon ? "" : "none";
  el("rowLag").style.display = m.lags ? "" : "none";
  el("insLevel").textContent = m.inLevels
    ? "This method works on levels."
    : "This method works on daily returns, not prices.";
}

function setPeriod(years) {
  endIdx = D.dates.length - 1;
  if (years == null) { startIdx = 0; }
  else {
    const target = new Date(TSi[endIdx]);
    target.setFullYear(target.getFullYear() - years);
    startIdx = TSi.findIndex(t => t >= target.getTime());
    if (startIdx < 0) startIdx = 0;
  }
  showPeriod();
}

function showPeriod() {
  el("insPeriod").innerHTML =
    `<b>${fmtD(D.dates[startIdx])}</b> to <b>${fmtD(D.dates[endIdx])}</b> · ` +
    `${(endIdx - startIdx + 1).toLocaleString("en-US")} days`;
}

// reuse whatever range is zoomed in the main chart above
function useChartWindow() {
  const g = document.getElementById("chart");
  const r = g && g.layout && g.layout.xaxis && g.layout.xaxis.range;
  if (!r) { setPeriod(null); return; }
  const ms = v => (typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T") + "Z"));
  const a = ms(r[0]), b = ms(r[1]);
  let i = TSi.findIndex(t => t >= a); if (i < 0) i = 0;
  let f = TSi.length - 1;
  for (let k = TSi.length - 1; k >= 0; k--) if (TSi[k] <= b) { f = k; break; }
  if (f - i < 40) { setPeriod(null); return; }
  startIdx = i; endIdx = f;
  showPeriod();
}

function bind() {
  el("selA").addEventListener("change", e => { seriesA = e.target.value; compute(); });
  el("selB").addEventListener("change", e => { seriesB = e.target.value; compute(); });
  el("inpWindow").addEventListener("change", e => {
    win = Math.max(10, Math.min(730, +e.target.value || 90)); e.target.value = win; compute();
  });
  el("inpHorizon").addEventListener("change", e => {
    horizon = Math.max(5, Math.min(500, +e.target.value || 90)); e.target.value = horizon; compute();
  });
  el("inpBins").addEventListener("change", e => {
    nBins = Math.max(2, Math.min(10, +e.target.value || 5)); e.target.value = nBins; compute();
  });
  el("inpLag").addEventListener("change", e => {
    maxLag = Math.max(5, Math.min(180, +e.target.value || 60)); e.target.value = maxLag; compute();
  });
  for (const b of document.querySelectorAll("[data-years]")) {
    b.addEventListener("click", () => {
      const v = b.dataset.years;
      v === "chart" ? useChartWindow() : setPeriod(v === "max" ? null : +v);
      compute();
    });
  }
}

function init(data, byKey, ts) {
  D = data; BYKEY = byKey; TSi = ts;
  buildSelects(); buildMethods(); bind();
  setPeriod(null);
  syncControls();
  compute();
}

return {init};
})();
