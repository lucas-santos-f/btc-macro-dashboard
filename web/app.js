"use strict";

// Two stacked panels sharing the x axis, so zoom/pan is synced for free:
//   top (y)     trend series rebased to 100, log scale, pick as many as you want
//   bottom (y2) ONE oscillator in its native scale. Single pick on purpose: MVRV
//               (0-4), Fear & Greed (0-100), rates (0-5%) and exchange flows
//               (±1e9 USD) can't share an axis without one flattening the rest.
// Base 100 is recomputed at the left edge of the visible window on every
// zoom/pan, so no arbitrary base date biases the comparison.

const gd = document.getElementById("grafico");
const LS_KEY = "btcmacro.selection";

let DATA, TS, BY_KEY;
let busy = false;
let baseIdx = 0;
let logScale = true;
let traceMeta = [];  // parallel to the plotly traces
let sel = {indexedKeys: new Set(), oscillator: null};

const fmtNum = (v, decimals) =>
  v.toLocaleString("pt-BR", {minimumFractionDigits: decimals, maximumFractionDigits: decimals});

function fmtValue(v, s) {
  const a = Math.abs(v);

  // spelled out, big numbers are unreadable in the tooltip
  if (a >= 1e12) return `${s.prefix}${fmtNum(v / 1e12, 2)} tri${s.suffix}`;
  if (a >= 1e9)  return `${s.prefix}${fmtNum(v / 1e9, 2)} bi${s.suffix}`;
  if (a >= 1e6)  return `${s.prefix}${fmtNum(v / 1e6, 2)} mi${s.suffix}`;
  return `${s.prefix}${fmtNum(v, s.decimals)}${s.suffix}`;
}

const fmtDate = iso => {
  const [a, m, d] = iso.split("-");
  return `${d}/${m}/${a}`;
};

function toMs(v) {
  if (typeof v === "number") return v;
  const s = String(v).trim().replace(" ", "T");
  const t = Date.parse(s.endsWith("Z") ? s : s + "Z");
  return Number.isNaN(t) ? Date.parse(String(v)) : t;
}

// first index with date >= t
function indexFor(t) {
  let lo = 0, hi = TS.length - 1;
  if (t <= TS[0]) return 0;
  if (t >= TS[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (TS[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// a series that didn't exist yet at the left edge gets rebased on its first
// real value, otherwise it would just vanish from the chart
function baseValue(s, start) {
  for (let i = start; i < s.values.length; i++) {
    if (s.values[i] != null && s.values[i] > 0) return s.values[i];
  }
  return null;
}

function indexedSeries(s, start) {
  const base = baseValue(s, start);
  if (base == null) return {y: s.values.map(() => null), cd: s.values.map(() => null)};
  return {
    y: s.values.map(v => (v == null ? null : (v / base) * 100)),
    cd: s.values.map(v => (v == null ? null : [fmtValue(v, s), (v / base - 1) * 100])),
  };
}

const nativeSeries = s => ({
  y: s.values,
  cd: s.values.map(v => (v == null ? null : [fmtValue(v, s), 0])),
});

function selectedTraces() {
  const t = [], meta = [];
  for (const s of DATA.series) {
    if (!sel.indexedKeys.has(s.key)) continue;
    const {y, cd} = indexedSeries(s, baseIdx);
    meta.push({series: s, indexed: true});
    t.push({
      type: "scatter", mode: "lines", name: s.label, x: DATA.dates, y,
      customdata: cd, yaxis: "y", line: {color: s.color, width: 1.5},
      connectgaps: false,
      hovertemplate: "<b>%{customdata[0]}</b>  (%{customdata[1]:+.1f}%)<extra>%{fullData.name}</extra>",
    });
  }
  if (sel.oscillator) {
    const s = BY_KEY[sel.oscillator];
    const {y, cd} = nativeSeries(s);
    meta.push({series: s, indexed: false});
    t.push({
      type: "scatter", mode: "lines", name: s.label, x: DATA.dates, y,
      customdata: cd, yaxis: "y2", line: {color: s.color, width: 1.5},
      connectgaps: false, showlegend: false,
      hovertemplate: "<b>%{customdata[0]}</b><extra>%{fullData.name}</extra>",
    });
  }
  traceMeta = meta;
  return t;
}

function buildLayout() {
  const hasOsc = !!sel.oscillator;
  const axis = {gridcolor: "#1c2128", linecolor: "#30363d", zeroline: false};

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: "#8b949e", size: 12},
    margin: {l: 64, r: 16, t: 8, b: 30},
    hovermode: "x unified",
    hoverlabel: {bgcolor: "#161b22", bordercolor: "#30363d",
                 font: {color: "#e6edf3", size: 12}},
    legend: {orientation: "h", x: 0, y: 1.05, font: {color: "#e6edf3"},
             bgcolor: "rgba(0,0,0,0)"},
    xaxis: {
      ...axis, type: "date", anchor: hasOsc ? "y2" : "y",
      rangeslider: {visible: true, thickness: .06,
                    bgcolor: "#161b22", bordercolor: "#30363d"},
      rangeselector: {
        buttons: [
          {count: 3, label: "3M", step: "month", stepmode: "backward"},
          {count: 1, label: "1A", step: "year", stepmode: "backward"},
          {count: 2, label: "2A", step: "year", stepmode: "backward"},
          {count: 5, label: "5A", step: "year", stepmode: "backward"},
          {step: "all", label: "Máx"},
        ],
        bgcolor: "#161b22", activecolor: "#F7931A",
        bordercolor: "#30363d", borderwidth: 1,
        font: {color: "#e6edf3", size: 11}, x: 0, y: 1.2,
      },
    },
    yaxis: {
      ...axis, anchor: "x", type: logScale ? "log" : "linear",
      domain: hasOsc ? [0.32, 1] : [0, 1],
      title: {text: "Índice (base 100)", font: {size: 11}},
    },
  };

  if (hasOsc) {
    const s = BY_KEY[sel.oscillator];
    layout.yaxis2 = {
      ...axis, anchor: "x", domain: [0, 0.25],
      title: {text: s.label, font: {size: 11, color: s.color}},
    };
  }
  return layout;
}

const CONFIG = {
  responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  toImageButtonOptions: {filename: "btc-macro", scale: 2},
};

function draw() {
  saveSelection();
  syncChecks();
  const t = selectedTraces();
  if (!t.length) {
    Plotly.purge(gd);
    gd.innerHTML = '<div id="vazio">Selecione ao menos uma série no painel à esquerda.</div>';
    document.getElementById("base").textContent = "";
    return Promise.resolve();
  }
  if (gd.querySelector("#vazio")) gd.innerHTML = "";
  busy = true;
  return Plotly.react(gd, t, buildLayout(), CONFIG)
    .then(() => { baseLabel(); bindRelayout(); })
    .finally(() => { busy = false; });
}

function baseLabel() {
  const el = document.getElementById("base");
  const n = sel.indexedKeys.size;
  el.innerHTML = n
    ? `Base <b>${fmtDate(DATA.dates[baseIdx])} = 100</b> · recalculada a cada ` +
      `zoom/pan · ${n} série${n > 1 ? "s" : ""} no painel superior` +
      (sel.oscillator ? ` · oscilador: <b>${BY_KEY[sel.oscillator].label}</b>` : "")
    : (sel.oscillator ? `Oscilador: <b>${BY_KEY[sel.oscillator].label}</b> (escala nativa)` : "");
}

function applyRebase(start) {
  baseIdx = start;
  const ys = [], cds = [];
  for (const m of traceMeta) {
    const r = m.indexed ? indexedSeries(m.series, start) : nativeSeries(m.series);
    ys.push(r.y); cds.push(r.cd);
  }
  baseLabel();
  return Plotly.restyle(gd, {y: ys, customdata: cds});
}

let relayoutBound = false;
function bindRelayout() {
  if (relayoutBound) return;
  relayoutBound = true;
  gd.on("plotly_relayout", ev => {
    if (busy) return;
    let start;
    if (ev["xaxis.autorange"] === true) start = 0;
    else if (ev["xaxis.range[0]"] !== undefined) start = indexFor(toMs(ev["xaxis.range[0]"]));
    else if (Array.isArray(ev["xaxis.range"])) start = indexFor(toMs(ev["xaxis.range"][0]));
    else return;  // y axis relayout / resize
    if (start === baseIdx) return;
    busy = true;
    Promise.resolve(applyRebase(start)).finally(() => { busy = false; });
  });
}

function buildPanel() {
  const target = {indexed: document.getElementById("listaIndexadas"),
                native: document.getElementById("listaNativas")};

  for (const g of DATA.groups) {
    for (const kind of ["indexed", "native"]) {
      const inGroup = DATA.series.filter(s => s.group === g.id && s.axis === kind);
      if (!inGroup.length) continue;

      const tit = document.createElement("div");
      tit.className = "grupo-tit";
      tit.textContent = g.name;
      target[kind].appendChild(tit);

      for (const s of inGroup) {
        const lab = document.createElement("label");
        lab.className = "item";
        lab.dataset.key = s.key;
        lab.title = `fonte: ${s.source}`;

        const inp = document.createElement("input");
        inp.type = kind === "indexed" ? "checkbox" : "radio";
        if (kind === "native") inp.name = "osc";
        inp.value = s.key;

        const pip = document.createElement("span");
        pip.className = "pip";
        pip.style.background = s.color;

        const txt = document.createElement("span");
        txt.textContent = s.label;

        lab.append(inp, pip, txt);
        target[kind].appendChild(lab);

        if (kind === "indexed") {
          inp.addEventListener("change", () => {
            inp.checked ? sel.indexedKeys.add(s.key) : sel.indexedKeys.delete(s.key);
            draw();
          });
        } else {
          // click, not change: a checked radio doesn't fire change, so there'd be
          // no way to turn the oscillator off by clicking it again
          inp.addEventListener("click", () => {
            sel.oscillator = sel.oscillator === s.key ? null : s.key;
            if (!sel.oscillator) inp.checked = false;
            draw();
          });
        }
      }
    }
  }
}

function syncChecks() {
  for (const lab of document.querySelectorAll("label.item")) {
    const k = lab.dataset.key;
    const on = sel.indexedKeys.has(k) || sel.oscillator === k;
    lab.classList.toggle("on", on);
    const inp = lab.querySelector("input");
    if (inp.checked !== on) inp.checked = on;
  }
}

function saveSelection() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      indexedKeys: [...sel.indexedKeys], oscillator: sel.oscillator, log: logScale,
    }));
  } catch (_) { /* private mode, just skip persisting */ }
}

function defaultSelection() {
  sel.indexedKeys = new Set(DATA.series.filter(s => s.default && s.axis === "indexed").map(s => s.key));
  const osc = DATA.series.find(s => s.default && s.axis === "native");
  sel.oscillator = osc ? osc.key : null;
}

function loadSelection() {
  let raw = null;
  try { raw = JSON.parse(localStorage.getItem(LS_KEY) || "null"); } catch (_) {}
  const isValid = k => !!BY_KEY[k];
  if (raw && Array.isArray(raw.indexedKeys)) {
    sel.indexedKeys = new Set(raw.indexedKeys.filter(isValid));
    sel.oscillator = isValid(raw.oscillator) ? raw.oscillator : null;
    logScale = raw.log !== false;
    if (sel.indexedKeys.size || sel.oscillator) return;
  }
  defaultSelection();
}

function bindControls() {
  const btnScale = document.getElementById("btnEscala");
  const syncScale = () => {
    btnScale.setAttribute("aria-pressed", String(logScale));
    btnScale.textContent = logScale ? "Escala log" : "Escala linear";
  };
  syncScale();
  btnScale.addEventListener("click", () => {
    logScale = !logScale;
    syncScale(); saveSelection();
    if (!sel.indexedKeys.size) return;
    busy = true;
    Promise.resolve(Plotly.relayout(gd, {"yaxis.type": logScale ? "log" : "linear"}))
      .finally(() => { busy = false; });
  });

  document.getElementById("btnLimpar").addEventListener("click", () => {
    sel.indexedKeys.clear(); sel.oscillator = null; draw();
  });
  document.getElementById("btnPadrao").addEventListener("click", () => {
    defaultSelection(); draw();
  });
  document.getElementById("btnInsights").addEventListener("click", () => {
    document.getElementById("secInsights").scrollIntoView({behavior: "smooth"});
  });
}

fetch("data.json", {cache: "no-store"})
  .then(r => {
    if (!r.ok) throw new Error(`data.json: HTTP ${r.status}`);
    return r.json();
  })
  .then(d => {
    DATA = d;
    TS = d.dates.map(x => Date.parse(x + "T00:00:00Z"));
    BY_KEY = Object.fromEntries(d.series.map(s => [s.key, s]));
    document.getElementById("periodo").textContent =
      `${fmtDate(d.start)} – ${fmtDate(d.end)} · ${d.n.toLocaleString("pt-BR")} dias úteis · ` +
      `${d.series.length} séries · atualizado em ${fmtDate(d.generated_at.slice(0, 10))}`;
    buildPanel();
    loadSelection();
    bindControls();
    return draw().then(() => Insights.init(DATA, BY_KEY, TS));
  })
  .catch(e => {
    document.getElementById("erro").style.display = "block";
    document.getElementById("erro").textContent =
      `Falha ao carregar os dados.\n\n${e.message}\n\nRode:  python fetch_data.py`;
  });
