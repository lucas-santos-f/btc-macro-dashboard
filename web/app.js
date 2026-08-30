"use strict";

/* ==========================================================================
   Dashboard BTC x Macro

   Dois paineis empilhados que COMPARTILHAM o mesmo eixo X (zoom sincronizado
   de graca):

     painel de cima  (y)  metricas de tendencia, normalizadas base 100 em
                          escala log. Multi-selecao: so faz sentido comparar
                          varias linhas entre si depois de normalizar.
     painel de baixo (y2) um oscilador de escala nativa. Selecao unica porque
                          MVRV (~0-4), Fear&Greed (0-100), juros (0-5%) e
                          fluxo em exchanges (+-1e9 USD) nao coexistem num
                          mesmo eixo sem que um esmague os outros.

   A base 100 e recalculada na borda esquerda da janela visivel a cada
   zoom/pan (rebase dinamico), o que elimina o vies de fixar uma data-base.
   ========================================================================== */

const gd = document.getElementById("grafico");
const CHAVE_LS = "btcmacro.selecao";

let DADOS, TS, PORCHAVE;
let travado = false;
let iniAtual = 0;
let escalaLog = true;
let traceMeta = [];              // paralelo aos traces do Plotly
let sel = {indexadas: new Set(), oscilador: null};

/* ---------- formatacao ---------- */

const fmtNum = (v, casas) =>
  v.toLocaleString("pt-BR", {minimumFractionDigits: casas, maximumFractionDigits: casas});

function fmtValor(v, s) {
  const a = Math.abs(v);
  // valores grandes ficam ilegiveis por extenso no tooltip
  if (a >= 1e12) return `${s.prefixo}${fmtNum(v / 1e12, 2)} tri${s.sufixo}`;
  if (a >= 1e9)  return `${s.prefixo}${fmtNum(v / 1e9, 2)} bi${s.sufixo}`;
  if (a >= 1e6)  return `${s.prefixo}${fmtNum(v / 1e6, 2)} mi${s.sufixo}`;
  return `${s.prefixo}${fmtNum(v, s.casas)}${s.sufixo}`;
}

const fmtData = iso => {
  const [a, m, d] = iso.split("-");
  return `${d}/${m}/${a}`;
};

/* ---------- eixo temporal ---------- */

function paraMs(v) {
  if (typeof v === "number") return v;
  const s = String(v).trim().replace(" ", "T");
  const t = Date.parse(s.endsWith("Z") ? s : s + "Z");
  return Number.isNaN(t) ? Date.parse(String(v)) : t;
}

/** Primeiro indice cuja data e >= t (busca binaria). */
function indicePara(t) {
  let lo = 0, hi = TS.length - 1;
  if (t <= TS[0]) return 0;
  if (t >= TS[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (TS[mid] < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* ---------- normalizacao ---------- */

/**
 * Valor de referencia da serie na borda esquerda da janela. Series que ainda
 * nao existiam naquela data (nulos a esquerda) sao normalizadas pelo primeiro
 * valor que de fato existe, para nao sumirem do grafico.
 */
function valorBase(s, ini) {
  for (let i = ini; i < s.valores.length; i++) {
    if (s.valores[i] != null && s.valores[i] > 0) return s.valores[i];
  }
  return null;
}

function serieIndexada(s, ini) {
  const base = valorBase(s, ini);
  if (base == null) return {y: s.valores.map(() => null), cd: s.valores.map(() => null)};
  return {
    y: s.valores.map(v => (v == null ? null : (v / base) * 100)),
    cd: s.valores.map(v => (v == null ? null : [fmtValor(v, s), (v / base - 1) * 100])),
  };
}

const serieNativa = s => ({
  y: s.valores,
  cd: s.valores.map(v => (v == null ? null : [fmtValor(v, s), 0])),
});

/* ---------- montagem ---------- */

function traçosSelecionados() {
  const t = [], meta = [];
  for (const s of DADOS.series) {
    if (!sel.indexadas.has(s.key)) continue;
    const {y, cd} = serieIndexada(s, iniAtual);
    meta.push({serie: s, indexado: true});
    t.push({
      type: "scatter", mode: "lines", name: s.label, x: DADOS.datas, y,
      customdata: cd, yaxis: "y", line: {color: s.cor, width: 1.5},
      connectgaps: false,
      hovertemplate: "<b>%{customdata[0]}</b>  (%{customdata[1]:+.1f}%)<extra>%{fullData.name}</extra>",
    });
  }
  if (sel.oscilador) {
    const s = PORCHAVE[sel.oscilador];
    const {y, cd} = serieNativa(s);
    meta.push({serie: s, indexado: false});
    t.push({
      type: "scatter", mode: "lines", name: s.label, x: DADOS.datas, y,
      customdata: cd, yaxis: "y2", line: {color: s.cor, width: 1.5},
      connectgaps: false, showlegend: false,
      hovertemplate: "<b>%{customdata[0]}</b><extra>%{fullData.name}</extra>",
    });
  }
  traceMeta = meta;
  return t;
}

function montarLayout() {
  const temOsc = !!sel.oscilador;
  const eixo = {gridcolor: "#1c2128", linecolor: "#30363d", zeroline: false};

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
      ...eixo, type: "date", anchor: temOsc ? "y2" : "y",
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
      ...eixo, anchor: "x", type: escalaLog ? "log" : "linear",
      domain: temOsc ? [0.32, 1] : [0, 1],
      title: {text: "Índice (base 100)", font: {size: 11}},
    },
  };

  if (temOsc) {
    const s = PORCHAVE[sel.oscilador];
    layout.yaxis2 = {
      ...eixo, anchor: "x", domain: [0, 0.25],
      title: {text: s.label, font: {size: 11, color: s.cor}},
    };
  }
  return layout;
}

const CONFIG = {
  responsive: true, displaylogo: false, scrollZoom: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  toImageButtonOptions: {filename: "btc-macro", scale: 2},
};

function desenhar() {
  guardarSelecao();
  atualizarMarcadores();
  const t = traçosSelecionados();
  if (!t.length) {
    Plotly.purge(gd);
    gd.innerHTML = '<div id="vazio">Selecione ao menos uma série no painel à esquerda.</div>';
    document.getElementById("base").textContent = "";
    return Promise.resolve();
  }
  if (gd.querySelector("#vazio")) gd.innerHTML = "";
  travado = true;
  return Plotly.react(gd, t, montarLayout(), CONFIG)
    .then(() => { rotuloBase(); ligarRelayout(); })
    .finally(() => { travado = false; });
}

/* ---------- rebase ---------- */

function rotuloBase() {
  const el = document.getElementById("base");
  const n = sel.indexadas.size;
  el.innerHTML = n
    ? `Base <b>${fmtData(DADOS.datas[iniAtual])} = 100</b> · recalculada a cada ` +
      `zoom/pan · ${n} série${n > 1 ? "s" : ""} no painel superior` +
      (sel.oscilador ? ` · oscilador: <b>${PORCHAVE[sel.oscilador].label}</b>` : "")
    : (sel.oscilador ? `Oscilador: <b>${PORCHAVE[sel.oscilador].label}</b> (escala nativa)` : "");
}

function aplicarRebase(ini) {
  iniAtual = ini;
  const ys = [], cds = [];
  for (const m of traceMeta) {
    const r = m.indexado ? serieIndexada(m.serie, ini) : serieNativa(m.serie);
    ys.push(r.y); cds.push(r.cd);
  }
  rotuloBase();
  return Plotly.restyle(gd, {y: ys, customdata: cds});
}

let relayoutLigado = false;
function ligarRelayout() {
  if (relayoutLigado) return;
  relayoutLigado = true;
  gd.on("plotly_relayout", ev => {
    if (travado) return;
    let ini;
    if (ev["xaxis.autorange"] === true) ini = 0;
    else if (ev["xaxis.range[0]"] !== undefined) ini = indicePara(paraMs(ev["xaxis.range[0]"]));
    else if (Array.isArray(ev["xaxis.range"])) ini = indicePara(paraMs(ev["xaxis.range"][0]));
    else return;                       // relayout de eixo Y / resize: ignorar
    if (ini === iniAtual) return;
    travado = true;
    Promise.resolve(aplicarRebase(ini)).finally(() => { travado = false; });
  });
}

/* ---------- painel lateral ---------- */

function montarPainel() {
  const alvo = {indexado: document.getElementById("listaIndexadas"),
                nativo: document.getElementById("listaNativas")};

  for (const g of DADOS.grupos) {
    for (const tipo of ["indexado", "nativo"]) {
      const doGrupo = DADOS.series.filter(s => s.grupo === g.id && s.eixo === tipo);
      if (!doGrupo.length) continue;

      const tit = document.createElement("div");
      tit.className = "grupo-tit";
      tit.textContent = g.nome;
      alvo[tipo].appendChild(tit);

      for (const s of doGrupo) {
        const lab = document.createElement("label");
        lab.className = "item";
        lab.dataset.key = s.key;
        lab.title = `fonte: ${s.fonte}`;

        const inp = document.createElement("input");
        inp.type = tipo === "indexado" ? "checkbox" : "radio";
        if (tipo === "nativo") inp.name = "osc";
        inp.value = s.key;

        const pip = document.createElement("span");
        pip.className = "pip";
        pip.style.background = s.cor;

        const txt = document.createElement("span");
        txt.textContent = s.label;

        lab.append(inp, pip, txt);
        alvo[tipo].appendChild(lab);

        if (tipo === "indexado") {
          inp.addEventListener("change", () => {
            inp.checked ? sel.indexadas.add(s.key) : sel.indexadas.delete(s.key);
            desenhar();
          });
        } else {
          // "click" e nao "change": um radio ja marcado nao dispara change, e
          // sem isso nao daria para desligar o oscilador clicando nele de novo.
          inp.addEventListener("click", () => {
            sel.oscilador = sel.oscilador === s.key ? null : s.key;
            if (!sel.oscilador) inp.checked = false;
            desenhar();
          });
        }
      }
    }
  }
}

function atualizarMarcadores() {
  for (const lab of document.querySelectorAll("label.item")) {
    const k = lab.dataset.key;
    const on = sel.indexadas.has(k) || sel.oscilador === k;
    lab.classList.toggle("on", on);
    const inp = lab.querySelector("input");
    if (inp.checked !== on) inp.checked = on;
  }
}

/* ---------- persistencia ---------- */

function guardarSelecao() {
  try {
    localStorage.setItem(CHAVE_LS, JSON.stringify({
      indexadas: [...sel.indexadas], oscilador: sel.oscilador, log: escalaLog,
    }));
  } catch (_) { /* modo privado: seguir sem persistir */ }
}

function selecaoPadrao() {
  sel.indexadas = new Set(DADOS.series.filter(s => s.padrao && s.eixo === "indexado").map(s => s.key));
  const osc = DADOS.series.find(s => s.padrao && s.eixo === "nativo");
  sel.oscilador = osc ? osc.key : null;
}

function carregarSelecao() {
  let bruto = null;
  try { bruto = JSON.parse(localStorage.getItem(CHAVE_LS) || "null"); } catch (_) {}
  const valida = k => !!PORCHAVE[k];
  if (bruto && Array.isArray(bruto.indexadas)) {
    sel.indexadas = new Set(bruto.indexadas.filter(valida));
    sel.oscilador = valida(bruto.oscilador) ? bruto.oscilador : null;
    escalaLog = bruto.log !== false;
    if (sel.indexadas.size || sel.oscilador) return;
  }
  selecaoPadrao();
}

/* ---------- controles do topo ---------- */

function ligarControles() {
  const btnEscala = document.getElementById("btnEscala");
  const sincEscala = () => {
    btnEscala.setAttribute("aria-pressed", String(escalaLog));
    btnEscala.textContent = escalaLog ? "Escala log" : "Escala linear";
  };
  sincEscala();
  btnEscala.addEventListener("click", () => {
    escalaLog = !escalaLog;
    sincEscala(); guardarSelecao();
    if (!sel.indexadas.size) return;
    travado = true;
    Promise.resolve(Plotly.relayout(gd, {"yaxis.type": escalaLog ? "log" : "linear"}))
      .finally(() => { travado = false; });
  });

  document.getElementById("btnLimpar").addEventListener("click", () => {
    sel.indexadas.clear(); sel.oscilador = null; desenhar();
  });
  document.getElementById("btnPadrao").addEventListener("click", () => {
    selecaoPadrao(); desenhar();
  });
  document.getElementById("btnInsights").addEventListener("click", () => {
    document.getElementById("secInsights").scrollIntoView({behavior: "smooth"});
  });
}

/* ---------- boot ---------- */

fetch("data.json", {cache: "no-store"})
  .then(r => {
    if (!r.ok) throw new Error(`data.json: HTTP ${r.status}`);
    return r.json();
  })
  .then(d => {
    DADOS = d;
    TS = d.datas.map(x => Date.parse(x + "T00:00:00Z"));
    PORCHAVE = Object.fromEntries(d.series.map(s => [s.key, s]));
    document.getElementById("periodo").textContent =
      `${fmtData(d.inicio)} – ${fmtData(d.fim)} · ${d.n.toLocaleString("pt-BR")} dias úteis · ` +
      `${d.series.length} séries · atualizado em ${fmtData(d.gerado_em.slice(0, 10))}`;
    montarPainel();
    carregarSelecao();
    ligarControles();
    return desenhar().then(() => Insights.iniciar(DADOS, PORCHAVE, TS));
  })
  .catch(e => {
    document.getElementById("erro").style.display = "block";
    document.getElementById("erro").textContent =
      `Falha ao carregar os dados.\n\n${e.message}\n\nRode:  python fetch_data.py`;
  });
