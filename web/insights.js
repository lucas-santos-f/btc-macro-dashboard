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
    id: "rolling_corr", name: "Correlação móvel (Pearson)", series: 2, win: true,
    technical: "Correlação de Pearson entre os log-retornos diários de A e B, calculada numa janela móvel. Varia de −1 a +1 e mede associação linear de curto prazo. Roda sobre retornos, não sobre preços, para evitar regressão espúria entre séries com tendência.",
    plain: "Mede se as duas coisas costumam subir e descer juntas no dia a dia. Perto de +1 andam juntas, perto de −1 andam ao contrário, perto de 0 não têm relação. A linha mostra como isso mudou ao longo do tempo — relação entre ativos não é fixa.",
  },
  {
    id: "rolling_spearman", name: "Correlação móvel (Spearman)", series: 2, win: true,
    technical: "Igual à anterior, mas sobre os postos dos retornos em vez dos valores. Captura qualquer relação monotônica, não só linear, e é robusta a outliers — relevante para BTC, cuja distribuição tem caudas pesadas.",
    plain: "Mesma ideia da correlação normal, mas resistente a dias absurdos. Um único crash de 30% distorce a correlação comum; esta aqui aguenta melhor porque olha a ordem dos movimentos, não o tamanho.",
  },
  {
    id: "lead_lag", name: "Lead/lag (quem antecipa quem)", series: 2, lags: true,
    technical: "Correlação cruzada entre os retornos de A em t e os de B em t+k, varrendo k. Pico em k>0 sugere que A antecipa B; em k<0, o contrário. É associação defasada, não causalidade — sem teste de Granger e sem controle de confundidores.",
    plain: "Testa se um dos dois se move primeiro e o outro só reage dias depois. A barra mais alta indica a defasagem mais provável. Cuidado: mesmo que apareça um padrão, isso não prova que um causa o outro — pode ser coincidência ou os dois reagindo a uma terceira coisa.",
  },
  {
    id: "beta", name: "Beta e R²", series: 2,
    technical: "Regressão OLS dos retornos de B sobre os de A. Beta é a sensibilidade (B varia beta% quando A varia 1%), R² é a fração da variância de B explicada por A, e o t do beta indica significância estatística.",
    plain: "Beta responde: quando A sobe 1%, quanto B costuma andar? R² responde: o quanto disso realmente se explica por A, de 0% a 100%. R² baixo com beta alto significa que a relação existe mas é barulhenta e pouco confiável.",
  },
  {
    id: "cointegration", name: "Cointegração (Engle-Granger)", series: 2, inLevels: true,
    technical: "Dois passos: ADF em nível para confirmar que ambas são I(1); depois OLS entre elas e ADF sobre os resíduos, comparando com os valores críticos de MacKinnon. Se os resíduos forem estacionários, existe relação de equilíbrio de longo prazo e desvios tendem a reverter.",
    plain: "Correlação vê se andam juntas no dia a dia; cointegração vê se estão amarradas no longo prazo, como um cachorro na coleira: ele se afasta, mas sempre volta. Se der positivo, distâncias grandes entre as duas tendem a se fechar. É um teste exigente e a maioria dos pares não passa.",
  },
  {
    id: "conditional", name: "Retornos futuros condicionais", series: 2, horizon: true, inLevels: true,
    technical: "Divide a métrica A em faixas por quantil e mede a distribuição dos retornos de B nos N dias úteis seguintes a cada observação. Mostra mediana, quartis e taxa de acerto por faixa. Amostras se sobrepõem, então os intervalos são otimistas e o número de ciclos independentes é pequeno.",
    plain: "Este é o de 'padrão que se repete'. Ele separa a história em faixas — por exemplo MVRV baixo, médio, alto — e mostra o que o Bitcoin fez nos meses seguintes em cada uma. A faixa onde estamos hoje vem destacada. Não é previsão: são poucos ciclos e o passado pode simplesmente não se repetir.",
  },
  {
    id: "zscore", name: "Z-score", series: 1, inLevels: true,
    technical: "Padroniza a série pela média e pelo desvio-padrão do período: z = (x − média) ÷ desvio. Indica quantos desvios o valor atual está do normal. Pressupõe distribuição razoavelmente estável, o que métricas em tendência estrutural violam.",
    plain: "Responde 'isso está esquisito?'. Zero é o normal do período, +2 é bem acima do normal, −2 bem abaixo. Serve para ver se uma métrica está em extremo histórico.",
  },
  {
    id: "percentile", name: "Percentil histórico", series: 1, inLevels: true,
    technical: "Posição de cada valor na distribuição empírica acumulada do período. Não assume normalidade, o que a torna mais adequada que o z-score para séries assimétricas como MVRV e volatilidade.",
    plain: "Diz em que posição da história o valor de hoje está, de 0 a 100. Percentil 95 significa que só 5% dos dias foram mais altos que hoje. Mais honesto que o z-score quando a métrica é torta.",
  },
  {
    id: "volatility", name: "Volatilidade realizada", series: 1, win: true,
    technical: "Desvio-padrão dos log-retornos na janela móvel, anualizado por √252. Mede risco realizado, olhando para trás — não é volatilidade implícita nem previsão.",
    plain: "Mede o quanto o preço balançou. Alto significa oscilações fortes, baixo significa calmaria. Períodos de calmaria costumam preceder movimentos grandes, mas não dizem para que lado.",
  },
  {
    id: "drawdown", name: "Drawdown", series: 1, inLevels: true,
    technical: "Queda percentual em relação ao máximo acumulado até cada data. Registra a profundidade e a duração das quedas dentro do período selecionado.",
    plain: "Mostra quanto o ativo está abaixo do próprio recorde. −50% significa que caiu pela metade desde o topo. É a medida mais direta da dor de segurar o ativo.",
  },
];
const byId = id => METHODS.find(m => m.id === id);

const el = id => document.getElementById(id);
const fmtD = iso => { const [a, m, d] = iso.split("-"); return `${d}/${m}/${a}`; };
const num = (v, c = 2) => (v == null || !isFinite(v)) ? "—"
  : v.toLocaleString("pt-BR", {minimumFractionDigits: c, maximumFractionDigits: c});

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
  const gr = el("insGrafico"), nm = el("insNumeros");
  nm.innerHTML = "";

  const warn = t => `<div class="aviso">${t}</div>`;
  const row = (r, k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const table = ls => `<table class="tab">${ls.join("")}</table>`;

  // under ~2 months of business days every number below is noise
  if (endIdx - startIdx < 40) {
    Plotly.purge(gr);
    gr.innerHTML = "";
    nm.innerHTML = warn("Período curto demais para estatística confiável. Escolha uma janela maior.");
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
      hovertemplate: "correlação %{y:.3f}<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: [-1, 1],
         title: {text: `corr. ${win}d`, font: {size: 10}}}}, CFG);
    const p = Stats.validPairs(rA, rB, startIdx, endIdx);
    nm.innerHTML = table([
      row(0, "Correlação no período inteiro", num(fn(p.x, p.y), 3)),
      row(0, "Média da janela móvel", num(Stats.mean(val), 3)),
      row(0, "Mínimo / Máximo", `${num(Math.min(...val), 3)} / ${num(Math.max(...val), 3)}`),
      row(0, "Valor mais recente", num(val[val.length - 1], 3)),
      row(0, "Dias usados", p.x.length.toLocaleString("pt-BR")),
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
      hovertemplate: "defasagem %{x}d<br>correlação %{y:.3f}<extra></extra>",
    }], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `← ${B.label} antecipa   |   ${A.label} antecipa →`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: "correlação", font: {size: 10}}}}, CFG);
    const zero = valid.find(r => r.k === 0);
    nm.innerHTML = table([
      row(0, "Pico de correlação", num(peak.r, 3)),
      row(0, "Defasagem do pico", `${peak.k} dias úteis`),
      row(0, "Leitura", peak.k === 0 ? "Movimento simultâneo"
        : peak.k > 0 ? `${A.label} tende a se mover ${peak.k}d antes`
                     : `${B.label} tende a se mover ${-peak.k}d antes`),
      row(0, "Correlação sem defasagem", num(zero ? zero.r : null, 3)),
    ]) + warn("Associação defasada não é causalidade. Com muitas defasagens testadas, algum pico aparece por acaso.");
    break;
  }

  case "beta": {
    const r = Stats.betaOLS(rA, rB, startIdx, endIdx);
    if (!r) { nm.innerHTML = warn("Dados insuficientes."); Plotly.purge(gr); gr.innerHTML = ""; break; }
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
      xaxis: {...LAYOUT.xaxis, title: {text: `retorno diário ${A.label} (%)`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: `retorno diário ${B.label} (%)`, font: {size: 10}}}}, CFG);
    // |t| > 1.96 ~ 5% two-sided, normal approximation is fine at these n
    const signif = Math.abs(r.tBeta) > 1.96;
    nm.innerHTML = table([
      row(0, "Beta", num(r.beta, 3)),
      row(0, "Erro-padrão / t", `${num(r.seBeta, 3)} / ${num(r.tBeta, 2)}`),
      row(0, "Significante a 5%", signif ? "sim" : "não"),
      row(0, "R²", `${num(r.r2 * 100, 1)}%`),
      row(0, "Correlação", num(r.corr, 3)),
      row(0, "Observações", r.n.toLocaleString("pt-BR")),
    ]) + warn(`Leitura: quando ${A.label} varia 1%, ${B.label} varia em média ${num(r.beta, 2)}%. `
      + `R² de ${num(r.r2 * 100, 1)}% indica que ${num(100 - r.r2 * 100, 1)}% do movimento vem de outros fatores.`);
    break;
  }

  case "cointegration": {
    const p = Stats.validPairs(A.values, B.values, startIdx, endIdx);
    const c = Stats.cointegration(p.x, p.y);
    if (!c) { nm.innerHTML = warn("Dados insuficientes."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const dts = p.idx.map(i => D.dates[i]);
    const sd = Stats.stdev(c.residuals);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: dts, y: c.residuals.map(v => v / sd),
      line: {color: "#F7931A", width: 1.4},
      hovertemplate: "desvio %{y:.2f}σ<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis,
         title: {text: "resíduo (desvios-padrão)", font: {size: 10}}}}, CFG);
    const ls = [
      row(0, `ADF em nível — ${A.label}`, `${num(c.adfX, 2)} (${c.stationaryX ? "estacionária" : "não estacionária"})`),
      row(0, `ADF em nível — ${B.label}`, `${num(c.adfY, 2)} (${c.stationaryY ? "estacionária" : "não estacionária"})`),
      row(0, "Relação estimada", `${B.label} = ${num(c.alpha, 2)} + ${num(c.beta, 4)} × ${A.label}`),
      row(0, "R² da regressão", `${num(c.r2 * 100, 1)}%`),
      row(0, "ADF nos resíduos", `${num(c.tResid, 3)} (${c.lags} defasagens)`),
      row(0, "Valores críticos", `1%: ${Stats.CV_EG[1]} · 5%: ${Stats.CV_EG[5]} · 10%: ${Stats.CV_EG[10]}`),
      row(0, "Desvio atual", `${num(c.residuals[c.residuals.length - 1] / sd, 2)}σ`),
    ];
    let verdict;
    if (!c.applicable) {
      verdict = warn("Teste NÃO aplicável: cointegração pressupõe que ambas as séries sejam não estacionárias em nível. "
        + "Ao menos uma já é estacionária, então o resultado abaixo não tem interpretação válida.");
    } else if (c.level) {
      verdict = `<div class="ok">Cointegradas ao nível de ${c.level}. Existe relação de equilíbrio de longo prazo: `
        + `desvios grandes tenderam a se fechar dentro deste período.</div>`;
    } else {
      verdict = warn("Não cointegradas. Não se rejeita a hipótese de que os desvios sejam permanentes — "
        + "as duas séries podem andar juntas às vezes, mas não há âncora de longo prazo.");
    }
    nm.innerHTML = table(ls) + verdict;
    break;
  }

  case "conditional": {
    const r = Stats.conditionalReturns(A.values, B.values, horizon, nBins, startIdx, endIdx);
    if (!r) { nm.innerHTML = warn("Amostra insuficiente para esse horizonte."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const labels = r.bins.map((f, i) =>
      `${A.prefix}${num(f.from, A.decimals)}–${A.prefix}${num(f.to, A.decimals)}`);
    Plotly.react(gr, [
      {type: "bar", name: "mediana", x: labels, y: r.bins.map(f => f.median),
       marker: {color: r.bins.map((_, i) => i === r.currentBin ? "#F7931A" : "#30475e")},
       error_y: {type: "data", symmetric: false,
                 array: r.bins.map(f => f.p75 - f.median),
                 arrayminus: r.bins.map(f => f.median - f.p25),
                 color: "#8b949e", thickness: 1, width: 4},
       hovertemplate: "%{x}<br>mediana %{y:.1f}%<extra></extra>"},
    ], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `faixa de ${A.label}`, font: {size: 10}}, type: "category"},
      yaxis: {...LAYOUT.yaxis, title: {text: `retorno de ${B.label} em ${horizon}d (%)`, font: {size: 10}}}}, CFG);
    const ls = [`<tr><th>Faixa de ${A.label}</th><th>n</th><th>Mediana</th><th>Média</th><th>% positivo</th></tr>`];
    r.bins.forEach((f, i) => {
      ls.push(`<tr class="${i === r.currentBin ? "atual" : ""}">`
        + `<td>${labels[i]}${i === r.currentBin ? " <b>← hoje</b>" : ""}</td>`
        + `<td>${f.n}</td><td>${num(f.median, 1)}%</td><td>${num(f.mean, 1)}%</td>`
        + `<td>${num(f.pctPositive, 0)}%</td></tr>`);
    });
    nm.innerHTML = table(ls)
      + warn(`Hoje ${A.label} está em ${A.prefix}${num(r.currentValue, A.decimals)}${A.suffix}. `
        + `As janelas de ${horizon} dias se sobrepõem, então as observações não são independentes: `
        + `os ${r.n} pontos representam bem menos que ${r.n} casos distintos. Trate como contexto histórico, não como previsão.`);
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
      hovertemplate: (zs ? "%{y:.2f}σ" : "percentil %{y:.0f}") + "<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: zs ? null : [0, 100],
         title: {text: zs ? "desvios-padrão" : "percentile", font: {size: 10}}}}, CFG);
    const current = val[val.length - 1];
    nm.innerHTML = table([
      row(0, "Valor atual", `${A.prefix}${num(A.values[endIdx], A.decimals)}${A.suffix}`),
      row(0, zs ? "Z-score atual" : "Percentil atual", zs ? `${num(current, 2)}σ` : `${num(current, 0)}`),
      ...(zs ? [row(0, "Média do período", num(z.mean, A.decimals)),
                row(0, "Desvio-padrão", num(z.stdev, A.decimals))]
             : [row(0, "Mediana do período", num(Stats.quantile(z.sorted, .5), A.decimals)),
                row(0, "Faixa p5–p95", `${num(Stats.quantile(z.sorted, .05), A.decimals)} – ${num(Stats.quantile(z.sorted, .95), A.decimals)}`)]),
      row(0, "Leitura", zs
        ? (Math.abs(current) > 2 ? "Extremo (>2σ)" : Math.abs(current) > 1 ? "Fora do normal" : "Dentro do normal")
        : (current > 95 ? "Topo histórico do período" : current < 5 ? "Fundo histórico do período" : "Faixa intermediária")),
    ]);
    break;
  }

  case "volatility": {
    const s = Stats.volatility(rA, win, startIdx, endIdx);
    const y = clip(s), val = y.filter(v => v != null);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: slicedDates(), y,
      line: {color: "#FB923C", width: 1.5}, connectgaps: false,
      hovertemplate: "%{y:.1f}% a.a.<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: `vol. ${win}d anualizada (%)`, font: {size: 10}}}}, CFG);
    const ord = [...val].sort((a, b) => a - b);
    nm.innerHTML = table([
      row(0, "Volatilidade atual", `${num(val[val.length - 1], 1)}% a.a.`),
      row(0, "Mediana do período", `${num(Stats.quantile(ord, .5), 1)}% a.a.`),
      row(0, "Mínima / Máxima", `${num(ord[0], 1)}% / ${num(ord[ord.length - 1], 1)}%`),
      row(0, "Percentil atual", `${num(ord.filter(v => v < val[val.length - 1]).length / ord.length * 100, 0)}`),
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
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: "queda desde o topo (%)", font: {size: 10}}}}, CFG);
    nm.innerHTML = table([
      row(0, "Drawdown atual", `${num(d.current, 1)}%`),
      row(0, "Pior drawdown do período", `${num(d.maxDD, 1)}%`),
      row(0, "Data do pior", d.iMax != null ? fmtD(D.dates[d.iMax]) : "—"),
    ]);
    break;
  }
  }

  el("insExplica").innerHTML =
    `<div class="exp"><div class="exp-tit">Como funciona</div><p>${m.technical}</p></div>` +
    `<div class="exp"><div class="exp-tit">Em palavras simples</div><p>${m.plain}</p></div>`;
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
  const c = el("insMetodos");
  c.innerHTML = "";
  for (const m of METHODS) {
    const b = document.createElement("button");
    b.className = "met"; b.dataset.id = m.id; b.textContent = m.name;
    b.addEventListener("click", () => { method = m.id; syncControls(); compute(); });
    c.appendChild(b);
  }
}

function syncControls() {
  const m = byId(method);
  for (const b of document.querySelectorAll(".met"))
    b.setAttribute("aria-pressed", String(b.dataset.id === method));
  el("linhaB").style.display = m.series === 2 ? "" : "none";
  el("rotuloB").textContent = method === "conditional" ? "Alvo (o que aconteceu depois)" : "Série B";
  el("rotuloA").textContent = method === "conditional" ? "Condição (métrica de referência)" : "Série A";
  el("linhaJanela").style.display = m.win ? "" : "none";
  el("linhaHorizonte").style.display = m.horizon ? "" : "none";
  el("linhaDefas").style.display = m.lags ? "" : "none";
  el("insNivel").textContent = m.inLevels
    ? "Este método opera sobre os valores em nível."
    : "Este método opera sobre retornos diários, não sobre preços.";
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
  el("insPeriodo").innerHTML =
    `<b>${fmtD(D.dates[startIdx])}</b> a <b>${fmtD(D.dates[endIdx])}</b> · ` +
    `${(endIdx - startIdx + 1).toLocaleString("pt-BR")} dias úteis`;
}

// reuse whatever range is zoomed in the main chart above
function useChartWindow() {
  const g = document.getElementById("grafico");
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
  el("inpJanela").addEventListener("change", e => {
    win = Math.max(10, Math.min(730, +e.target.value || 90)); e.target.value = win; compute();
  });
  el("inpHorizonte").addEventListener("change", e => {
    horizon = Math.max(5, Math.min(500, +e.target.value || 90)); e.target.value = horizon; compute();
  });
  el("inpFaixas").addEventListener("change", e => {
    nBins = Math.max(2, Math.min(10, +e.target.value || 5)); e.target.value = nBins; compute();
  });
  el("inpDefas").addEventListener("change", e => {
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
