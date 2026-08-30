"use strict";

/* ==========================================================================
   Secao de insights: escolhe duas series, um periodo e um metodo estatistico.

   Regra que atravessa tudo: correlacao e regressao rodam sobre RETORNOS, nunca
   sobre precos. Duas series com tendencia dao correlacao perto de 1 sem
   relacao nenhuma entre si (regressao espuria, Granger & Newbold 1974). Os
   metodos que operam em nivel — z-score, percentil, cointegracao, drawdown —
   estao marcados com `emNivel` e sao os unicos onde isso e correto.
   ========================================================================== */

const Insights = (() => {

let D, PC, TSi;
let serieA = "btc", serieB = "dxy", metodo = "corr_movel";
let iniIdx = 0, fimIdx = 0;
let janela = 90, horizonte = 90, nFaixas = 5, maxDefas = 60;

/* ---------- metodos ---------- */

const METODOS = [
  {
    id: "corr_movel", nome: "Correlação móvel (Pearson)", series: 2, janela: true,
    tecnica: "Correlação de Pearson entre os log-retornos diários de A e B, calculada numa janela móvel. Varia de −1 a +1 e mede associação linear de curto prazo. Roda sobre retornos, não sobre preços, para evitar regressão espúria entre séries com tendência.",
    leigo: "Mede se as duas coisas costumam subir e descer juntas no dia a dia. Perto de +1 andam juntas, perto de −1 andam ao contrário, perto de 0 não têm relação. A linha mostra como isso mudou ao longo do tempo — relação entre ativos não é fixa.",
  },
  {
    id: "corr_spearman", nome: "Correlação móvel (Spearman)", series: 2, janela: true,
    tecnica: "Igual à anterior, mas sobre os postos dos retornos em vez dos valores. Captura qualquer relação monotônica, não só linear, e é robusta a outliers — relevante para BTC, cuja distribuição tem caudas pesadas.",
    leigo: "Mesma ideia da correlação normal, mas resistente a dias absurdos. Um único crash de 30% distorce a correlação comum; esta aqui aguenta melhor porque olha a ordem dos movimentos, não o tamanho.",
  },
  {
    id: "lead_lag", nome: "Lead/lag (quem antecipa quem)", series: 2, defas: true,
    tecnica: "Correlação cruzada entre os retornos de A em t e os de B em t+k, varrendo k. Pico em k>0 sugere que A antecipa B; em k<0, o contrário. É associação defasada, não causalidade — sem teste de Granger e sem controle de confundidores.",
    leigo: "Testa se um dos dois se move primeiro e o outro só reage dias depois. A barra mais alta indica a defasagem mais provável. Cuidado: mesmo que apareça um padrão, isso não prova que um causa o outro — pode ser coincidência ou os dois reagindo a uma terceira coisa.",
  },
  {
    id: "beta", nome: "Beta e R²", series: 2,
    tecnica: "Regressão OLS dos retornos de B sobre os de A. Beta é a sensibilidade (B varia beta% quando A varia 1%), R² é a fração da variância de B explicada por A, e o t do beta indica significância estatística.",
    leigo: "Beta responde: quando A sobe 1%, quanto B costuma andar? R² responde: o quanto disso realmente se explica por A, de 0% a 100%. R² baixo com beta alto significa que a relação existe mas é barulhenta e pouco confiável.",
  },
  {
    id: "cointegracao", nome: "Cointegração (Engle-Granger)", series: 2, emNivel: true,
    tecnica: "Dois passos: ADF em nível para confirmar que ambas são I(1); depois OLS entre elas e ADF sobre os resíduos, comparando com os valores críticos de MacKinnon. Se os resíduos forem estacionários, existe relação de equilíbrio de longo prazo e desvios tendem a reverter.",
    leigo: "Correlação vê se andam juntas no dia a dia; cointegração vê se estão amarradas no longo prazo, como um cachorro na coleira: ele se afasta, mas sempre volta. Se der positivo, distâncias grandes entre as duas tendem a se fechar. É um teste exigente e a maioria dos pares não passa.",
  },
  {
    id: "condicional", nome: "Retornos futuros condicionais", series: 2, horizonte: true, emNivel: true,
    tecnica: "Divide a métrica A em faixas por quantil e mede a distribuição dos retornos de B nos N dias úteis seguintes a cada observação. Mostra mediana, quartis e taxa de acerto por faixa. Amostras se sobrepõem, então os intervalos são otimistas e o número de ciclos independentes é pequeno.",
    leigo: "Este é o de 'padrão que se repete'. Ele separa a história em faixas — por exemplo MVRV baixo, médio, alto — e mostra o que o Bitcoin fez nos meses seguintes em cada uma. A faixa onde estamos hoje vem destacada. Não é previsão: são poucos ciclos e o passado pode simplesmente não se repetir.",
  },
  {
    id: "zscore", nome: "Z-score", series: 1, emNivel: true,
    tecnica: "Padroniza a série pela média e pelo desvio-padrão do período: z = (x − média) ÷ desvio. Indica quantos desvios o valor atual está do normal. Pressupõe distribuição razoavelmente estável, o que métricas em tendência estrutural violam.",
    leigo: "Responde 'isso está esquisito?'. Zero é o normal do período, +2 é bem acima do normal, −2 bem abaixo. Serve para ver se uma métrica está em extremo histórico.",
  },
  {
    id: "percentil", nome: "Percentil histórico", series: 1, emNivel: true,
    tecnica: "Posição de cada valor na distribuição empírica acumulada do período. Não assume normalidade, o que a torna mais adequada que o z-score para séries assimétricas como MVRV e volatilidade.",
    leigo: "Diz em que posição da história o valor de hoje está, de 0 a 100. Percentil 95 significa que só 5% dos dias foram mais altos que hoje. Mais honesto que o z-score quando a métrica é torta.",
  },
  {
    id: "volatilidade", nome: "Volatilidade realizada", series: 1, janela: true,
    tecnica: "Desvio-padrão dos log-retornos na janela móvel, anualizado por √252. Mede risco realizado, olhando para trás — não é volatilidade implícita nem previsão.",
    leigo: "Mede o quanto o preço balançou. Alto significa oscilações fortes, baixo significa calmaria. Períodos de calmaria costumam preceder movimentos grandes, mas não dizem para que lado.",
  },
  {
    id: "drawdown", nome: "Drawdown", series: 1, emNivel: true,
    tecnica: "Queda percentual em relação ao máximo acumulado até cada data. Registra a profundidade e a duração das quedas dentro do período selecionado.",
    leigo: "Mostra quanto o ativo está abaixo do próprio recorde. −50% significa que caiu pela metade desde o topo. É a medida mais direta da dor de segurar o ativo.",
  },
];
const porId = id => METODOS.find(m => m.id === id);

/* ---------- utilidades ---------- */

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

const datasFatia = () => D.datas.slice(iniIdx, fimIdx + 1);
const fatia = arr => arr.slice(iniIdx, fimIdx + 1);

/* ---------- calculo + render ---------- */

function calcular() {
  const m = porId(metodo);
  const A = PC[serieA], B = PC[serieB];
  const rA = Est.retornos(A.valores).valores;
  const rB = B ? Est.retornos(B.valores).valores : null;
  const gr = el("insGrafico"), nm = el("insNumeros");
  nm.innerHTML = "";

  const aviso = t => `<div class="aviso">${t}</div>`;
  const linha = (r, k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const tabela = ls => `<table class="tab">${ls.join("")}</table>`;

  if (fimIdx - iniIdx < 40) {
    Plotly.purge(gr);
    gr.innerHTML = "";
    nm.innerHTML = aviso("Período curto demais para estatística confiável. Escolha uma janela maior.");
    return;
  }

  switch (metodo) {
  case "corr_movel":
  case "corr_spearman": {
    const fn = metodo === "corr_movel" ? Est.pearson : Est.spearman;
    const s = Est.corrMovel(rA, rB, janela, iniIdx, fimIdx, fn);
    const y = fatia(s);
    const val = y.filter(v => v != null);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: datasFatia(), y,
      line: {color: "#F7931A", width: 1.6}, connectgaps: false,
      hovertemplate: "correlação %{y:.3f}<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: [-1, 1],
         title: {text: `corr. ${janela}d`, font: {size: 10}}}}, CFG);
    const p = Est.paresValidos(rA, rB, iniIdx, fimIdx);
    nm.innerHTML = tabela([
      linha(0, "Correlação no período inteiro", num(fn(p.x, p.y), 3)),
      linha(0, "Média da janela móvel", num(Est.media(val), 3)),
      linha(0, "Mínimo / Máximo", `${num(Math.min(...val), 3)} / ${num(Math.max(...val), 3)}`),
      linha(0, "Valor mais recente", num(val[val.length - 1], 3)),
      linha(0, "Dias usados", p.x.length.toLocaleString("pt-BR")),
    ]);
    break;
  }

  case "lead_lag": {
    const res = Est.leadLag(rA, rB, maxDefas, iniIdx, fimIdx);
    const com = res.filter(r => r.r != null);
    const pico = com.reduce((a, b) => Math.abs(b.r) > Math.abs(a.r) ? b : a, com[0]);
    Plotly.react(gr, [{
      type: "bar", x: com.map(r => r.k), y: com.map(r => r.r),
      marker: {color: com.map(r => r.k === pico.k ? "#F7931A" : "#30475e")},
      hovertemplate: "defasagem %{x}d<br>correlação %{y:.3f}<extra></extra>",
    }], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `← ${B.label} antecipa   |   ${A.label} antecipa →`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: "correlação", font: {size: 10}}}}, CFG);
    const zero = com.find(r => r.k === 0);
    nm.innerHTML = tabela([
      linha(0, "Pico de correlação", num(pico.r, 3)),
      linha(0, "Defasagem do pico", `${pico.k} dias úteis`),
      linha(0, "Leitura", pico.k === 0 ? "Movimento simultâneo"
        : pico.k > 0 ? `${A.label} tende a se mover ${pico.k}d antes`
                     : `${B.label} tende a se mover ${-pico.k}d antes`),
      linha(0, "Correlação sem defasagem", num(zero ? zero.r : null, 3)),
    ]) + aviso("Associação defasada não é causalidade. Com muitas defasagens testadas, algum pico aparece por acaso.");
    break;
  }

  case "beta": {
    const r = Est.betaOLS(rA, rB, iniIdx, fimIdx);
    if (!r) { nm.innerHTML = aviso("Dados insuficientes."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const p = Est.paresValidos(rA, rB, iniIdx, fimIdx);
    const xs = p.x, ys = p.y;
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    Plotly.react(gr, [
      {type: "scattergl", mode: "markers", x: xs.map(v => v * 100), y: ys.map(v => v * 100),
       marker: {color: "#F7931A", size: 3, opacity: .35},
       hovertemplate: `${A.label} %{x:.2f}%<br>${B.label} %{y:.2f}%<extra></extra>`},
      {type: "scatter", mode: "lines",
       x: [minX * 100, maxX * 100],
       y: [(r.alfa + r.beta * minX) * 100, (r.alfa + r.beta * maxX) * 100],
       line: {color: "#e6edf3", width: 1.5}, hoverinfo: "skip"},
    ], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `retorno diário ${A.label} (%)`, font: {size: 10}}},
      yaxis: {...LAYOUT.yaxis, title: {text: `retorno diário ${B.label} (%)`, font: {size: 10}}}}, CFG);
    const signif = Math.abs(r.tBeta) > 1.96;
    nm.innerHTML = tabela([
      linha(0, "Beta", num(r.beta, 3)),
      linha(0, "Erro-padrão / t", `${num(r.seBeta, 3)} / ${num(r.tBeta, 2)}`),
      linha(0, "Significante a 5%", signif ? "sim" : "não"),
      linha(0, "R²", `${num(r.r2 * 100, 1)}%`),
      linha(0, "Correlação", num(r.corr, 3)),
      linha(0, "Observações", r.n.toLocaleString("pt-BR")),
    ]) + aviso(`Leitura: quando ${A.label} varia 1%, ${B.label} varia em média ${num(r.beta, 2)}%. `
      + `R² de ${num(r.r2 * 100, 1)}% indica que ${num(100 - r.r2 * 100, 1)}% do movimento vem de outros fatores.`);
    break;
  }

  case "cointegracao": {
    const p = Est.paresValidos(A.valores, B.valores, iniIdx, fimIdx);
    const c = Est.cointegracao(p.x, p.y);
    if (!c) { nm.innerHTML = aviso("Dados insuficientes."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const dts = p.idx.map(i => D.datas[i]);
    const sd = Est.desvio(c.residuos);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: dts, y: c.residuos.map(v => v / sd),
      line: {color: "#F7931A", width: 1.4},
      hovertemplate: "desvio %{y:.2f}σ<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis,
         title: {text: "resíduo (desvios-padrão)", font: {size: 10}}}}, CFG);
    const ls = [
      linha(0, `ADF em nível — ${A.label}`, `${num(c.adfX, 2)} (${c.estX ? "estacionária" : "não estacionária"})`),
      linha(0, `ADF em nível — ${B.label}`, `${num(c.adfY, 2)} (${c.estY ? "estacionária" : "não estacionária"})`),
      linha(0, "Relação estimada", `${B.label} = ${num(c.alfa, 2)} + ${num(c.beta, 4)} × ${A.label}`),
      linha(0, "R² da regressão", `${num(c.r2 * 100, 1)}%`),
      linha(0, "ADF nos resíduos", `${num(c.tResiduo, 3)} (${c.defas} defasagens)`),
      linha(0, "Valores críticos", `1%: ${Est.CV_EG[1]} · 5%: ${Est.CV_EG[5]} · 10%: ${Est.CV_EG[10]}`),
      linha(0, "Desvio atual", `${num(c.residuos[c.residuos.length - 1] / sd, 2)}σ`),
    ];
    let veredito;
    if (!c.aplicavel) {
      veredito = aviso("Teste NÃO aplicável: cointegração pressupõe que ambas as séries sejam não estacionárias em nível. "
        + "Ao menos uma já é estacionária, então o resultado abaixo não tem interpretação válida.");
    } else if (c.nivel) {
      veredito = `<div class="ok">Cointegradas ao nível de ${c.nivel}. Existe relação de equilíbrio de longo prazo: `
        + `desvios grandes tenderam a se fechar dentro deste período.</div>`;
    } else {
      veredito = aviso("Não cointegradas. Não se rejeita a hipótese de que os desvios sejam permanentes — "
        + "as duas séries podem andar juntas às vezes, mas não há âncora de longo prazo.");
    }
    nm.innerHTML = tabela(ls) + veredito;
    break;
  }

  case "condicional": {
    const r = Est.retornosCondicionais(A.valores, B.valores, horizonte, nFaixas, iniIdx, fimIdx);
    if (!r) { nm.innerHTML = aviso("Amostra insuficiente para esse horizonte."); Plotly.purge(gr); gr.innerHTML = ""; break; }
    const rot = r.faixas.map((f, i) =>
      `${A.prefixo}${num(f.de, A.casas)}–${A.prefixo}${num(f.ate, A.casas)}`);
    Plotly.react(gr, [
      {type: "bar", name: "mediana", x: rot, y: r.faixas.map(f => f.mediana),
       marker: {color: r.faixas.map((_, i) => i === r.faixaAtual ? "#F7931A" : "#30475e")},
       error_y: {type: "data", symmetric: false,
                 array: r.faixas.map(f => f.p75 - f.mediana),
                 arrayminus: r.faixas.map(f => f.mediana - f.p25),
                 color: "#8b949e", thickness: 1, width: 4},
       hovertemplate: "%{x}<br>mediana %{y:.1f}%<extra></extra>"},
    ], {...LAYOUT,
      xaxis: {...LAYOUT.xaxis, title: {text: `faixa de ${A.label}`, font: {size: 10}}, type: "category"},
      yaxis: {...LAYOUT.yaxis, title: {text: `retorno de ${B.label} em ${horizonte}d (%)`, font: {size: 10}}}}, CFG);
    const ls = [`<tr><th>Faixa de ${A.label}</th><th>n</th><th>Mediana</th><th>Média</th><th>% positivo</th></tr>`];
    r.faixas.forEach((f, i) => {
      ls.push(`<tr class="${i === r.faixaAtual ? "atual" : ""}">`
        + `<td>${rot[i]}${i === r.faixaAtual ? " <b>← hoje</b>" : ""}</td>`
        + `<td>${f.n}</td><td>${num(f.mediana, 1)}%</td><td>${num(f.media, 1)}%</td>`
        + `<td>${num(f.pctPositivo, 0)}%</td></tr>`);
    });
    nm.innerHTML = tabela(ls)
      + aviso(`Hoje ${A.label} está em ${A.prefixo}${num(r.valorAtual, A.casas)}${A.sufixo}. `
        + `As janelas de ${horizonte} dias se sobrepõem, então as observações não são independentes: `
        + `os ${r.n} pontos representam bem menos que ${r.n} casos distintos. Trate como contexto histórico, não como previsão.`);
    break;
  }

  case "zscore":
  case "percentil": {
    const z = metodo === "zscore" ? Est.zscore(A.valores, iniIdx, fimIdx)
                                  : Est.percentil(A.valores, iniIdx, fimIdx);
    const y = fatia(z.serie);
    const val = y.filter(v => v != null);
    const zs = metodo === "zscore";
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: datasFatia(), y,
      line: {color: A.cor, width: 1.5}, connectgaps: false,
      hovertemplate: (zs ? "%{y:.2f}σ" : "percentil %{y:.0f}") + "<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, range: zs ? null : [0, 100],
         title: {text: zs ? "desvios-padrão" : "percentil", font: {size: 10}}}}, CFG);
    const atual = val[val.length - 1];
    nm.innerHTML = tabela([
      linha(0, "Valor atual", `${A.prefixo}${num(A.valores[fimIdx], A.casas)}${A.sufixo}`),
      linha(0, zs ? "Z-score atual" : "Percentil atual", zs ? `${num(atual, 2)}σ` : `${num(atual, 0)}`),
      ...(zs ? [linha(0, "Média do período", num(z.media, A.casas)),
                linha(0, "Desvio-padrão", num(z.desvio, A.casas))]
             : [linha(0, "Mediana do período", num(Est.quantil(z.ordenada, .5), A.casas)),
                linha(0, "Faixa p5–p95", `${num(Est.quantil(z.ordenada, .05), A.casas)} – ${num(Est.quantil(z.ordenada, .95), A.casas)}`)]),
      linha(0, "Leitura", zs
        ? (Math.abs(atual) > 2 ? "Extremo (>2σ)" : Math.abs(atual) > 1 ? "Fora do normal" : "Dentro do normal")
        : (atual > 95 ? "Topo histórico do período" : atual < 5 ? "Fundo histórico do período" : "Faixa intermediária")),
    ]);
    break;
  }

  case "volatilidade": {
    const s = Est.volatilidade(rA, janela, iniIdx, fimIdx);
    const y = fatia(s), val = y.filter(v => v != null);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: datasFatia(), y,
      line: {color: "#FB923C", width: 1.5}, connectgaps: false,
      hovertemplate: "%{y:.1f}% a.a.<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: `vol. ${janela}d anualizada (%)`, font: {size: 10}}}}, CFG);
    const ord = [...val].sort((a, b) => a - b);
    nm.innerHTML = tabela([
      linha(0, "Volatilidade atual", `${num(val[val.length - 1], 1)}% a.a.`),
      linha(0, "Mediana do período", `${num(Est.quantil(ord, .5), 1)}% a.a.`),
      linha(0, "Mínima / Máxima", `${num(ord[0], 1)}% / ${num(ord[ord.length - 1], 1)}%`),
      linha(0, "Percentil atual", `${num(ord.filter(v => v < val[val.length - 1]).length / ord.length * 100, 0)}`),
    ]);
    break;
  }

  case "drawdown": {
    const d = Est.drawdown(A.valores, iniIdx, fimIdx);
    Plotly.react(gr, [{
      type: "scatter", mode: "lines", x: datasFatia(), y: fatia(d.serie),
      fill: "tozeroy", fillcolor: "rgba(248,81,73,.15)",
      line: {color: "#f85149", width: 1.3}, connectgaps: false,
      hovertemplate: "%{y:.1f}%<extra></extra>",
    }], {...LAYOUT, yaxis: {...LAYOUT.yaxis, title: {text: "queda desde o topo (%)", font: {size: 10}}}}, CFG);
    nm.innerHTML = tabela([
      linha(0, "Drawdown atual", `${num(d.atual, 1)}%`),
      linha(0, "Pior drawdown do período", `${num(d.maxDD, 1)}%`),
      linha(0, "Data do pior", d.iMax != null ? fmtD(D.datas[d.iMax]) : "—"),
    ]);
    break;
  }
  }

  el("insExplica").innerHTML =
    `<div class="exp"><div class="exp-tit">Como funciona</div><p>${m.tecnica}</p></div>` +
    `<div class="exp"><div class="exp-tit">Em palavras simples</div><p>${m.leigo}</p></div>`;
}

/* ---------- interface ---------- */

function montarSelects() {
  for (const sid of ["selA", "selB"]) {
    const s = el(sid);
    s.innerHTML = "";
    for (const g of D.grupos) {
      const og = document.createElement("optgroup");
      og.label = g.nome;
      for (const x of D.series.filter(v => v.grupo === g.id)) {
        const o = document.createElement("option");
        o.value = x.key; o.textContent = x.label;
        og.appendChild(o);
      }
      s.appendChild(og);
    }
  }
  el("selA").value = serieA;
  el("selB").value = serieB;
}

function montarMetodos() {
  const c = el("insMetodos");
  c.innerHTML = "";
  for (const m of METODOS) {
    const b = document.createElement("button");
    b.className = "met"; b.dataset.id = m.id; b.textContent = m.nome;
    b.addEventListener("click", () => { metodo = m.id; sincronizar(); calcular(); });
    c.appendChild(b);
  }
}

function sincronizar() {
  const m = porId(metodo);
  for (const b of document.querySelectorAll(".met"))
    b.setAttribute("aria-pressed", String(b.dataset.id === metodo));
  el("linhaB").style.display = m.series === 2 ? "" : "none";
  el("rotuloB").textContent = metodo === "condicional" ? "Alvo (o que aconteceu depois)" : "Série B";
  el("rotuloA").textContent = metodo === "condicional" ? "Condição (métrica de referência)" : "Série A";
  el("linhaJanela").style.display = m.janela ? "" : "none";
  el("linhaHorizonte").style.display = m.horizonte ? "" : "none";
  el("linhaDefas").style.display = m.defas ? "" : "none";
  el("insNivel").textContent = m.emNivel
    ? "Este método opera sobre os valores em nível."
    : "Este método opera sobre retornos diários, não sobre preços.";
}

function definirPeriodo(anos) {
  fimIdx = D.datas.length - 1;
  if (anos == null) { iniIdx = 0; }
  else {
    const alvo = new Date(TSi[fimIdx]);
    alvo.setFullYear(alvo.getFullYear() - anos);
    iniIdx = TSi.findIndex(t => t >= alvo.getTime());
    if (iniIdx < 0) iniIdx = 0;
  }
  mostrarPeriodo();
}

function mostrarPeriodo() {
  el("insPeriodo").innerHTML =
    `<b>${fmtD(D.datas[iniIdx])}</b> a <b>${fmtD(D.datas[fimIdx])}</b> · ` +
    `${(fimIdx - iniIdx + 1).toLocaleString("pt-BR")} dias úteis`;
}

/** Usa a janela que estiver visível no gráfico principal. */
function usarJanelaDoGrafico() {
  const g = document.getElementById("grafico");
  const r = g && g.layout && g.layout.xaxis && g.layout.xaxis.range;
  if (!r) { definirPeriodo(null); return; }
  const ms = v => (typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T") + "Z"));
  const a = ms(r[0]), b = ms(r[1]);
  let i = TSi.findIndex(t => t >= a); if (i < 0) i = 0;
  let f = TSi.length - 1;
  for (let k = TSi.length - 1; k >= 0; k--) if (TSi[k] <= b) { f = k; break; }
  if (f - i < 40) { definirPeriodo(null); return; }
  iniIdx = i; fimIdx = f;
  mostrarPeriodo();
}

function ligar() {
  el("selA").addEventListener("change", e => { serieA = e.target.value; calcular(); });
  el("selB").addEventListener("change", e => { serieB = e.target.value; calcular(); });
  el("inpJanela").addEventListener("change", e => {
    janela = Math.max(10, Math.min(730, +e.target.value || 90)); e.target.value = janela; calcular();
  });
  el("inpHorizonte").addEventListener("change", e => {
    horizonte = Math.max(5, Math.min(500, +e.target.value || 90)); e.target.value = horizonte; calcular();
  });
  el("inpFaixas").addEventListener("change", e => {
    nFaixas = Math.max(2, Math.min(10, +e.target.value || 5)); e.target.value = nFaixas; calcular();
  });
  el("inpDefas").addEventListener("change", e => {
    maxDefas = Math.max(5, Math.min(180, +e.target.value || 60)); e.target.value = maxDefas; calcular();
  });
  for (const b of document.querySelectorAll("[data-anos]")) {
    b.addEventListener("click", () => {
      const v = b.dataset.anos;
      v === "grafico" ? usarJanelaDoGrafico() : definirPeriodo(v === "max" ? null : +v);
      calcular();
    });
  }
}

function iniciar(dados, porChave, ts) {
  D = dados; PC = porChave; TSi = ts;
  montarSelects(); montarMetodos(); ligar();
  definirPeriodo(null);
  sincronizar();
  calcular();
}

return {iniciar};
})();
