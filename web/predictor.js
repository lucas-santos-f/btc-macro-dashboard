"use strict";

(() => {
  let payload = null;
  const el = id => document.getElementById(id);
  const pct = value => value == null ? "—" : `${(value * 100).toFixed(1).replace(".", ",")}%`;
  const num = value => value == null ? "—" : Number(value).toLocaleString("pt-BR", {maximumFractionDigits: 2});
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function setView(name) {
    const predictor = name === "preditor";
    el("area").hidden = predictor;
    el("preditorView").hidden = !predictor;
    el("menuMercado").hidden = predictor;
    el("menuPreditor").hidden = !predictor;
    document.querySelectorAll(".side-tab").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.view === name));
    });
    document.querySelector("header .controles").style.visibility = predictor ? "hidden" : "visible";
    document.body.dataset.view = name;
    if (predictor) el("preditorView").focus({preventScroll: true});
  }

  function metricLine(metrics) {
    if (!metrics) return "Sem backtest suficiente";
    return `Brier ${num(metrics.brier)} · AUC ${num(metrics.auc)} · teste ${metrics.amostras_teste ?? "—"}`;
  }

  function renderForecast(item) {
    if (item.status !== "ok") {
      return `<article class="forecast-card insufficient">
        <div class="forecast-kind">${escapeHtml(item.tipo)} · ${escapeHtml(item.horizonte)}</div>
        <h4>${escapeHtml(item.titulo)}</h4>
        <div class="forecast-prob">Dados insuficientes</div>
        <p>${escapeHtml(item.motivo || `${item.amostras || 0} observações; evento raro ou cobertura incompleta.`)}</p>
      </article>`;
    }
    const drivers = (item.drivers || []).slice(0, 4).map(driver =>
      `<li class="${driver.impacto >= 0 ? "up" : "down"}">${escapeHtml(driver.nome.replaceAll("_", " "))}</li>`).join("");
    return `<article class="forecast-card">
      <div class="forecast-kind">${escapeHtml(item.tipo)} · ${escapeHtml(item.horizonte)}</div>
      <h4>${escapeHtml(item.titulo)}</h4>
      <div class="forecast-prob">${pct(item.probabilidade)}</div>
      <div class="forecast-range">intervalo ${pct(item.intervalo?.[0])}–${pct(item.intervalo?.[1])}</div>
      <p>${escapeHtml(item.descricao)}</p>
      <div class="metric-line">Qualidade ${escapeHtml(item.qualidade || "não classificada")} · ${metricLine(item.metricas)}</div>
      <details><summary>Fatores dominantes</summary><ul class="driver-list">${drivers}</ul></details>
    </article>`;
  }


  const chartLayout = (xTitle, yTitle) => ({
    margin: {l: 48, r: 16, t: 18, b: 42},
    paper_bgcolor: "transparent", plot_bgcolor: "transparent",
    font: {color: "#8b949e", size: 11},
    xaxis: {title: xTitle, gridcolor: "#21262d", zerolinecolor: "#30363d"},
    yaxis: {title: yTitle, gridcolor: "#21262d", zerolinecolor: "#30363d", range: [0, 1]},
    showlegend: true, legend: {orientation: "h", y: 1.15},
  });
  const chartConfig = {responsive: true, displayModeBar: false};

  function renderScenarios(data) {
    const scenarios = data.cenarios || [];
    el("scenarioGrid").innerHTML = scenarios.length ? scenarios.map(scenario => {
      const rows = Object.values(scenario.previsoes || {}).map(item =>
        `<li><span>${escapeHtml(item.titulo)} · ${escapeHtml(item.horizonte)}</span><b>${pct(item.probabilidade)}</b></li>`).join("");
      return `<article class="scenario-card"><strong>${escapeHtml(scenario.nome)}</strong>
        <p>${escapeHtml(scenario.premissa)}</p><ul>${rows}</ul>
        <small>${escapeHtml(scenario.limitacao)}</small></article>`;
    }).join("") : `<p class="empty-state">Sem intervalos suficientes para cenários.</p>`;
  }

  function renderSimilarEpisodes(data) {
    const episodes = data.episodios_similares || [];
    el("similarEpisodes").innerHTML = episodes.length ? episodes.map(item => {
      const outcome = item.desfecho_observado || {};
      return `<article class="episode-row">
        <div><strong>${escapeHtml(item.data)}</strong><small>similaridade ${pct(item.similaridade)}</small></div>
        <div><span>Recessão 12m: ${outcome.recessao_12m == null ? "—" : outcome.recessao_12m ? "sim" : "não"}</span>
        <span>S&P 6m: ${outcome.sp500_6m == null ? "—" : pct(outcome.sp500_6m)}</span>
        <span>BTC 6m: ${outcome.btc_6m == null ? "—" : pct(outcome.btc_6m)}</span></div>
      </article>`;
    }).join("") : `<p class="empty-state">Cobertura insuficiente para localizar episódios comparáveis.</p>`;
  }

  function renderCalibration(forecastId) {
    const item = (payload?.previsoes || []).find(row => row.id === forecastId);
    const points = item?.calibracao || [];
    if (!points.length) {
      Plotly.purge(el("calibrationChart"));
      el("calibrationSummary").textContent = "Sem pontos fora da amostra suficientes.";
      return;
    }
    Plotly.newPlot(el("calibrationChart"), [
      {x: [0, 1], y: [0, 1], mode: "lines", name: "calibração perfeita",
       line: {color: "#565f6b", dash: "dot"}},
      {x: points.map(row => row.previsto), y: points.map(row => row.observado),
       mode: "lines+markers", name: "observado", line: {color: "#F7931A"}},
    ], chartLayout("Probabilidade prevista", "Frequência observada"), chartConfig);
    el("calibrationSummary").textContent =
      `${item.titulo} · ${points.length} faixas · ${item.metricas?.amostras_teste || 0} previsões fora da amostra.`;
  }

  function renderHistory(seriesId) {
    const series = (payload?.historico_previsoes?.series || []).find(row => row.id === seriesId);
    const points = series?.pontos || [];
    if (!points.length) {
      Plotly.purge(el("historyChart"));
      el("historySummary").textContent = "O histórico será preenchido nas próximas execuções válidas.";
      return;
    }
    Plotly.newPlot(el("historyChart"), [{
      x: points.map(row => row.data), y: points.map(row => row.probabilidade),
      mode: "lines+markers", name: series.titulo, line: {color: "#58a6ff"},
    }], chartLayout("Execução", "Probabilidade"), chartConfig);
    el("historySummary").textContent = `${series.titulo} · ${points.length} registros persistidos.`;
  }

  function setupCharts(data) {
    const calibrated = (data.previsoes || []).filter(item => (item.calibracao || []).length);
    el("calibrationSelect").innerHTML = calibrated.map(item =>
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.titulo)} · ${escapeHtml(item.horizonte)}</option>`).join("");
    const history = data.historico_previsoes?.series || [];
    el("historySelect").innerHTML = history.map(item =>
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.titulo)}</option>`).join("");
    el("calibrationSelect").onchange = event => renderCalibration(event.target.value);
    el("historySelect").onchange = event => renderHistory(event.target.value);
    renderCalibration(calibrated[0]?.id);
    renderHistory(history[0]?.id);
  }

  function render(data) {
    payload = data;
    const generated = new Date(data.gerado_em).toLocaleString("pt-BR");
    el("predictMeta").textContent = `Atualizado em ${generated} · ${data.cobertura.fontes_ok}/${data.cobertura.fontes_total} fontes · mercado ${data.cobertura.mercado_inicio}–${data.cobertura.mercado_fim}`;
    el("predictWarnings").innerHTML = (data.avisos || []).map(text => `<div class="predict-warning">${escapeHtml(text)}</div>`).join("");
    el("forecastCards").innerHTML = data.previsoes.map(renderForecast).join("");
    renderScenarios(data);
    renderSimilarEpisodes(data);
    setupCharts(data);
    el("benchmarkList").innerHTML = data.benchmarks.map(item => `<article class="benchmark-row">
      <div><strong>${escapeHtml(item.nome)}</strong><small>${escapeHtml(item.horizonte)}</small></div>
      <b>${item.probabilidade != null ? pct(item.probabilidade) : item.valor != null ? num(item.valor) : item.status === "dados_insuficientes" ? "amostra insuficiente" : "—"}</b>
    </article>`).join("");
    el("indicatorGrid").innerHTML = data.indicadores.map(item => `<article class="indicator">
      <span>${escapeHtml(item.nome)}</span><strong>${num(item.valor)}${escapeHtml(item.unidade)}</strong>
      <small>${item.variacao == null ? "—" : `${item.variacao >= 0 ? "+" : ""}${num(item.variacao)}`} · ${escapeHtml(item.data)}</small>
    </article>`).join("");
    el("methodology").innerHTML = Object.entries(data.metodologia).map(([key, value]) =>
      `<div class="method-row"><strong>${escapeHtml(key.replaceAll("_", " "))}</strong><span>${escapeHtml(value)}</span></div>`).join("");
    const sources = [...Object.entries(data.fontes.mercado), ...Object.entries(data.fontes.macro)];
    el("sourceStatus").innerHTML = `<div class="source-table-wrap" role="region" aria-label="Estado e governança das fontes" tabindex="0"><table class="source-table"><thead><tr><th>Série</th><th>Fonte</th><th>Estado</th><th>Referência</th><th>Revisável</th></tr></thead><tbody>${sources.map(([key, source]) => {
      const governance = source.governanca || {};
      return `<tr><td>${escapeHtml(source.rotulo || key)}</td><td>${escapeHtml(source.fonte)}</td><td class="${source.ok ? "source-ok" : "source-fail"}">${source.ok ? "OK" : "Falhou"}</td><td>${escapeHtml(source.ultima_data || source.erro || "—")}</td><td>${governance.revisavel == null ? "—" : governance.revisavel ? "sim" : "não"}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }

  function aiContext() {
    if (!payload) return;
    const compact = {
      gerado_em: payload.gerado_em,
      cobertura: payload.cobertura,
      previsoes: payload.previsoes.map(x => ({alvo:x.titulo, horizonte:x.horizonte,
        probabilidade:x.probabilidade, intervalo:x.intervalo, metricas:x.metricas,
        fatores:(x.drivers || []).slice(0, 5)})),
      benchmarks: payload.benchmarks,
      cenarios: payload.cenarios,
      episodios_similares: payload.episodios_similares,
      comparacoes_historicas: payload.comparacoes_historicas,
      indicadores: payload.indicadores,
      avisos: payload.avisos,
    };
    const text = `Analise o quadro quantitativo abaixo. Separe fatos, inferências e incertezas; não trate correlação como causalidade nem dê recomendação financeira.\n\n${JSON.stringify(compact, null, 2)}`;
    navigator.clipboard.writeText(text).then(() => {
      [el("btnCopyAi"), el("btnCopyAiSide")].forEach(button => {
        const old = button.textContent; button.textContent = "Contexto copiado";
        setTimeout(() => button.textContent = old, 1800);
      });
    });
  }

  document.querySelectorAll(".side-tab").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  el("btnCopyAi").addEventListener("click", aiContext);
  el("btnCopyAiSide").addEventListener("click", aiContext);
  fetch("predictor.json", {cache:"no-store"}).then(response => {
    if (!response.ok) throw new Error(`predictor.json: HTTP ${response.status}`);
    return response.json();
  }).then(render).catch(error => {
    el("predictMeta").textContent = `Preditor ainda não gerado: ${error.message}. Rode python predictor.py.`;
  });
})();
