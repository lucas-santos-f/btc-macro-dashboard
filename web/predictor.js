"use strict";

(() => {
  let payload = null;
  const el = id => document.getElementById(id);
  const pct = value => value == null ? "—" : `${(value * 100).toFixed(1).replace(".", ",")}%`;
  const num = value => value == null ? "—" : Number(value).toLocaleString("pt-BR", {maximumFractionDigits: 2});
  // the payload uses english ids, the screen speaks portuguese
  const KIND = {macro: "macro", stress: "estresse", direction: "direção", inflation: "inflação", jobs: "emprego"};
  const QUALITY = {high: "alta", moderate: "moderada", low: "baixa"};
  const METHOD = {validation: "validação", ensemble: "ensemble", no_leakage: "anti-vazamento", revisions: "revisões",
    idempotency: "idempotência", metrics: "métricas", comparisons: "comparações"};
  const label = (map, key) => map[key] ?? String(key ?? "").replaceAll("_", " ");
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
    return `Brier ${num(metrics.brier)} · AUC ${num(metrics.auc)} · teste ${metrics.test_samples ?? "—"}`;
  }

  function renderForecast(item) {
    if (item.status !== "ok") {
      return `<article class="forecast-card insufficient">
        <div class="forecast-kind">${escapeHtml(label(KIND, item.kind))} · ${escapeHtml(item.horizon)}</div>
        <h4>${escapeHtml(item.title)}</h4>
        <div class="forecast-prob">Dados insuficientes</div>
        <p>${escapeHtml(item.reason || `${item.samples || 0} observações; evento raro ou cobertura incompleta.`)}</p>
      </article>`;
    }
    const drivers = (item.drivers || []).slice(0, 4).map(driver =>
      `<li class="${driver.impact >= 0 ? "up" : "down"}">${escapeHtml(driver.name.replaceAll("_", " "))}</li>`).join("");
    return `<article class="forecast-card">
      <div class="forecast-kind">${escapeHtml(label(KIND, item.kind))} · ${escapeHtml(item.horizon)}</div>
      <h4>${escapeHtml(item.title)}</h4>
      <div class="forecast-prob">${pct(item.probability)}</div>
      <div class="forecast-range">intervalo ${pct(item.interval?.[0])}–${pct(item.interval?.[1])}</div>
      <p>${escapeHtml(item.description)}</p>
      <div class="metric-line">Qualidade ${escapeHtml(item.quality ? label(QUALITY, item.quality) : "não classificada")} · ${metricLine(item.metrics)}</div>
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
    const scenarios = data.scenarios || [];
    el("scenarioGrid").innerHTML = scenarios.length ? scenarios.map(scenario => {
      const rows = Object.values(scenario.forecasts || {}).map(item =>
        `<li><span>${escapeHtml(item.title)} · ${escapeHtml(item.horizon)}</span><b>${pct(item.probability)}</b></li>`).join("");
      return `<article class="scenario-card"><strong>${escapeHtml(scenario.name)}</strong>
        <p>${escapeHtml(scenario.assumption)}</p><ul>${rows}</ul>
        <small>${escapeHtml(scenario.limitation)}</small></article>`;
    }).join("") : `<p class="empty-state">Sem intervalos suficientes para cenários.</p>`;
  }

  function renderSimilarEpisodes(data) {
    const episodes = data.similar_episodes || [];
    el("similarEpisodes").innerHTML = episodes.length ? episodes.map(item => {
      const outcome = item.observed_outcome || {};
      return `<article class="episode-row">
        <div><strong>${escapeHtml(item.date)}</strong><small>similaridade ${pct(item.similarity)}</small></div>
        <div><span>Recessão 12m: ${outcome.recession_12m == null ? "—" : outcome.recession_12m ? "sim" : "não"}</span>
        <span>S&P 6m: ${outcome.sp500_6m == null ? "—" : pct(outcome.sp500_6m)}</span>
        <span>BTC 6m: ${outcome.btc_6m == null ? "—" : pct(outcome.btc_6m)}</span></div>
      </article>`;
    }).join("") : `<p class="empty-state">Cobertura insuficiente para localizar episódios comparáveis.</p>`;
  }

  function renderCalibration(forecastId) {
    const item = (payload?.forecasts || []).find(row => row.id === forecastId);
    const points = item?.calibration || [];
    if (!points.length) {
      Plotly.purge(el("calibrationChart"));
      el("calibrationSummary").textContent = "Sem pontos fora da amostra suficientes.";
      return;
    }
    Plotly.newPlot(el("calibrationChart"), [
      {x: [0, 1], y: [0, 1], mode: "lines", name: "calibração perfeita",
       line: {color: "#565f6b", dash: "dot"}},
      {x: points.map(row => row.predicted), y: points.map(row => row.observed),
       mode: "lines+markers", name: "observado", line: {color: "#F7931A"}},
    ], chartLayout("Probabilidade prevista", "Frequência observada"), chartConfig);
    el("calibrationSummary").textContent =
      `${item.title} · ${points.length} faixas · ${item.metrics?.test_samples || 0} previsões fora da amostra.`;
  }

  function renderHistory(seriesId) {
    const series = (payload?.forecast_history?.series || []).find(row => row.id === seriesId);
    const points = series?.points || [];
    if (!points.length) {
      Plotly.purge(el("historyChart"));
      el("historySummary").textContent = "O histórico será preenchido nas próximas execuções válidas.";
      return;
    }
    Plotly.newPlot(el("historyChart"), [{
      x: points.map(row => row.date), y: points.map(row => row.probability),
      mode: "lines+markers", name: series.title, line: {color: "#58a6ff"},
    }], chartLayout("Execução", "Probabilidade"), chartConfig);
    el("historySummary").textContent = `${series.title} · ${points.length} registros persistidos.`;
  }

  function setupCharts(data) {
    const calibrated = (data.forecasts || []).filter(item => (item.calibration || []).length);
    el("calibrationSelect").innerHTML = calibrated.map(item =>
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)} · ${escapeHtml(item.horizon)}</option>`).join("");
    const history = data.forecast_history?.series || [];
    el("historySelect").innerHTML = history.map(item =>
      `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)}</option>`).join("");
    el("calibrationSelect").onchange = event => renderCalibration(event.target.value);
    el("historySelect").onchange = event => renderHistory(event.target.value);
    renderCalibration(calibrated[0]?.id);
    renderHistory(history[0]?.id);
  }

  function render(data) {
    payload = data;
    const generated = new Date(data.generated_at).toLocaleString("pt-BR");
    el("predictMeta").textContent = `Atualizado em ${generated} · ${data.coverage.sources_ok}/${data.coverage.sources_total} fontes · mercado ${data.coverage.market_start}–${data.coverage.market_end}`;
    el("predictWarnings").innerHTML = (data.warnings || []).map(text => `<div class="predict-warning">${escapeHtml(text)}</div>`).join("");
    el("forecastCards").innerHTML = data.forecasts.map(renderForecast).join("");
    renderScenarios(data);
    renderSimilarEpisodes(data);
    setupCharts(data);
    el("benchmarkList").innerHTML = data.benchmarks.map(item => `<article class="benchmark-row">
      <div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.horizon)}</small></div>
      <b>${item.probability != null ? pct(item.probability) : item.value != null ? num(item.value) : item.status === "insufficient_data" ? "amostra insuficiente" : "—"}</b>
    </article>`).join("");
    el("indicatorGrid").innerHTML = data.indicators.map(item => `<article class="indicator">
      <span>${escapeHtml(item.name)}</span><strong>${num(item.value)}${escapeHtml(item.unit)}</strong>
      <small>${item.change == null ? "—" : `${item.change >= 0 ? "+" : ""}${num(item.change)}`} · ${escapeHtml(item.date)}</small>
    </article>`).join("");
    el("methodology").innerHTML = Object.entries(data.methodology).map(([key, value]) =>
      `<div class="method-row"><strong>${escapeHtml(label(METHOD, key))}</strong><span>${escapeHtml(value)}</span></div>`).join("");
    const sources = [...Object.entries(data.sources.market), ...Object.entries(data.sources.macro)];
    el("sourceStatus").innerHTML = `<div class="source-table-wrap" role="region" aria-label="Estado e governança das fontes" tabindex="0"><table class="source-table"><thead><tr><th>Série</th><th>Fonte</th><th>Estado</th><th>Referência</th><th>Revisável</th></tr></thead><tbody>${sources.map(([key, source]) => {
      const governance = source.governance || {};
      return `<tr><td>${escapeHtml(source.label || key)}</td><td>${escapeHtml(source.source)}</td><td class="${source.ok ? "source-ok" : "source-fail"}">${source.ok ? "OK" : "Falhou"}</td><td>${escapeHtml(source.last_date || source.error || "—")}</td><td>${governance.revisable == null ? "—" : governance.revisable ? "sim" : "não"}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }

  function aiContext() {
    if (!payload) return;
    const compact = {
      generated_at: payload.generated_at,
      coverage: payload.coverage,
      forecasts: payload.forecasts.map(x => ({target:x.title, horizon:x.horizon,
        probability:x.probability, interval:x.interval, metrics:x.metrics,
        factors:(x.drivers || []).slice(0, 5)})),
      benchmarks: payload.benchmarks,
      scenarios: payload.scenarios,
      similar_episodes: payload.similar_episodes,
      historical_comparisons: payload.historical_comparisons,
      indicators: payload.indicators,
      warnings: payload.warnings,
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
