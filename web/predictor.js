"use strict";

(() => {
  let payload = null;
  const el = id => document.getElementById(id);
  const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
  const num = value => value == null ? "—" : Number(value).toLocaleString("en-US", {maximumFractionDigits: 2});
  const label = key => String(key ?? "").replaceAll("_", " ");
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  function setView(name) {
    const predictor = name === "predictor";
    el("area").hidden = predictor;
    el("predictorView").hidden = !predictor;
    el("menuMarket").hidden = predictor;
    el("menuPredictor").hidden = !predictor;
    document.querySelectorAll(".side-tab").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.view === name));
    });
    document.querySelector("header .controls").style.visibility = predictor ? "hidden" : "visible";
    document.body.dataset.view = name;
    if (predictor) el("predictorView").focus({preventScroll: true});
  }

  function metricLine(metrics) {
    if (!metrics) return "Not enough backtest";
    // raw_auc is the model before shrinking toward the base rate; overlapping
    // horizons make the test set look bigger than it is
    const eff = metrics.effective_samples;
    const independent = eff != null && eff < metrics.test_samples ? ` (~${Math.round(eff)} independent)` : "";
    return `Brier ${num(metrics.brier)} · AUC ${num(metrics.raw_auc ?? metrics.auc)} · test ${metrics.test_samples ?? "—"}${independent}`;
  }

  function renderForecast(item) {
    if (item.status !== "ok") {
      return `<article class="forecast-card insufficient">
        <div class="forecast-kind">${escapeHtml(label(item.kind))} · ${escapeHtml(item.horizon)}</div>
        <h4>${escapeHtml(item.title)}</h4>
        <div class="forecast-prob">Not enough data</div>
        <p>${escapeHtml(item.reason || `${item.samples || 0} observations; rare event or incomplete coverage.`)}</p>
      </article>`;
    }
    const drivers = (item.drivers || []).slice(0, 4).map(driver =>
      `<li class="${driver.impact >= 0 ? "up" : "down"}">${escapeHtml(driver.name.replaceAll("_", " "))}</li>`).join("");
    return `<article class="forecast-card">
      <div class="forecast-kind">${escapeHtml(label(item.kind))} · ${escapeHtml(item.horizon)}</div>
      <h4>${escapeHtml(item.title)}</h4>
      <div class="forecast-prob">${pct(item.probability)}</div>
      <div class="forecast-range">range ${pct(item.interval?.[0])}–${pct(item.interval?.[1])}</div>
      <p>${escapeHtml(item.description)}</p>
      <div class="metric-line">Quality ${escapeHtml(item.quality || "unrated")} · ${metricLine(item.metrics)}</div>
      <details><summary>Main drivers</summary><ul class="driver-list">${drivers}</ul></details>
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
    }).join("") : `<p class="empty-state">Not enough intervals for scenarios.</p>`;
  }

  function renderSimilarEpisodes(data) {
    const episodes = data.similar_episodes || [];
    el("similarEpisodes").innerHTML = episodes.length ? episodes.map(item => {
      const outcome = item.observed_outcome || {};
      return `<article class="episode-row">
        <div><strong>${escapeHtml(item.date)}</strong><small>similarity ${pct(item.similarity)}</small></div>
        <div><span>Recession 12m: ${outcome.recession_12m == null ? "—" : outcome.recession_12m ? "yes" : "no"}</span>
        <span>S&P 6m: ${outcome.sp500_6m == null ? "—" : pct(outcome.sp500_6m)}</span>
        <span>BTC 6m: ${outcome.btc_6m == null ? "—" : pct(outcome.btc_6m)}</span></div>
      </article>`;
    }).join("") : `<p class="empty-state">Not enough coverage to find comparable episodes.</p>`;
  }

  function renderCalibration(forecastId) {
    const item = (payload?.forecasts || []).find(row => row.id === forecastId);
    const points = item?.calibration || [];
    if (!points.length) {
      Plotly.purge(el("calibrationChart"));
      el("calibrationSummary").textContent = "Not enough out-of-sample points.";
      return;
    }
    Plotly.newPlot(el("calibrationChart"), [
      {x: [0, 1], y: [0, 1], mode: "lines", name: "perfect calibration",
       line: {color: "#565f6b", dash: "dot"}},
      {x: points.map(row => row.predicted), y: points.map(row => row.observed),
       mode: "lines+markers", name: "observed", line: {color: "#F7931A"}},
    ], chartLayout("Predicted probability", "Observed frequency"), chartConfig);
    el("calibrationSummary").textContent =
      `${item.title} · ${points.length} bins · ${item.metrics?.test_samples || 0} out-of-sample forecasts.`;
  }

  function renderHistory(seriesId) {
    const series = (payload?.forecast_history?.series || []).find(row => row.id === seriesId);
    const points = series?.points || [];
    if (!points.length) {
      Plotly.purge(el("historyChart"));
      el("historySummary").textContent = "The history fills in with the next valid runs.";
      return;
    }
    Plotly.newPlot(el("historyChart"), [{
      x: points.map(row => row.date), y: points.map(row => row.probability),
      mode: "lines+markers", name: series.title, line: {color: "#58a6ff"},
    }], chartLayout("Run", "Probability"), chartConfig);
    el("historySummary").textContent = `${series.title} · ${points.length} saved runs.`;
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
    const generated = new Date(data.generated_at).toLocaleString("en-US");
    el("predictMeta").textContent = `Updated ${generated} · ${data.coverage.sources_ok}/${data.coverage.sources_total} sources · market ${data.coverage.market_start}–${data.coverage.market_end}`;
    el("predictWarnings").innerHTML = (data.warnings || []).map(text => `<div class="predict-warning">${escapeHtml(text)}</div>`).join("");
    el("forecastCards").innerHTML = data.forecasts.map(renderForecast).join("");
    renderScenarios(data);
    renderSimilarEpisodes(data);
    setupCharts(data);
    el("benchmarkList").innerHTML = data.benchmarks.map(item => `<article class="benchmark-row">
      <div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.horizon)}</small></div>
      <b>${item.probability != null ? pct(item.probability) : item.value != null ? num(item.value) : item.status === "insufficient_data" ? "not enough sample" : "—"}</b>
    </article>`).join("");
    el("indicatorGrid").innerHTML = data.indicators.map(item => `<article class="indicator">
      <span>${escapeHtml(item.name)}</span><strong>${num(item.value)}${escapeHtml(item.unit)}</strong>
      <small>${item.change == null ? "—" : `${item.change >= 0 ? "+" : ""}${num(item.change)}`} · ${escapeHtml(item.date)}</small>
    </article>`).join("");
    el("methodology").innerHTML = Object.entries(data.methodology).map(([key, value]) =>
      `<div class="method-row"><strong>${escapeHtml(label(key))}</strong><span>${escapeHtml(value)}</span></div>`).join("");
    const sources = [...Object.entries(data.sources.market), ...Object.entries(data.sources.macro)];
    el("sourceStatus").innerHTML = `<div class="source-table-wrap" role="region" aria-label="Data sources and governance" tabindex="0"><table class="source-table"><thead><tr><th>Series</th><th>Source</th><th>Status</th><th>Reference</th><th>Revisable</th></tr></thead><tbody>${sources.map(([key, source]) => {
      const governance = source.governance || {};
      return `<tr><td>${escapeHtml(source.label || key)}</td><td>${escapeHtml(source.source)}</td><td class="${source.ok ? "source-ok" : "source-fail"}">${source.ok ? "OK" : "Failed"}</td><td>${escapeHtml(source.last_date || source.error || "—")}</td><td>${governance.revisable == null ? "—" : governance.revisable ? "yes" : "no"}</td></tr>`;
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
    const text = `Go through the quantitative picture below. Keep facts, inferences and uncertainty apart; don't treat correlation as causation and don't give financial advice.\n\n${JSON.stringify(compact, null, 2)}`;
    navigator.clipboard.writeText(text).then(() => {
      [el("btnCopyAi"), el("btnCopyAiSide")].forEach(button => {
        const old = button.textContent; button.textContent = "Context copied";
        setTimeout(() => button.textContent = old, 1800);
      });
    });
  }

  document.querySelectorAll(".side-tab").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  // deep link, e.g. index.html#predictor
  if (location.hash === "#predictor") setView("predictor");
  el("btnCopyAi").addEventListener("click", aiContext);
  el("btnCopyAiSide").addEventListener("click", aiContext);
  fetch("predictor.json", {cache:"no-store"}).then(response => {
    if (!response.ok) throw new Error(`predictor.json: HTTP ${response.status}`);
    return response.json();
  }).then(render).catch(error => {
    el("predictMeta").textContent = `No predictor output yet: ${error.message}. Run python predictor.py.`;
  });
})();
