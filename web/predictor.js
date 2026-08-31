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

  function render(data) {
    payload = data;
    const generated = new Date(data.gerado_em).toLocaleString("pt-BR");
    el("predictMeta").textContent = `Atualizado em ${generated} · ${data.cobertura.fontes_ok}/${data.cobertura.fontes_total} fontes · mercado ${data.cobertura.mercado_inicio}–${data.cobertura.mercado_fim}`;
    el("predictWarnings").innerHTML = (data.avisos || []).map(text => `<div class="predict-warning">${escapeHtml(text)}</div>`).join("");
    el("forecastCards").innerHTML = data.previsoes.map(renderForecast).join("");
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
    el("sourceStatus").innerHTML = `<table class="source-table"><thead><tr><th>Série</th><th>Fonte</th><th>Estado</th><th>Última data</th></tr></thead><tbody>${sources.map(([key, source]) =>
      `<tr><td>${escapeHtml(source.rotulo || key)}</td><td>${escapeHtml(source.fonte)}</td><td class="${source.ok ? "source-ok" : "source-fail"}">${source.ok ? "OK" : "Falhou"}</td><td>${escapeHtml(source.ultima_data || source.erro || "—")}</td></tr>`).join("")}</tbody></table>`;
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
