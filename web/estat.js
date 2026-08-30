"use strict";

/* ==========================================================================
   Motor estatistico. Sem dependencias, roda inteiro no navegador.

   Convencao: series podem ter nulos (a metrica ainda nao existia). Toda funcao
   trabalha sobre pares completos e devolve nulo onde nao da para calcular.
   ========================================================================== */

const Est = (() => {

/* ---------- basico ---------- */

const soma = a => a.reduce((s, v) => s + v, 0);
const media = a => (a.length ? soma(a) / a.length : NaN);

function variancia(a, amostral = true) {
  const n = a.length;
  if (n < (amostral ? 2 : 1)) return NaN;
  const m = media(a);
  return soma(a.map(v => (v - m) ** 2)) / (n - (amostral ? 1 : 0));
}
const desvio = (a, amostral = true) => Math.sqrt(variancia(a, amostral));

function quantil(ordenado, p) {
  const n = ordenado.length;
  if (!n) return NaN;
  const i = (n - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? ordenado[lo] : ordenado[lo] * (hi - i) + ordenado[hi] * (i - lo);
}

function pearson(x, y) {
  const n = x.length;
  if (n < 3) return NaN;
  const mx = media(x), my = media(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx, b = y[i] - my;
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  return (sxx <= 0 || syy <= 0) ? NaN : sxy / Math.sqrt(sxx * syy);
}

/** Postos com media em caso de empate (necessario para Spearman correto). */
function postos(a) {
  const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
  const r = new Array(a.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    const m = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[idx[k][1]] = m;
    i = j + 1;
  }
  return r;
}
const spearman = (x, y) => pearson(postos(x), postos(y));

/* ---------- algebra para OLS ---------- */

/** Inverte matriz quadrada por Gauss-Jordan com pivoteamento parcial. */
function inverte(M) {
  const n = M.length;
  const A = M.map((l, i) => [...l, ...Array.from({length: n}, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;   // singular
    [A[c], A[p]] = [A[p], A[c]];
    const piv = A[c][c];
    for (let j = 0; j < 2 * n; j++) A[c][j] /= piv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (!f) continue;
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map(l => l.slice(n));
}

/**
 * OLS de y sobre X (X ja deve incluir a coluna de 1s se houver intercepto).
 * Devolve coeficientes, erros-padrao, estatisticas t, residuos e R2.
 */
function ols(y, X) {
  const n = y.length, k = X[0].length;
  if (n <= k) return null;
  const XtX = Array.from({length: k}, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      Xty[a] += X[i][a] * y[i];
      for (let b = 0; b < k; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  const inv = inverte(XtX);
  if (!inv) return null;
  const beta = inv.map(l => soma(l.map((v, j) => v * Xty[j])));
  const resid = y.map((v, i) => v - soma(X[i].map((x, j) => x * beta[j])));
  const sse = soma(resid.map(r => r * r));
  const gl = n - k;
  const s2 = sse / gl;
  const se = inv.map((l, i) => Math.sqrt(Math.max(s2 * l[i], 0)));
  const my = media(y);
  const sst = soma(y.map(v => (v - my) ** 2));
  return {beta, se, t: beta.map((b, i) => b / se[i]), resid, r2: sst > 0 ? 1 - sse / sst : NaN, n, gl};
}

/* ---------- raiz unitaria e cointegracao ---------- */

/** Regra de Schwert para numero de defasagens do ADF. */
const defasagensADF = n => Math.max(1, Math.floor(12 * Math.pow(n / 100, 0.25)));

/**
 * ADF: Dy_t = a + g*y_{t-1} + sum(d_i * Dy_{t-i}) + u.
 * Devolve a estatistica t de g. `comConstante=false` para residuos de
 * cointegracao, que ja tem media zero por construcao.
 */
function adf(y, defas = null, comConstante = true) {
  const p = defas == null ? defasagensADF(y.length) : defas;
  const dy = [];
  for (let i = 1; i < y.length; i++) dy.push(y[i] - y[i - 1]);
  const lin = [], alvo = [];
  for (let t = p; t < dy.length; t++) {
    const l = [];
    if (comConstante) l.push(1);
    l.push(y[t]);                                  // y_{t-1} alinhado a dy[t]
    for (let i = 1; i <= p; i++) l.push(dy[t - i]);
    lin.push(l); alvo.push(dy[t]);
  }
  if (lin.length < p + 5) return null;
  const r = ols(alvo, lin);
  if (!r) return null;
  const iG = comConstante ? 1 : 0;
  return {tEstat: r.t[iG], gama: r.beta[iG], n: r.n, defas: p};
}

// Dickey-Fuller com constante (assintoticos)
const CV_ADF = {1: -3.43, 5: -2.86, 10: -2.57};
// Engle-Granger, 2 variaveis, regressao com constante (MacKinnon)
const CV_EG = {1: -3.90, 5: -3.34, 10: -3.04};

const classifica = (t, cv) =>
  t < cv[1] ? "1%" : t < cv[5] ? "5%" : t < cv[10] ? "10%" : null;

/**
 * Engle-Granger em dois passos. Pre-requisito: ambas as series I(1), isto e,
 * NAO estacionarias em nivel. Se alguma ja for estacionaria, o teste de
 * cointegracao nao se aplica e devolvemos esse aviso em vez de um numero.
 */
function cointegracao(x, y) {
  const adfX = adf(x), adfY = adf(y);
  if (!adfX || !adfY) return null;
  const estX = adfX.tEstat < CV_ADF[5], estY = adfY.tEstat < CV_ADF[5];

  const reg = ols(y, x.map(v => [1, v]));
  if (!reg) return null;
  const adfR = adf(reg.resid, null, false);
  if (!adfR) return null;

  return {
    adfX: adfX.tEstat, adfY: adfY.tEstat, estX, estY,
    aplicavel: !estX && !estY,
    alfa: reg.beta[0], beta: reg.beta[1], r2: reg.r2,
    tResiduo: adfR.tEstat, defas: adfR.defas,
    nivel: classifica(adfR.tEstat, CV_EG),
    residuos: reg.resid,
  };
}

/* ---------- transformacoes ---------- */

/**
 * Retornos. Series estritamente positivas usam log-retorno; series que passam
 * por zero ou negativo (fluxo liquido, curva de juros, ROI) usam a primeira
 * diferenca, porque log de numero nao positivo nao existe.
 */
function retornos(v) {
  const positiva = v.every(x => x == null || x > 0);
  const out = [null];
  for (let i = 1; i < v.length; i++) {
    const a = v[i - 1], b = v[i];
    out.push(a == null || b == null ? null : (positiva ? Math.log(b / a) : b - a));
  }
  return {valores: out, tipo: positiva ? "log" : "dif"};
}

/** Pares (x,y) sem nulos dentro de [ini, fim]. */
function paresValidos(x, y, ini, fim) {
  const a = [], b = [], idx = [];
  for (let i = ini; i <= fim; i++) {
    if (x[i] == null || y[i] == null || !isFinite(x[i]) || !isFinite(y[i])) continue;
    a.push(x[i]); b.push(y[i]); idx.push(i);
  }
  return {x: a, y: b, idx};
}

/* ---------- metodos ---------- */

/** Correlacao movel entre retornos. fn = pearson ou spearman. */
function corrMovel(rx, ry, janela, ini, fim, fn = pearson) {
  const out = new Array(rx.length).fill(null);
  for (let t = ini; t <= fim; t++) {
    const a = [], b = [];
    for (let i = t - janela + 1; i <= t; i++) {
      if (i < 0 || rx[i] == null || ry[i] == null) continue;
      a.push(rx[i]); b.push(ry[i]);
    }
    if (a.length >= Math.max(10, janela * 0.6)) out[t] = fn(a, b);
  }
  return out;
}

/**
 * Correlacao cruzada defasada. k > 0 significa que X antecipa Y:
 * comparamos X em t contra Y em t+k.
 */
function leadLag(rx, ry, maxK, ini, fim) {
  const res = [];
  for (let k = -maxK; k <= maxK; k++) {
    const a = [], b = [];
    for (let t = ini; t <= fim; t++) {
      const u = t, w = t + k;
      if (w < 0 || w >= ry.length) continue;
      if (rx[u] == null || ry[w] == null) continue;
      a.push(rx[u]); b.push(ry[w]);
    }
    res.push({k, r: a.length >= 30 ? pearson(a, b) : null, n: a.length});
  }
  return res;
}

/** Beta e R2 de ry sobre rx (retornos). */
function betaOLS(rx, ry, ini, fim) {
  const p = paresValidos(rx, ry, ini, fim);
  if (p.x.length < 10) return null;
  const r = ols(p.y, p.x.map(v => [1, v]));
  if (!r) return null;
  return {alfa: r.beta[0], beta: r.beta[1], seBeta: r.se[1], tBeta: r.t[1],
          r2: r.r2, n: r.n, corr: pearson(p.x, p.y)};
}

function zscore(v, ini, fim) {
  const amostra = [];
  for (let i = ini; i <= fim; i++) if (v[i] != null) amostra.push(v[i]);
  const m = media(amostra), s = desvio(amostra);
  const out = new Array(v.length).fill(null);
  for (let i = ini; i <= fim; i++) if (v[i] != null) out[i] = (v[i] - m) / s;
  return {serie: out, media: m, desvio: s};
}

function percentil(v, ini, fim) {
  const amostra = [];
  for (let i = ini; i <= fim; i++) if (v[i] != null) amostra.push(v[i]);
  amostra.sort((a, b) => a - b);
  const out = new Array(v.length).fill(null);
  for (let i = ini; i <= fim; i++) {
    if (v[i] == null) continue;
    let lo = 0, hi = amostra.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (amostra[m] < v[i]) lo = m + 1; else hi = m; }
    out[i] = (lo / (amostra.length - 1)) * 100;
  }
  return {serie: out, ordenada: amostra};
}

/**
 * Retornos futuros condicionais: divide a metrica em faixas por quantil e,
 * para cada faixa, mede o que o alvo fez nos `horizonte` dias uteis seguintes.
 * E o teste direto de "quando isso aconteceu antes, o que veio depois".
 */
function retornosCondicionais(metrica, alvo, horizonte, nFaixas, ini, fim) {
  const obs = [];
  for (let i = ini; i <= fim - horizonte; i++) {
    const m = metrica[i], a0 = alvo[i], a1 = alvo[i + horizonte];
    if (m == null || a0 == null || a1 == null || a0 <= 0) continue;
    obs.push({m, ret: (a1 / a0 - 1) * 100});
  }
  if (obs.length < nFaixas * 10) return null;

  const cortes = [];
  const ord = obs.map(o => o.m).sort((a, b) => a - b);
  for (let q = 1; q < nFaixas; q++) cortes.push(quantil(ord, q / nFaixas));

  const faixa = m => { let f = 0; while (f < cortes.length && m > cortes[f]) f++; return f; };
  const baldes = Array.from({length: nFaixas}, () => []);
  for (const o of obs) baldes[faixa(o.m)].push(o.ret);

  const lim = [ord[0], ...cortes, ord[ord.length - 1]];
  return {
    horizonte, n: obs.length,
    faixas: baldes.map((b, i) => {
      const s = [...b].sort((p, q) => p - q);
      return {
        de: lim[i], ate: lim[i + 1], n: b.length,
        mediana: quantil(s, .5), media: media(b),
        p25: quantil(s, .25), p75: quantil(s, .75),
        pctPositivo: b.length ? b.filter(r => r > 0).length / b.length * 100 : NaN,
      };
    }),
    faixaAtual: metrica[fim] != null ? faixa(metrica[fim]) : null,
    valorAtual: metrica[fim],
  };
}

/** Volatilidade realizada anualizada (252 dias uteis). */
function volatilidade(r, janela, ini, fim) {
  const out = new Array(r.length).fill(null);
  for (let t = ini; t <= fim; t++) {
    const a = [];
    for (let i = t - janela + 1; i <= t; i++) if (i >= 0 && r[i] != null) a.push(r[i]);
    if (a.length >= janela * 0.6) out[t] = desvio(a) * Math.sqrt(252) * 100;
  }
  return out;
}

function drawdown(v, ini, fim) {
  const out = new Array(v.length).fill(null);
  let pico = -Infinity, maxDD = 0, iMax = null;
  for (let i = ini; i <= fim; i++) {
    if (v[i] == null) continue;
    if (v[i] > pico) pico = v[i];
    out[i] = (v[i] / pico - 1) * 100;
    if (out[i] < maxDD) { maxDD = out[i]; iMax = i; }
  }
  return {serie: out, maxDD, iMax, atual: out[fim]};
}

return {media, desvio, quantil, pearson, spearman, postos, ols, adf, cointegracao,
        retornos, paresValidos, corrMovel, leadLag, betaOLS, zscore, percentil,
        retornosCondicionais, volatilidade, drawdown, CV_ADF, CV_EG};
})();
