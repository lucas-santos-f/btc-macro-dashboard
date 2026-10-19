"use strict";

// Stats engine, no dependencies, runs entirely in the browser. Series can have
// nulls (the metric didn't exist yet), so everything works on complete pairs
// and returns null/NaN where it can't compute.

const Stats = (() => {
const sum = a => a.reduce((s, v) => s + v, 0);
const mean = a => (a.length ? sum(a) / a.length : NaN);

function variance(a, isSample = true) {
  const n = a.length;
  if (n < (isSample ? 2 : 1)) return NaN;
  const m = mean(a);
  return sum(a.map(v => (v - m) ** 2)) / (n - (isSample ? 1 : 0));
}
const stdev = (a, isSample = true) => Math.sqrt(variance(a, isSample));

function quantile(sorted, p) {
  const n = sorted.length;
  if (!n) return NaN;
  const i = (n - 1) * p, lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] * (hi - i) + sorted[hi] * (i - lo);
}

function pearson(x, y) {
  const n = x.length;
  if (n < 3) return NaN;
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx, b = y[i] - my;
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  return (sxx <= 0 || syy <= 0) ? NaN : sxy / Math.sqrt(sxx * syy);
}

// ties get the average rank, otherwise spearman is off on flat stretches
function ranks(a) {
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
const spearman = (x, y) => pearson(ranks(x), ranks(y));

// Gauss-Jordan with partial pivoting. k is 2 or 3 here, no need for anything smarter
function invert(M) {
  const n = M.length;
  const A = M.map((l, i) => [...l, ...Array.from({length: n}, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;  // singular
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

// X must already carry the column of 1s if there's an intercept
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
  const inv = invert(XtX);
  if (!inv) return null;
  const beta = inv.map(l => sum(l.map((v, j) => v * Xty[j])));
  const resid = y.map((v, i) => v - sum(X[i].map((x, j) => x * beta[j])));
  const sse = sum(resid.map(r => r * r));
  const dof = n - k;
  const s2 = sse / dof;
  const se = inv.map((l, i) => Math.sqrt(Math.max(s2 * l[i], 0)));
  const my = mean(y);
  const sst = sum(y.map(v => (v - my) ** 2));
  return {beta, se, t: beta.map((b, i) => b / se[i]), resid, r2: sst > 0 ? 1 - sse / sst : NaN, n, dof};
}

// Schwert rule
const adfLags = n => Math.max(1, Math.floor(12 * Math.pow(n / 100, 0.25)));

// dy_t = a + g*y_{t-1} + sum(d_i * dy_{t-i}) + u, returns the t stat of g.
// withConst=false for cointegration residuals (zero mean by construction)
function adf(y, lags = null, withConst = true) {
  const p = lags == null ? adfLags(y.length) : lags;
  const dy = [];
  for (let i = 1; i < y.length; i++) dy.push(y[i] - y[i - 1]);
  const design = [], target = [];
  for (let t = p; t < dy.length; t++) {
    const l = [];
    if (withConst) l.push(1);
    l.push(y[t]);  // y_{t-1} lines up with dy[t]
    for (let i = 1; i <= p; i++) l.push(dy[t - i]);
    design.push(l); target.push(dy[t]);
  }
  if (design.length < p + 5) return null;
  const r = ols(target, design);
  if (!r) return null;
  const iGamma = withConst ? 1 : 0;
  return {tStat: r.t[iGamma], gamma: r.beta[iGamma], n: r.n, lags: p};
}

// Dickey-Fuller with constant, asymptotic
const CV_ADF = {1: -3.43, 5: -2.86, 10: -2.57};

// Engle-Granger, 2 variables, with constant (MacKinnon)
const CV_EG = {1: -3.90, 5: -3.34, 10: -3.04};

const classify = (t, cv) =>
  t < cv[1] ? "1%" : t < cv[5] ? "5%" : t < cv[10] ? "10%" : null;

// Engle-Granger two step. Only meaningful if BOTH series are I(1); if either
// is already stationary in levels the caller gets applicable=false instead of
// a number that looks like an answer
function cointegration(x, y) {
  const adfX = adf(x), adfY = adf(y);
  if (!adfX || !adfY) return null;
  const stationaryX = adfX.tStat < CV_ADF[5], stationaryY = adfY.tStat < CV_ADF[5];

  const reg = ols(y, x.map(v => [1, v]));
  if (!reg) return null;
  const adfR = adf(reg.resid, null, false);
  if (!adfR) return null;

  return {
    adfX: adfX.tStat, adfY: adfY.tStat, stationaryX, stationaryY,
    applicable: !stationaryX && !stationaryY,
    alpha: reg.beta[0], beta: reg.beta[1], r2: reg.r2,
    tResid: adfR.tStat, lags: adfR.lags,
    level: classify(adfR.tStat, CV_EG),
    residuals: reg.resid,
  };
}

// log returns for strictly positive series; anything that crosses zero (net
// flow, the yield curve, ROI) gets first differences, log of <= 0 doesn't exist
function returns(v) {
  const positive = v.every(x => x == null || x > 0);
  const out = [null];
  for (let i = 1; i < v.length; i++) {
    const a = v[i - 1], b = v[i];
    out.push(a == null || b == null ? null : (positive ? Math.log(b / a) : b - a));
  }
  return {values: out, kind: positive ? "log" : "diff"};
}

function validPairs(x, y, start, end) {
  const a = [], b = [], idx = [];
  for (let i = start; i <= end; i++) {
    if (x[i] == null || y[i] == null || !isFinite(x[i]) || !isFinite(y[i])) continue;
    a.push(x[i]); b.push(y[i]); idx.push(i);
  }
  return {x: a, y: b, idx};
}

function rollingCorr(rx, ry, win, start, end, fn = pearson) {
  const out = new Array(rx.length).fill(null);
  for (let t = start; t <= end; t++) {
    const a = [], b = [];
    for (let i = t - win + 1; i <= t; i++) {
      if (i < 0 || rx[i] == null || ry[i] == null) continue;
      a.push(rx[i]); b.push(ry[i]);
    }
    if (a.length >= Math.max(10, win * 0.6)) out[t] = fn(a, b);
  }
  return out;
}

// k > 0 means X leads: X at t against Y at t+k
function leadLag(rx, ry, maxK, start, end) {
  const res = [];
  for (let k = -maxK; k <= maxK; k++) {
    const a = [], b = [];
    for (let t = start; t <= end; t++) {
      const u = t, w = t + k;
      if (w < 0 || w >= ry.length) continue;
      if (rx[u] == null || ry[w] == null) continue;
      a.push(rx[u]); b.push(ry[w]);
    }
    res.push({k, r: a.length >= 30 ? pearson(a, b) : null, n: a.length});
  }
  return res;
}

function betaOLS(rx, ry, start, end) {
  const p = validPairs(rx, ry, start, end);
  if (p.x.length < 10) return null;
  const r = ols(p.y, p.x.map(v => [1, v]));
  if (!r) return null;
  return {alpha: r.beta[0], beta: r.beta[1], seBeta: r.se[1], tBeta: r.t[1],
          r2: r.r2, n: r.n, corr: pearson(p.x, p.y)};
}

function zscore(v, start, end) {
  const vals = [];
  for (let i = start; i <= end; i++) if (v[i] != null) vals.push(v[i]);
  const m = mean(vals), s = stdev(vals);
  const out = new Array(v.length).fill(null);
  for (let i = start; i <= end; i++) if (v[i] != null) out[i] = (v[i] - m) / s;
  return {series: out, mean: m, stdev: s};
}

function percentile(v, start, end) {
  const vals = [];
  for (let i = start; i <= end; i++) if (v[i] != null) vals.push(v[i]);
  vals.sort((a, b) => a - b);
  const out = new Array(v.length).fill(null);
  for (let i = start; i <= end; i++) {
    if (v[i] == null) continue;
    let lo = 0, hi = vals.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (vals[m] < v[i]) lo = m + 1; else hi = m; }
    out[i] = (lo / (vals.length - 1)) * 100;
  }
  return {series: out, sorted: vals};
}

// "last time this metric was here, what did the target do over the next N
// days". Quantile bins over the metric, return distribution per bin.
function conditionalReturns(metric, target, horizon, nBins, start, end) {
  const obs = [];
  for (let i = start; i <= end - horizon; i++) {
    const m = metric[i], a0 = target[i], a1 = target[i + horizon];
    if (m == null || a0 == null || a1 == null || a0 <= 0) continue;
    obs.push({m, ret: (a1 / a0 - 1) * 100});
  }
  if (obs.length < nBins * 10) return null;

  const cuts = [];
  const ord = obs.map(o => o.m).sort((a, b) => a - b);
  for (let q = 1; q < nBins; q++) cuts.push(quantile(ord, q / nBins));

  const binOf = m => { let f = 0; while (f < cuts.length && m > cuts[f]) f++; return f; };
  const buckets = Array.from({length: nBins}, () => []);
  for (const o of obs) buckets[binOf(o.m)].push(o.ret);

  const lim = [ord[0], ...cuts, ord[ord.length - 1]];
  return {
    horizon, n: obs.length,
    bins: buckets.map((b, i) => {
      const s = [...b].sort((p, q) => p - q);
      return {
        from: lim[i], to: lim[i + 1], n: b.length,
        median: quantile(s, .5), mean: mean(b),
        p25: quantile(s, .25), p75: quantile(s, .75),
        pctPositive: b.length ? b.filter(r => r > 0).length / b.length * 100 : NaN,
      };
    }),
    currentBin: metric[end] != null ? binOf(metric[end]) : null,
    currentValue: metric[end],
  };
}

// annualized with 252 business days
function volatility(r, win, start, end) {
  const out = new Array(r.length).fill(null);
  for (let t = start; t <= end; t++) {
    const a = [];
    for (let i = t - win + 1; i <= t; i++) if (i >= 0 && r[i] != null) a.push(r[i]);
    if (a.length >= win * 0.6) out[t] = stdev(a) * Math.sqrt(252) * 100;
  }
  return out;
}

function drawdown(v, start, end) {
  const out = new Array(v.length).fill(null);
  let peak = -Infinity, maxDD = 0, iMax = null;
  for (let i = start; i <= end; i++) {
    if (v[i] == null) continue;
    if (v[i] > peak) peak = v[i];
    out[i] = (v[i] / peak - 1) * 100;
    if (out[i] < maxDD) { maxDD = out[i]; iMax = i; }
  }
  return {series: out, maxDD, iMax, current: out[end]};
}

return {mean, stdev, quantile, pearson, spearman, ranks, ols, adf, cointegration,
        returns, validPairs, rollingCorr, leadLag, betaOLS, zscore, percentile,
        conditionalReturns, volatility, drawdown, CV_ADF, CV_EG};
})();
