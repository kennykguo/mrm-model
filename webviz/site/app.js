// MRM bank viewer: state, navigation between the views, controls and playback. Every view follows the time
// scrubber. Views (one at a time, chosen in the top bar; the address keeps the view):
//   scenario  js/scenario.js   what the run tests: description, inputs over time, timing, every input signal
//   die       js/scene3d.js, js/bank.js, js/spectrum.js, js/heatmap.js   the die at the scrubber frame
//   parts     js/parts.js      every quantity of one part over the whole run, one ring or all 8
//   stats     js/stats.js      statistics, time in each state and counts of the whole run
//   metrics   js/metrics.js    the engine's metrics, and the BER
//   limits    js/limits.js, js/constants.js   the hardware limits and how near the run comes to them
//   xtalk     js/xtalk.js      thermal and optical crosstalk

import { installMath } from './js/math.js';
import { getJSON, loadScenario, K0 } from './js/data.js';
import { SpectrumView } from './js/spectrum.js';
import { fmtNum } from './js/traces.js';
import { HeatmapView } from './js/heatmap.js';
import { BistabilityView } from './js/bistability.js';
import { MetricsView, BerView } from './js/metrics.js';
import { BankView } from './js/bank.js';
import { PartsView } from './js/parts.js';
import { ScenarioView } from './js/scenario.js';
import { StatsView } from './js/stats.js';
import { LimitsView } from './js/limits.js';
import { CrosstalkView } from './js/xtalk.js';
import { renderConstants, renderDistances } from './js/constants.js';
import { initThemeToggle, onThemeChange, token } from './js/theme.js';

installMath(document.body);

const BASE_FPS = 30;          // frames per second at 1x
const VIEWS = ['scenario', 'die', 'parts', 'stats', 'metrics', 'limits', 'xtalk'];
const ALL_VIEWS = new Set(['parts', 'stats', 'die', 'xtalk']);           // views with a bank-comparison / selected-ring toggle
const PARTS = ['ring', 'heater', 'pd', 'laser', 'ctrl', 'data'];
const $ = (s) => document.querySelector(s);
const state = {
  meta: null, S: null, frame: 0, ring: 0, all: true, view: 'scenario', part: 'ring', playing: false, speed: 1,
  colorMode: 'temp', mapMode: 'flat', hover: null,
};
const V = {};                 // the view objects
const dirty = new Set(VIEWS); // views to render again when they are shown
let scene = null;
let bist = null;
let plotsDirty = false;
let lastPlotT = 0;
let lastTs = null;
let framesDue = 0;   // playback frames owed since the last animation tick (the fraction carries over)
let loadSeq = 0;

const sgn = (v, d = 2) => (v == null ? '–' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`);
const ms = (t) => t.toFixed(2);

function setStatus(text, error = false) {
  const el = $('#status');
  el.textContent = text;
  el.classList.toggle('error', error);
}

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return {
    v: VIEWS.includes(h.get('v')) ? h.get('v') : null, s: h.get('s'), r: h.has('r') ? Number(h.get('r')) : null,
    a: h.get('a') !== '0', f: h.has('f') ? Number(h.get('f')) : null, p: PARTS.includes(h.get('p')) ? h.get('p') : null,
    c: ['temp', 'err', 'det'].includes(h.get('c')) ? h.get('c') : null, m: ['flat', 'height', 'off'].includes(h.get('m')) ? h.get('m') : null,
  };
}

function hashNow() {
  const S = state.S;
  return `#v=${state.view}&s=${S ? S.id : ''}&r=${state.ring}&a=${state.all ? '1' : '0'}&f=${state.frame}&p=${state.part}&c=${state.colorMode}&m=${state.mapMode}`;
}

let hashTimer = 0;
function writeHash(now = false) {
  clearTimeout(hashTimer);
  const go = () => { if (location.hash !== hashNow()) history.replaceState(null, '', hashNow()); };
  if (now) go(); else hashTimer = setTimeout(go, 250);
}

async function main() {
  initThemeToggle($('#theme'));
  setStatus('Loading the viewer…');
  if (typeof Plotly === 'undefined') {
    setStatus('Plotly did not load from cdn.jsdelivr.net (offline?). The page needs it for the 2-D views.', true);
    return;
  }
  let meta;
  try {
    meta = await getJSON('data/meta.json');
  } catch (e) {
    setStatus(location.protocol === 'file:'
      ? 'Browsers block data loading from file:// pages. Serve it instead: cd ~/mrm-model/webviz && npm run dev, then open the address it prints.'
      : `Could not load data/meta.json (${e.message}). Regenerate the data: cd ~/mrm-model/webviz && npm run regen (see webviz/README.md).`, true);
    return;
  }
  state.meta = meta;
  const h = readHash();
  if (h.c) state.colorMode = h.c;
  if (h.m) state.mapMode = h.m;
  if (h.p) state.part = h.p;
  state.all = h.a;
  state.view = h.v || 'scenario';

  // controls
  const sel = $('#scenario');
  for (const s of meta.scenarios) {
    const o = document.createElement('option');
    o.value = s.id;
    o.textContent = s.title;
    sel.appendChild(o);
  }
  const rings = $('#rings');
  for (let r = 0; r < meta.n_rings; r++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = `<span class="sw" style="background:var(--cat-${r + 1})"></span>R${r}`;
    b.setAttribute('aria-label', `Select ring ${r}`);
    b.addEventListener('click', () => setRing(r));
    rings.appendChild(b);
  }
  const allB = document.createElement('button');
  allB.type = 'button';
  allB.id = 'allRings';
  allB.textContent = 'All 8';
  allB.title = 'Compare all 8 rings; click again to focus on the selected ring';
  allB.setAttribute('aria-label', 'Show all rings together');
  allB.addEventListener('click', () => setAll(!state.all));
  rings.appendChild(allB);
  sel.addEventListener('change', () => openScenario(sel.value, 0));
  $('#play').addEventListener('click', () => setPlaying(!state.playing));
  $('#speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); updateRate(); });
  $('#scrub').addEventListener('input', (e) => { setPlaying(false); setFrame(Number(e.target.value)); });
  segGroup('#colorMode', state.colorMode, (v) => { state.colorMode = v; if (scene) scene.setColorMode(v); renderLegend(); if (state.S) setFrame(state.frame); });
  segGroup('#mapMode', state.mapMode, (v) => { state.mapMode = v; if (scene) scene.setMapMode(v); if (state.S) setFrame(state.frame); });
  $('#reset').addEventListener('click', () => scene && scene.resetView());
  $('#top').addEventListener('click', () => scene && scene.topView());
  document.addEventListener('keydown', onKey);
  $('#nav').addEventListener('click', (e) => {
    const a = e.target.closest('a[data-v]');
    if (!a) return;
    e.preventDefault();
    showView(a.dataset.v);
  });

  // model limits and provenance (scenario view)
  $('#limits').innerHTML = '';
  for (const l of meta.limits) {
    const li = document.createElement('li');
    li.textContent = l;
    $('#limits').appendChild(li);
  }
  $('#prov').textContent = `Preset ${meta.preset} · model commit ${meta.git_commit}${meta.tracked_files_modified ? ' (modified)' : ''} · data generated ${meta.generated} by ${meta.generator} · nominal world, 25 °C. Details: webviz/README.md.`;

  // views
  const seek = (tMs) => seekMs(tMs);
  V.scenario = new ScenarioView($('[data-view=scenario]'), { onSeek: seek });
  V.spectrum = new SpectrumView($('#cardSpec'));
  V.heat = new HeatmapView($('#cardHeat'), meta);
  V.bank = new BankView($('#cardBank'), { onPick: (r) => setRing(r) });
  V.parts = new PartsView($('#cardParts'), $('#cardBist'), { onSeek: seek, onPart: (p) => setPart(p) });
  V.stats = new StatsView($('[data-view=stats]'), { onPart: (p) => setPart(p) });
  V.metrics = new MetricsView($('#cardMetrics'), { onPick: (r) => setRing(r), onSeek: seek });
  V.ber = new BerView($('#cardBer'), { onSeek: seek, onPick: (r) => setRing(r) });
  V.limits = new LimitsView($('[data-view=limits]'), { onSeek: seek });
  V.xtalk = new CrosstalkView($('[data-view=xtalk]'), { onSeek: seek });
  V.parts.setPart(state.part);
  V.stats.setPart(state.part);
  renderConstants($('#cardConst'), meta);
  V.limits.renderHardware(meta);
  try {
    const { Scene3D } = await import('./js/scene3d.js');
    scene = new Scene3D($('#stage'), meta, { onPick: (r) => setRing(r), onHover: onHover3D });
    scene.setMapMode(state.mapMode);
    scene.setColorMode(state.colorMode);
  } catch (e) {
    console.error(e);
    setStatus(`The 3-D view did not start (three.js from cdn.jsdelivr.net, or WebGL): ${e.message}. The 2-D views still work.`, true);
  }
  getJSON('data/bistability.json').then((d) => { bist = new BistabilityView($('#cardBist'), d); if (state.view === 'parts' && state.part === 'ring') bist.render(); })
    .catch((e) => { $('#cardBist .readout').textContent = `bistability.json did not load: ${e.message}`; });

  onThemeChange(() => {
    if (scene) scene.setTheme();
    VIEWS.forEach((v) => dirty.add(v));
    if (V.xtalk) V.xtalk.invalidate();
    if (state.S) renderView(state.view);
    renderLegend();
  });

  const sid = meta.scenarios.some((s) => s.id === h.s) ? h.s : (meta.scenarios.find((s) => s.id === 'a1_aggressor') || meta.scenarios[0]).id;
  const r0 = Math.min(meta.n_rings - 1, Math.max(0, Number.isInteger(h.r) ? h.r : 0));
  state.ring = r0;
  markRingButtons();
  markNav();
  if (scene) scene.setRing(r0);
  await openScenario(sid, h.f || 0);
  requestAnimationFrame(loop);

  // a hash typed or pasted by the viewer (replaceState does not fire this)
  window.addEventListener('hashchange', () => {
    const q = readHash();
    if (q.c && q.c !== state.colorMode) document.querySelector(`#colorMode button[data-v="${q.c}"]`)?.click();
    if (q.m && q.m !== state.mapMode) document.querySelector(`#mapMode button[data-v="${q.m}"]`)?.click();
    if (q.p && q.p !== state.part) setPart(q.p);
    if (q.a !== state.all) setAll(q.a);
    if (Number.isInteger(q.r) && q.r >= 0 && q.r < meta.n_rings && q.r !== state.ring) setRing(q.r);
    if (q.v && q.v !== state.view) showView(q.v);
    if (q.s && q.s !== (state.S && state.S.id) && meta.scenarios.some((x) => x.id === q.s)) openScenario(q.s, q.f || 0);
    else if (q.f != null && state.S) setFrame(Math.max(0, Math.min(state.S.nF - 1, q.f)));
  });
}

function segGroup(sel, initial, fn) {
  const g = $(sel);
  const btns = [...g.querySelectorAll('button')];
  const mark = (v) => btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
  mark(initial);
  btns.forEach((b) => b.addEventListener('click', () => { mark(b.dataset.v); fn(b.dataset.v); }));
}

// ---------------------------------------------------------------- navigation

function markNav() {
  document.querySelectorAll('#nav a').forEach((a) => {
    const on = a.dataset.v === state.view;
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  $('#allRings').hidden = !ALL_VIEWS.has(state.view);
}

function showView(v) {
  if (!VIEWS.includes(v)) return;
  state.view = v;
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.dataset.view !== v; });
  markNav();
  markRingButtons();
  // the 3-D stage had no size while hidden: fit the camera once the view has its layout
  if (v === 'die' && scene) requestAnimationFrame(() => requestAnimationFrame(() => scene.resize()));
  if (state.S) {
    renderView(v);
    setFrame(state.frame, true);
  }
  writeHash(true);
}

// render a view's whole-run content (only when shown: Plotly needs a visible box to size its plots)
function renderView(v) {
  const S = state.S;
  if (!S || !dirty.has(v)) return;
  dirty.delete(v);
  const r = state.ring;
  if (v === 'scenario') V.scenario.render(S, r);
  else if (v === 'die') {
    V.bank.render(S, r);
    renderDistances($('#distances'), state.meta, r);
    renderLegend();
  } else if (v === 'parts') {
    V.parts.setState(S, r, state.all, state.frame);
    V.parts.render();
    if (bist && state.part === 'ring') bist.render();
  } else if (v === 'stats') {
    V.stats.setState(S, r, state.all);
    V.stats.render();
  } else if (v === 'metrics') {
    V.metrics.render(S, r);
    V.ber.render(S, r);
  } else if (v === 'limits') {
    V.limits.setState(S, r);
    V.limits.render();
  } else if (v === 'xtalk') {
    V.xtalk.setState(S, r, state.frame, state.all);
    V.xtalk.invalidate();
    V.xtalk.render();
  }
}

function setPart(p) {
  if (!PARTS.includes(p)) p = 'ring';
  state.part = p;
  V.parts.setPart(p);
  V.stats.setPart(p);
  dirty.add('parts');
  dirty.add('stats');
  if (state.view === 'parts' || state.view === 'stats') {
    renderView(state.view);
    if (state.view === 'parts') { if (bist && p === 'ring') bist.render(); plotsDirty = true; }
  }
  writeHash(true);
}

function setAll(on) {
  state.all = !!on;
  markRingButtons();
  ALL_VIEWS.forEach((v) => dirty.add(v));
  renderView(state.view);
  if (state.S) setFrame(state.frame, true);
  writeHash(true);
}

async function openScenario(id, frame) {
  const seq = ++loadSeq;
  setPlaying(false);
  $('#scenario').value = id;
  setStatus('Loading the scenario (about 8 MB of model output)…');
  let S;
  try {
    S = await loadScenario(state.meta, id);
  } catch (e) {
    setStatus(`Could not load scenario ${id}: ${e.message}`, true);
    return;
  }
  if (seq !== loadSeq) return;
  state.S = S;
  $('#scrub').max = String(S.nF - 1);
  renderEvents(S);
  if (scene) scene.setScenario(S);
  V.spectrum.setScenario(S);
  VIEWS.forEach((v) => dirty.add(v));
  state.frame = Math.max(0, Math.min(S.nF - 1, frame));
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.dataset.view !== state.view; });
  if (state.view === 'die' && scene) requestAnimationFrame(() => requestAnimationFrame(() => scene.resize()));
  renderView(state.view);
  const miss = [!S.dev && 'devices', !S.st && 'stats'].filter(Boolean);
  if (miss.length) setStatus(`${miss.map((m) => `${m}_${id}.json`).join(' and ')} missing: regenerate the data (npm run regen). Some views stay empty.`, true);
  updateRate();
  setFrame(state.frame, true);
  if (!miss.length) setStatus('');
}

function renderEvents(S) {
  const box = $('#events');
  box.innerHTML = '';
  const tmax = S.tMs[S.nF - 1];
  for (const ev of S.scn.events) {
    const t = ev.t_s * 1e3;
    const b = document.createElement('button');
    b.type = 'button';
    b.style.left = `${Math.max(0, Math.min(100, (t / tmax) * 100))}%`;
    b.title = `${ms(t)} ms: ${ev.label}`;
    b.tabIndex = -1;
    b.addEventListener('click', () => { setPlaying(false); seekMs(t); });
    box.appendChild(b);
  }
}

function updateRate() {
  if (!state.S) return;
  const msPerS = BASE_FPS * state.speed * state.S.frameMs;
  $('#rate').textContent = `${fmtNum(msPerS)} ms of model time per second`;
}

function seekMs(tMs) {
  const S = state.S;
  if (!S) return;
  let best = 0;
  for (let f = 0; f < S.nF; f++) if (Math.abs(S.tMs[f] - tMs) < Math.abs(S.tMs[best] - tMs)) best = f;
  setFrame(best);
}

function setFrame(f, force = false) {
  const S = state.S;
  if (!S) return;
  state.frame = f;
  $('#scrub').value = String(f);
  const t = S.tMs[f];
  $('#time').textContent = `t = ${ms(t)} ms / ${ms(S.m.duration_s * 1e3)} ms`;
  const v = state.view;
  if (v === 'die') {
    if (scene) scene.setFrame(f);
    V.bank.setFrame(f);
    const k = S.nearestKey[f];
    $('#keynote').textContent = `rings, heaters, PDs, aggressor: frame ${ms(t)} ms · map, bus light: key frame ${ms(S.tMs[S.keys[k]])} ms`;
    if (state.hover) showTooltip(state.hover);
  } else if (v === 'scenario') V.scenario.setTime(t);
  else if (v === 'parts') V.parts.setFrame(f);
  else if (v === 'metrics') { V.metrics.setTime(t); V.ber.setTime(t); }
  else if (v === 'limits') V.limits.setTime(t);
  else if (v === 'xtalk') V.xtalk.setFrame(f);
  plotsDirty = true;
  if (force) drawKeyPlots();
  writeHash();
}

// the key-frame views of the visible view (spectra, map, comb, matrices)
function drawKeyPlots() {
  const S = state.S;
  if (!S) return;
  const r = state.ring;
  const k = S.nearestKey[state.frame];
  const tk = S.tMs[S.keys[k]];
  if (state.view === 'die') {
    const res = V.spectrum.update(S, r, k, state.all);
    const lres = S.spec.lam_res_nm[k][r];
    $('#cardSpec .readout').textContent = state.all ? `All ${S.nR} rings · same key frame ${ms(tk)} ms · R${r} outlined · common transmission scale` : `R${r} on its declared line L${res.decl} · resonance ${lres.toFixed(3)} nm · detuning ${sgn(res.detuning)} pm (resonance − line) · key frame ${ms(tk)} ms`;
    V.heat.update(S, r, k);
    const base = S.therm.t_base_k[k] - K0;
    let mx = -Infinity;
    for (const row of S.therm.rise_k[k]) for (const v of row) if (v != null && v > mx) mx = v;
    $('#cardHeat .readout').textContent = `key frame ${ms(tk)} ms · colour ${S.scales.temp.lo}–${S.scales.temp.hi} °C, fixed for this scenario` +
      (mx + base > S.scales.temp.hi ? ` · the point law peaks at point sources (${fmtNum(mx + base)} °C on the node beside one; clipped)` : '');
  } else if (state.view === 'parts') V.parts.slowUpdate();
  else if (state.view === 'xtalk') V.xtalk.slowUpdate();
}

function setRing(r) {
  state.ring = r;
  markRingButtons();
  if (scene) scene.setRing(r);
  VIEWS.forEach((v) => dirty.add(v));
  // Scenario per-ring inputs emphasize the selected ring too.
  if (state.S) {
    if (state.view === 'metrics') { V.metrics.setRing(r); V.ber.render(state.S, r); dirty.delete('metrics'); }
    if (state.view === 'die') { V.bank.setRing(r); renderDistances($('#distances'), state.meta, r); dirty.delete('die'); }
    renderView(state.view);
    if (state.view === 'parts' && bist && state.part === 'ring') bist.render();
    setFrame(state.frame, true);
  }
  writeHash(true);
}

function markRingButtons() {
  [...$('#rings').children].forEach((b, i) => {
    if (b.id === 'allRings') b.setAttribute('aria-pressed', String(state.all));
    else b.setAttribute('aria-pressed', String(i === state.ring));
  });
}

function setPlaying(on) {
  const S = state.S;
  if (on && S && state.frame >= S.nF - 1) setFrame(0);
  state.playing = on && !!S;
  lastTs = null;
  framesDue = 0;
  const b = $('#play');
  if (!state.playing && S) writeHash(true);
  b.textContent = state.playing ? 'Pause' : 'Play';
  b.setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
}

function loop(ts) {
  requestAnimationFrame(loop);
  const S = state.S;
  if (state.playing && S) {
    if (lastTs == null) lastTs = ts;
    framesDue += Math.min(0.25, (ts - lastTs) / 1000) * BASE_FPS * state.speed;
    lastTs = ts;
    if (framesDue >= 1) {
      const n = Math.floor(framesDue);
      framesDue -= n;
      let f = state.frame + n;
      if (f >= S.nF - 1) { f = S.nF - 1; setPlaying(false); }
      setFrame(f);
    }
  }
  // the key-frame plots redraw at most about 10 times a second while playing
  if (plotsDirty && ts - lastPlotT > (state.playing ? 100 : 0)) {
    plotsDirty = false;
    lastPlotT = ts;
    drawKeyPlots();
  }
}

function onKey(e) {
  const tag = e.target.tagName;
  if (tag === 'SELECT' || tag === 'TEXTAREA' || (tag === 'INPUT' && e.target.type !== 'range')) return;
  if (e.key === ' ' && tag !== 'BUTTON' && tag !== 'A') { e.preventDefault(); setPlaying(!state.playing); return; }
  if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && tag !== 'INPUT' && state.S) {
    e.preventDefault();
    setPlaying(false);
    const step = (e.shiftKey ? 10 : 1) * (e.key === 'ArrowRight' ? 1 : -1);
    setFrame(Math.max(0, Math.min(state.S.nF - 1, state.frame + step)));
  }
}

// ---------------------------------------------------------------- 3-D legend and hover

function legendItem(title, gradient, lo, hi, note) {
  return `<div class="lg"><b>${title}</b><div class="bar" style="background:${gradient}"></div>` +
    `<div class="ends"><span>${lo}</span><span>${hi}</span></div>${note ? `<span>${note}</span>` : ''}</div>`;
}

function renderLegend() {
  const S = state.S;
  if (!S) return;
  const sc = S.scales;
  let html = legendItem('Temperature (°C)', sc.temp.gradient(), sc.temp.lo, sc.temp.hi,
    state.colorMode === 'temp' ? 'rings, heaters, PDs and map; clipped at the ends' : 'heaters, PDs and map; clipped at the ends');
  if (state.colorMode === 'err') {
    html += legendItem('Ring lock error (pm)', sc.err.gradient(), `−${fmtNum(sc.err.hi)}`, `+${fmtNum(sc.err.hi)}`, 'scale = the lock band; beyond it: clipped');
  } else if (state.colorMode === 'det') {
    html += legendItem('Ring detuning (pm)', sc.det.gradient(), `−${fmtNum(sc.det.hi)}`, `+${fmtNum(sc.det.hi)}`, 'resonance − line; + is red of the line');
  }
  const unlit = token('--bus-unlit');
  const lit = token('--bus-lit');
  html += legendItem('Bus light (mW, all 8 lines)', `linear-gradient(to right, ${unlit}, ${lit})`, '0', fmtNum(S.busMax), 'key frame');
  html += `<div class="lg"><b>Heater</b><span>annulus inside each ring: colour = heater-resistor temperature (scale above)</span><span>copper column: a power gauge, not a structure; 30 µm tall = ${fmtNum(S.pHeatMax)} mW</span></div>`;
  html += '<div class="lg"><b>Monitor PD, driver</b><span>PD disc: colour = PD temperature (scale above)</span><span>driver disc: grey, a heat source</span></div>';
  if (S.aggAny > 0) html += `<div class="lg"><b>Aggressor glow</b><span>size = power</span><span>largest ${fmtNum(S.aggAny)} W</span></div>`;
  $('#legend3d').innerHTML = html;
}

function onHover3D(hit) {
  state.hover = hit;
  if (!hit) { $('#tooltip').hidden = true; return; }
  showTooltip(hit);
}

function showTooltip(hit) {
  const S = state.S;
  const tip = $('#tooltip');
  if (!S || !hit) { tip.hidden = true; return; }
  tip.innerHTML = describe(hit, S, state.frame);
  tip.hidden = false;
  const pad = 14;
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  let x = hit.clientX + pad;
  let y = hit.clientY + pad;
  if (x + w > window.innerWidth - 8) x = hit.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = hit.clientY - h - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}

function describe(hit, S, f) {
  const R = S.scn.rings;
  const celsiusText = (kelvin) => (kelvin - K0).toFixed(2);
  const row = (k, v) => `<div><span class="k">${k}</span> ${v}</div>`;
  const r = hit.idx;
  if (hit.kind === 'ring') {
    const lo = R.line_on[f][r];
    return `<b>Ring R${r}</b> · ${lo >= 0 ? `on line L${lo}` : 'on no line'} (declared L${R.declared_line[f][r]})` +
      row('detuning', `${sgn(R.detuning_pm[f][r])} pm`) +
      row('lock error', `${sgn(S.err[f][r])} pm (band ±${fmtNum(S.thr)})`) +
      row('ring', `${celsiusText(R.t_ring_k[f][r])} °C`) +
      row('heater', `code ${R.heater_code[f][r]} · ${fmtNum(R.p_heater_mw[f][r])} mW`) +
      row('light in', `${fmtNum(R.p_in_mw[f][r])} mW · absorbed ${fmtNum(R.p_abs_mw[f][r])} mW`) +
      row('monitor', `code ${R.monitor_code[f][r]} · drop ${fmtNum(R.p_drop_uw[f][r])} µW`) +
      row('relOMA', `${fmtNum(R.reloma_db[f][r])} dB (best ${fmtNum(R.oracle_reloma_db[f][r])} dB)`);
  }
  const g = state.meta.geometry;
  const D = S.dev;
  if (hit.kind === 'heater') {
    const hs = g.heater_structure;
    const q = hs ? hs.rects[0] : null;
    return `<b>Heater of R${r}</b>` + row('code', `${R.heater_code[f][r]}`) + row('power', `${fmtNum(R.p_heater_mw[f][r])} mW`) +
      (D ? row('drive', `${fmtNum(D.heater.v_v[f][r])} V · ${fmtNum(D.heater.i_ma[f][r])} mA · ${fmtNum(D.heater.r_ohm[f][r])} Ω`) : '') +
      row('resistor', `${celsiusText(R.t_heater_k[f][r])} °C`) +
      (q ? `<div class="k">${hs.family}: annulus r ${q.r_inner_um.toFixed(2)}–${q.r_outer_um.toFixed(2)} µm, ${(q.z_top_um - q.z_bottom_um).toFixed(2)} µm thick (assumed geometry of the FEM extraction)</div>` : '');
  }
  if (hit.kind === 'pd') {
    return `<b>Monitor PD of R${r}</b>` + row('PD', `${celsiusText(R.t_pd_k[f][r])} °C`) + row('light', `${fmtNum(R.p_drop_uw[f][r])} µW`) +
      row('current', `${fmtNum(R.i_monitor_ua[f][r])} µA` + (D ? ` (dark ${fmtNum(D.pd.i_dark_na[f][r])} nA)` : '')) + row('ADC', `code ${R.monitor_code[f][r]}`) +
      `<div class="k">${fmtNum(g.distances.ring_to_own_pd_um[r])} µm from the ring centre on the drop side; position and ${g.site_sizes_um.monitor_pd_radius} µm radius assumed</div>`;
  }
  if (hit.kind === 'driver') {
    return `<b>Driver of R${r}</b><div class="k">heat source of the junction driver, ${fmtNum(g.distances.ring_to_own_driver_um[r])} µm from the ring; position and ${g.site_sizes_um.driver_radius} µm radius assumed</div>`;
  }
  if (hit.kind === 'aggressor') {
    const p = S.scn.drives.aggressor_w[hit.idx];
    return `<b>${hit.idx === 'aggressor' ? 'Aggressor site (A1/A2 default)' : 'Aggressor stress site'}</b>` +
      row('power', `${p ? fmtNum(p[f]) : '0'} W`) +
      `<div class="k">${fmtNum(g.distances.ring_to_site_um[hit.idx][0])} µm from R0, ${fmtNum(g.distances.pd_to_site_um[hit.idx][0])} µm from its PD${hit.idx === 'aggressor' ? "; the same point as R0's driver" : '; a stress test, not a real layout'}. Position assumed.</div>`;
  }
  if (hit.kind === 'bus' && S.bus) {
    const k = S.nearestKey[f];
    const j = r;
    const order = state.meta.geometry.bus_order;
    const where = j === 0 ? `input → R${order[0]}` : j === order.length ? `after R${order[order.length - 1]} → output` : `R${order[j - 1]} → R${order[j]}`;
    const per = j < order.length ? S.spec.bus.p_in_mw[k][order[j]] : S.spec.bus.p_thru_mw[k][order[order.length - 1]];
    const lines = per.map((v, i) => `L${i} ${fmtNum(v)}`).join(' · ');
    return `<b>Bus, ${where}</b>` + row('all lines', `${fmtNum(S.bus[k][j])} mW`) + `<div class="k">${lines} mW</div>` +
      `<div class="k">key frame ${ms(S.tMs[S.keys[k]])} ms</div>`;
  }
  return '';
}

main().catch((e) => { console.error(e); setStatus(`The viewer stopped: ${e.message}`, true); });
