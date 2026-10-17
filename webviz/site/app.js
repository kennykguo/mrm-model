// MRM bank viewer: state, controls and playback. Every view follows the time scrubber.
// Views: js/scene3d.js (3-D), js/spectrum.js, js/traces.js, js/heatmap.js, js/bistability.js, js/summary.js.

import { getJSON, loadScenario, K0 } from './js/data.js';
import { SpectrumView } from './js/spectrum.js';
import { TracesView, fmtNum, shortEvent } from './js/traces.js';
import { HeatmapView } from './js/heatmap.js';
import { BistabilityView } from './js/bistability.js';
import { SummaryView } from './js/summary.js';
import { initThemeToggle, onThemeChange, token } from './js/theme.js';

const BASE_FPS = 30;          // frames per second at 1x
const $ = (s) => document.querySelector(s);
const state = { meta: null, S: null, frame: 0, ring: 0, playing: false, speed: 1, colorMode: 'temp', mapMode: 'flat', hover: null };
let scene = null;
let spectrum;
let traces;
let heat;
let bist = null;
let summary;
let plotsDirty = false;
let lastPlotT = 0;
let lastTs = null;
let acc = 0;
let loadSeq = 0;

const sgn = (v, d = 2) => (v == null ? '–' : `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)}`);
const ms = (t) => t.toFixed(2);

function setStatus(text, error = false) {
  const el = $('#status');
  el.textContent = text;
  el.classList.toggle('error', error);
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
      ? 'Browsers block data loading from file:// pages. Serve the folder instead: cd webviz/site && python3 -m http.server 8000, then open http://localhost:8000.'
      : `Could not load data/meta.json (${e.message}). Regenerate the data with webviz/generate.py (see webviz/README.md).`, true);
    return;
  }
  state.meta = meta;

  // initial state from the URL hash (#s=<scenario>&r=<ring>&f=<frame>&c=<ring colour>&m=<map mode>)
  const h = new URLSearchParams(location.hash.slice(1));
  if (['temp', 'err', 'det'].includes(h.get('c'))) state.colorMode = h.get('c');
  if (['flat', 'height', 'off'].includes(h.get('m'))) state.mapMode = h.get('m');

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
    b.textContent = `R${r}`;
    b.setAttribute('aria-label', `Select ring ${r}`);
    b.addEventListener('click', () => setRing(r));
    rings.appendChild(b);
  }
  sel.addEventListener('change', () => openScenario(sel.value, 0));
  $('#play').addEventListener('click', () => setPlaying(!state.playing));
  $('#speed').addEventListener('change', (e) => { state.speed = Number(e.target.value); updateRate(); });
  $('#scrub').addEventListener('input', (e) => { setPlaying(false); setFrame(Number(e.target.value)); });
  segGroup('#colorMode', state.colorMode, (v) => { state.colorMode = v; if (scene) scene.setColorMode(v); renderLegend(); if (state.S) setFrame(state.frame); });
  segGroup('#mapMode', state.mapMode, (v) => { state.mapMode = v; if (scene) scene.setMapMode(v); if (state.S) setFrame(state.frame); });
  $('#reset').addEventListener('click', () => scene && scene.resetView());
  $('#top').addEventListener('click', () => scene && scene.topView());
  document.addEventListener('keydown', onKey);

  // limits and provenance
  $('#limits').innerHTML = '';
  for (const l of meta.limits) {
    const li = document.createElement('li');
    li.textContent = l;
    $('#limits').appendChild(li);
  }
  $('#prov').textContent = `Preset ${meta.preset} · model commit ${meta.git_commit}${meta.tracked_files_modified ? ' (modified)' : ''} · data generated ${meta.generated} by ${meta.generator} · nominal world, 25 °C. Details: webviz/README.md.`;

  // views
  spectrum = new SpectrumView($('#cardSpec'));
  traces = new TracesView($('#cardTraces'), { onSeek: (tMs) => seekMs(tMs) });
  heat = new HeatmapView($('#cardHeat'), meta);
  summary = new SummaryView($('#cardSum'), { onPick: (r) => setRing(r) });
  try {
    const { Scene3D } = await import('./js/scene3d.js');
    scene = new Scene3D($('#stage'), meta, { onPick: (r) => setRing(r), onHover: onHover3D });
    scene.setMapMode(state.mapMode);
    scene.setColorMode(state.colorMode);
  } catch (e) {
    console.error(e);
    setStatus(`The 3-D view did not start (three.js from cdn.jsdelivr.net, or WebGL): ${e.message}. The 2-D views still work.`, true);
  }
  getJSON('data/bistability.json').then((d) => { bist = new BistabilityView($('#cardBist'), d); bist.render(); })
    .catch((e) => { $('#cardBist .readout').textContent = `bistability.json did not load: ${e.message}`; });

  onThemeChange(() => {
    if (scene) scene.setTheme();
    if (state.S) { traces.render(state.S, state.ring); plotsDirty = true; }
    if (bist) bist.render();
    renderLegend();
  });

  const sid = meta.scenarios.some((s) => s.id === h.get('s')) ? h.get('s') : (meta.scenarios.find((s) => s.id === 'a1_aggressor') || meta.scenarios[0]).id;
  const r0 = Math.min(meta.n_rings - 1, Math.max(0, Number(h.get('r')) || 0));
  state.ring = r0;
  markRingButtons();
  if (scene) scene.setRing(r0);
  await openScenario(sid, Number(h.get('f')) || 0);
  requestAnimationFrame(loop);

  // a hash typed or pasted by the viewer (replaceState above does not fire this)
  window.addEventListener('hashchange', () => {
    const q = new URLSearchParams(location.hash.slice(1));
    const s = q.get('s');
    const r = Number(q.get('r'));
    if (Number.isInteger(r) && r >= 0 && r < meta.n_rings && r !== state.ring) setRing(r);
    const c = q.get('c'), m = q.get('m');
    if (['temp', 'err', 'det'].includes(c) && c !== state.colorMode) document.querySelector(`#colorMode button[data-v="${c}"]`)?.click();
    if (['flat', 'height', 'off'].includes(m) && m !== state.mapMode) document.querySelector(`#mapMode button[data-v="${m}"]`)?.click();
    if (s && s !== (state.S && state.S.id) && meta.scenarios.some((x) => x.id === s)) openScenario(s, Number(q.get('f')) || 0);
    else if (q.has('f') && state.S) setFrame(Math.max(0, Math.min(state.S.nF - 1, Number(q.get('f')) || 0)));
  });
}

function segGroup(sel, initial, fn) {
  const g = $(sel);
  const btns = [...g.querySelectorAll('button')];
  const mark = (v) => btns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
  mark(initial);
  btns.forEach((b) => b.addEventListener('click', () => { mark(b.dataset.v); fn(b.dataset.v); }));
}

async function openScenario(id, frame) {
  const seq = ++loadSeq;
  setPlaying(false);
  $('#scenario').value = id;
  setStatus('Loading the scenario (about 7 MB of model output)…');
  let S;
  try {
    S = await loadScenario(state.meta, id);
  } catch (e) {
    setStatus(`Could not load scenario ${id}: ${e.message}`, true);
    return;
  }
  if (seq !== loadSeq) return;
  state.S = S;
  const scrub = $('#scrub');
  scrub.max = String(S.nF - 1);
  $('#desc').textContent = S.m.description;
  renderEvents(S);
  if (scene) scene.setScenario(S);
  spectrum.setScenario(S);
  traces.render(S, state.ring);
  updateTraceReadout();
  summary.render(S, state.ring);
  renderLegend();
  updateRate();
  setFrame(Math.max(0, Math.min(S.nF - 1, frame)), true);
  setStatus('');
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

let hashTimer = 0;
function setFrame(f, force = false) {
  const S = state.S;
  if (!S) return;
  state.frame = f;
  $('#scrub').value = String(f);
  const t = S.tMs[f];
  $('#time').textContent = `t = ${ms(t)} ms / ${ms(S.m.duration_s * 1e3)} ms`;
  if (scene) scene.setFrame(f);
  traces.setTime(t);
  const k = S.nearestKey[f];
  $('#keynote').textContent = `rings, heaters, PDs, aggressor: frame ${ms(t)} ms · map, bus light: key frame ${ms(S.tMs[S.keys[k]])} ms`;
  plotsDirty = true;
  if (force) drawKeyPlots();
  if (state.hover) showTooltip(state.hover);
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    history.replaceState(null, '', `#s=${S.id}&r=${state.ring}&f=${state.frame}&c=${state.colorMode}&m=${state.mapMode}`);
  }, 250);
}

function drawKeyPlots() {
  const S = state.S;
  if (!S) return;
  const r = state.ring;
  const k = S.nearestKey[state.frame];
  const tk = S.tMs[S.keys[k]];
  const res = spectrum.update(S, r, k);
  const lres = S.spec.lam_res_nm[k][r];
  $('#cardSpec .readout').textContent = `R${r} on its declared line L${res.decl} · resonance ${lres.toFixed(3)} nm · ` +
    `detuning ${sgn(res.detuning)} pm (resonance − line) · key frame ${ms(tk)} ms`;
  heat.update(S, r, k);
  const base = S.therm.t_base_k[k] - K0;
  let mx = -Infinity;
  for (const row of S.therm.rise_k[k]) for (const v of row) if (v != null && v > mx) mx = v;
  $('#cardHeat .readout').textContent = `key frame ${ms(tk)} ms · colour ${S.scales.temp.lo}–${S.scales.temp.hi} °C, fixed for this scenario` +
    (mx + base > S.scales.temp.hi ? ` · the point law peaks at point sources (${fmtNum(mx + base)} °C on the node beside one; clipped)` : '');
}

function setRing(r) {
  state.ring = r;
  markRingButtons();
  if (scene) scene.setRing(r);
  if (state.S) {
    traces.render(state.S, r);
    summary.setRing(r);
    updateTraceReadout();
    plotsDirty = true;
    drawKeyPlots();
    setFrame(state.frame);
  }
}

function updateTraceReadout() {
  const S = state.S;
  const r = state.ring;
  const lr = S.scn.lock_reference;
  const ref = lr.kind === 'line' ? 'the declared line (lock error = detuning)' : `its operating point, ${sgn(S.dRef[r])} pm detuning`;
  $('#cardTraces .readout').textContent = `R${r} · ${S.scn.controller} · lock reference: ${ref} · ` +
    `green band: the lock band, ±${fmtNum(S.thr)} pm about it · bottom panel: the scenario drive`;
}

function markRingButtons() {
  [...$('#rings').children].forEach((b, i) => b.setAttribute('aria-pressed', String(i === state.ring)));
}

function setPlaying(on) {
  const S = state.S;
  if (on && S && state.frame >= S.nF - 1) setFrame(0);
  state.playing = on && !!S;
  lastTs = null;
  acc = 0;
  const b = $('#play');
  if (!state.playing && S) history.replaceState(null, '', `#s=${S.id}&r=${state.ring}&f=${state.frame}&c=${state.colorMode}&m=${state.mapMode}`);
  b.textContent = state.playing ? 'Pause' : 'Play';
  b.setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
}

function loop(ts) {
  requestAnimationFrame(loop);
  const S = state.S;
  if (state.playing && S) {
    if (lastTs == null) lastTs = ts;
    acc += Math.min(0.25, (ts - lastTs) / 1000) * BASE_FPS * state.speed;
    lastTs = ts;
    if (acc >= 1) {
      const n = Math.floor(acc);
      acc -= n;
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
  if (e.key === ' ' && tag !== 'BUTTON') { e.preventDefault(); setPlaying(!state.playing); return; }
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
    state.colorMode === 'temp' ? 'rings, PDs and map; clipped at the ends' : 'PDs and map; clipped at the ends');
  if (state.colorMode === 'err') {
    html += legendItem('Ring lock error (pm)', sc.err.gradient(), `−${fmtNum(sc.err.hi)}`, `+${fmtNum(sc.err.hi)}`, 'scale = the lock band; beyond it: clipped');
  } else if (state.colorMode === 'det') {
    html += legendItem('Ring detuning (pm)', sc.det.gradient(), `−${fmtNum(sc.det.hi)}`, `+${fmtNum(sc.det.hi)}`, 'resonance − line; + is red of the line');
  }
  const unlit = token('--bus-unlit');
  const lit = token('--bus-lit');
  html += legendItem('Bus light (mW, all 8 lines)', `linear-gradient(to right, ${unlit}, ${lit})`, '0', fmtNum(S.busMax), 'key frame');
  html += `<div class="lg"><b>Heater column</b><span>height = heater power</span><span>30 µm tall = ${fmtNum(S.pHeatMax)} mW</span></div>`;
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
  const C = (k) => (k - K0).toFixed(2);
  const row = (k, v) => `<div><span class="k">${k}</span> ${v}</div>`;
  const r = hit.idx;
  if (hit.kind === 'ring') {
    const lo = R.line_on[f][r];
    return `<b>Ring R${r}</b> · ${lo >= 0 ? `on line L${lo}` : 'on no line'} (declared L${R.declared_line[f][r]})` +
      row('detuning', `${sgn(R.detuning_pm[f][r])} pm`) +
      row('lock error', `${sgn(S.err[f][r])} pm (band ±${fmtNum(S.thr)})`) +
      row('ring', `${C(R.t_ring_k[f][r])} °C`) +
      row('heater', `code ${R.heater_code[f][r]} · ${fmtNum(R.p_heater_mw[f][r])} mW`) +
      row('light in', `${fmtNum(R.p_in_mw[f][r])} mW · absorbed ${fmtNum(R.p_abs_mw[f][r])} mW`) +
      row('monitor', `code ${R.monitor_code[f][r]} · drop ${fmtNum(R.p_drop_uw[f][r])} µW`) +
      row('relOMA', `${fmtNum(R.reloma_db[f][r])} dB (best ${fmtNum(R.oracle_reloma_db[f][r])} dB)`);
  }
  if (hit.kind === 'heater') {
    return `<b>Heater of R${r}</b>` + row('code', `${R.heater_code[f][r]}`) + row('power', `${fmtNum(R.p_heater_mw[f][r])} mW`) +
      row('resistor', `${C(R.t_heater_k[f][r])} °C`);
  }
  if (hit.kind === 'pd') {
    return `<b>Monitor PD of R${r}</b>` + row('PD', `${C(R.t_pd_k[f][r])} °C`) + row('light', `${fmtNum(R.p_drop_uw[f][r])} µW`) +
      row('current', `${fmtNum(R.i_monitor_ua[f][r])} µA`) + row('ADC', `code ${R.monitor_code[f][r]}`);
  }
  if (hit.kind === 'driver') {
    return `<b>Driver of R${r}</b><div class="k">heat source (junction driver); position assumed</div>`;
  }
  if (hit.kind === 'aggressor') {
    const p = S.scn.drives.aggressor_w[hit.idx];
    return `<b>${hit.idx === 'aggressor' ? 'Aggressor site (A1/A2 default)' : 'Aggressor stress site'}</b>` +
      row('power', `${p ? fmtNum(p[f]) : '0'} W`) +
      `<div class="k">${hit.idx === 'aggressor' ? "30 µm from R0; the same point as R0's driver" : "5 µm from R0's PD; a stress test, not a real layout"}</div>`;
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
