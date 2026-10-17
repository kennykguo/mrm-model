// Loading the data files (site/data/*.json, written by webviz/generate.py) and the few derived arrays the
// views share. Nothing here adds physics: every derived number is a difference, a sum or a lookup.

import { Scale, niceUp } from './scales.js';

export const K0 = 273.15;
const cache = new Map();

export function getJSON(path) {
  if (!cache.has(path)) {
    // no-cache: revalidate, so regenerated data files are picked up at once
    cache.set(path, fetch(path, { cache: 'no-cache' }).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    }).catch((e) => { cache.delete(path); throw e; }));
  }
  return cache.get(path);
}

const col = (rows, r) => rows.map((row) => row[r]);

function maxOf(rows) {
  let m = -Infinity;
  for (const row of rows) {
    if (Array.isArray(row)) { for (const v of row) if (v != null && v > m) m = v; } else if (row != null && row > m) m = row;
  }
  return m;
}

export async function loadScenario(meta, id) {
  const m = meta.scenarios.find((s) => s.id === id);
  if (!m) throw new Error(`unknown scenario ${id}`);
  const [scn, spec, therm] = await Promise.all([
    getJSON(`data/${m.files.scenario}`), getJSON(`data/${m.files.spectra}`), getJSON(`data/${m.files.thermal}`),
  ]);
  return derive(meta, m, scn, spec, therm);
}

function derive(meta, m, scn, spec, therm) {
  const nF = scn.t_s.length;
  const nR = meta.n_rings;
  const R = scn.rings;
  const tMs = scn.t_s.map((t) => t * 1e3);
  const keys = scn.key_frames;
  const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  if (!same(keys, spec.frames) || !same(keys, therm.frames)) {
    console.warn('key frames differ between scenario, spectra and thermal files');
  }

  // nearest key frame (spectra, thermal map, bus light) for every frame
  const nearestKey = new Int32Array(nF);
  for (let f = 0, j = 0; f < nF; f++) {
    while (j + 1 < keys.length && Math.abs(keys[j + 1] - f) < Math.abs(keys[j] - f)) j++;
    nearestKey[f] = j;
  }

  // lock error = detuning minus the ring's lock reference (its operating point, or 0 = the declared line)
  const dRef = scn.lock_reference.d_ref_pm;
  const err = R.detuning_pm.map((row) => row.map((v, r) => v - dRef[r]));
  const thr = scn.summary.lock_threshold_pm;

  // fixed colour ranges per scenario
  const baseC = Math.round((Math.min(...scn.drives.t_base_k) - K0) * 100) / 100;
  const ringMaxC = Math.max(maxOf(R.t_ring_k) - K0, maxOf(therm.observers.ring_rise_k) + baseC);
  const tLo = Math.floor(baseC / 5) * 5;
  const tHi = Math.ceil(ringMaxC / 10) * 10;
  const temp = new Scale('inferno', tLo, tHi, '°C', 'Temperature');
  const errScale = new Scale('coolwarm', -thr, thr, 'pm', 'Lock error');
  let detMax = 0;
  for (const row of R.detuning_pm) for (const v of row) if (v != null && Math.abs(v) > detMax) detMax = Math.abs(v);
  const det = new Scale('coolwarm', -niceUp(detMax), niceUp(detMax), 'pm', 'Detuning');

  // light on the bus: segment j arrives at ring bus_order[j]; the last segment leaves the last ring
  const order = meta.geometry.bus_order;
  const sum = (a) => a.reduce((s, v) => s + (v || 0), 0);
  const bus = spec.bus ? spec.bus.p_in_mw.map((pin, k) => {
    const thru = spec.bus.p_thru_mw[k];
    const seg = order.map((r) => sum(pin[r]));
    seg.push(sum(thru[order[order.length - 1]]));
    return seg;
  }) : null;
  const busMax = bus ? Math.ceil(maxOf(bus)) : 1;   // whole mW, so the brightness uses the full range

  const pHeatMax = niceUp(maxOf(R.p_heater_mw));
  const aggSites = Object.keys(scn.drives.aggressor_w);
  const aggMax = {};
  for (const s of aggSites) aggMax[s] = maxOf(scn.drives.aggressor_w[s]);
  const aggAny = Math.max(0, ...Object.values(aggMax));

  // the scenario drive for the traces: aggressor power, else laser power factor, else light at the ring
  let drive;
  const activeSites = aggSites.filter((s) => aggMax[s] > 0);
  const lpf = scn.drives.laser_power_factor;
  let lpfVaries = false;
  for (const row of lpf) for (const v of row) if (Math.abs(v - lpf[0][0]) > 1e-9) lpfVaries = true;
  if (activeSites.length) {
    drive = { kind: 'aggressor', label: 'aggressor (W)', series: (r) => scn.drives.aggressor_w[activeSites[0]] };
  } else if (lpfVaries) {
    // the factor of the line the ring is declared on, frame by frame
    drive = { kind: 'laser', label: 'laser (× nominal)', series: (r) => lpf.map((row, f) => row[Math.max(0, R.declared_line[f][r])]) };
  } else {
    drive = { kind: 'pin', label: 'light at ring (mW)', series: (r) => col(R.p_in_mw, r) };
  }

  return {
    id: m.id, m, scn, spec, therm, nF, nR, tMs, keys, nearestKey, err, dRef, thr, baseC,
    scales: { temp, err: errScale, det }, bus, busMax, pHeatMax, aggSites, aggMax, aggAny, drive,
    col: (name, r) => col(R[name], r),
    frameMs: scn.frame_s * 1e3,
  };
}
