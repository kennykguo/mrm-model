// Device definitions (photodiode, heater, ring, laser and WDM, controller view, data path, thermal crosstalk,
// disturbances): the stat tiles of the selected ring at the scrubber frame, used by the "Parts over the run" view
// (js/parts.js), and the key-frame drawings (comb, bus matrix, assignment, scatter, crosstalk matrices).
// Every value is read from the data files (model output). The few differences shown (headroom, command minus DAC
// code) are differences of two exported values.
// Names used throughout: S = the scenario object of js/data.js, r = ring index, f = frame index, k = key-frame index.

import { restylePlot, plotBase, axis, PLOT_CONFIG, token, renderPlot } from './theme.js';
import { fmt, sgn, sci, esc, cat, colOf, mapRows, K0, hexA, formatMicroseconds } from './ui.js';

const toCelsius = (kelvin) => (kelvin == null ? null : kelvin - K0);
const pct = (v) => (v == null ? '–' : `${(100 * v).toFixed(1)} %`);
const lineName = (l) => (l == null || l < 0 ? 'none' : `L${l}`);

// ------------------------------------------------------------------ helpers on the scenario object
const ringsOf = (S) => S.scn.rings;                     // per-ring traces [frame][ring] (scenario_<id>.json)
const devicesOf = (S) => S.dev;                         // devices_<id>.json
const settingsOf = (S) => S.meta.devices;               // fixed device settings (meta.json devices)
const keyT = (S) => S.keys.map((f) => S.tMs[f]);        // key-frame times (ms)

function lockState(S, r, f) {
  const lk = devicesOf(S).running.locked[f];
  if (lk == null || lk[r] == null) return { v: 'not scored yet' };
  return lk[r] ? { v: 'locked' } : { v: 'unlocked', flag: 'UNLOCKED (engine lock state)' };
}

function lineStatus(S, r, f) {
  const R = ringsOf(S);
  const on = R.line_on[f][r];
  const dl = R.declared_line[f][r];
  if (on < 0) return { v: `on no line (declared ${lineName(dl)})`, flag: 'OFF-LINE', bad: false };
  if (on !== dl) return { v: `on ${lineName(on)}, declared ${lineName(dl)}`, flag: 'FOREIGN LINE', bad: true };
  return { v: `on its own line ${lineName(on)}`, bad: false };
}

// rise of ring r at key frame k grouped by source kind and owner
function riseGroups(S, r, k, which = 'ring') {
  const kk = devicesOf(S).key;
  const row = (which === 'pd' ? kk.pd_rise_by_source_k : kk.ring_rise_by_source_k)[k][r];
  const g = { ownHeater: 0, ownLight: 0, ownJD: 0, otherHeater: 0, otherLight: 0, otherJD: 0, aggressor: 0, total: 0 };
  kk.src_names.forEach((nm, s) => {
    const v = row[s] || 0;
    g.total += v;
    const m = nm.match(/^(heater|waveguide_absorption|junction|driver)(\d+)$/);
    if (!m) { g.aggressor += v; return; }
    const own = Number(m[2]) === r;
    const kind = m[1] === 'heater' ? 'Heater' : m[1] === 'waveguide_absorption' ? 'Light' : 'JD';
    g[(own ? 'own' : 'other') + kind] += v;
  });
  return g;
}

// ------------------------------------------------------------------ the tab definitions
function pdDef() {
  return {
    id: 'pd', label: 'Photodiode',
    question: 'What does the monitor photodiode see, and what does it send to the controller?',
    explain: 'The PD sits at the end of the drop waveguide. Its current is integrated, digitized and sent to the controller as a code.',
    tiles: (S) => {
      const mo = settingsOf(S).monitor;
      const ct = settingsOf(S).controller;
      return [
        { label: 'Drop power into the PD', note: 'Light from the drop port, all lines and orders, averaged over the data (frame mean).', get: (S, r, f) => fmt(ringsOf(S).p_drop_uw[f][r], 'µW') },
        { label: 'Signal photocurrent', note: `Responsivity ${mo.responsivity_a_per_w} A/W × drop power (the model's PD law).`, get: (S, r, f) => fmt(devicesOf(S).pd.i_signal_ua[f][r], 'µA') },
        { label: 'Dark current', note: `${mo.dark_i0_na} nA at 25 °C; it increases with the PD temperature (activation energy ${mo.dark_ea_ev} eV).`, get: (S, r, f) => fmt(devicesOf(S).pd.i_dark_na[f][r], 'nA') },
        { label: 'PD temperature', note: 'The PD thermal observer (the point-site heat law).', get: (S, r, f) => fmt(toCelsius(ringsOf(S).t_pd_k[f][r]), '°C', 2) },
        { label: 'Total PD current', note: 'Signal plus dark current: the current that the ADC integrates (frame mean).', get: (S, r, f) => fmt(ringsOf(S).i_monitor_ua[f][r], 'µA') },
        { label: 'ADC code to the controller', note: `Latest released code at the frame end; full scale ${mo.code_max}.`, get: (S, r, f) => {
          const c = ringsOf(S).monitor_code[f][r];
          const clip = ringsOf(S).monitor_code_max[f][r] >= mo.code_max || devicesOf(S).pd.overrange_frac[f][r] > 0;
          return { v: `${c} (${pct(c / mo.code_max)})`, flag: clip ? 'CLIPPED: a sample reached full scale' : '' };
        } },
        { label: 'ADC full scale', note: `${mo.adc_bits}-bit ADC, LSB ${mo.lsb_na} nA at PGA gain 1 (gains ${mo.pga_gains.join(', ')}).`, get: (S, r, f) => {
          const p = devicesOf(S).pd;
          return `${fmt(p.full_scale_ua[f][r], 'µA')} = ${fmt(p.full_scale_drop_uw[f][r], 'µW')} drop (PGA ×${mo.pga_gains[p.pga[f][r]] || 1})`;
        } },
        { label: 'Noise of one sample', note: `Shot noise plus ${mo.input_noise_pa_per_rt_hz} pA/√Hz input noise over the ${formatMicroseconds(mo.integration_window_s)} integration (the model's white-noise law).`, get: (S, r, f) => `${fmt(devicesOf(S).pd.noise_rms_na[f][r], 'nA')} rms = ${fmt(devicesOf(S).pd.noise_rms_codes[f][r], 'codes')}` },
        { label: 'Code spread in this frame', note: 'Lowest and highest released code in the frame: noise plus change.', get: (S, r, f) => `${ringsOf(S).monitor_code_min[f][r]} to ${ringsOf(S).monitor_code_max[f][r]}` },
        { label: 'Noise-free code minus released code', note: 'The ideal code of the frame-mean current (no noise, no clip) minus the code at the frame end.', get: (S, r, f) => fmt(devicesOf(S).pd.ideal_code[f][r] - ringsOf(S).monitor_code[f][r], 'codes') },
        { label: 'ADC samples per controller step', note: `One sample every ${formatMicroseconds(mo.sample_period_s)}, one controller step every ${formatMicroseconds(ct.period_s)} (room for ${ct.samples_per_step_capacity}). Frame mean.`, get: (S, r, f) => fmt(devicesOf(S).pd.samples_per_step[f][r]) },
        { label: 'Over-range samples', note: 'Share of saved device samples whose latest ADC reading was flagged over-range; not a count of all ADC conversions.', get: (S, r, f) => pct(devicesOf(S).pd.overrange_frac[f][r]) },
      ];
    },
    plots: (S, r) => {
      const R = ringsOf(S);
      const p = devicesOf(S).pd;
      const fs = settingsOf(S).monitor.code_max;
      return [
        { title: 'drop (µW)', traces: [{ y: colOf(R.p_drop_uw, r), name: 'drop power', unit: 'µW' }] },
        { title: 'current (µA)', traces: [{ y: colOf(R.i_monitor_ua, r), name: 'total', unit: 'µA' }, { y: colOf(p.i_signal_ua, r), name: 'signal', unit: 'µA', color: cat(1), dash: 'dot' }] },
        { title: 'dark (nA)', traces: [{ y: colOf(p.i_dark_na, r), name: 'dark current', unit: 'nA' }] },
        { title: 'PD (°C)', traces: [{ y: colOf(R.t_pd_k, r).map(toCelsius), name: 'PD temperature', unit: '°C' }] },
        { title: 'ADC (code)', traces: [{ y: colOf(R.monitor_code, r), band: [colOf(R.monitor_code_min, r), colOf(R.monitor_code_max, r)], name: 'code', shape: 'hv', fmt: '.0f' }],
          hlines: [{ y: fs, label: `full scale ${fs}` }, { y: 0, label: '0' }] },
        { title: 'noise (code rms)', traces: [{ y: colOf(p.noise_rms_codes, r), name: 'noise of one sample', unit: 'codes' }] },
      ];
    },
  };
}

function heaterDef() {
  return {
    id: 'heater', label: 'Heater',
    question: 'What is the heater driver doing, and how much range is left?',
    explain: 'The controller writes a DAC code; the driver puts a voltage on the heater resistor; the resistance changes with temperature (TCR).',
    tiles: (S) => {
      const h = settingsOf(S).heater;
      const ct = settingsOf(S).controller;
      return [
        { label: 'Heater code (applied)', note: `DAC register, ${h.bits} bits; rails ${h.code_lo} and ${h.code_hi}.`, get: (S, r, f) => {
          const c = devicesOf(S).heater.reg[f][r];
          return { v: `${c} (${pct((c - h.code_lo) / (h.code_hi - h.code_lo))})`, flag: c <= h.code_lo ? 'AT THE 0 RAIL' : c >= h.code_hi ? 'AT THE FULL-SCALE RAIL' : '' };
        } },
        { label: 'Headroom to the rails', note: 'Codes left below the lowest code (to 0) and above the highest code (to full scale) in this frame.', get: (S, r, f) => {
          const lo = ringsOf(S).heater_code_min[f][r] - h.code_lo;
          const hi = h.code_hi - ringsOf(S).heater_code_max[f][r];
          return { v: `down ${lo} · up ${hi} codes`, flag: Math.min(lo, hi) <= 0 ? 'NO HEADROOM' : '' };
        } },
        { label: 'Command → DAC code', note: `Controller command (after the channel clip) and the code at the DAC after its ${formatMicroseconds(h.dac_latency_s)} latency.`, get: (S, r, f) => `${devicesOf(S).heater.cmd[f][r]} → ${devicesOf(S).heater.dac_code[f][r]}` },
        { label: 'Heater voltage', note: `${h.law} drive: V = code / ${h.code_hi} × ${h.v_full_scale_v} V; compliance ${h.compliance_v} V.`, get: (S, r, f) => fmt(devicesOf(S).heater.v_v[f][r], 'V', 4) },
        { label: 'Heater current', note: 'I = V / R (frame mean of 10 µs samples).', get: (S, r, f) => fmt(devicesOf(S).heater.i_ma[f][r], 'mA') },
        { label: 'Heater resistance', note: `R = ${h.r_cold_ohm[0]} Ω × (1 + ${h.tcr_per_k}/K × (T_heater − 25 °C)).`, get: (S, r, f) => fmt(devicesOf(S).heater.r_ohm[f][r], 'Ω') },
        { label: 'Heater power', note: 'Electrical power, tick average (frame mean).', get: (S, r, f) => fmt(ringsOf(S).p_heater_mw[f][r], 'mW') },
        { label: 'Power at full code now', note: 'The driver law at code full scale and the present heater temperature; the difference is the power still available.', get: (S, r, f) => {
          const pf = devicesOf(S).heater.p_full_code_mw[f][r];
          return `${fmt(pf, 'mW')} (${fmt(pf - ringsOf(S).p_heater_mw[f][r], 'mW')} left)`;
        } },
        { label: 'Heater temperature', note: 'The heater-resistor observer; it sets R through the TCR.', get: (S, r, f) => fmt(toCelsius(ringsOf(S).t_heater_k[f][r]), '°C', 2) },
        { label: 'Largest code step per controller step', note: `Largest register change between two steps (${formatMicroseconds(ct.period_s)}) in this frame. Hardware slew limit: ${h.slew_codes_per_s == null ? 'none' : `${h.slew_codes_per_s} codes/s`}.`, get: (S, r, f) => fmt(devicesOf(S).heater.step_max_codes[f][r], 'codes') },
        { label: 'Bank supply use', note: `Sum of the 8 heaters against the supply limits ${h.supply_power_max_w} W and ${h.supply_current_max_a * 1e3} mA (largest in the frame).`, get: (S, r, f) => {
          const d = devicesOf(S).heater;
          return { v: `power ${pct(d.supply_p_frac_max[f])} · current ${pct(d.supply_i_frac_max[f])}`, flag: d.supply_scale_min[f] < 1 ? `SUPPLY LIMIT: drive scaled ×${d.supply_scale_min[f]}` : '' };
        } },
        { label: 'Clipped commands so far', note: 'Commands outside 0 to full scale that were clipped (cumulative, from the reset).', get: (S, r, f) => fmt(devicesOf(S).heater.sat_count[f][r]) },
      ];
    },
    plots: (S, r) => {
      const R = ringsOf(S);
      const d = devicesOf(S).heater;
      const h = settingsOf(S).heater;
      return [
        { title: 'code', traces: [{ y: colOf(d.reg, r), band: [colOf(R.heater_code_min, r), colOf(R.heater_code_max, r)], name: 'register', shape: 'hv', fmt: '.0f' }],
          hlines: [{ y: h.code_hi, label: `rail ${h.code_hi}` }, { y: h.code_lo, label: 'rail 0' }] },
        { title: 'voltage (V)', traces: [{ y: colOf(d.v_v, r), name: 'voltage', unit: 'V' }], hlines: [{ y: h.v_full_scale_v, label: `full scale ${h.v_full_scale_v} V` }] },
        { title: 'current (mA)', traces: [{ y: colOf(d.i_ma, r), name: 'current', unit: 'mA' }] },
        { title: 'R (Ω)', traces: [{ y: colOf(d.r_ohm, r), name: 'resistance', unit: 'Ω' }] },
        { title: 'power (mW)', traces: [{ y: colOf(R.p_heater_mw, r), name: 'power', unit: 'mW' }, { y: colOf(d.p_full_code_mw, r), name: 'at full code', unit: 'mW', dash: 'dash', color: cat(1) }] },
        { title: 'heater (°C)', traces: [{ y: colOf(R.t_heater_k, r).map(toCelsius), name: 'heater temperature', unit: '°C' }] },
        { title: 'step (codes)', traces: [{ y: colOf(d.step_max_codes, r), name: 'largest step per controller step', unit: 'codes', shape: 'hv' }] },
        { title: 'supply (%)', traces: [{ y: d.supply_p_frac_max.map((v) => v * 100), name: 'bank power', unit: '%' }, { y: d.supply_i_frac_max.map((v) => v * 100), name: 'bank current', unit: '%', color: cat(1), dash: 'dot' }], hlines: [{ y: 100, label: 'limit' }] },
      ];
    },
  };
}

function ringDef() {
  return {
    id: 'ring', label: 'Ring',
    question: 'Where is the resonance, and is the ring stable where it sits?',
    explain: 'Truth values of the ring: the controller never sees them. The light absorbed in the ring heats it (self-heating).',
    tiles: () => [
      { label: 'Detuning (true)', note: 'Resonance minus its declared line; + is red of the line (frame mean).', get: (S, r, f) => `${sgn(ringsOf(S).detuning_pm[f][r], 3)} pm` },
      { label: 'Lock error', note: 'Detuning minus the lock reference (the ring\'s operating point, or its line in the cold start).', get: (S, r, f) => {
        const e = S.err[f][r];
        return { v: `${sgn(e, 3)} pm (band ±${S.thr} pm)`, flag: Math.abs(e) > S.thr ? 'OUTSIDE THE LOCK BAND' : '' };
      } },
      { label: 'Engine lock state', note: 'Locked / unlocked with hysteresis, as the engine counts it (mrm/sim/metrics.py).', get: (S, r, f) => lockState(S, r, f) },
      { label: 'Ring temperature', note: 'The mode-weighted ring observer (frame mean).', get: (S, r, f) => fmt(toCelsius(ringsOf(S).t_ring_k[f][r]), '°C', 2) },
      { label: 'Absorbed light', note: 'Optical power absorbed in the ring, every line (frame mean). It heats the ring.', get: (S, r, f) => fmt(ringsOf(S).p_abs_mw[f][r], 'mW') },
      { label: 'Light in on its line', note: 'Power of the declared line at the ring input, after the rings before it.', get: (S, r, f) => fmt(ringsOf(S).p_in_mw[f][r], 'mW') },
      { label: 'Self-heating loop gain', note: 'Z_opt,self × dP_abs/dT at the nearest key frame. Below 1: the ring\'s own heating is a stable loop here.', get: (S, r, f) => {
        const g = devicesOf(S).key.selfheat_loop_gain[S.nearestKey[f]][r];
        return { v: fmt(g), flag: g >= 1 ? 'UNSTABLE (above 1)' : '' };
      } },
      { label: 'Self-heating branch', note: 'Red of the line, more heat moves the resonance away from the line; blue of it, toward the line.', get: (S, r, f) => (ringsOf(S).detuning_pm[f][r] > 0 ? 'red of its line (hot branch)' : 'blue of its line (cold branch)') },
      { label: 'relOMA (best possible)', note: 'OMA / P_in at the through port; the oracle is the best stable point at the same time.', get: (S, r, f) => `${fmt(ringsOf(S).reloma_db[f][r], 'dB')} (${fmt(ringsOf(S).oracle_reloma_db[f][r], 'dB')})` },
      { label: 'OMA lost to the oracle now', note: 'Oracle relOMA minus relOMA in this frame (dB).', get: (S, r, f) => fmt(ringsOf(S).oracle_reloma_db[f][r] - ringsOf(S).reloma_db[f][r], 'dB') },
      { label: 'Line', note: 'The line the ring sits on (within 1 FWHM) and its declared line.', get: (S, r, f) => lineStatus(S, r, f) },
    ],
    plots: (S, r) => {
      const R = ringsOf(S);
      const kt = keyT(S);
      const dref = S.dRef[r];
      return [
        { title: 'detuning (pm)', traces: [{ y: colOf(R.detuning_pm, r), band: [colOf(R.detuning_min_pm, r), colOf(R.detuning_max_pm, r)], name: 'detuning', unit: 'pm' }],
          hlines: [{ y: dref + S.thr, label: `lock band ±${S.thr} pm`, color: token('--series-3') }, { y: dref - S.thr, color: token('--series-3') }] },
        { title: 'ring (°C)', traces: [{ y: colOf(R.t_ring_k, r).map(toCelsius), name: 'ring temperature', unit: '°C' }] },
        { title: 'absorbed (mW)', traces: [{ y: colOf(R.p_abs_mw, r), name: 'absorbed light', unit: 'mW' }] },
        { title: 'relOMA (dB)', traces: [{ y: colOf(R.reloma_db, r), name: 'relOMA', unit: 'dB' }, { y: colOf(R.oracle_reloma_db, r), name: 'oracle', unit: 'dB', color: cat(1), dash: 'dot' }] },
        { title: 'loop gain', traces: [{ x: kt, y: devicesOf(S).key.selfheat_loop_gain.map((row) => row[r]), name: 'self-heating loop gain (key frames)', mode: 'lines+markers', marker: { size: 5 } }], hlines: [{ y: 1, label: 'stability limit 1' }] },
      ];
    },
  };
}

function laserDef() {
  return {
    id: 'laser', label: 'Laser and WDM',
    question: 'Which line reaches which ring, and how much light does each ring get?',
    explain: '8 lines on a 400 GHz grid enter the bus; each ring takes part of its own line and passes the rest on.',
    tiles: (S) => {
      const L = settingsOf(S).laser;
      return [
        { label: 'Own line', note: 'The declared line of the ring at this frame.', get: (S, r, f) => {
          const l = ringsOf(S).declared_line[f][r];
          return l < 0 ? 'none' : `L${l} · ${S.meta.lasers.line_nm[l].toFixed(3)} nm · ${S.meta.lasers.line_thz[l].toFixed(3)} THz`;
        } },
        { label: 'Line status', note: 'Own line, a foreign line (within 1 FWHM of another line), or no line.', get: (S, r, f) => lineStatus(S, r, f) },
        { label: 'Laser output, own line', note: 'Power at the laser module output (frame mean of 10 µs samples).', get: (S, r, f) => { const l = Math.max(0, ringsOf(S).declared_line[f][r]); return fmt(devicesOf(S).laser.p_mw[f][l], 'mW', 4); } },
        { label: 'Into the chip, own line', note: `× ${L.input_transmission} chip input transmission (${(10 * Math.log10(L.input_transmission)).toFixed(1)} dB)${S.drive.kind === 'laser' ? '; the laser step acts here' : ''}.`, get: (S, r, f) => { const l = Math.max(0, ringsOf(S).declared_line[f][r]); return fmt(devicesOf(S).laser.p_chip_mw[f][l], 'mW', 4); } },
        { label: 'At this ring, own line', note: 'After the rings before it on the bus (each takes some of its own line).', get: (S, r, f) => fmt(ringsOf(S).p_in_mw[f][r], 'mW') },
        { label: 'Frequency offset from the grid', note: `Drift ${L.f_rate_hz_per_s[0]} Hz/s and random walk ${L.f_rw_hz_per_rt_s[0]} Hz/√s in this world.`, get: (S, r, f) => { const l = Math.max(0, ringsOf(S).declared_line[f][r]); return fmt(devicesOf(S).laser.f_dev_mhz[f][l], 'MHz'); } },
        { label: 'Power spread in the frame (RIN)', note: `RIN ${L.rin_db_per_hz == null ? 'off' : `${L.rin_db_per_hz} dB/Hz`}${L.provenance['laser.rin_db_per_hz'] ? ` (${L.provenance['laser.rin_db_per_hz']})` : ''}, drawn per 1 µs tick; sampled every 10 µs.`, get: (S, r, f) => { const l = Math.max(0, ringsOf(S).declared_line[f][r]); return `${sgn(devicesOf(S).laser.p_ppm_min[f][l], 1)} to ${sgn(devicesOf(S).laser.p_ppm_max[f][l], 1)} ppm`; } },
        { label: 'Rings on a foreign line now', note: 'Count over the bank (engine wrong-line rule, 1 FWHM window).', get: (S, r, f) => {
          const bad = [];
          for (let q = 0; q < S.nR; q++) if (lineStatus(S, q, f).bad) bad.push(`R${q}`);
          return { v: bad.length ? `${bad.length}: ${bad.join(', ')}` : '0', flag: bad.length ? 'FOREIGN LINE' : '' };
        } },
      ];
    },
    plots: (S, r) => {
      const d = devicesOf(S).laser;
      const R = ringsOf(S);
      const lines = d.p_chip_mw[0].length;
      const pc = [];
      const fd = [];
      for (let l = 0; l < lines; l++) {
        pc.push({ y: colOf(d.p_chip_mw, l), name: `L${l}`, unit: 'mW', color: cat(l) });
        fd.push({ y: colOf(d.f_dev_mhz, l), name: `L${l}`, unit: 'MHz', color: cat(l), legend: false });
      }
      const pin = [];
      for (let q = 0; q < S.nR; q++) pin.push({ y: colOf(R.p_in_mw, q), name: `R${q}`, unit: 'mW', color: cat(q), legend: false, width: q === r ? 2.4 : 1 });
      const l0 = Math.max(0, R.declared_line[S.nF - 1][r]);
      return [
        { title: 'into chip (mW)', traces: pc },
        { title: 'f − grid (MHz)', traces: fd },
        { title: `L${l0} spread (ppm)`, traces: [{ y: colOf(d.p_ppm_max, l0), band: [colOf(d.p_ppm_min, l0), colOf(d.p_ppm_max, l0)], name: 'max', unit: 'ppm', legend: false }, { y: colOf(d.p_ppm_min, l0), name: 'min', unit: 'ppm', legend: false }] },
        { title: 'own line at ring (mW)', traces: pin },
      ];
    },
    legend: true,
    extra: {
      html: '<div class="sub2">Whole comb at the nearest key frame: each ring alone in its window (through port, data-averaged), the lines with their power at the chip input, every resonance.</div><div class="plot comb"></div>' +
        '<div class="grid2 tight"><div><div class="sub2">Power of each line reaching each ring (mW, key frame; rings in bus order)</div><div class="plot busmat"></div></div>' +
        '<div><div class="sub2">Line assignment at this frame</div><div class="tablewrap"><table class="mini assign"></table></div></div></div>',
      update: (host, S, r, k, f) => { drawComb(host.querySelector('.comb'), S, r, k); drawBusMatrix(host.querySelector('.busmat'), S, r, k); drawAssign(host.querySelector('.assign'), S, f); },
      perFrame: (host, S, r, k, f) => drawAssign(host.querySelector('.assign'), S, f),
    },
  };
}

function controllerDef() {
  return {
    id: 'ctrl', label: 'Controller vs Truth',
    question: 'What does the controller see, compared with what is true?',
    explain: 'The controller reads only ADC codes and writes heater codes. The detuning is Truth: it is never given to the controller.',
    tiles: (S) => {
      const mo = settingsOf(S).monitor;
      const ct = settingsOf(S).controller;
      const h = settingsOf(S).heater;
      const lat = S.scn.summary.controller_latency_ticks;
      return [
        { label: 'Monitor code (controller input)', note: 'Latest released ADC code at the frame end.', get: (S, r, f) => `${ringsOf(S).monitor_code[f][r]}` },
        { label: 'True detuning (Truth)', note: 'Resonance minus its declared line; the controller never sees it.', get: (S, r, f) => `${sgn(ringsOf(S).detuning_pm[f][r], 3)} pm` },
        { label: 'Lock error (Truth)', note: 'Detuning minus the lock reference.', get: (S, r, f) => `${sgn(S.err[f][r], 3)} pm` },
        { label: 'Command / register / DAC code', note: 'Controller command after the clip, latched register, code at the DAC after its latency (frame end).', get: (S, r, f) => `${devicesOf(S).heater.cmd[f][r]} / ${devicesOf(S).heater.reg[f][r]} / ${devicesOf(S).heater.dac_code[f][r]}` },
        { label: 'Command minus DAC code', note: 'Largest |command − DAC code| at the samples of this frame (latency and clip).', get: (S, r, f) => fmt(devicesOf(S).heater.cmd_minus_dac_max_abs[f][r], 'codes') },
        { label: 'Timing chain', note: 'From the light to the heater, as the engine schedules it.', get: () => `ADC ${formatMicroseconds(mo.integration_window_s)} window, sample every ${formatMicroseconds(mo.sample_period_s)}, ${formatMicroseconds(mo.latency_s)} latency → step every ${formatMicroseconds(ct.period_s)} (+${lat == null ? '–' : lat} ticks) → DAC ${formatMicroseconds(h.dac_latency_s)}` },
      ];
    },
    plots: (S, r) => {
      const R = ringsOf(S);
      const d = devicesOf(S).heater;
      return [
        { title: 'monitor (code)', traces: [{ y: colOf(R.monitor_code, r), band: [colOf(R.monitor_code_min, r), colOf(R.monitor_code_max, r)], name: 'controller input', shape: 'hv', fmt: '.0f' }] },
        { title: 'detuning (pm)', traces: [{ y: colOf(R.detuning_pm, r), name: 'Truth', unit: 'pm', color: cat(1) }] },
        { title: 'heater (code)', traces: [{ y: colOf(d.cmd, r), name: 'command', shape: 'hv', fmt: '.0f' }, { y: colOf(d.reg, r), name: 'register', shape: 'hv', dash: 'dot', color: cat(1), fmt: '.0f' }, { y: colOf(d.dac_code, r), name: 'DAC code', shape: 'hv', dash: 'dash', color: cat(2), fmt: '.0f' }] },
      ];
    },
    legend: true,
    extra: {
      html: '<div class="sub2">What the controller sees against the truth: monitor code against detuning, one dot per frame (the large dot is the scrubber frame). The shape is the discriminator the controller has to work with.</div><div class="plot scatter"></div>',
      update: (host, S, r, k, f) => drawScatter(host.querySelector('.scatter'), S, r, f),
      perFrame: (host, S, r, k, f) => moveScatter(host.querySelector('.scatter'), S, r, f),
    },
  };
}

function datapathDef() {
  return {
    id: 'data', label: 'Data path',
    question: 'Is the data link good?',
    explain: `NRZ data at ${'{br}'} Gb/s on every ring. The BER comes from the OMA at the bus output (mrm.plant.datapath).`,
    tiles: (S) => {
      const dp = settingsOf(S).datapath;
      return [
        { label: 'Ones density', note: 'Fraction of ones in the data (10 µs samples, frame mean).', get: (S, r, f) => fmt(devicesOf(S).datapath.ones_density[f][r], '', 4) },
        { label: 'BER (frame mean)', note: `Instantaneous BER of the declared line; limit ${S.scn.summary.ber_limit}.`, get: (S, r, f) => { const b = ringsOf(S).ber[f][r]; return { v: sci(b), flag: b > S.scn.summary.ber_limit ? 'ABOVE THE BER LIMIT' : '' }; } },
        { label: 'OMA at the bus output', note: '|P_out,1 − P_out,0| of the declared line at the end of the bus.', get: (S, r, f) => fmt(ringsOf(S).oma_uw[f][r], 'µW') },
        { label: 'OMA at the receiver', note: `× ${dp.link_transmission} link transmission; ${dp.oma_sensitivity_uw} µW gives Q = ${dp.q_ref}.`, get: (S, r, f) => fmt(ringsOf(S).oma_uw[f][r] * dp.link_transmission, 'µW') },
        { label: 'Extinction ratio', note: 'At the ring\'s through port, 10 log10(high / low level).', get: (S, r, f) => fmt(ringsOf(S).er_db[f][r], 'dB') },
        { label: 'Insertion loss', note: 'Of the high level at the ring\'s through port.', get: (S, r, f) => fmt(ringsOf(S).il_db[f][r], 'dB') },
        { label: 'Junction bias', note: 'Tick-average reverse bias at the junction.', get: (S, r, f) => fmt(ringsOf(S).bias_v[f][r], 'V', 3) },
        { label: 'Expected errors so far', note: 'Sum of BER × bits since scoring started (engine accumulator).', get: (S, r, f) => { const v = devicesOf(S).running.expected_errors[f]; return v == null ? 'not scored yet' : sci(v[r]); } },
      ];
    },
    plots: (S, r) => {
      const R = ringsOf(S);
      const floor = 1e-300;
      return [
        { title: 'BER', log: true, traces: [{ y: colOf(R.ber, r).map((v) => (v == null ? null : Math.max(v, floor))), name: 'BER', fmt: '.3g' }], hlines: [{ y: S.scn.summary.ber_limit, label: `limit ${S.scn.summary.ber_limit}` }] },
        { title: 'OMA (µW)', traces: [{ y: colOf(R.oma_uw, r), name: 'OMA at bus output', unit: 'µW' }] },
        { title: 'ER (dB)', traces: [{ y: colOf(R.er_db, r), name: 'extinction ratio', unit: 'dB' }] },
        { title: 'IL (dB)', traces: [{ y: colOf(R.il_db, r), name: 'insertion loss', unit: 'dB' }] },
        { title: 'ones density', traces: [{ y: colOf(devicesOf(S).datapath.ones_density, r), name: 'ones density', fmt: '.4f' }] },
      ];
    },
  };
}

function crosstalkDef() {
  return {
    id: 'xt', label: 'Thermal crosstalk',
    question: 'Who heats whom?',
    explain: 'Each ring temperature is the base plus the sum of its thermal lags. Grouping the lags by their heat source gives an exact split (key frames).',
    tiles: () => {
      const g = (key, label, note) => ({ label, note, get: (S, r, f) => fmt(riseGroups(S, r, S.nearestKey[f])[key], 'K', 2) });
      return [
        g('total', 'Rise above the base', 'Ring temperature minus the base temperature (key frame).'),
        g('ownHeater', 'From its own heater', 'Part of the rise caused by the ring\'s own heater.'),
        g('ownLight', 'From its own absorbed light', 'Self-heating: light absorbed in the ring.'),
        g('ownJD', 'From its own junction and driver', 'Data-activity heat of the junction and the driver.'),
        g('otherHeater', 'From the other heaters', 'Heater crosstalk from the other 7 rings.'),
        g('otherLight', 'From the other rings\' light', 'Absorbed light of the other rings.'),
        g('otherJD', 'From other junctions and drivers', 'Data heat of the other rings.'),
        g('aggressor', 'From the aggressor sites', 'The on-die aggressor (0 when it is off).'),
      ];
    },
    plots: (S, r) => {
      const kt = keyT(S);
      const K = S.keys.length;
      const ser = (key) => Array.from({ length: K }, (_, k) => riseGroups(S, r, k)[key]);
      return [
        { title: 'own (K)', traces: [{ x: kt, y: ser('ownHeater'), name: 'own heater', unit: 'K', mode: 'lines+markers', marker: { size: 4 } }, { x: kt, y: ser('ownLight'), name: 'own light', unit: 'K', color: cat(1), mode: 'lines+markers', marker: { size: 4, color: cat(1) } }] },
        { title: 'others (K)', traces: [{ x: kt, y: ser('otherHeater'), name: 'other heaters', unit: 'K', color: cat(2), mode: 'lines+markers', marker: { size: 4, color: cat(2) } }, { x: kt, y: ser('aggressor'), name: 'aggressor', unit: 'K', color: cat(7), mode: 'lines+markers', marker: { size: 4, color: cat(7) } }, { x: kt, y: ser('otherJD'), name: 'other junction + driver', unit: 'K', color: cat(3), mode: 'lines+markers', marker: { size: 4, color: cat(3) } }] },
      ];
    },
    legend: true,
    extra: {
      html: '<div class="tools left"><div class="segs" role="group" aria-label="Heat source kind" data-role="xtkind">' +
        '<button type="button" data-v="all" aria-pressed="true">All sources</button><button type="button" data-v="heater" aria-pressed="false">Heaters</button>' +
        '<button type="button" data-v="light" aria-pressed="false">Absorbed light</button><button type="button" data-v="jd" aria-pressed="false">Junction + driver</button></div></div>' +
        '<div class="grid2 tight"><div><div class="sub2">Rise of each ring (row) caused by each source (column), K, nearest key frame. Exact split of the engine\'s lag states.</div><div class="plot xtmat"></div></div>' +
        '<div><div class="sub2">Static coupling: K of ring rise per mW into each heater (DC gain of the model\'s thermal system). The diagonal is the heater efficiency.</div><div class="plot dcmat"></div></div></div>',
      bind: (host, panel) => {
        host.querySelectorAll('[data-role=xtkind] button').forEach((b) => b.addEventListener('click', () => {
          host.querySelectorAll('[data-role=xtkind] button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
          panel.xtKind = b.dataset.v;
          panel.refreshExtra();
        }));
      },
      update: (host, S, r, k, f, panel) => { drawXtMatrix(host.querySelector('.xtmat'), S, r, k, panel.xtKind || 'all'); drawDcMatrix(host.querySelector('.dcmat'), S, r); },
    },
  };
}

function disturbDef() {
  return {
    id: 'dist', label: 'Disturbances',
    question: 'What is pushing the rings in this scenario?',
    explain: 'The scenario inputs: aggressor power, die base temperature and laser power. They act on the plant, not on the controller.',
    tiles: () => [
      { label: 'Aggressor power', note: 'Power into each aggressor site (frame mean).', get: (S, r, f) => Object.entries(S.scn.drives.aggressor_w).map(([s, v]) => `${s === 'aggressor' ? 'default' : 'stress'} ${fmt(v[f], 'W')}`).join(' · ') || '0' },
      { label: 'Distance to the aggressor', note: 'From this ring\'s centre to the default site (assumed position).', get: (S, r) => fmt(S.meta.geometry.distances.ring_to_site_um.aggressor[r], 'µm') },
      { label: 'Aggressor heat at this ring', note: 'Part of the ring rise caused by the aggressor sites (key frame).', get: (S, r, f) => fmt(riseGroups(S, r, S.nearestKey[f]).aggressor, 'K', 3) },
      { label: 'Aggressor heat at its PD', note: 'The same at the monitor PD (key frame).', get: (S, r, f) => fmt(riseGroups(S, r, S.nearestKey[f], 'pd').aggressor, 'K', 3) },
      { label: 'Die base temperature', note: 'The package base that every rise is added to.', get: (S, r, f) => fmt(toCelsius(S.scn.drives.t_base_k[f]), '°C', 2) },
      { label: 'Laser power factor, own line', note: 'Chip power relative to nominal (scenario step × line enable).', get: (S, r, f) => fmt(S.scn.drives.laser_power_factor[f][Math.max(0, ringsOf(S).declared_line[f][r])], '×', 4) },
    ],
    plots: (S, r) => {
      const dr = S.scn.drives;
      const kt = keyT(S);
      const K = S.keys.length;
      const agg = Object.entries(dr.aggressor_w).map(([s, v], i) => ({ y: v, name: s, unit: 'W', color: cat(i) }));
      const ar = [];
      for (let q = 0; q < S.nR; q++) ar.push({ x: kt, y: Array.from({ length: K }, (_, k) => riseGroups(S, q, k).aggressor), name: `R${q}`, unit: 'K', color: cat(q), width: q === r ? 2.4 : 1, legend: false });
      const lpf = [];
      const L = dr.laser_power_factor[0].length;
      for (let l = 0; l < L; l++) lpf.push({ y: colOf(dr.laser_power_factor, l), name: `L${l}`, color: cat(l), shape: 'hv', legend: false });
      return [
        { title: 'aggressor (W)', traces: agg.length ? agg : [{ y: S.tMs.map(() => 0), name: 'none' }] },
        { title: 'aggressor heat (K)', traces: ar },
        { title: 'base (°C)', traces: [{ y: dr.t_base_k.map(toCelsius), name: 'base temperature', unit: '°C', legend: false }] },
        { title: 'laser (× nominal)', traces: lpf },
      ];
    },
    legend: true,
  };
}

export const DEFS = [pdDef, heaterDef, ringDef, laserDef, controllerDef, datapathDef, crosstalkDef, disturbDef];

// ------------------------------------------------------------------ key-frame views
function baseLayout(h, extra = {}) {
  const b = plotBase();
  return { paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel, height: h, margin: { l: 58, r: 16, t: 10, b: 44 }, ...extra };
}

function drawComb(gd, S, r, k) {
  const sp = S.spec;
  const b = plotBase();
  const ink = token('--text-primary');
  const muted = token('--text-muted');
  const data = [];
  const off = sp.grid.offset_nm;
  for (let q = 0; q < S.nR; q++) {
    const cen = sp.center_nm[k][q];
    const rho = sp.levels.ones_density[k][q];
    const y = sp.thru0_db[k][q].map((v0, i) => {
      const v1 = sp.thru1_db[k][q][i];
      return 10 * Math.log10(rho * 10 ** (v1 / 10) + (1 - rho) * 10 ** (v0 / 10));
    });
    data.push({ x: off.map((o) => cen + o), y, mode: 'lines', name: `R${q}`, line: { width: q === r ? 2.2 : 1.2, color: q === r ? ink : cat(q) },
      hovertemplate: `R${q}<br>%{x:.3f} nm<br>%{y:.2f} dB<extra></extra>` });
  }
  const lines = sp.line_nm[k];
  const f = S.keys[k];
  const pc = S.dev.laser.p_chip_mw[f];
  const shapes = lines.map((x) => ({ type: 'line', xref: 'x', yref: 'y', x0: x, x1: x, y0: -30, y1: 2, line: { width: 1, dash: 'dash', color: muted } }));
  const ann = lines.map((x, l) => ({ x, y: 2, xref: 'x', yref: 'y', yanchor: 'bottom', showarrow: false, text: `L${l} ${pc ? pc[l].toFixed(2) : ''} mW`, font: { size: 10, color: token('--text-secondary') } }));
  data.push({ x: sp.lam_res_nm[k], y: sp.lam_res_nm[k].map(() => -29), mode: 'markers', name: 'resonances', marker: { size: 7, symbol: 'triangle-up', color: sp.lam_res_nm[k].map((_, q) => (q === r ? ink : muted)) },
    hovertemplate: sp.lam_res_nm[k].map((_, q) => `R${q} resonance<br>%{x:.3f} nm<extra></extra>`), showlegend: false });
  renderPlot(gd, data, {
    ...baseLayout(400), showlegend: true, legend: { ...b.legend, orientation: 'h', y: -0.25 },
    xaxis: axis(b, { title: { text: 'wavelength (nm)' }, tickformat: '.1f' }), yaxis: axis(b, { title: { text: 'through (dB)' }, range: [-31, 6] }), shapes, annotations: ann,
  }, PLOT_CONFIG);
}

function drawBusMatrix(gd, S, r, k) {
  const b = plotBase();
  const order = S.meta.geometry.bus_order;
  const z = order.map((q) => S.spec.bus.p_in_mw[k][q]);
  const L = z[0].length;
  renderPlot(gd, [{ type: 'heatmap', z, x: Array.from({ length: L }, (_, l) => `L${l}`), y: order.map((q) => `R${q}`), colorscale: seqScale(), zmin: 0,
    text: z.map((row) => row.map((v) => v.toFixed(2))), texttemplate: '%{text}', textfont: { size: 12 }, hovertemplate: '%{y} input, %{x}: %{z:.4f} mW<extra></extra>', colorbar: { thickness: 10, title: { text: 'mW', side: 'right' } } }], {
    ...baseLayout(400), xaxis: axis(b, { side: 'bottom' }), yaxis: axis(b, { autorange: 'reversed' }),
  }, PLOT_CONFIG);
}

function drawAssign(tbl, S, f) {
  const R = ringsOf(S);
  let h = '<thead><tr><th>Ring</th><th>Declared</th><th>On now</th><th>Status</th><th>At ring (mW)</th></tr></thead><tbody>';
  for (let q = 0; q < S.nR; q++) {
    const st = lineStatus(S, q, f);
    h += `<tr><th>R${q}</th><td>${lineName(R.declared_line[f][q])}</td><td>${lineName(R.line_on[f][q])}</td><td>${st.flag ? `<span class="tag">${esc(st.flag)}</span>` : 'own line'}</td><td>${fmt(R.p_in_mw[f][q])}</td></tr>`;
  }
  tbl.innerHTML = `${h}</tbody>`;
}

function drawScatter(gd, S, r, f, all = false) {
  const b = plotBase();
  const R = ringsOf(S);
  if (all) {
    const data = Array.from({ length: S.nR }, (_, q) => ({ x: colOf(R.detuning_pm, q), y: colOf(R.monitor_code, q), mode: 'markers', name: `R${q}`, legendgroup: `R${q}`,
      marker: { size: q === r ? 6 : 4, color: cat(q), opacity: q === r ? 0.85 : 0.4 }, hovertemplate: `R${q}: %{x:.3f} pm → code %{y}<extra></extra>` }));
    data.push({ x: R.detuning_pm[f], y: R.monitor_code[f], mode: 'markers', name: 'scrubber frame', showlegend: false,
      marker: { size: Array.from({ length: S.nR }, (_, q) => q === r ? 14 : 10), color: Array.from({ length: S.nR }, (_, q) => cat(q)), line: { width: 2, color: token('--surface-1') } },
      text: Array.from({ length: S.nR }, (_, q) => `R${q}`), hovertemplate: '%{text} now: %{x:.3f} pm → code %{y}<extra></extra>' });
    renderPlot(gd, data, { ...baseLayout(420), showlegend: true, legend: { ...b.legend, orientation: 'h', y: -0.24 }, xaxis: axis(b, { title: { text: 'true detuning (pm)' } }), yaxis: axis(b, { title: { text: 'monitor code' } }) }, PLOT_CONFIG);
    return;
  }
  const x = colOf(R.detuning_pm, r);
  const y = colOf(R.monitor_code, r);
  const c = token('--series-1');
  renderPlot(gd, [
    { x, y, mode: 'markers', type: 'scatter', name: 'frames', marker: { size: 5, color: S.tMs, colorscale: [[0, hexA(c, 0.3)], [1, c]], showscale: true, colorbar: { thickness: 10, title: { text: 'ms', side: 'right' } } },
      hovertemplate: '%{x:.3f} pm → code %{y}<extra></extra>' },
    { x: [x[f]], y: [y[f]], mode: 'markers', name: 'now', marker: { size: 13, color: c, line: { width: 2, color: token('--surface-1') } }, hoverinfo: 'skip' },
  ], { ...baseLayout(400), showlegend: false, xaxis: axis(b, { title: { text: 'true detuning (pm)' } }), yaxis: axis(b, { title: { text: 'monitor code' } }) }, PLOT_CONFIG);
}

function moveScatter(gd, S, r, f, all = false) {
  if (!gd.data || gd.data.length < 2) return;
  const R = ringsOf(S);
  restylePlot(gd, all ? { x: [R.detuning_pm[f]], y: [R.monitor_code[f]] } : { x: [[R.detuning_pm[f][r]]], y: [[R.monitor_code[f][r]]] }, [all ? S.nR : 1]);
}

function seqScale() {
  const lo = token('--surface-2');
  const hi = token('--series-1');
  return [[0, lo], [1, hi]];
}

function drawXtMatrix(gd, S, r, k, kind) {
  const b = plotBase();
  const kk = S.dev.key;
  const names = kk.src_names;
  const cols = [];
  for (let j = 0; j < S.nR; j++) cols.push(`R${j}`);
  const sites = names.filter((n) => !/\d$/.test(n) || /^aggressor/.test(n)).filter((n) => !/^(heater|waveguide_absorption|junction|driver)\d+$/.test(n));
  const want = { all: /^(heater|waveguide_absorption|junction|driver)$/, heater: /^heater$/, light: /^waveguide_absorption$/, jd: /^(junction|driver)$/ }[kind];
  const z = [];
  for (let i = 0; i < S.nR; i++) {
    const row = new Array(S.nR + sites.length).fill(0);
    kk.ring_rise_by_source_k[k][i].forEach((v, s) => {
      const m = names[s].match(/^(heater|waveguide_absorption|junction|driver)(\d+)$/);
      if (m) { if (want.test(m[1])) row[Number(m[2])] += v || 0; } else if (kind === 'all') row[S.nR + sites.indexOf(names[s])] += v || 0;
    });
    z.push(row);
  }
  const x = cols.concat(sites.map((s) => (s === 'aggressor' ? 'aggr.' : 'stress')));
  renderPlot(gd, [{ type: 'heatmap', z, x, y: cols, colorscale: seqScale(), zmin: 0, text: z.map((row) => row.map((v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2)))), texttemplate: '%{text}', textfont: { size: 12 },
    hovertemplate: 'ring %{y} heated by %{x}: %{z:.4f} K<extra></extra>', colorbar: { thickness: 10, title: { text: 'K', side: 'right' } } }], {
    ...baseLayout(440), xaxis: axis(b, { title: { text: 'heat source (owner ring or site)' } }), yaxis: axis(b, { title: { text: 'heated ring' }, autorange: 'reversed' }),
    shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: r - 0.5, y1: r + 0.5, line: { width: 1.5, color: token('--text-primary') } }],
  }, PLOT_CONFIG);
}

function drawDcMatrix(gd, S, r) {
  const b = plotBase();
  const tc = S.meta.thermal_coupling;
  const hs = tc.src_names.map((n, s) => [n, s]).filter(([n]) => /^heater\d+$/.test(n)).map(([, s]) => s);
  const z = tc.ring_k_per_mw.map((row) => hs.map((s) => row[s]));
  const lab = hs.map((s) => `H${tc.src_names[s].replace('heater', '')}`);
  renderPlot(gd, [{ type: 'heatmap', z: z.map((row) => row.map((v) => Math.log10(Math.max(v, 1e-6)))), x: lab, y: z.map((_, i) => `R${i}`), colorscale: seqScale(),
    text: z.map((row) => row.map((v) => (v >= 1 ? v.toFixed(2) : v.toFixed(3)))), texttemplate: '%{text}', textfont: { size: 12 }, customdata: z,
    hovertemplate: 'ring %{y}, heater %{x}: %{customdata:.4f} K/mW<extra></extra>', colorbar: { thickness: 10, title: { text: 'log10 K/mW', side: 'right' } } }], {
    ...baseLayout(440), xaxis: axis(b, { title: { text: 'heater' } }), yaxis: axis(b, { title: { text: 'ring' }, autorange: 'reversed' }),
    shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: r - 0.5, y1: r + 0.5, line: { width: 1.5, color: token('--text-primary') } }],
  }, PLOT_CONFIG);
}

export { riseGroups, lineStatus, lockState, hexA, mapRows, drawComb, drawBusMatrix, drawAssign, drawScatter, moveScatter, drawXtMatrix, drawDcMatrix, baseLayout, seqScale };
