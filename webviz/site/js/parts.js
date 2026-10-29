// "Parts over the run": one screen per part of the bank (ring, heater, photodiode and ADC, laser / WDM / bus,
// controller I/O, data path). Each screen has stat tiles (the selected ring at the scrubber frame) and one stacked
// time plot of every quantity of that part over the whole run, for the selected ring or for all 8 rings overlaid.
// Every value comes from the data files (model output); the few differences (headroom, OMA lost, |cmd - DAC|)
// are differences of two exported values.

import { relayoutPlot, plotBase, axis, PLOT_CONFIG, token, renderPlot } from './theme.js';
import { Tiles, TimeStack, esc, cat, colOf, K0 } from './ui.js';
import { DEFS, lineStatus, drawComb, drawBusMatrix, drawAssign, drawScatter, moveScatter, baseLayout } from './panels.js';

const toCelsius = (kelvin) => (kelvin == null ? null : kelvin - K0);
// helpers on the scenario object S (the same names as in js/panels.js)
const ringsOf = (S) => S.scn.rings;                     // per-ring traces [frame][ring] (scenario_<id>.json)
const devicesOf = (S) => S.dev;                         // devices_<id>.json
const settingsOf = (S) => S.meta.devices;               // fixed device settings (meta.json devices)
const keyT = (S) => S.keys.map((f) => S.tMs[f]);        // key-frame times (ms)
const base = Object.fromEntries(DEFS.map((d) => { const x = d(); return [x.id, x]; }));
const tr = (y, name, extra = {}) => ({ y, name, ...extra });

// ------------------------------------------------------------------ the parts
const PARTS = [
  {
    id: 'ring', label: 'Ring', tiles: 'ring',
    question: 'Where is the resonance, and is the ring stable where it sits?',
    explain: 'Truth values of the ring: the controller never sees them. The light absorbed in the ring heats it (self-heating).',
    panels: [
      { title: 'detuning (pm)', question: "Where is the resonance relative to its laser?", description: "Resonance minus laser wavelength; positive values put the resonance at a longer wavelength. The band shows the frame’s minimum and maximum.", ring: (S, q, one) => [tr(colOf(ringsOf(S).detuning_pm, q), 'detuning', { unit: 'pm', band: one ? [colOf(ringsOf(S).detuning_min_pm, q), colOf(ringsOf(S).detuning_max_pm, q)] : null })],
        hlines: (S, r, all) => (all ? [{ y: 0, label: 'line' }] : [{ y: S.dRef[r] + S.thr, label: `lock band ±${S.thr} pm`, color: token('--series-3') }, { y: S.dRef[r] - S.thr, color: token('--series-3') }]) },
      { title: 'lock error (pm)', question: "How far is the ring from its lock reference?", description: "Detuning minus the run’s reference. Dashed lines mark the allowed lock-error band.", ring: (S, q) => [tr(S.err.map((row) => row[q]), 'lock error', { unit: 'pm' })],
        hlines: (S) => [{ y: S.thr, label: `±${S.thr} pm`, color: token('--series-3') }, { y: -S.thr, color: token('--series-3') }] },
      { title: 'ring (°C)', question: "How does the ring temperature change?", description: "The model’s mode-weighted ring temperature, averaged within each frame.", ring: (S, q) => [tr(colOf(ringsOf(S).t_ring_k, q).map(toCelsius), 'ring temperature', { unit: '°C' })] },
      { title: 'absorbed (mW)', question: "How much optical power heats the ring?", description: "Light absorbed in the ring and converted to heat, averaged within each frame.", ring: (S, q) => [tr(colOf(ringsOf(S).p_abs_mw, q), 'absorbed light', { unit: 'mW' })] },
      { title: 'relOMA (dB)', question: "How much data contrast does the ring preserve?", description: "Optical modulation amplitude divided by input power, in dB. In the single-ring view, the dotted trace is the model’s best stable value.", ring: (S, q, one) => [tr(colOf(ringsOf(S).reloma_db, q), 'relOMA', { unit: 'dB' })].concat(one ? [tr(colOf(ringsOf(S).oracle_reloma_db, q), 'oracle (best stable)', { unit: 'dB', color: cat(1), dash: 'dot' })] : []) },
      { title: 'OMA lost (dB)', question: "How much data contrast is lost?", description: "Best stable relOMA minus actual relOMA. Larger positive values mean more contrast has been lost.", ring: (S, q) => [tr(ringsOf(S).oracle_reloma_db.map((row, f) => row[q] - ringsOf(S).reloma_db[f][q]), 'oracle − relOMA', { unit: 'dB' })] },
      { title: 'branch', question: "Which side of the laser is the resonance on?", description: "+1 means resonance at a longer wavelength; −1 means shorter. This sign alone does not prove stability.", ring: (S, q) => [tr(colOf(ringsOf(S).detuning_pm, q).map((v) => (v == null ? null : v > 0 ? 1 : -1)), 'branch (+1 red, −1 blue)', { shape: 'hv', fmt: '.0f' })],
        range: [-1.4, 1.4] },
      { title: 'loop gain', question: "Does optical self-heating restore or amplify a disturbance?", description: "The local self-heating loop gain at the saved key frames. Below 1 is thermally stable; 1 is the local stability limit.", ring: (S, q) => [tr(devicesOf(S).key.selfheat_loop_gain.map((row) => row[q]), 'self-heating loop gain', { x: keyT(S), mode: 'lines+markers', marker: { size: 4 } })],
        hlines: () => [{ y: 1, label: 'stability limit 1' }] },
    ],
    note: (S) => `Branch: +1 when the resonance is red of its line (the hot branch: more heat moves it away from the line), −1 when blue. Loop gain at the ${S.keys.length} key frames; below 1 the ring's own heating is stable.`,
    bist: true,
  },
  {
    id: 'heater', label: 'Heater', tiles: 'heater',
    question: 'What is the heater driver doing, and how much range is left?',
    explain: 'The controller writes a DAC code; the driver puts a voltage on the heater resistor; the resistance changes with temperature (TCR).',
    panels: [
      { title: 'code', question: "What heater setting is being applied?", description: "The heater register code; the band shows its lowest and highest values within each frame.", ring: (S, q, one) => [tr(colOf(devicesOf(S).heater.reg, q), 'register', { shape: 'hv', fmt: '.0f', band: one ? [colOf(ringsOf(S).heater_code_min, q), colOf(ringsOf(S).heater_code_max, q)] : null })],
        hlines: (S) => [{ y: settingsOf(S).heater.code_hi, label: `rail ${settingsOf(S).heater.code_hi}` }, { y: settingsOf(S).heater.code_lo, label: `rail ${settingsOf(S).heater.code_lo}` }] },
      { title: 'headroom (code)', question: "How much adjustment remains before a heater rail?", description: "Distance in codes to the nearer command limit, using the frame’s extremes. Zero means a rail was reached.", ring: (S, q) => [tr(ringsOf(S).heater_code_min.map((row, f) => Math.min(row[q] - settingsOf(S).heater.code_lo, settingsOf(S).heater.code_hi - ringsOf(S).heater_code_max[f][q])), 'codes to the nearest rail', { shape: 'hv', fmt: '.0f' })],
        hlines: () => [{ y: 0, label: 'at a rail' }] },
      { title: 'voltage (V)', question: "What voltage reaches the heater?", description: "Actual heater voltage over time; the reference line marks the driver’s full-scale voltage.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.v_v, q), 'voltage', { unit: 'V' })], hlines: (S) => [{ y: settingsOf(S).heater.v_full_scale_v, label: `full scale ${settingsOf(S).heater.v_full_scale_v} V` }] },
      { title: 'current (mA)', question: "How much current flows through the heater?", description: "Electrical current through the heater resistor, averaged within each frame.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.i_ma, q), 'current', { unit: 'mA' })] },
      { title: 'R (Ω)', question: "Does warming change the heater resistance?", description: "The heater’s temperature-dependent resistance. Compare it with heater temperature and power.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.r_ohm, q), 'resistance', { unit: 'Ω' })] },
      { title: 'power (mW)', question: "How much electrical heating is delivered?", description: "Actual heater power. In the single-ring view, the dashed trace is the power available at full code at that temperature.", ring: (S, q, one) => [tr(colOf(ringsOf(S).p_heater_mw, q), 'power', { unit: 'mW' })].concat(one ? [tr(colOf(devicesOf(S).heater.p_full_code_mw, q), 'at full code', { unit: 'mW', dash: 'dash', color: cat(1) })] : []) },
      { title: 'heater (°C)', question: "How hot is the heater resistor?", description: "The heater resistor’s own temperature, which differs from the ring’s mode-weighted temperature.", ring: (S, q) => [tr(colOf(ringsOf(S).t_heater_k, q).map(toCelsius), 'heater temperature', { unit: '°C' })] },
      { title: 'step (code)', question: "How large are the controller’s heater adjustments?", description: "The largest register-code change during a controller step within each displayed frame.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.step_max_codes, q), 'largest step per controller step', { unit: 'codes', shape: 'hv' })] },
      { title: 'supply (%)', question: "Are the eight heaters approaching the shared supply limits?", description: "The highest bank power and current usage in each frame, as percentages of their respective limits. The limit is 100%.", common: (S) => [tr(devicesOf(S).heater.supply_p_frac_max.map((v) => v * 100), 'bank power', { unit: '%', color: token('--text-secondary'), legend: true }), tr(devicesOf(S).heater.supply_i_frac_max.map((v) => v * 100), 'bank current', { unit: '%', color: token('--text-muted'), dash: 'dot', legend: true })],
        hlines: () => [{ y: 100, label: 'supply limit' }] },
    ],
    note: () => 'Headroom: the smaller of (lowest code in the frame − low rail) and (high rail − highest code). Supply: the 8 heaters together against the bank supply limits (largest in each frame).',
  },
  {
    id: 'pd', label: 'Photodiode and ADC', tiles: 'pd',
    question: 'What does the monitor photodiode see, and what does it send to the controller?',
    explain: 'The PD sits at the end of the drop waveguide. Its current is integrated, digitized and sent to the controller as a code.',
    panels: [
      { title: 'drop (µW)', question: "How much light reaches the monitor?", description: "Optical power entering the monitor photodiode from the drop port, including all laser lines, averaged over the data.", ring: (S, q) => [tr(colOf(ringsOf(S).p_drop_uw, q), 'drop power', { unit: 'µW' })] },
      { title: 'current (µA)', question: "How much current comes from light, and how much is background?", description: "Total photodiode current includes signal plus dark current. The single-ring view also shows the optical signal alone as a dotted trace.", ring: (S, q, one) => [tr(colOf(ringsOf(S).i_monitor_ua, q), 'total', { unit: 'µA' })].concat(one ? [tr(colOf(devicesOf(S).pd.i_signal_ua, q), 'signal', { unit: 'µA', color: cat(1), dash: 'dot' })] : []) },
      { title: 'dark (nA)', question: "How much current exists without light?", description: "Modelled dark current in nA. It rises with photodiode temperature; compare it with the temperature plot below.", ring: (S, q) => [tr(colOf(devicesOf(S).pd.i_dark_na, q), 'dark current', { unit: 'nA' })] },
      { title: 'PD (°C)', question: "How warm is the photodiode?", description: "The photodiode’s own temperature, rather than the ring temperature. This temperature sets its dark-current contribution.", ring: (S, q) => [tr(colOf(ringsOf(S).t_pd_k, q).map(toCelsius), 'PD temperature', { unit: '°C' })] },
      { title: 'ADC (code)', question: "What reading is actually sent to the controller?", description: "The latest released ADC code at each frame end. The band shows the frame’s code range; the reference lines mark 0 and full scale.", ring: (S, q, one) => [tr(colOf(ringsOf(S).monitor_code, q), 'code', { shape: 'hv', fmt: '.0f', band: one ? [colOf(ringsOf(S).monitor_code_min, q), colOf(ringsOf(S).monitor_code_max, q)] : null })],
        hlines: (S) => [{ y: settingsOf(S).monitor.code_max, label: `full scale ${settingsOf(S).monitor.code_max}` }, { y: 0, label: '0' }] },
      { title: 'over-range (%)', question: "How often does the ADC exceed its measurement range?", description: "Percentage of saved device samples whose latest ADC reading was flagged over-range. It indicates clipping, rather than counting every ADC conversion.", ring: (S, q) => [tr(colOf(devicesOf(S).pd.overrange_frac, q, 100), 'over-range samples', { unit: '%', shape: 'hv' })] },
      { title: 'noise (code RMS)', question: "How much does one ADC reading fluctuate?", description: "Expected noise size (RMS, or root mean square) of one ADC sample, in code units. This differs from the full code spread within a frame.", ring: (S, q) => [tr(colOf(devicesOf(S).pd.noise_rms_codes, q), 'white noise of one sample', { unit: 'codes rms' })] },
    ],
    note: (S) => `ADC band: lowest to highest released code in the frame. Clipping: the code reaches full scale (${settingsOf(S).monitor.code_max}); the over-range panel shows how often saved device samples have the over-range flag.`,
  },
  {
    id: 'laser', label: 'Laser, WDM and bus', tiles: 'laser',
    question: 'Which line reaches which ring, and how much light does each ring get?',
    explain: '8 lines on a 400 GHz grid enter the bus; each ring takes part of its own line and passes the rest on.',
    panels: [
      { title: 'into chip (mW)', question: "How much optical power enters the chip on each line?", description: "Input power after chip coupling, with one trace per laser line. These are powers at the chip input, before upstream rings take light.", common: (S) => devicesOf(S).laser.p_chip_mw[0].map((_, l) => tr(colOf(devicesOf(S).laser.p_chip_mw, l), `L${l}`, { unit: 'mW', color: cat(l), group: `L${l}`, legend: true })) },
      { title: 'f − grid (MHz)', question: "Does a laser move away from its grid frequency?", description: "Each line’s frequency offset from its nominal grid position, in MHz. Zero means no frequency shift.", common: (S) => devicesOf(S).laser.f_dev_mhz[0].map((_, l) => tr(colOf(devicesOf(S).laser.f_dev_mhz, l), `L${l}`, { unit: 'MHz', color: cat(l), group: `L${l}`, legend: false })) },
      { title: 'own line at ring (mW)', question: "How much of the assigned laser line reaches each ring?", description: "Power on the declared line after attenuation by upstream rings and the bus waveguide.", ring: (S, q) => [tr(colOf(ringsOf(S).p_in_mw, q), 'light in on its declared line', { unit: 'mW' })] },
      { title: 'declared line', question: "Which laser line has the controller assigned?", description: "The declared line index over time. A declaration is an assignment, not proof that the ring is resonant with that line.", ring: (S, q) => [tr(colOf(ringsOf(S).declared_line, q), 'declared line', { shape: 'hv', fmt: '.0f' })] },
      { title: 'line on (−1 none)', question: "Which laser line is the ring actually near?", description: "The nearest line within one linewidth of resonance; −1 means none. Compare this index with the declared line.", ring: (S, q) => [tr(colOf(ringsOf(S).line_on, q), 'line it sits on', { shape: 'hv', fmt: '.0f' })] },
      { title: 'status', question: "Is the ring near its assigned line?", description: "0 means its assigned line, 1 means a foreign line, and −1 means no line within one linewidth. This is a line-assignment check, not the lock-error test.", ring: (S, q) => [tr(S.tMs.map((_, f) => { const st = lineStatus(S, q, f); return st.bad ? 1 : st.flag ? -1 : 0; }), 'status: 0 own line, 1 foreign, −1 none', { shape: 'hv', fmt: '.0f' })],
        range: [-1.4, 1.4] },
    ],
    note: () => 'Status: 0 = on its own line, 1 = on a foreign line (within 1 FWHM of another line: it would carry another channel), −1 = on no line. Line indices: L0 is the reddest line.',
    extra: {
      html: '<h3>Where does light disappear along the bus?</h3><p class="sub">Read time horizontally and bus segments vertically, from input to output. Stronger color means more total optical power across all laser lines. Compare successive rows to see what remains after each ring; the vertical line marks the scrubber.</p><div class="plot busrun"></div>' +
        '<h3>Where do the rings sit relative to the laser comb?</h3><p class="sub">Each curve is one ring’s through spectrum at the nearest key frame, averaged over its data. Dips locate resonances; vertical lines locate laser wavelengths and label their chip-input power. The selected ring is thicker. These are individual ring spectra, not the combined bus transmission.</p><div class="plot comb"></div>' +
        '<div class="grid2 tight"><div><h3>How much of each laser line reaches each ring?</h3><p class="sub">Columns are laser lines; rows follow the rings along the bus. Cell values show incoming power in mW at the nearest key frame. Compare down a column to see attenuation by upstream rings and the waveguide.</p><div class="plot busmat"></div></div>' +
        '<div><div class="sub2">Line assignment at this frame</div><div class="tablewrap"><table class="mini assign"></table></div></div></div>',
      once: (host, S) => drawBusRun(host.querySelector('.busrun'), S),
      update: (host, S, r, k, f) => { drawComb(host.querySelector('.comb'), S, r, k); drawBusMatrix(host.querySelector('.busmat'), S, r, k); drawAssign(host.querySelector('.assign'), S, f); moveBusCursor(host.querySelector('.busrun'), S, f); },
      perFrame: (host, S, r, k, f) => { drawAssign(host.querySelector('.assign'), S, f); moveBusCursor(host.querySelector('.busrun'), S, f); },
    },
  },
  {
    id: 'ctrl', label: 'Controller I/O', tiles: 'ctrl',
    question: 'What does the controller see, compared with what is true?',
    explain: 'The controller reads only ADC codes and writes heater codes. The detuning is Truth: it is never given to the controller.',
    panels: [
      { title: 'input (code)', question: "What measurement can the controller use?", description: "Released monitor ADC codes, including measurement imperfections. The band shows the code range within each frame.", ring: (S, q, one) => [tr(colOf(ringsOf(S).monitor_code, q), 'monitor code (controller input)', { shape: 'hv', fmt: '.0f', band: one ? [colOf(ringsOf(S).monitor_code_min, q), colOf(ringsOf(S).monitor_code_max, q)] : null })] },
      { title: 'detuning (pm)', question: "What is the actual optical detuning?", description: "The simulator’s true resonance-minus-laser detuning. This diagnostic is not given to the controller.", ring: (S, q) => [tr(colOf(ringsOf(S).detuning_pm, q), 'Truth detuning', { unit: 'pm' })] },
      { title: 'lock error (pm)', question: "Does the true operating point remain inside the lock band?", description: "True detuning minus the run’s lock reference; dashed lines show the allowed error.", ring: (S, q) => [tr(S.err.map((row) => row[q]), 'Truth lock error', { unit: 'pm' })], hlines: (S) => [{ y: S.thr, color: token('--series-3') }, { y: -S.thr, color: token('--series-3') }] },
      { title: 'output (code)', question: "Does the heater output follow the requested command?", description: "In the single-ring view, compare controller command, register, and applied DAC code. Their differences include command timing and actuator limits.", ring: (S, q, one) => [tr(colOf(devicesOf(S).heater.cmd, q), 'command', { shape: 'hv', fmt: '.0f' })].concat(one ? [tr(colOf(devicesOf(S).heater.reg, q), 'register', { shape: 'hv', dash: 'dot', color: cat(1), fmt: '.0f' }), tr(colOf(devicesOf(S).heater.dac_code, q), 'DAC code', { shape: 'hv', dash: 'dash', color: cat(2), fmt: '.0f' })] : []) },
      { title: '|cmd − DAC| (code)', question: "How far does the applied DAC code lag the command?", description: "Absolute command-to-DAC difference at the last device sample in each frame. This is a difference in codes, not a delay measured in time.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.cmd_minus_dac_max_abs, q), '|command − DAC code| at last device sample', { shape: 'hv', fmt: '.0f' })] },
      { title: 'step (code)', question: "How aggressively is the controller changing the heater?", description: "The largest register-code step during a controller update within each frame.", ring: (S, q) => [tr(colOf(devicesOf(S).heater.step_max_codes, q), 'largest register step', { shape: 'hv', fmt: '.0f' })] },
    ],
    note: (S) => {
      const t = settingsOf(S).timing;
      if (!t) return '';
      const us = (n) => `${n * t.dt_s * 1e6} µs`;
      return `Timing chain: the ADC integrates ${us(t.adc_window_ticks)} and gives a code every ${us(t.adc_period_ticks)}, ${us(t.adc_latency_ticks)} later; the controller steps every ${us(t.controller_period_ticks)} (latency ${us(t.controller_latency_ticks || 0)}); the DAC output follows ${us(t.dac_latency_ticks)} after the latch.`;
    },
    extra: {
      html: '<h3>Can the monitor reading distinguish the true operating point?</h3><p class="sub">Each dot pairs true detuning with the released monitor code for one frame. In All 8 mode, colors identify rings; larger dots mark their scrubber frames. In single-ring mode, color shows time. Where readings repeat at different detunings, the code alone is ambiguous. The controller receives the code, not the true detuning.</p><div class="plot scatter"></div>',
      update: (host, S, r, k, f, all) => drawScatter(host.querySelector('.scatter'), S, r, f, all),
      perFrame: (host, S, r, k, f, all) => moveScatter(host.querySelector('.scatter'), S, r, f, all),
    },
  },
  {
    id: 'data', label: 'Data path', tiles: 'data',
    question: 'Is the data link good?',
    explain: 'NRZ data on every ring. The BER comes from the OMA of the declared line at the bus output (mrm.plant.datapath).',
    panels: [
      { title: 'ones density', question: "What fraction of transmitted bits are ones?", description: "Mean ones density in the selected data stream. A value of 0.5 means equal proportions of zeros and ones.", ring: (S, q) => [tr(colOf(devicesOf(S).datapath.ones_density, q), 'ones density', { fmt: '.4f' })] },
      { title: 'BER', question: "What bit-error rate does the model predict?", description: "Predicted BER of the declared line, averaged within each frame and shown on a logarithmic axis. The reference line marks the run’s BER limit.", log: true, ring: (S, q) => [tr(colOf(ringsOf(S).ber, q).map((v) => (v == null ? null : Math.max(v, 1e-300))), 'BER', { fmt: '.3g' })], hlines: (S) => [{ y: S.scn.summary.ber_limit, label: `limit ${S.scn.summary.ber_limit}` }] },
      { title: 'OMA bus out (µW)', question: "How much optical separation remains between zero and one?", description: "Optical modulation amplitude (OMA): the difference between the two data power levels at the bus output.", ring: (S, q) => [tr(colOf(ringsOf(S).oma_uw, q), 'OMA at the bus output', { unit: 'µW' })] },
      { title: 'OMA at Rx (µW)', question: "How much of that data contrast reaches the receiver?", description: "Bus-output OMA after link attenuation. The reference line is the receiver’s OMA sensitivity at BER 10⁻⁹.", ring: (S, q) => [tr(colOf(ringsOf(S).oma_uw, q, settingsOf(S).datapath.link_transmission), 'OMA at the receiver', { unit: 'µW' })], hlines: (S) => [{ y: settingsOf(S).datapath.oma_sensitivity_uw, label: `sensitivity ${settingsOf(S).datapath.oma_sensitivity_uw} µW (BER 1e-9)` }] },
      { title: 'ER (dB)', question: "How different are the optical one and zero levels?", description: "Extinction ratio (ER), expressed in dB: the ratio of the higher optical level to the lower level.", ring: (S, q) => [tr(colOf(ringsOf(S).er_db, q), 'extinction ratio', { unit: 'dB' })] },
      { title: 'IL (dB)', question: "How much optical power is lost through the modulator?", description: "Insertion loss (IL) in dB, using the model’s data-path definition. Larger positive loss means less transmitted power.", ring: (S, q) => [tr(colOf(ringsOf(S).il_db, q), 'insertion loss', { unit: 'dB' })] },
      { title: 'bias (V)', question: "What electrical bias sets the data operating point?", description: "The PN-junction bias voltage, averaged within each frame. This is separate from the heater command.", ring: (S, q) => [tr(colOf(ringsOf(S).bias_v, q), 'junction bias', { unit: 'V', fmt: '.4f' })] },
      { title: 'errors so far', question: "How many bit errors are expected to have accumulated?", description: "The sum of predicted BER times transmitted bits since scoring began. It is an expected count, not a measured integer count.", log: true, ring: (S, q) => [tr(devicesOf(S).running.expected_errors.map((row) => (row == null || row[q] == null ? null : Math.max(row[q], 1e-300))), 'expected errors since scoring started', { fmt: '.3g' })] },
    ],
    note: (S) => `OMA at the receiver = bus-output OMA × ${settingsOf(S).datapath.link_transmission} link transmission; at ${settingsOf(S).datapath.oma_sensitivity_uw} µW the BER is 1e-9 (Q = ${settingsOf(S).datapath.q_ref}).`,
  },
];

export const PART_IDS = PARTS.map((p) => [p.id, p.label]);

// ------------------------------------------------------------------ bus light over the run (key frames)
function drawBusRun(gd, S) {
  if (!S.bus) { gd.innerHTML = ''; return; }
  const b = plotBase();
  const order = S.meta.geometry.bus_order;
  const segs = order.map((q, j) => (j === 0 ? `in → R${q}` : `R${order[j - 1]} → R${q}`)).concat([`R${order[order.length - 1]} → out`]);
  const z = segs.map((_, j) => S.bus.map((row) => row[j]));
  const kt = keyT(S);
  renderPlot(gd, [{ type: 'heatmap', x: kt, y: segs, z, colorscale: [[0, token('--surface-2')], [1, token('--bus-lit')]], zmin: 0,
    hovertemplate: '%{y}<br>%{x:.2f} ms: %{z:.3f} mW<extra></extra>', colorbar: { thickness: 10, title: { text: 'mW', side: 'right' } } }], {
    ...baseLayout(400), margin: { l: 90, r: 16, t: 10, b: 40 }, xaxis: axis(b, { title: { text: 'time (ms)' }, range: [0, S.m.duration_s * 1e3] }), yaxis: axis(b, { autorange: 'reversed' }),
    shapes: [{ type: 'line', xref: 'x', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1, line: { width: 1.5, color: token('--text-primary') } }],
  }, PLOT_CONFIG);
}

function moveBusCursor(gd, S, f) {
  if (!gd || !gd.layout || !gd.layout.shapes) return;
  relayoutPlot(gd, { 'shapes[0].x0': S.tMs[f], 'shapes[0].x1': S.tMs[f] });
}

// ------------------------------------------------------------------ the view
export class PartsView {
  constructor(card, bistCard, { onSeek, onPart }) {
    this.card = card;
    this.bistCard = bistCard;
    this.tabs = card.querySelector('[data-role=parts]');
    this.q = card.querySelector('.dq');
    this.ex = card.querySelector('.dx');
    this.note = card.querySelector('[data-role=ringnote]');
    this.tiles = new Tiles(card.querySelector('.tiles'));
    this.extraHost = card.querySelector('.extra');
    this.stack = new TimeStack(card.querySelector('.plotwrap'), { onSeek, height: 104 });
    this.onPart = onPart;
    this.part = PARTS[0];
    this.tabs.innerHTML = PARTS.map((p) => `<button type="button" role="tab" data-id="${p.id}" aria-pressed="false">${esc(p.label)}</button>`).join('');
    this.tabs.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-id]');
      if (b) { this.setPart(b.dataset.id); this.onPart(b.dataset.id); }
    });
  }

  setPart(id) {
    this.part = PARTS.find((p) => p.id === id) || PARTS[0];
    this.tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === this.part.id)));
    if (this.S) this.render();
  }

  setState(S, r, all, f) { this.S = S; this.r = r; this.all = all; this.f = f; }

  render() {
    const S = this.S;
    const p = this.part;
    if (!S || !S.dev) return;
    this.tabs.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === p.id)));
    this.q.textContent = p.question;
    this.ex.textContent = p.explain;
    this.note.textContent = `${this.all ? `All ${S.nR} rings compared (R${this.r} emphasized). Identical curves overlap; use the legend to isolate a ring.` : `Ring R${this.r}. Toggle All 8 to compare the bank.`} ${p.note ? p.note(S) : ''}`;
    const bt = base[p.tiles];
    this.tiles.build(bt.tiles(S), { rings: this.all ? S.nR : 0, selected: this.r });
    this.tiles.update(S, this.r, this.f);
    this.stack.render(S, this.panels(S), { legend: true, ring: this.r });
    this.stack.setTime(S.tMs[this.f]);
    this.extraHost.innerHTML = p.extra ? p.extra.html : '';
    this.extraHost.hidden = !p.extra;
    if (p.extra && p.extra.once) p.extra.once(this.extraHost, S);
    this.lastKey = -1;
    this.refreshExtra();
    this.bistCard.hidden = !p.bist;
  }

  panels(S) {
    const r = this.r;
    let first = true;
    return this.part.panels.flatMap((pn) => {
      let traces = [];
      const extraPanels = [];
      if (pn.ring) {
        if (this.all) {
          const perRing = Array.from({ length: S.nR }, (_, q) => pn.ring(S, q, true));
          const compare = (index) => perRing.map((ts, q) => {
            const t = ts[index];
            return { ...t, band: null, name: `R${q}`, group: `R${q}`, color: cat(q), marker: { ...t.marker, color: cat(q) }, width: q === r ? 2.6 : 1.1, legend: first };
          });
          traces = compare(0);
          const copy = {
            'oracle (best stable)': ['How does the best stable modulation compare across rings?', 'The model’s best stable normalized modulation at the same instants. Compare with each ring’s actual relOMA above.'],
            'at full code': ['How much power could each heater deliver at full code?', 'Available power at each heater’s current temperature. Compare with actual power above to see remaining authority.'],
            'signal': ['How much photodiode current comes from light alone?', 'Signal photocurrent for every ring, excluding dark current. Compare with total current above and dark current below.'],
            'register': ['What heater code is latched for each ring?', 'The heater register after command clipping and latching. Compare with requested commands above and applied DAC codes below.'],
            'DAC code': ['What heater code actually reaches each DAC?', 'Applied DAC codes for every ring, after actuator latency. Compare with each ring’s command and register.'],
          };
          for (let j = 1; j < perRing[0].length; j++) {
            const t = perRing[0][j];
            const [question, description] = copy[t.name];
            extraPanels.push({ title: t.name + (t.unit ? ` (${t.unit})` : ' (code)'), question, description, traces: compare(j) });
          }
          first = false;
        } else traces = pn.ring(S, r, true).map((t, i) => ({ ...t, legend: i > 0 || !!t.legend, color: t.color || token('--series-1') }));
      }
      if (pn.common) traces = traces.concat(pn.common(S, r, this.all));
      let description = pn.description;
      if (this.all && pn.ring) description = description.replace(/In the single-ring view, compare controller command, register, and applied DAC code\./, 'Controller command for each ring; compare with register and DAC rows below.').replace(/ The band shows[^.]*\./, '').replace(/; the band shows[^.]*\./, '.').replace(/ In the single-ring view,[^.]*\./, '').replace(/ The single-ring view[^.]*\./, '');
      return [{ title: pn.title, question: pn.question, description, traces, hlines: pn.hlines ? pn.hlines(S, r, this.all) : [], log: pn.log, range: pn.range }, ...extraPanels];
    });
  }

  refreshExtra() {
    const p = this.part;
    const S = this.S;
    if (!p.extra || !S) return;
    this.lastKey = S.nearestKey[this.f];
    p.extra.update(this.extraHost, S, this.r, this.lastKey, this.f, this.all);
  }

  setFrame(f) {
    this.f = f;
    const S = this.S;
    if (!S || !S.dev) return;
    this.tiles.update(S, this.r, f);
    this.stack.setTime(S.tMs[f]);
  }

  slowUpdate() {
    const p = this.part;
    const S = this.S;
    if (!p.extra || !S || !S.dev) return;
    const k = S.nearestKey[this.f];
    if (k !== this.lastKey) this.refreshExtra();
    else if (p.extra.perFrame) p.extra.perFrame(this.extraHost, S, this.r, k, this.f, this.all);
  }
}
