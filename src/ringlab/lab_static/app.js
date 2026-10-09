"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const form = byId("run-form");
  const deviceCanvas = byId("device-canvas");
  const controlKeys = ["target_code", "initial_code", "step_codes", "deadband_codes", "warmup_ms", "ambient_step_k"];
  const state = {
    metadata: null, result: null, index: 0, busy: false, playing: false,
    lastFrame: 0, playPosition: 0, frame: 0, minTemperature: 300, maxTemperature: 301,
    camera: { yaw: -0.45, pitch: 0.73, zoom: 1 }, drag: null, colors: {},
  };
  const charts = [
    { id: "chart-monitor", name: "monitor", key: "monitor_code", scale: 1, decimals: 0, units: "code" },
    { id: "chart-heater", name: "heater", key: "heater_code", scale: 1, decimals: 0, units: "code" },
    { id: "chart-detuning", name: "detuning", key: "detuning_hz", scale: 1e-9, decimals: 3, units: "GHz" },
    { id: "chart-oma", name: "oma", key: "oma_w", scale: 1e6, decimals: 3, units: "µW" },
  ];

  const finite = (v) => Number.isFinite(Number(v));
  const scalar = (value) => Array.isArray(value) ? Number(value[0]) : Number(value);
  const at = (trace, key, index) => scalar(trace[key]?.[index]);
  const format = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "—";
  const currentTrace = () => state.result?.runs[byId("device-run").value]?.trace;
  const escapeNumber = (value) => typeof value === "number" ? (Math.abs(value) >= 1e6 || (Math.abs(value) < 0.0001 && value !== 0) ? value.toExponential(4) : String(value)) : String(value);

  function resolveColors() {
    const style = getComputedStyle(document.documentElement);
    for (const name of ["ink", "muted", "line", "page", "panel", "accent", "chart-hold", "warn", "accent-soft", "viewer", "viewer-ink", "viewer-muted", "mesh-edge", "waveguide", "substrate-edge", "heat-low", "heat-high"]) {
      state.colors[name] = style.getPropertyValue(`--${name}`).trim();
    }
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { context, width, height };
  }

  function renderParameterObject(parent, value) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const primitiveEntries = Object.entries(value).filter(([, item]) => !item || typeof item !== "object" || Array.isArray(item));
      if (primitiveEntries.length) {
        const table = document.createElement("table");
        table.className = "parameter-table";
        const body = document.createElement("tbody");
        for (const [name, item] of primitiveEntries) {
          const row = document.createElement("tr");
          const key = document.createElement("td"), cell = document.createElement("td");
          key.textContent = name;
          cell.textContent = Array.isArray(item) ? JSON.stringify(item) : escapeNumber(item);
          row.append(key, cell);
          body.append(row);
        }
        table.append(body);
        parent.append(table);
      }
      for (const [name, item] of Object.entries(value)) {
        if (!item || typeof item !== "object" || Array.isArray(item)) continue;
        const details = document.createElement("details"), summary = document.createElement("summary");
        summary.textContent = name.replaceAll("_", " ");
        details.append(summary);
        renderParameterObject(details, item);
        parent.append(details);
      }
    }
  }

  function showModel(metadata) {
    state.metadata = metadata;
    byId("model-id").textContent = metadata.model_id;
    byId("model-short-hash").textContent = `SHA-256 ${metadata.model_sha256.slice(0, 16)}…`;
    byId("model-hash").textContent = metadata.model_sha256;
    byId("implementation-hash").textContent = metadata.implementation_sha256 || "Not supplied";
    byId("resolved-model").replaceChildren();
    renderParameterObject(byId("resolved-model"), { ...metadata.model, provenance: metadata.provenance || {} });
    for (const key of controlKeys) {
      const input = form.elements.namedItem(key);
      const limit = metadata.limits?.[key];
      if (limit) {
        for (const attribute of ["min", "max", "step"]) {
          if (limit[attribute] !== undefined) input.setAttribute(attribute, String(limit[attribute]));
        }
      }
      if (metadata.defaults?.[key] !== undefined) input.value = String(metadata.defaults[key]);
    }
    updateExperimentDescription();
  }

  function readControls() {
    return Object.fromEntries(controlKeys.map((key) => [key, Number(form.elements.namedItem(key).value)]));
  }

  function updateExperimentDescription() {
    const value = Number(byId("warmup-ms").value);
    byId("experiment-description").textContent = `Hold the initial code for ${finite(value) ? value : "—"} ms, then enable feedback. The ambient temperature steps at 2 ms. Warmup is not acquisition.`;
  }

  function setBusy(busy) {
    state.busy = busy;
    byId("run-button").disabled = busy || !state.metadata;
    byId("run-button-label").textContent = busy ? "Simulating both runs…" : "Run comparison";
    byId("download").disabled = busy || !state.result;
    byId("results").setAttribute("aria-busy", String(busy));
    for (const key of controlKeys) form.elements.namedItem(key).disabled = busy;
    byId("play").disabled = busy || !state.result;
    byId("time-index").disabled = busy || !state.result;
  }

  function showError(message) {
    byId("error-message").textContent = message;
    byId("error-message").hidden = false;
    byId("form-state").textContent = state.result ? "Run failed. The last completed comparison is still shown." : "No simulation result is available.";
  }

  async function fetchJson(path, options = {}) {
    const response = await fetch(path, options);
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`The local server returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(data.error || `The local request failed (${response.status}).`);
    return data;
  }

  function validateResult(result) {
    if (result.model_sha256 !== state.metadata.model_sha256) throw new Error("The physical model changed while this page was open. Reload to inspect the new model.");
    if (state.metadata.implementation_sha256 && result.implementation_sha256 !== state.metadata.implementation_sha256) throw new Error("The model implementation changed while this page was open. Reload to inspect the current version.");
    for (const name of ["step", "hold"]) {
      const trace = result.runs?.[name]?.trace;
      if (!Array.isArray(trace?.time_s) || trace.time_s.length < 2) throw new Error(`The ${name} simulation returned no usable recorded trace.`);
      for (const key of ["ring_temperature_k", "heater_temperature_k", "package_temperature_k", "monitor_code", "heater_code", "detuning_hz", "oma_w"]) {
        if (!Array.isArray(trace[key]) || trace[key].length !== trace.time_s.length) throw new Error(`The ${name} trace is missing ${key} samples.`);
      }
    }
  }

  async function runSimulation() {
    if (state.busy || !form.reportValidity()) return;
    stopPlayback();
    byId("error-message").hidden = true;
    const controls = readControls();
    setBusy(true);
    byId("form-state").textContent = "Solving the Python model twice. This can take a few seconds.";
    try {
      const result = await fetchJson("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(controls) });
      validateResult(result);
      state.result = result;
      const trace = result.runs.step.trace;
      state.index = trace.time_s.length - 1;
      byId("time-index").max = String(state.index);
      let minimum = Infinity, maximum = -Infinity;
      for (const run of Object.values(result.runs)) {
        for (const key of ["ring_temperature_k", "heater_temperature_k", "package_temperature_k"]) {
          for (const value of run.trace[key]) { const temperature = scalar(value); if (Number.isFinite(temperature)) { minimum = Math.min(minimum, temperature); maximum = Math.max(maximum, temperature); } }
        }
      }
      state.minTemperature = minimum;
      state.maxTemperature = Math.max(minimum + 0.01, maximum);
      byId("temperature-min").textContent = `${minimum.toFixed(1)} K`;
      byId("temperature-max").textContent = `${maximum.toFixed(1)} K`;
      byId("target-caption").textContent = `Target ${result.controls.target_code} ± ${result.controls.deadband_codes}`;
      const elapsed = Number(result.runs.step.diagnostics?.wall_s || 0) + Number(result.runs.hold.diagnostics?.wall_s || 0);
      byId("form-state").textContent = `${trace.time_s.length} recorded samples per run${elapsed > 0 ? ` · ${elapsed.toFixed(2)} s solver time` : ""}. Comparison complete.`;
      updateAssessment();
      updateDiagnostics();
      updateSample();
    } catch (error) {
      showError(error.message || "The simulation could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  function updateAssessment() {
    const box = byId("run-assessment");
    box.replaceChildren();
    box.className = "run-assessment";
    const result = state.result;
    const step = result.runs.step, hold = result.runs.hold;
    const warnings = (step.diagnostics?.warnings || []).filter((item) => typeof item === "string");
    const finalIndex = step.trace.time_s.length - 1;
    const finalDetuning = at(step.trace, "detuning_hz", finalIndex);
    const finalCode = at(step.trace, "monitor_code", finalIndex);
    const error = finalCode - result.controls.target_code;
    const code = at(step.trace, "heater_code", finalIndex);
    if (finalDetuning >= 0 && !warnings.some((item) => /side|detun|resonan/i.test(item))) warnings.push("The final detuning is outside the assumed negative-detuning operating side. Choose a suitable initial heater code.");
    if ((code <= 0 || code >= 4095) && !warnings.some((item) => /rail/i.test(item))) warnings.push("The heater has reached a DAC rail; the requested target may be infeasible from this prebias.");
    if (warnings.length) {
      box.classList.add(finalDetuning >= 0 ? "danger" : "warning");
      const intro = document.createElement("p"), heading = document.createElement("strong");
      heading.textContent = finalDetuning >= 0 ? "Operating assumption violated. " : "Check this operating point. ";
      intro.append(heading, document.createTextNode(warnings.join(" ")));
      box.append(intro);
    }
    const summary = document.createElement("p");
    const intro = document.createElement("strong");
    intro.textContent = `Final monitor error: ${error >= 0 ? "+" : ""}${format(error, 1)} codes. `;
    summary.append(intro);
    const stepControl = step.metrics?.control, holdControl = hold.metrics?.control;
    if (finite(stepControl?.post_disturbance_rms_error_code) && finite(holdControl?.post_disturbance_rms_error_code)) {
      summary.append(document.createTextNode(`Over 2–4 ms (after the ambient step), RMS monitor error is ${format(Number(stepControl.post_disturbance_rms_error_code), 2)} codes with control and ${format(Number(holdControl.post_disturbance_rms_error_code), 2)} with the heater held. `));
    }
    summary.append(document.createTextNode("Matching an ADC target alone does not establish resonance lock."));
    box.append(summary);
  }

  function updateDiagnostics() {
    const target = byId("run-diagnostics");
    target.replaceChildren();
    const assumptions = state.result.assumptions || state.metadata.assumptions || [];
    if (assumptions.length) {
      const list = document.createElement("ul");
      for (const text of assumptions) { const item = document.createElement("li"); item.textContent = typeof text === "string" ? text : JSON.stringify(text); list.append(item); }
      target.append(list);
    }
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify({ controls: state.result.controls, experiment: state.result.experiment, step: { metrics: state.result.runs.step.metrics, diagnostics: state.result.runs.step.diagnostics }, hold: { metrics: state.result.runs.hold.metrics, diagnostics: state.result.runs.hold.diagnostics } }, null, 2);
    target.append(pre);
  }

  function tempColor(temperature) {
    if (!Number.isFinite(temperature) || !state.result) return state.colors["waveguide"];
    const t = Math.min(1, Math.max(0, (temperature - state.minTemperature) / (state.maxTemperature - state.minTemperature)));
    const parse = (color) => [parseInt(color.slice(1, 3), 16), parseInt(color.slice(3, 5), 16), parseInt(color.slice(5, 7), 16)];
    const low = parse(state.colors["heat-low"]), high = parse(state.colors["heat-high"]);
    return `rgb(${low.map((value, i) => Math.round(value + (high[i] - value) * t)).join(",")})`;
  }

  function updateSample() {
    if (!state.result) { scheduleRender(); return; }
    const trace = currentTrace();
    state.index = Math.max(0, Math.min(state.index, trace.time_s.length - 1));
    byId("time-index").value = String(state.index);
    const time = trace.time_s[state.index];
    byId("time-value").textContent = `${format(time * 1000, 3)} ms`;
    const warmup = state.result.experiment.warmup_s ?? state.result.controls.warmup_ms / 1000;
    const disturbance = state.result.experiment.disturbance_time_s;
    byId("phase-label").textContent = byId("device-run").value === "hold" ? "Fixed heater hold" : time < warmup ? "Warmup · hold" : time < disturbance ? "Feedback active" : "After ambient step";
    for (const [name, key] of [["ring", "ring_temperature_k"], ["heater", "heater_temperature_k"], ["package", "package_temperature_k"]]) {
      const value = at(trace, key, state.index);
      byId(`${name}-temperature`).textContent = `${format(value, 3)} K`;
      byId(`${name}-dot`).style.background = tempColor(value);
    }
    for (const chart of charts) {
      for (const run of ["step", "hold"]) {
        byId(`value-${chart.name}-${run}`).textContent = format(at(state.result.runs[run].trace, chart.key, state.index) * chart.scale, chart.decimals);
      }
    }
    const sampledAt = at(state.result.runs.step.trace, "sample_time_s", state.index);
    byId("sample-time").textContent = Number.isFinite(sampledAt) && sampledAt >= 0 ? `ADC sample time: ${format(sampledAt * 1000, 3)} ms` : "No ADC sample yet";
    deviceCanvas.setAttribute("aria-label", `${byId("device-run").selectedOptions[0].text}, recorded at ${format(time * 1000)} milliseconds. Ring ${byId("ring-temperature").textContent}, heater ${byId("heater-temperature").textContent}, package ${byId("package-temperature").textContent}. Schematic geometry, not a 3D field solver.`);
    scheduleRender();
  }

  function stopPlayback() {
    state.playing = false;
    byId("play").textContent = "Play";
    byId("play").setAttribute("aria-label", "Play recorded simulation");
  }

  function togglePlayback() {
    if (!state.result || state.busy) return;
    if (state.playing) { stopPlayback(); return; }
    const count = currentTrace().time_s.length;
    if (state.index >= count - 1) state.index = 0;
    state.playPosition = state.index;
    state.playing = true;
    state.lastFrame = performance.now();
    byId("play").textContent = "Pause";
    byId("play").setAttribute("aria-label", "Pause recorded simulation");
    requestAnimationFrame(playFrame);
  }

  function playFrame(now) {
    if (!state.playing || !state.result) return;
    const count = currentTrace().time_s.length;
    state.playPosition += (now - state.lastFrame) * (count - 1) / 10000;
    state.lastFrame = now;
    state.index = Math.min(count - 1, Math.floor(state.playPosition));
    updateSample();
    if (state.index >= count - 1) stopPlayback();
    else requestAnimationFrame(playFrame);
  }

  function niceTicks(min, max, count) {
    const rough = (max - min) / Math.max(1, count - 1);
    const power = 10 ** Math.floor(Math.log10(rough));
    const fraction = rough / power;
    const step = (fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10) * power;
    const ticks = [];
    for (let value = Math.ceil(min / step) * step; value <= max + step * 0.0001 && ticks.length < 20; value += step) ticks.push(value);
    return ticks;
  }

  function tickLabel(value, spacing) {
    // Precision follows the interval, so adjacent fractional ticks stay distinct
    // even when their values have a large magnitude (for example, -12 and -11.5).
    const step = Math.abs(spacing) || 1;
    const decimals = Math.max(0, Math.min(12, -Math.floor(Math.log10(step) + 1e-9)));
    const rounded = Number(value.toFixed(decimals));
    return rounded.toFixed(decimals);
  }

  function plotDomain() {
    if (!state.result) return [0, 4];
    return [byId("focus-warmup").checked ? state.result.controls.warmup_ms : 0, state.result.experiment.duration_s * 1000];
  }

  function drawChart(chart) {
    const canvas = byId(chart.id);
    const { context: ctx, width, height } = fitCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const pad = { left: 43, right: 7, top: 19, bottom: 36 };
    const plot = { x: pad.left, y: pad.top, width: Math.max(10, width - pad.left - pad.right), height: height - pad.top - pad.bottom };
    chart.plot = plot;
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    const [xmin, xmax] = plotDomain();
    let min = Infinity, max = -Infinity;
    if (state.result) {
      for (const runName of ["step", "hold"]) {
        const trace = state.result.runs[runName].trace;
        trace[chart.key].forEach((value, i) => {
          if (trace.time_s[i] * 1000 < xmin - 1e-9) return;
          const number = scalar(value) * chart.scale;
          if (Number.isFinite(number)) { min = Math.min(min, number); max = Math.max(max, number); }
        });
      }
      if (chart.name === "monitor") {
        const { target_code: target, deadband_codes: band } = state.result.controls;
        min = Math.min(min, target - band); max = Math.max(max, target + band);
      }
    }
    if (!Number.isFinite(min)) { min = 0; max = 1; }
    const margin = Math.max((max - min) * 0.11, chart.scale === 1 ? 1 : 0.005);
    min -= margin; max += margin;
    const yTicks = niceTicks(min, max, 4);
    const tickSpacing = yTicks.length > 1 ? yTicks[1] - yTicks[0] : max - min;
    const tickLabels = yTicks.map((value) => tickLabel(value, tickSpacing));
    const labelWidth = Math.max(0, ...tickLabels.map((label) => ctx.measureText(label).width));
    plot.x = Math.max(pad.left, Math.ceil(labelWidth + 9));
    plot.width = Math.max(10, width - plot.x - pad.right);
    const x = (value) => plot.x + (value - xmin) / (xmax - xmin) * plot.width;
    const y = (value) => plot.y + plot.height - (value - min) / (max - min) * plot.height;
    ctx.lineWidth = 0.7;
    yTicks.forEach((value, index) => {
      const py = y(value);
      ctx.strokeStyle = state.colors.line; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(plot.x, py); ctx.lineTo(plot.x + plot.width, py); ctx.stroke();
      ctx.fillStyle = state.colors.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(tickLabels[index], plot.x - 7, py);
    });
    const ticks = width < 240 ? [xmin, (xmin + xmax) / 2, xmax] : [xmin, xmin + (xmax - xmin) / 3, xmin + 2 * (xmax - xmin) / 3, xmax];
    ticks.forEach((value, i) => {
      ctx.fillStyle = state.colors.muted; ctx.textAlign = i === 0 ? "left" : i === ticks.length - 1 ? "right" : "center"; ctx.textBaseline = "top";
      ctx.fillText(`${Number(value.toFixed(2))}`, x(value), plot.y + plot.height + 7);
    });
    ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillStyle = state.colors.muted; ctx.fillText("Time (ms)", plot.x + plot.width / 2, height - 1);
    if (!state.result) {
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("Waiting for recorded data", plot.x + plot.width / 2, plot.y + plot.height / 2);
      return;
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(plot.x, plot.y, plot.width, plot.height); ctx.clip();
    if (chart.name === "monitor") {
      const { target_code: target, deadband_codes: band } = state.result.controls;
      ctx.fillStyle = state.colors["accent-soft"];
      ctx.fillRect(plot.x, y(target + band), plot.width, Math.max(1, y(target - band) - y(target + band)));
      ctx.strokeStyle = state.colors.accent; ctx.globalAlpha = 0.4; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(plot.x, y(target)); ctx.lineTo(plot.x + plot.width, y(target)); ctx.stroke(); ctx.globalAlpha = 1;
    }
    const warmup = state.result.controls.warmup_ms;
    if (warmup >= xmin) { ctx.fillStyle = state.colors.muted; ctx.globalAlpha = 0.055; ctx.fillRect(plot.x, plot.y, x(warmup) - plot.x, plot.height); ctx.globalAlpha = 1; }
    for (const [position, dashed, color] of [[warmup, true, state.colors.muted], [state.result.experiment.disturbance_time_s * 1000, false, state.colors.warn]]) {
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash(dashed ? [3, 4] : []); ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(x(position), plot.y); ctx.lineTo(x(position), plot.y + plot.height); ctx.stroke(); ctx.globalAlpha = 1;
    }
    for (const name of ["hold", "step"]) {
      const trace = state.result.runs[name].trace;
      ctx.strokeStyle = name === "step" ? state.colors.accent : state.colors["chart-hold"];
      ctx.lineWidth = name === "step" ? 1.8 : 1.5; ctx.setLineDash(name === "step" ? [] : [4, 3]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < trace.time_s.length; i++) {
        const px = x(trace.time_s[i] * 1000), py = y(at(trace, chart.key, i) * chart.scale);
        if (!Number.isFinite(py)) { started = false; continue; }
        if (!started) { ctx.moveTo(px, py); started = true; }
        else if (chart.name === "heater" || chart.name === "monitor") { ctx.lineTo(px, y(at(trace, chart.key, i - 1) * chart.scale)); ctx.lineTo(px, py); }
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
    const cursorTime = state.result.runs.step.trace.time_s[state.index] * 1000;
    const cursorX = x(cursorTime);
    ctx.setLineDash([]); ctx.strokeStyle = state.colors.ink; ctx.globalAlpha = 0.3; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cursorX, plot.y); ctx.lineTo(cursorX, plot.y + plot.height); ctx.stroke(); ctx.globalAlpha = 1;
    for (const name of ["hold", "step"]) {
      const py = y(at(state.result.runs[name].trace, chart.key, state.index) * chart.scale);
      ctx.fillStyle = name === "step" ? state.colors.accent : state.colors["chart-hold"];
      ctx.beginPath(); ctx.arc(cursorX, py, name === "step" ? 3 : 2.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    ctx.strokeStyle = state.colors.line; ctx.lineWidth = 0.7; ctx.setLineDash([]); ctx.strokeRect(plot.x, plot.y, plot.width, plot.height);
    if (cursorTime >= xmin) {
      ctx.fillStyle = state.colors.ink; ctx.textBaseline = "bottom";
      ctx.textAlign = cursorX > width - 58 ? "right" : cursorX < plot.x + 35 ? "left" : "center";
      ctx.fillText(`${cursorTime.toFixed(3)} ms`, cursorX, plot.y - 4);
    }
  }

  function scrubChart(event, chart) {
    if (!state.result || state.busy || !chart.plot) return;
    stopPlayback();
    const rect = byId(chart.id).getBoundingClientRect();
    const [min, max] = plotDomain();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - chart.plot.x) / chart.plot.width));
    const time = (min + fraction * (max - min)) / 1000;
    const values = state.result.runs.step.trace.time_s;
    let low = 0, high = values.length - 1;
    while (low < high) { const middle = Math.floor((low + high) / 2); if (values[middle] < time) low = middle + 1; else high = middle; }
    state.index = low > 0 && time - values[low - 1] < values[low] - time ? low - 1 : low;
    updateSample();
  }

  function ringMesh(faces, radius, tubeWidth, height, y, color, segments = 72, start = 0, sweep = Math.PI * 2) {
    const outer = radius + tubeWidth / 2, inner = radius - tubeWidth / 2;
    for (let i = 0; i < segments; i++) {
      const a = start + i * sweep / segments, b = start + (i + 1) * sweep / segments;
      const point = (r, angle, vertical) => [r * Math.cos(angle), vertical, r * Math.sin(angle)];
      const ot1 = point(outer, a, y + height), ot2 = point(outer, b, y + height), it1 = point(inner, a, y + height), it2 = point(inner, b, y + height);
      const ob1 = point(outer, a, y), ob2 = point(outer, b, y), ib1 = point(inner, a, y), ib2 = point(inner, b, y);
      faces.push({ points: [ot1, ot2, it2, it1], color }, { points: [ob1, ob2, ot2, ot1], color }, { points: [ib2, ib1, it1, it2], color });
    }
  }

  function boxMesh(faces, center, size, color, outline = true) {
    const [x, y, z] = center, [w, h, d] = size;
    const points = [[x - w/2,y-h/2,z-d/2],[x+w/2,y-h/2,z-d/2],[x+w/2,y+h/2,z-d/2],[x-w/2,y+h/2,z-d/2],[x-w/2,y-h/2,z+d/2],[x+w/2,y-h/2,z+d/2],[x+w/2,y+h/2,z+d/2],[x-w/2,y+h/2,z+d/2]];
    for (const indices of [[0,1,2,3],[5,4,7,6],[4,0,3,7],[1,5,6,2],[3,2,6,7],[4,5,1,0]]) faces.push({ points: indices.map((index) => points[index]), color, outline });
  }

  function drawDevice() {
    const { context: ctx, width, height } = fitCanvas(deviceCanvas);
    ctx.clearRect(0, 0, width, height);
    const { yaw, pitch, zoom } = state.camera;
    const cosYaw = Math.cos(yaw), sinYaw = Math.sin(yaw), cosPitch = Math.cos(pitch), sinPitch = Math.sin(pitch);
    const exploded = byId("explode").checked;
    const ringY = exploded ? 0.25 : -0.02;
    const heaterY = exploded ? 0.9 : 0.12;
    const packageY = exploded ? -0.65 : -0.35;
    const rotate = ([x, y, z]) => {
      const rotatedX = x * cosYaw + z * sinYaw;
      const rotatedZ = -x * sinYaw + z * cosYaw;
      return { x: rotatedX, y: rotatedZ * sinPitch - y * cosPitch, depth: rotatedZ * cosPitch + y * sinPitch };
    };
    // Fit the complete schematic between the controls at the default zoom.
    // Explicit user zoom can enlarge it beyond these bounds.
    const bounds = [];
    for (const x of [-2.15, 2.15]) for (const y of [packageY - 0.12, heaterY + 0.08]) for (const z of [-1.45, 1.45]) bounds.push(rotate([x, y, z]));
    const minX = Math.min(...bounds.map((point) => point.x)), maxX = Math.max(...bounds.map((point) => point.x));
    const minY = Math.min(...bounds.map((point) => point.y)), maxY = Math.max(...bounds.map((point) => point.y));
    const top = width < 430 ? 103 : 85, bottom = height - 77;
    const scale = Math.min((width - 88) / (maxX - minX), (bottom - top) / (maxY - minY)) * zoom;
    const project = (point) => {
      const rotated = rotate(point);
      return { x: width / 2 + (rotated.x - (minX + maxX) / 2) * scale, y: (top + bottom) / 2 + (rotated.y - (minY + maxY) / 2) * scale, depth: rotated.depth };
    };
    const trace = currentTrace();
    const ringTemperature = trace ? at(trace, "ring_temperature_k", state.index) : NaN;
    const heaterTemperature = trace ? at(trace, "heater_temperature_k", state.index) : NaN;
    const packageTemperature = trace ? at(trace, "package_temperature_k", state.index) : NaN;
    const faces = [];
    if (byId("show-package").checked) boxMesh(faces, [0, packageY, 0], [4.05, 0.24, 2.9], tempColor(packageTemperature));
    if (byId("show-waveguides").checked) {
      boxMesh(faces, [0, ringY + 0.025, 1.14], [4.3, 0.07, 0.10], state.colors.waveguide);
      boxMesh(faces, [0, ringY + 0.025, -1.14], [4.3, 0.07, 0.10], state.colors.waveguide);
    }
    ringMesh(faces, 0.97, 0.11, 0.08, ringY, tempColor(ringTemperature));
    if (byId("show-heater").checked) {
      ringMesh(faces, 0.91, 0.15, 0.07, heaterY, tempColor(heaterTemperature), 64, 0.34, Math.PI * 2 - 0.68);
      boxMesh(faces, [1.28, heaterY + 0.035, 0.30], [0.83, 0.07, 0.14], tempColor(heaterTemperature));
      boxMesh(faces, [1.28, heaterY + 0.035, -0.30], [0.83, 0.07, 0.14], tempColor(heaterTemperature));
      boxMesh(faces, [1.73, heaterY + 0.035, 0.30], [0.25, 0.07, 0.29], tempColor(heaterTemperature));
      boxMesh(faces, [1.73, heaterY + 0.035, -0.30], [0.25, 0.07, 0.29], tempColor(heaterTemperature));
    }
    const projectedFaces = faces.map((face) => ({ ...face, projected: face.points.map(project) }));
    projectedFaces.sort((a, b) => a.projected.reduce((sum, point) => sum + point.depth, 0) / a.projected.length - b.projected.reduce((sum, point) => sum + point.depth, 0) / b.projected.length);
    for (const face of projectedFaces) {
      ctx.beginPath(); face.projected.forEach((point, i) => i ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y)); ctx.closePath();
      ctx.fillStyle = face.color; ctx.fill();
      if (face.outline) { ctx.strokeStyle = state.colors["mesh-edge"]; ctx.lineWidth = 0.7; ctx.stroke(); }
    }
    // Each thermal component has one uniform color. Geometry carries no simulated spatial gradient.
    const labels = [
      { text: "RING", point: [-0.94, ringY + 0.09, -0.1], side: -1, offset: -5 },
      { text: "HEATER", point: [-0.77, heaterY + 0.08, -0.50], side: -1, offset: -18, visible: byId("show-heater").checked },
      { text: "PACKAGE", point: [1.86, packageY + 0.12, 0.75], side: 1, offset: 5, visible: byId("show-package").checked },
    ];
    if (width > 430 && byId("show-waveguides").checked) labels.push({ text: "BUS", point: [-1.8, ringY + 0.08, 1.14], side: -1, offset: 19 }, { text: "DROP", point: [1.8, ringY + 0.08, -1.14], side: 1, offset: -8 });
    ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
    for (const label of labels) {
      if (label.visible === false) continue;
      const point = project(label.point);
      const targetX = Math.max(73, Math.min(width - 80, point.x + label.side * Math.min(42, width * 0.09)));
      const targetY = Math.max(width < 430 ? 98 : 82, Math.min(height - 76, point.y + label.offset));
      ctx.strokeStyle = state.colors["viewer-muted"]; ctx.globalAlpha = 0.55; ctx.lineWidth = 0.7; ctx.beginPath(); ctx.moveTo(point.x, point.y); ctx.lineTo(targetX, targetY); ctx.lineTo(targetX + label.side * 9, targetY); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = state.colors["viewer-ink"]; ctx.textAlign = label.side < 0 ? "right" : "left"; ctx.textBaseline = "middle"; ctx.fillText(label.text, targetX + label.side * 13, targetY);
    }
    const center = project([0, packageY - 0.15, 0]);
    if (width > 500) {
      ctx.fillStyle = state.colors["viewer-muted"]; ctx.textAlign = "center"; ctx.font = "11px ui-sans-serif, system-ui, sans-serif"; ctx.fillText("ONE COLOR PER RECORDED THERMAL NODE", width / 2, Math.min(height - 14, Math.max(center.y + 94, height - 16)));
    }
  }

  function scheduleRender() {
    if (state.frame) return;
    state.frame = requestAnimationFrame(() => { state.frame = 0; drawDevice(); charts.forEach(drawChart); });
  }

  function moveCamera(action) {
    if (action === "left") state.camera.yaw -= 0.2;
    if (action === "right") state.camera.yaw += 0.2;
    if (action === "up") state.camera.pitch = Math.min(1.4, state.camera.pitch + 0.14);
    if (action === "down") state.camera.pitch = Math.max(0.18, state.camera.pitch - 0.14);
    if (action === "in") state.camera.zoom = Math.min(1.7, state.camera.zoom * 1.1);
    if (action === "out") state.camera.zoom = Math.max(0.6, state.camera.zoom / 1.1);
    if (action === "reset") state.camera = { yaw: -0.45, pitch: 0.73, zoom: 1 };
    scheduleRender();
  }

  form.addEventListener("submit", (event) => { event.preventDefault(); runSimulation(); });
  form.addEventListener("input", () => {
    updateExperimentDescription();
    if (state.result && !state.busy) {
      const changed = controlKeys.some((key) => readControls()[key] !== state.result.controls[key]);
      byId("form-state").textContent = changed ? "Inputs changed. Run comparison to update the recorded results." : "The completed comparison matches these inputs.";
    }
  });
  byId("play").addEventListener("click", togglePlayback);
  byId("time-index").addEventListener("input", (event) => { stopPlayback(); state.index = Number(event.target.value); updateSample(); });
  byId("device-run").addEventListener("change", updateSample);
  byId("focus-warmup").addEventListener("change", () => {
    if (state.result && byId("focus-warmup").checked) {
      const times = state.result.runs.step.trace.time_s;
      if (times[state.index] * 1000 < state.result.controls.warmup_ms) state.index = times.findIndex((time) => time * 1000 >= state.result.controls.warmup_ms - 1e-9);
      updateSample();
    } else scheduleRender();
  });
  for (const id of ["explode", "show-heater", "show-waveguides", "show-package"]) byId(id).addEventListener("change", scheduleRender);
  document.querySelectorAll("[data-camera]").forEach((button) => button.addEventListener("click", () => moveCamera(button.dataset.camera)));
  deviceCanvas.addEventListener("pointerdown", (event) => { state.drag = { x: event.clientX, y: event.clientY }; deviceCanvas.setPointerCapture(event.pointerId); });
  deviceCanvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    state.camera.yaw += (event.clientX - state.drag.x) * 0.007;
    state.camera.pitch = Math.max(0.18, Math.min(1.4, state.camera.pitch - (event.clientY - state.drag.y) * 0.006));
    state.drag = { x: event.clientX, y: event.clientY }; scheduleRender();
  });
  deviceCanvas.addEventListener("pointerup", () => { state.drag = null; });
  deviceCanvas.addEventListener("pointercancel", () => { state.drag = null; });
  deviceCanvas.addEventListener("lostpointercapture", () => { state.drag = null; });
  deviceCanvas.addEventListener("wheel", (event) => { event.preventDefault(); state.camera.zoom = Math.max(0.6, Math.min(1.7, state.camera.zoom * Math.exp(-event.deltaY * 0.001))); scheduleRender(); }, { passive: false });
  for (const chart of charts) {
    const canvas = byId(chart.id);
    canvas.addEventListener("pointermove", (event) => { if (event.pointerType !== "touch" || event.buttons) scrubChart(event, chart); });
    canvas.addEventListener("pointerdown", (event) => scrubChart(event, chart));
  }
  byId("download").addEventListener("click", () => {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob), anchor = document.createElement("a");
    anchor.href = url; anchor.download = `mrm-control-${state.result.model_sha256.slice(0, 12)}.json`; document.body.append(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const observer = new ResizeObserver(scheduleRender);
  observer.observe(deviceCanvas); charts.forEach((chart) => observer.observe(byId(chart.id)));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { resolveColors(); scheduleRender(); });
  resolveColors(); scheduleRender();
  fetchJson("/api/model").then((metadata) => { showModel(metadata); setBusy(false); return runSimulation(); }).catch((error) => { setBusy(false); showError(error.message); byId("model-id").textContent = "Model unavailable"; byId("run-button-label").textContent = "Model unavailable"; });
})();
