/* Shared LaTeX rendering for prose, live readouts, and chart annotations.
   Equations are authored explicitly; the small token dictionary only typesets
   already-written variable names and units, never infers a physical formula. */
(function (root) {
  "use strict";
  const engine = root.katex;
  const options = {
    throwOnError: true,
    strict: "error",
    trust: false,
    output: "htmlAndMathml",
  };
  const cache = new Map();
  const escape = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const symbols = {
    α: "\\alpha",
    β: "\\beta",
    γ: "\\gamma",
    δ: "\\delta",
    ε: "\\varepsilon",
    θ: "\\theta",
    κ: "\\kappa",
    λ: "\\lambda",
    π: "\\pi",
    σ: "\\sigma",
    τ: "\\tau",
    φ: "\\phi",
    ω: "\\omega",
    Γ: "\\Gamma",
    Δ: "\\Delta",
    Ω: "\\Omega",
    Σ: "\\Sigma",
  };
  const subs = { "₀": "0", "₁": "1", "₂": "2", "₃": "3", ₓ: "x", ᵧ: "y" };
  const named = {
    "%": String.raw`\%`, "×": String.raw`\times`, "−": "-",
    "≤": String.raw`\leq`, "≥": String.raw`\geq`, "=": "=",

    "V = code / code_max × V_FS": String.raw`V=\frac{\mathrm{code}}{\mathrm{code}_{\max}}V_{\mathrm{FS}}`,
    "R = R0 (1 + TCR × (T_heater − 25 °C))": String.raw`R=R_0[1+\mathrm{TCR}(T_{\mathrm{heater}}-25\,{}^\circ\mathrm C)]`,
    "Z(0)": String.raw`Z(0)`, "K": String.raw`\mathrm K`,

    "pA/√Hz": String.raw`\mathrm{pA}/\sqrt{\mathrm{Hz}}`,
    "Hz/√s": String.raw`\mathrm{Hz}/\sqrt{\mathrm{s}}`,
    "log10": String.raw`\log_{10}`,
    "(K)": String.raw`(\mathrm K)`, "(V)": String.raw`(\mathrm V)`,
    "K/mW": String.raw`\mathrm{K/mW}`, "K/W": String.raw`\mathrm{K/W}`,
    "pm/K": String.raw`\mathrm{pm/K}`, "A/W": String.raw`\mathrm{A/W}`,
    "rad/µm": String.raw`\mathrm{rad}/\mu\mathrm{m}`,
    "Gb/s": String.raw`\mathrm{Gb/s}`, "Hz/s": String.raw`\mathrm{Hz/s}`,
    "MHz": String.raw`\mathrm{MHz}`, "kHz": String.raw`\mathrm{kHz}`,
    "µW": String.raw`\mu\mathrm W`, "µA": String.raw`\mu\mathrm A`,
    "µs": String.raw`\mu\mathrm s`, "µm": String.raw`\mu\mathrm m`,
    "mW": String.raw`\mathrm{mW}`, "nA": String.raw`\mathrm{nA}`,
    "mA": String.raw`\mathrm{mA}`, "pA": String.raw`\mathrm{pA}`,
    "nm": String.raw`\mathrm{nm}`, "pm": String.raw`\mathrm{pm}`,
    "ms": String.raw`\mathrm{ms}`, "ns": String.raw`\mathrm{ns}`,
    "eV": String.raw`\mathrm{eV}`, "dBm": String.raw`\mathrm{dBm}`,
    "dB": String.raw`\mathrm{dB}`, "Ω": String.raw`\Omega`,
    "t10": String.raw`t_{10}`, "t63": String.raw`t_{63}`, "t90": String.raw`t_{90}`,
    "10⁻⁹": String.raw`10^{-9}`, "½": String.raw`\tfrac12`,
    "√2": String.raw`\sqrt2`, "±": String.raw`\pm`,

    "°C": String.raw`{}^\circ\mathrm C`,
    "T₀(t)": String.raw`T_0(t)`,
    "T(0)": String.raw`T(0)`,
    "T*": String.raw`T^*`,
    "K/µs": String.raw`\mathrm{K}/(\mu\mathrm{s})`,
    "dT/dT₀": String.raw`\frac{dT}{dT_0}`,
    "dT/dt": String.raw`\frac{dT}{dt}`,
    "dn/dλ": String.raw`\frac{dn}{d\lambda}`,
    "dn_eff/dλ": String.raw`\frac{dn_{\mathrm{eff}}}{d\lambda}`,
    "dP_abs/dT": String.raw`\frac{dP_{\mathrm{abs}}}{dT}`,
    "|E|²": String.raw`|E|^2`,
    "|F|²": String.raw`|F|^2`,
    "|A|²": String.raw`|A|^2`,
    "|B|²": String.raw`|B|^2`,
    "f′(T*)": String.raw`f'(T^*)`,
    "f(T*)": String.raw`f(T^*)`,
    "(1, dT/dt)": String.raw`\left(1,\frac{dT}{dt}\right)`,
    "(T − T₀)/R": String.raw`\frac{T-T_0}{R}`,
    "(x,y)": "(x,y)",
    "(x,v)": "(x,v)",
    "λ_laser < λ_res": String.raw`\lambda_{\mathrm{laser}}<\lambda_{\mathrm{res}}`,
    "ωt − βz = 0": String.raw`\omega t-\beta z=0`,
    "z = (ω/β)t": String.raw`z=\frac{\omega}{\beta}t`,
    "kₓ² + β² = (nk₀)²": String.raw`k_x^2+\beta^2=(nk_0)^2`,
    "n − λ dn/dλ": String.raw`n-\lambda\frac{dn}{d\lambda}`,
    "β/k₀": String.raw`\beta/k_0`,
    "β₂z/T₀²": String.raw`\beta_2 z/T_0^2`,
    "f = f′ = 0": String.raw`f=f'=0`,
    "f′(T*) < 0": String.raw`f'(T^*)<0`,
    "(v, cos(Ωt) − Γv − x)": String.raw`(v,\cos(\Omega t)-\Gamma v-x)`,
    Fv: "Fv",
    "Γv²": String.raw`\Gamma v^2`,
  };
  const reEscape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const names = Object.keys(named)
    .sort((a, b) => b.length - a.length)
    .map((name) => (/[A-Za-z0-9_]/.test(name[0]) ? '(?<![A-Za-z0-9_])' : '') + reEscape(name) + (/[A-Za-z0-9_]/.test(name.at(-1)) ? '(?![A-Za-z0-9_])' : ''))
    .join("|");
  // A quantity is a number followed by a stated unit; prose numbers remain prose.
  const quantity = String.raw`[+−-]?\d+(?:\.\d+)?(?:\s*)(?:A/W|K/W|W/K|pm/K|K/mW|K/K|K/µs|ps²/km|rad/µm|pA/√Hz|Hz/√s|Hz/s|Gb/s|MHz|kHz|dBm|dB|mA|µA|nA|pA|eV|Ω|mW|µW|nm|pm|ms|ns|µm(?:⁻²|²)?|µs|ps|fs|km|°C|V|W|A|K|%|°)(?![\w])`;
  // Bare A is usually the English article; amperes are handled as quantities,
  // and a field amplitude A should be authored in explicit LaTeX.
  const identifier = String.raw`(?:[A-Za-zαβγδεθκλπστφωΓΔΩ]+_[A-Za-z0-9]+|(?:TE|TM)[₀₁₂₃]|[A-Za-zαβγδεθκλπστφωΓΔΩ][₀₁₂₃ₓᵧ]+(?:²|³)?|[αβγδεθκλπστφωΓΔΩ](?:²|³)?|[A-Za-z][²³]|[A-Za-z][′″*]|\b[ETCFVRPBxzyztuvwnhr]\b)`;
  const scientific = String.raw`[+−-]?\d+(?:\.\d+)?(?:\s*×\s*10\^[+−-]?\d+|e[+−-]?\d+)`;
  const tokens = new RegExp(
    `${scientific}|${quantity}|${names}|(?<![\\w])${identifier}(?![\\w])`,
    "gu",
  );
  function tokenTex(token) {
    if (named[token]) return named[token];
    const sci = token.match(/^([+−-]?\d+(?:\.\d+)?)(?:\s*×\s*10\^|e)([+−-]?\d+)$/);
    if (sci) return sci[1].replace('−','-') + String.raw`\times 10^{` + sci[2].replace('−','-') + '}';
    const q = token.match(/^([+−-]?\d+(?:\.\d+)?)\s*(.*)$/u);
    if (q) {
      const unit = q[2]
        .replace(/√Hz/g, "\\sqrt{Hz}")
        .replace(/√s/g, "\\sqrt{s}")
        .replace(/Ω/g, "\\Omega")
        .replace(/µ/g, "\\mu ")
        .replace(/⁻²/g, "^{-2}")
        .replace(/²/g, "^2")
        .replace(/°C/g, "{}^{\\circ}C")
        .replace(/°/g, "{}^{\\circ}")
        .replace(/%/g, "\\%");
      return q[1].replace("−", "-") + String.raw`\,\mathrm{` + unit + "}";
    }
    let out = token.replace(
      /_([A-Za-z]+[0-9]*)/g,
      (_, s) => `_{\\mathrm{${s}}}`,
    );
    out = out.replace(/[αβγδεθκλπστφωΓΔΩ]/g, (c) => symbols[c]);
    out = out
      .replace(/[₀₁₂₃ₓᵧ]+/g, (s) => `_{${[...s].map((c) => subs[c]).join("")}}`)
      .replace(/²/g, "^2")
      .replace(/³/g, "^3")
      .replace(/′/g, "'")
      .replace(/″/g, "''");
    if (/^(TE|TM)/.test(out)) out = out.replace(/^(TE|TM)/, "\\mathrm{$1}");
    return out;
  }
  function texHTML(tex, display = false) {
    const key = (display ? "D:" : "I:") + tex;
    if (!cache.has(key)) {
      if (cache.size > 2500) cache.clear();
      cache.set(
        key,
        // Use KaTeX's typewriter metrics for letters, digits and upright units.
        // Operators, Greek symbols and stretchy delimiters retain math-font
        // fallbacks; replacing their CSS fonts would break KaTeX's geometry.
        engine.renderToString(String.raw`\mathtt{` + tex.replace(/\\mathrm\b/g, String.raw`\mathtt`) + '}', { ...options, displayMode: display }),
      );
    }
    return cache.get(key);
  }
  function plainParts(text) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(tokens)) {
      if (m.index > last) out.push({ text: text.slice(last, m.index) });
      out.push({ tex: tokenTex(m[0]), source: m[0] });
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push({ text: text.slice(last) });
    return out;
  }
  function parts(text) {
    const out = [];
    let last = 0;
    for (const m of String(text).matchAll(
      /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g,
    )) {
      out.push(...plainParts(text.slice(last, m.index)));
      out.push({ tex: m[1] ?? m[2], display: m[2] !== undefined });
      last = m.index + m[0].length;
    }
    out.push(...plainParts(String(text).slice(last)));
    return out;
  }
  function inline(text) {
    if (/^[+−-]?\d+(?:\.\d+)?(?:e[+−-]?\d+)?$/i.test(String(text))) {
      const [mantissa, exponent] = String(text).replace(/−/g, "-").split(/e/i);
      return texHTML(
        exponent === undefined
          ? mantissa
          : mantissa + String.raw`\times 10^{` + exponent + "}",
      );
    }
    return parts(String(text))
      .map((p) =>
        p.tex !== undefined ? texHTML(p.tex, p.display) : escape(p.text),
      )
      .join("");
  }
  function hasMath(text) {
    if (/^[+−-]?\d+(?:\.\d+)?(?:e[+−-]?\d+)?$/i.test(String(text))) return true;
    return parts(String(text)).some((p) => p.tex !== undefined);
  }
  function block(el, tex) {
    el.classList.add("lab-math-block");
    el.innerHTML = texHTML(tex, true);
  }
  function refresh(scope) {
    if (!root.document?.createTreeWalker) return;
    const walk = document.createTreeWalker(scope, 4),
      pending = [];
    while (walk.nextNode()) {
      const node = walk.currentNode,
        parent = node.parentElement;
      if (
        !parent ||
        parent.namespaceURI === "http://www.w3.org/2000/svg" ||
        parent.closest(
          ".lab-math-inline,.lab-math-block,.katex,.js-plotly-plot,.math-plot-label,script,style,pre,code,option,textarea",
        )
      )
        continue;
      // Plain table values and ADC codes stay in tabular digits.
      if (/^[+−-]?[\d.,]+$/.test(node.nodeValue.trim()) && parent.closest('td,.tv,.time,button')) continue;
      if (hasMath(node.nodeValue)) pending.push(node);
    }
    for (const node of pending) {
      const span = document.createElement("span");
      span.className = "lab-math-inline";
      span.innerHTML = inline(node.nodeValue);
      node.replaceWith(span);
    }
  }
  const api = { inline, hasMath, parts, block, refresh, texHTML };
  if (typeof module === "object" && module.exports) module.exports = api;
  root.MathText = api;
})(typeof window !== "undefined" ? window : globalThis);
