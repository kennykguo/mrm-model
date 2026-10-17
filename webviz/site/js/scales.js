// Colour scales shared by the 3-D view and the 2-D plots (one lookup table, so a colour means the same
// number everywhere). Temperatures: inferno. Lock error and detuning: coolwarm (two hues, grey middle).

import { INFERNO, COOLWARM } from './colormaps.js';

const hexToRgb = (h) => [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

const LUTS = {};
for (const [name, hexes] of Object.entries({ inferno: INFERNO, coolwarm: COOLWARM })) {
  const rgb = hexes.map(hexToRgb);
  LUTS[name] = { rgb, lin: rgb.map((c) => c.map((v) => srgbToLinear(v / 255))) };
}

export class Scale {
  // lut: 'inferno' | 'coolwarm'; [lo, hi] the fixed range; values outside are clamped (the legend says so)
  constructor(lut, lo, hi, unit, label) {
    this.lut = LUTS[lut];
    this.lutName = lut;
    this.lo = lo;
    this.hi = hi;
    this.unit = unit;
    this.label = label;
  }

  frac(v) {
    if (v == null || !Number.isFinite(v)) return 0;
    const t = (v - this.lo) / (this.hi - this.lo);
    return t < 0 ? 0 : t > 1 ? 1 : t;
  }

  index(v) { return Math.round(this.frac(v) * 255); }
  rgb(v) { return this.lut.rgb[this.index(v)]; }
  lin(v) { return this.lut.lin[this.index(v)]; }

  css(v) {
    const [r, g, b] = this.rgb(v);
    return `rgb(${r},${g},${b})`;
  }

  // Plotly colorscale over [lo, hi]
  plotly(n = 33) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const [r, g, b] = this.lut.rgb[Math.round(t * 255)];
      out.push([t, `rgb(${r},${g},${b})`]);
    }
    return out;
  }

  // CSS gradient for the HTML legends
  gradient(n = 17) {
    const stops = [];
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const [r, g, b] = this.lut.rgb[Math.round(t * 255)];
      stops.push(`rgb(${r},${g},${b}) ${(t * 100).toFixed(1)}%`);
    }
    return `linear-gradient(to right, ${stops.join(', ')})`;
  }
}

// Linear mix of two sRGB colours (hex strings), for the bus-light brightness.
export function mixHex(a, b, t) {
  const A = hexToRgb(a.replace('#', ''));
  const B = hexToRgb(b.replace('#', ''));
  const c = A.map((v, i) => Math.round(v + (B[i] - v) * Math.max(0, Math.min(1, t))));
  return c;
}

// "Nice" upper bound: 1, 2, 2.5, 5 x 10^n at or above v.
export function niceUp(v) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * e >= v * (1 - 1e-9)) return m * e;
  return 10 * e;
}
