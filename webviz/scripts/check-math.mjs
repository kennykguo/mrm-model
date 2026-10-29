// Check the site's math rendering without a browser: node webviz/scripts/check-math.mjs
// It loads the bundled KaTeX and js/math-text.js into a small fake page, then checks:
//   - Plotly tick labels (scientific exponents, Unicode minus, thousands separators) become the right LaTeX;
//   - quantities with units are typeset, plain words are not;
//   - the monospace (\mathtt) styling keeps fractions, fallback symbols and exponents;
//   - horizontalYTitles (js/theme.js) moves y-axis titles above the plot without changing the axis;
//   - js/parts.js has 44 plot questions and 44 descriptions;
//   - every quoted string in the other js/*.js files (not math-text.js, math.js, colormaps.js) goes through
//     MathText.inline without an exception (strings with a backslash are counted, not rendered).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../site');
const context = { console, document: { compatMode: 'CSS1Compat' }, matchMedia: () => ({ addEventListener() {} }) };
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(root, 'vendor/katex/katex.min.js'), 'utf8'), context);
globalThis.window = context;
globalThis.document = context.document;
const { plotMathSource } = await import('../site/js/math.js');
const M = context.MathText;

// ---- Plotly tick labels -> LaTeX
const label = (raw) => ({ getAttribute: () => raw, textContent: raw });
const cases = [
  ['5×10<sup>4</sup>', String.raw`\(5\times 10^{4}\)`],
  ['10<sup>−9</sup>', String.raw`\(10^{-9}\)`],
  ['−5000', String.raw`\(-5000\)`],
  ['−1', String.raw`\(-1\)`],
  ['10k', String.raw`\(10\mathrm{k}\)`],
  ['1,000', String.raw`\(1{,}000\)`],
];
for (const [raw, expected] of cases) {
  const tex = plotMathSource(label(raw));
  assert.equal(tex, expected);
  assert(M.inline(tex).includes('katex'));
}

// ---- quantities are typeset, plain words are not
assert(!M.hasMath('sometimes sums forms equipment'));
assert(!M.hasMath('3-D die'));
assert(!M.hasMath('A step changes the input.'));
assert(M.inline('1 A').includes('katex'));
assert.equal(M.parts('3.2×10^-9').find((p) => p.tex).tex, String.raw`3.2\times 10^{-9}`);
assert(M.inline('V = code / code_max × V_FS').includes('mfrac'));
for (const text of ['0.95 A/W', '1 pA/√Hz', '2 µs', '45.41 Ω', '10⁻⁹', String.raw`\(r(1-e^{-t/\tau})\)`,
  String.raw`\(\mathrm{BER}=\frac12\operatorname{erfc}(Q/\sqrt2)\)`]) {
  assert(M.inline(text).includes('katex'));
}
assert.equal(M.parts('0.95 A/W').filter((p) => p.tex).length, 1);

// ---- the styling must keep genuine math layout, including fallback symbols and exponents
for (const tex of [String.raw`\frac{dT}{dt}`, String.raw`r(1-e^{-t/\tau})`,
  String.raw`\mathrm{BER}=\frac12\operatorname{erfc}(Q/\sqrt2)`, String.raw`\left(\frac{T-T_0}{R}\right)`,
  String.raw`\int_0^t P(\tau)\,d\tau`, String.raw`10^{-227}`, String.raw`25\,{}^\circ\mathrm C`]) {
  const html = M.texHTML(tex);
  assert(html.includes('katex'));
  assert(html.includes('mathtt'));
}
assert(M.texHTML(String.raw`\frac{dT}{dt}`).includes('mfrac'));
assert(M.texHTML(String.raw`2\,\mathrm{ms}`).includes('mathvariant="monospace"'));

// ---- horizontal y-axis titles
const { horizontalYTitles } = await import('../site/js/theme.js');
const original = {
  margin: { t: 10 }, annotations: [{ text: 'fold' }],
  yaxis: { title: { text: 'noise (code RMS)' }, range: [0, 3], type: 'log' },
  yaxis2: { anchor: 'x', domain: [0, 0.5], title: { text: 'temperature (°C)' } },
};
const result = horizontalYTitles(original);
assert.equal(original.yaxis.title.text, 'noise (code RMS)');
assert.equal(result.yaxis.title.text, '');
assert.equal(result.yaxis.tickangle, 0);
assert.deepEqual(result.yaxis.range, [0, 3]);
assert.equal(result.yaxis.type, 'log');
assert.equal(result.annotations.length, 3);
assert.equal(result.annotations[2].yref, 'y2 domain');

// ---- the 44 plot explanations of the Parts view
const parts = fs.readFileSync(path.join(root, 'js/parts.js'), 'utf8');
assert.equal([...parts.matchAll(/question: "/g)].length, 44);
assert.equal([...parts.matchAll(/description: "/g)].length, 44);

// ---- every quoted string in js/*.js renders (the math modules and the colour tables are skipped)
let count = 0;
const skip = ['math-text.js', 'math.js', 'colormaps.js'];
for (const file of fs.readdirSync(path.join(root, 'js')).filter((f) => f.endsWith('.js') && !skip.includes(f))) {
  const source = fs.readFileSync(path.join(root, 'js', file), 'utf8');
  for (const match of source.matchAll(/"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/g)) {
    let value;
    try { value = vm.runInNewContext(match[0]); } catch { continue; }
    if (!value.includes('\\')) M.inline(value);
    count++;
  }
}
console.log(`PASS: monospace math, fractions, fallback symbols, scientific exponents, negative ticks, quantities, 44 plot explanations, horizontal axes, and ${count} authored strings.`);
