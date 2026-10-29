// LaTeX on the page and in the Plotly charts.
// Authored notation and stated quantities are typeset with KaTeX (js/math-text.js); plain table values and ADC
// codes keep their tabular digits. Plotly keeps its own SVG text, because it measures that text for the layout.
// This module hides that text and draws a KaTeX copy at the same place, again after every redraw.
import './math-text.js';
const M = window.MathText;
const SVG = 'http://www.w3.org/2000/svg';
const XHTML = 'http://www.w3.org/1999/xhtml';
const labels = new WeakMap();
const bound = new WeakSet();
const sizes = new Map();
let measure;
function mathSize(html, style, source) {
  const key = style.font + source;
  if (!sizes.has(key)) {
    if (!measure) {
      measure = document.createElement('div');measure.className = 'math-measure';
      Object.assign(measure.style, { position:'fixed', left:'-100000px', top:'0', visibility:'hidden', whiteSpace:'pre', width:'max-content', pointerEvents:'none' });
      document.body.append(measure);
    }
    measure.style.font = style.font;measure.innerHTML = html;
    const box = measure.getBoundingClientRect();
    if (sizes.size > 2500) sizes.clear();
    sizes.set(key, { width: Math.ceil(box.width) + 2, height: Math.ceil(box.height) + 4 });
  }
  return sizes.get(key);
}
document.fonts?.addEventListener('loadingdone', () => {
  sizes.clear();
  requestAnimationFrame(() => document.querySelectorAll('.js-plotly-plot').forEach(renderPlotMath));
});

export function plotMathSource(text) {
  const raw = text.getAttribute('data-unformatted') || text.textContent;
  // Plotly stores scientific exponents in <sup> elements. Preserve their value:
  // flattening textContent would turn 5×10⁴ into the incorrect number 5×104.
  const scientific = raw.match(/^([+−-]?[\d.]+)\s*×\s*10<sup>([+−-]?\d+)<\/sup>$/);
  if (scientific) return `\\(${scientific[1].replace('−', '-')}\\times 10^{${scientific[2].replace('−', '-')}}\\)`;
  const power = raw.match(/^10<sup>([+−-]?\d+)<\/sup>$/);
  if (power) return `\\(10^{${power[1].replace('−', '-')}}\\)`;
  const numeric = raw.match(/^([+−-]?[\d,.]+)([kMGTmunpµ])?$/);
  if (numeric) {
    const number = numeric[1].replace('−', '-').replace(/,/g, '{,}');
    const suffix = numeric[2] === 'µ' ? '\\mu' : numeric[2] ? `\\mathrm{${numeric[2]}}` : '';
    return `\\(${number}${suffix}\\)`;
  }
  return text.textContent.replace(/\u200b/g, '');
}

export function renderPlotMath(gd) {
  if (!gd?.isConnected) return;
  for (const text of gd.querySelectorAll('svg text')) {
    if (text.closest('.math-plot-label,.hoverlayer')) continue;
    const source = plotMathSource(text);
    if (!M.hasMath(source)) continue;
    const box = text.getBBox();
    if (!box.width || !box.height) continue;
    let record = labels.get(text);
    if (!record || !record.group.isConnected) {
      const group = document.createElementNS(SVG, 'g');
      group.setAttribute('class', 'math-plot-label');
      group.style.pointerEvents = 'none';
      const foreign = document.createElementNS(SVG, 'foreignObject');
      const content = document.createElementNS(XHTML, 'div');
      foreign.appendChild(content);group.appendChild(foreign);
      text.after(group);record = { group, foreign, content };labels.set(text, record);
    }
    const style = getComputedStyle(text);
    const font = parseFloat(style.fontSize) || 12;
    const anchor = text.getAttribute('text-anchor') || style.textAnchor;
    const html = M.inline(source);
    const size = mathSize(html, style, source);
    const width = size.width;
    const origin = anchor === 'middle' ? box.x + box.width / 2 : anchor === 'end' ? box.x + box.width : box.x;
    record.group.setAttribute('transform', text.getAttribute('transform') || '');
    record.foreign.setAttribute('x', origin - (anchor === 'middle' ? width / 2 : anchor === 'end' ? width : 0));
    record.foreign.setAttribute('y', box.y - font * .1);
    record.foreign.setAttribute('width', width);
    record.foreign.setAttribute('height', Math.max(size.height, box.height + font));
    Object.assign(record.content.style, { fontSize: `${font}px`, color: style.fill,
      textAlign: anchor === 'middle' ? 'center' : anchor === 'end' ? 'right' : 'left' });
    if (record.source !== source) { record.content.innerHTML = html;record.source = source; }
    text.style.visibility = 'hidden';
    text.setAttribute('aria-hidden', 'true');
  }
}

export function watchPlotMath(gd) {
  if (bound.has(gd)) return;
  bound.add(gd);
  gd.on('plotly_afterplot', () => renderPlotMath(gd));
  gd.on('plotly_relayout', () => renderPlotMath(gd));
}

export function installMath(root = document.body) {
  let scheduled = false;
  const pending = new Set([root]);
  const observer = new MutationObserver((records) => {
    for (const record of records) {
      const el = record.target.nodeType === 1 ? record.target : record.target.parentElement;
      if (!el || el.closest('.katex,.lab-math-inline,.math-plot-label,.math-measure,.js-plotly-plot,script,style')) continue;
      pending.add(el);
    }
    if (pending.size && !scheduled) { scheduled = true;requestAnimationFrame(flush); }
  });
  const observe = () => observer.observe(root, { subtree: true, childList: true, characterData: true });
  function flush() {
    scheduled = false;observer.disconnect();
    try {
      for (const scope of pending) if (scope.isConnected) M.refresh(scope);
    } finally { pending.clear();observe(); }
  }
  flush();
  return observer;
}
