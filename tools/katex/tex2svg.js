// Typeset LaTeX symbols to SVG with MathJax (mathjax-full, pinned in package.json) for the boxes of the variable
// map viewer (docs/variables/index.html draws each SVG as the background image of its box). Used by
// tools/gen_variable_map.py, which writes the result to docs/variables/symbols.json.
//
//   node tools/katex/tex2svg.js < strings.json     # strings.json: ["\\lambda_L", ...]
//
// Prints a JSON object {"version": "...", "items": [{"tex", "svg", "w", "h"}], "errors": [{"tex", "error"}]}.
// w and h are the size of the drawing in em (1000 viewBox units = 1 em). The SVG has no width, height or style
// attribute (the viewer sets the size) and draws in currentColor (the viewer puts the theme's text colour there).
// fontCache "none" puts every glyph path inline: no ids, so the output is the same on each run.
"use strict";
const {mathjax} = require("mathjax-full/js/mathjax.js");
const {TeX} = require("mathjax-full/js/input/tex.js");
const {SVG} = require("mathjax-full/js/output/svg.js");
const {liteAdaptor} = require("mathjax-full/js/adaptors/liteAdaptor.js");
const {RegisterHTMLHandler} = require("mathjax-full/js/handlers/html.js");
require("mathjax-full/js/input/tex/base/BaseConfiguration.js");
require("mathjax-full/js/input/tex/ams/AmsConfiguration.js");

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
// base and ams cover every command of mrm.params.symbols.KNOWN_COMMANDS; without the noundefined package an
// unknown command is an error (formatError throws it) and not a red name in the drawing
const tex = new TeX({packages: ["base", "ams"], formatError: (_jax, err) => { throw err; }});
const svg = new SVG({fontCache: "none"});
const doc = mathjax.document("", {InputJax: tex, OutputJax: svg});

function render(src) {
  const node = doc.convert(src, {display: false, em: 16, ex: 8, containerWidth: 1000});
  const el = adaptor.firstChild(node);              // <mjx-container><svg ...>
  const html = adaptor.outerHTML(el);
  if (/data-mml-node="merror"|data-mjx-error/.test(html)) throw new Error("MathJax error node in the output");
  const vb = adaptor.getAttribute(el, "viewBox").split(/\s+/).map(Number);
  // keep only what the drawing needs: the viewBox, the namespace and the paths (no size, role, data-* attributes)
  let out = html
    .replace(/^<svg[^>]*>/, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${vb.join(" ")}">`)
    .replace(/ data-[a-z-]+="[^"]*"/g, "")
    .replace(/<title>[^<]*<\/title>/g, "")
    .replace(/<defs><\/defs>/g, "");
  return {tex: src, svg: out, w: +(vb[2] / 1000).toFixed(3), h: +(vb[3] / 1000).toFixed(3)};
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => { input += c; });
process.stdin.on("end", () => {
  const items = [], errors = [];
  for (const src of JSON.parse(input)) {
    try { items.push(render(src)); } catch (e) { errors.push({tex: src, error: String(e.message || e)}); }
  }
  const version = require("mathjax-full/package.json").version;
  process.stdout.write(JSON.stringify({version, items, errors}) + "\n");
});
