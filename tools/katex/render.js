// Render LaTeX strings with KaTeX (the same version the viewer docs/variables/index.html loads) and report the
// ones that fail. Used by tests/verification/test_variable_map.py.
//
//   node tools/katex/render.js < strings.json     # strings.json: ["\\lambda_L", ...]
//
// Prints a JSON object {"version": "...", "errors": [{"tex": ..., "error": ...}]}. strict mode "error" makes
// KaTeX fail on input that it would only warn about (unknown commands, text-mode characters in math, ...).
"use strict";
const katex = require("katex");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => { input += c; });
process.stdin.on("end", () => {
  const errors = [];
  for (const tex of JSON.parse(input)) {
    try {
      katex.renderToString(tex, {throwOnError: true, strict: "error", displayMode: false});
    } catch (e) {
      errors.push({tex, error: String(e.message || e)});
    }
  }
  process.stdout.write(JSON.stringify({version: katex.version, errors}) + "\n");
});
