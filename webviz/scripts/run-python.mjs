#!/usr/bin/env node
// npm run regen [-- --only steady,a1_aggressor] [-- --quick]  -> webviz/generate.py (every data file)
// npm run check [-- steady|a1_key|cold_key|a1_full|all]       -> webviz/check_data.py (default: all)
// Both run from the repository root through tools/memguard.py with the repository's .venv Python.
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PY = path.join(REPO, '.venv', 'bin', 'python');
const [what, ...rest] = process.argv.slice(2);
if (!fs.existsSync(PY)) {
  console.error(`No Python environment at ${PY}. Create it in the repository first (see the repository README).`);
  process.exit(1);
}
let args;
if (what === 'generate') {
  const d = new Date();
  const date = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  args = ['webviz/generate.py', ...(rest.includes('--date') ? [] : ['--date', date]), ...rest];
} else if (what === 'check') {
  args = ['webviz/check_data.py', ...(rest.length ? rest : ['all'])];
} else {
  console.error('usage: run-python.mjs generate|check [args]');
  process.exit(2);
}
const cmd = [path.join('tools', 'memguard.py'), '--', PY, ...args];
console.log(`(in ${REPO}) .venv/bin/python ${cmd.join(' ').replace(PY, '.venv/bin/python')}`);
const r = spawnSync(PY, cmd, { cwd: REPO, stdio: 'inherit' });
process.exit(r.status == null ? 1 : r.status);
