#!/usr/bin/env node
// Zero-dependency dev server for webviz/site with live reload (npm run dev).
//   PORT=8001 npm run dev      another port (default 8000; the next free port is used if it is busy)
// Serves webviz/site on 127.0.0.1 only, never caches, and reloads the open page when a file under site/ changes
// (the HTML, CSS, JS, or the data files that `npm run regen` writes).
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'site');
const START_PORT = Number(process.env.PORT || 8000);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.ico': 'image/x-icon', '.txt': 'text/plain; charset=utf-8', '.map': 'application/json',
};
const RELOAD = `<script>(() => { const es = new EventSource('/__livereload'); es.onmessage = (e) => { if (e.data === 'reload') location.reload(); }; })();</script>`;
const clients = new Set();

function send(res, code, body, type = 'text/plain; charset=utf-8') {
  res.writeHead(code, { 'Content-Type': type, 'Cache-Control': 'no-store' });
  res.end(body);
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/__livereload') {
    res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-store', Connection: 'keep-alive' });
    res.write('retry: 1000\n\n');
    clients.add(res);
    req.on('close', () => clients.delete(res));
    return;
  }
  let rel = decodeURIComponent(url.pathname);
  if (rel.endsWith('/')) rel += 'index.html';
  const file = path.resolve(ROOT, `.${rel}`);
  if (file !== ROOT && !file.startsWith(ROOT + path.sep)) return send(res, 403, 'forbidden');
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return send(res, 404, `not found: ${rel}`);
    const type = TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream';
    if (type.startsWith('text/html')) {
      fs.readFile(file, 'utf8', (e, txt) => {
        if (e) return send(res, 500, String(e));
        send(res, 200, txt.includes('</body>') ? txt.replace('</body>', `${RELOAD}\n</body>`) : txt + RELOAD, type);
      });
      return;
    }
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': st.size, 'Cache-Control': 'no-store' });
    fs.createReadStream(file).pipe(res);
  });
});

// live reload: one message per burst of changes (the generator writes many files)
let timer = null;
try {
  fs.watch(ROOT, { recursive: true }, (_ev, name) => {
    if (name && /(^|[/\\])\./.test(name)) return;
    clearTimeout(timer);
    timer = setTimeout(() => { for (const c of clients) c.write('data: reload\n\n'); }, 400);
  });
} catch (e) {
  console.warn(`live reload is off (fs.watch: ${e.message}); reload the page by hand`);
}

function listen(port, tries = 10) {
  server.once('error', (e) => {
    if (e.code === 'EADDRINUSE' && tries > 0) listen(port + 1, tries - 1);
    else { console.error(e.message); process.exit(1); }
  });
  server.listen(port, '127.0.0.1', () => {
    if (!fs.existsSync(path.join(ROOT, 'data', 'meta.json'))) console.warn('site/data/meta.json is missing: run `npm run regen` first.');
    console.log(`webviz: http://localhost:${port}  (serving ${ROOT}; live reload on; Ctrl+C to stop)`);
  });
}
listen(START_PORT);
