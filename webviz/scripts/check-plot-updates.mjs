// Regression check for resize/update races, without a browser or model rerun.
import assert from 'node:assert/strict';
globalThis.document = {};
globalThis.window = { matchMedia: () => ({ addEventListener() {} }) };
const { relayoutPlot, restylePlot } = await import('../site/js/theme.js');
const calls = [];
let releaseFirst;
const firstUpdate = new Promise((resolve) => { releaseFirst = resolve; });
globalThis.Plotly = {
  relayout(gd, update) {
    calls.push(['layout', update]);
    return calls.length === 1 ? firstUpdate : Promise.resolve();
  },
  restyle(gd, update, indices) { calls.push(['style', update, indices]); return Promise.resolve(); },
};
const chart = { isConnected: true, offsetParent: {}, clientWidth: 400, layout: {}, data: [] };
const first = relayoutPlot(chart, { width: 900 });
const second = relayoutPlot(chart, { width: 400 });
const third = restylePlot(chart, { x: [[1]] }, [0]);
assert.equal(calls.length, 1, 'later updates must wait for the first update');
chart.clientWidth = 620;
releaseFirst();
await Promise.all([first, second, third]);
assert.deepEqual(calls.map((entry) => entry[0]), ['layout', 'layout', 'style']);
assert.equal(calls[0][1].width, 400);
assert.equal(calls[1][1].width, 620, 'queued resize must use the latest measured width');
chart.offsetParent = null;
await relayoutPlot(chart, { width: 0 });
await restylePlot(chart, { x: [[2]] }, [0]);
chart.offsetParent = {};
chart.clientWidth = 0;
await relayoutPlot(chart, { width: 0 });
chart.clientWidth = 400;
chart.isConnected = false;
await restylePlot(chart, { x: [[3]] }, [0]);
assert.equal(calls.length, 3, 'hidden, zero-width and detached charts must not update');
console.log('PASS: chart updates serialize, resize uses current width, and hidden/zero-width/detached charts skip updates.');
