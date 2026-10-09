"use strict";

const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

// Exercise the production helpers without requiring a DOM or a build dependency.
const source = readFileSync(join(__dirname, "app.js"), "utf8");
const start = source.indexOf("  function niceTicks(");
const end = source.indexOf("  function plotDomain(");
assert.ok(start >= 0 && end > start, "Chart formatting helpers must be present");
const helpers = {};
vm.runInNewContext(source.slice(start, end), helpers);

test("fractional GHz ticks retain distinct labels at negative magnitudes", () => {
  assert.equal(helpers.tickLabel(-12, 0.5), "-12.0");
  assert.equal(helpers.tickLabel(-11.5, 0.5), "-11.5");
  assert.equal(helpers.tickLabel(-11.52, 0.02), "-11.52");
});

test("adjacent generated ticks never share a rounded label", () => {
  for (const [minimum, maximum] of [[-12.2, -11.3], [-11.59, -11.48], [10, 10.8], [0, 0.012], [-0.003, 0.003], [2600, 2660]]) {
    const ticks = Array.from(helpers.niceTicks(minimum, maximum, 4));
    assert.ok(ticks.length > 1);
    const spacing = ticks[1] - ticks[0];
    const labels = ticks.map((value) => helpers.tickLabel(value, spacing));
    assert.equal(new Set(labels).size, ticks.length, `Duplicate labels in ${labels.join(", ")}`);
    labels.forEach((label, index) => assert.ok(Math.abs(Number(label) - ticks[index]) < spacing * 1e-6));
  }
});

test("integer codes and floating-point zero format cleanly", () => {
  assert.equal(helpers.tickLabel(2660, 20), "2660");
  assert.equal(helpers.tickLabel(-1e-16, 0.1), "0.0");
  assert.equal(helpers.tickLabel(1.1, 0.09999999999999987), "1.1");
});
