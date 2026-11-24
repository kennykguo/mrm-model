'use strict';

const groups = [
  ['Device', [
    ['08', 'Silicon rings, slabs and bus', '#58baff', 0],
    ['09', 'Heaters', '#ff9560', 0],
    ['10', 'Vias', '#e3b7ff', 18],
    ['11', 'Metal leads', '#fbd86a', 36],
  ]],
  ['Surrounding layers', [
    ['06', 'Passivation', '#8393b8', 144],
    ['05', 'Upper metal stack', '#7786b0', 108],
    ['04', 'Lower metal stack', '#65729b', 72],
    ['07', 'Metal keep-out regions', '#b684e9', 86],
    ['03', 'Cladding', '#5a9ba2', -36],
    ['02', 'Buried oxide', '#7bb5a6', -72],
    ['01', 'Silicon substrate', '#667184', -108],
  ]],
];

const device = groups[0][1].map(([id]) => id);
const all = groups.flatMap(([, rows]) => rows.map(([id]) => id));
const views = {
  bank: {
    target: [175, 26, 0], width: 430, azimuth: 0.06, elevation: 1.18, visible: device,
    caption: 'Follow the blue bus past all eight rings. The yellow leads extend from each ring toward the same side of the chip.',
  },
  ring: {
    target: [150, 1, 0], width: 24, azimuth: 0.01, elevation: 1.29, visible: device,
    caption: 'Look for the narrow gap between the blue ring and straight bus. The orange heater warms the ring. Purple vias connect the heater region to the yellow metal leads.',
  },
  stack: {
    target: [175, 0, -12], width: 1240, azimuth: -0.23, elevation: 0.34, visible: all,
    caption: 'Layers are displaced vertically for inspection. Use the checkboxes to trace the stack from passivation down to the silicon substrate.',
  },
  assembled: {
    target: [175, 0, -96], width: 1300, azimuth: -0.27, elevation: 0.54, visible: all,
    caption: 'The surrounding layers enclose the rings. Hide the upper layers or switch to the ring bank to see the buried device.',
  },
};

const canvas = document.querySelector('#model');
const caption = document.querySelector('#caption');
const gl = canvas.getContext('webgl2', { antialias: true });
if (!gl) {
  caption.textContent = 'This browser could not start WebGL. Open the viewer in a browser with hardware acceleration enabled.';
} else {
  startViewer(gl);
}

function startViewer(gl) {
  const vertexSource = `#version 300 es
    in vec3 position;
    uniform vec3 target, right, up, toward;
    uniform vec2 span;
    uniform float offset;
    out vec3 location;
    void main() {
      location = position + vec3(0.0, 0.0, offset);
      vec3 p = location - target;
      gl_Position = vec4(dot(p, right)/span.x, dot(p, up)/span.y, -dot(p, toward)/2000.0, 1.0);
    }`;
  const fragmentSource = `#version 300 es
    precision highp float;
    in vec3 location;
    uniform vec3 color, toward;
    out vec4 pixel;
    void main() {
      vec3 normal = normalize(cross(dFdx(location), dFdy(location)));
      if (dot(normal, toward) < 0.0) normal = -normal;
      float light = 0.64 + 0.36 * max(0.0, dot(normal, normalize(vec3(-0.28, -0.42, 0.87))));
      pixel = vec4(color * light, 1.0);
    }`;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(16/255, 17/255, 20/255, 1);

  const position = gl.getAttribLocation(program, 'position');
  const uniforms = Object.fromEntries(
    ['target', 'right', 'up', 'toward', 'span', 'offset', 'color']
      .map(name => [name, gl.getUniformLocation(program, name)])
  );

  const panels = document.querySelector('#layers');
  const layers = [];
  for (const [title, entries] of groups) {
    const section = document.createElement('section');
    section.className = 'group';
    const heading = document.createElement('h2');
    heading.textContent = title;
    section.append(heading);
    for (const [id, label, hex, offset] of entries) {
      const row = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.background = hex;
      swatch.setAttribute('aria-hidden', 'true');
      row.append(input, swatch, label);
      section.append(row);
      const color = hex.slice(1).match(/../g).map(pair => parseInt(pair, 16) / 255);
      const [pointData, triangleData] = geometry[id].map(data => Uint8Array.from(atob(data), char => char.charCodeAt(0)).buffer);
      const triangles = new Uint32Array(triangleData);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pointData), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triangles, gl.STATIC_DRAW);
      layers.push({ id, input, color, offset, vao, count: triangles.length });
      input.addEventListener('change', draw);
    }
    panels.append(section);
  }

  let selected, camera, drag;
  function draw() {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * scale);
    const height = Math.round(canvas.clientHeight * scale);
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { target, azimuth, elevation } = camera;
    const sinA = Math.sin(azimuth), cosA = Math.cos(azimuth);
    const sinE = Math.sin(elevation), cosE = Math.cos(elevation);
    gl.uniform3fv(uniforms.target, target);
    gl.uniform3f(uniforms.right, cosA, sinA, 0);
    gl.uniform3f(uniforms.up, -sinA * sinE, cosA * sinE, cosE);
    gl.uniform3f(uniforms.toward, sinA * cosE, -cosA * cosE, sinE);
    const halfWidth = camera.width * Math.max(1, width / height / 1.9) / 2;
    gl.uniform2f(uniforms.span, halfWidth, halfWidth * height / width);
    for (const layer of layers) {
      if (!layer.input.checked) continue;
      gl.uniform1f(uniforms.offset, selected === 'stack' ? layer.offset : 0);
      gl.uniform3fv(uniforms.color, layer.color);
      gl.bindVertexArray(layer.vao);
      gl.drawElements(gl.TRIANGLES, layer.count, gl.UNSIGNED_INT, 0);
    }
  }

  const buttons = document.querySelectorAll('[data-view]');
  function choose(name) {
    selected = name;
    camera = { ...views[name] };
    caption.textContent = camera.caption;
    for (const button of buttons) button.setAttribute('aria-pressed', String(button.dataset.view === name));
    for (const layer of layers) layer.input.checked = camera.visible.includes(layer.id);
    draw();
  }

  function zoom(factor) {
    camera.width = Math.max(3, Math.min(6000, camera.width * factor));
    draw();
  }

  for (const button of buttons) button.addEventListener('click', () => choose(button.dataset.view));
  document.querySelector('#reset').addEventListener('click', () => { camera = { ...views[selected] }; draw(); });
  document.querySelector('#zoom-in').addEventListener('click', () => zoom(0.8));
  document.querySelector('#zoom-out').addEventListener('click', () => zoom(1.25));
  canvas.addEventListener('wheel', event => { event.preventDefault(); zoom(Math.exp(event.deltaY * 0.001)); }, { passive: false });
  canvas.addEventListener('pointerdown', event => {
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    camera.azimuth -= (event.clientX - drag[0]) * 0.008;
    camera.elevation = Math.max(-1.55, Math.min(1.55, camera.elevation + (event.clientY - drag[1]) * 0.008));
    drag = [event.clientX, event.clientY];
    draw();
  });
  canvas.addEventListener('lostpointercapture', () => { drag = null; });
  choose('bank');
  new ResizeObserver(draw).observe(canvas);
}
