// The 3-D view: the die, the bus, the 8 rings to scale, heaters, monitor PDs, drivers, the aggressor site,
// the bus light and the temperature map (flat colour on the die, or a height field under it).
// Meshes are built once from meta.json; a frame change only writes colours, heights and the map texture.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { K0 } from './data.js';
import { token } from './theme.js';
import { mixHex } from './scales.js';

export const WG_DRAW_UM = 1.2;   // drawn waveguide width; the rib is 0.4 um (drawn 3x so it shows at full view)
export const COL_MAX_UM = 30;    // heater-power column height at the top of its scale
const FIELD_DEPTH = 40;          // the height field spans this depth below the die surface
const PD_SIZE = 7;               // drawn PD marker (the model has no PD size)

export class Scene3D {
  constructor(host, meta, { onPick, onHover }) {
    this.host = host;
    this.meta = meta;
    this.onPick = onPick;
    this.onHover = onHover;
    const tg = meta.thermal_grid;
    this.x0 = tg.x_um[0];
    this.x1 = tg.x_um[1];
    this.y0 = tg.y_um[0];
    this.y1 = tg.y_um[1];
    this.nx = tg.nx;
    this.ny = tg.ny;
    this.cx = (this.x0 + this.x1) / 2;
    this.cy = (this.y0 + this.y1) / 2;
    this.S = null;
    this.key = -1;
    this.frame = 0;
    this.ring = 0;
    this.colorMode = 'temp';
    this.mapMode = 'flat';
    this.dirty = true;
    this.pickables = [];

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.domElement.className = 'gl3d';
    host.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels3d';
    host.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 2, 1, 8000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.minDistance = 25;
    this.controls.maxDistance = 2500;
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.controls.addEventListener('start', () => { this.userMoved = true; });

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8);
    sun.position.set(120, 400, 260);
    this.scene.add(sun);

    this.build();
    this.resetView();
    this.setTheme();

    this.raycaster = new THREE.Raycaster();
    this.pointer = null;
    const el = this.renderer.domElement;
    el.addEventListener('pointermove', (e) => { this.pointer = e; });
    el.addEventListener('pointerleave', () => { this.pointer = null; this.onHover(null); });
    el.addEventListener('pointerdown', (e) => { this.down = [e.clientX, e.clientY]; });
    el.addEventListener('pointerup', (e) => {
      if (!this.down) return;
      const moved = Math.hypot(e.clientX - this.down[0], e.clientY - this.down[1]);
      this.down = null;
      if (moved > 4) return;
      const hit = this.pick(e);
      if (hit && hit.kind === 'ring') this.onPick(hit.idx);
    });

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    const loop = () => {
      requestAnimationFrame(loop);
      if (this.controls.update()) this.dirty = true;
      if (this.pointer) {
        const e = this.pointer;
        this.pointer = null;
        const hit = this.pick(e);
        this.onHover(hit ? { ...hit, clientX: e.clientX, clientY: e.clientY } : null);
      }
      if (this.dirty) {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
        this.labels.render(this.scene, this.camera);
      }
    };
    loop();
  }

  X(x) { return x - this.cx; }
  Z(y) { return -(y - this.cy); }

  label(text, x, y, h, cls = '') {
    const div = document.createElement('div');
    div.className = `label3d ${cls}`;
    div.textContent = text;
    const obj = new CSS2DObject(div);
    obj.position.set(this.X(x), h, this.Z(y));
    this.scene.add(obj);
    div.obj = obj;   // CSS2DRenderer drives display from obj.visible
    return div;
  }

  flat(geo) { geo.rotateX(-Math.PI / 2); return geo; }

  build() {
    const g = this.meta.geometry;
    const S = this.scene;
    const dx = (this.x1 - this.x0) / (this.nx - 1);
    const dy = (this.y1 - this.y0) / (this.ny - 1);
    // the map plane is half a node wider on every side, so each texel centre sits on its grid node
    const W = this.x1 - this.x0 + dx;
    const H = this.y1 - this.y0 + dy;

    this.slab = new THREE.Mesh(new THREE.BoxGeometry(W, 4, H),
      new THREE.MeshStandardMaterial({ color: 0x77756f, roughness: 0.95 }));
    this.slab.position.set(0, -2.03, 0);
    S.add(this.slab);

    // flat temperature map on the die surface
    this.texData = new Uint8Array(this.nx * this.ny * 4);
    this.tex = new THREE.DataTexture(this.texData, this.nx, this.ny, THREE.RGBAFormat);
    this.tex.colorSpace = THREE.SRGBColorSpace;
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.needsUpdate = true;
    this.mapMesh = new THREE.Mesh(this.flat(new THREE.PlaneGeometry(W, H)), new THREE.MeshBasicMaterial({ map: this.tex }));
    S.add(this.mapMesh);

    // neutral die surface (map off) and the glass die surface (height-field mode)
    this.plainMat = new THREE.MeshBasicMaterial({ color: 0x9a9892 });
    this.plain = new THREE.Mesh(this.flat(new THREE.PlaneGeometry(W, H)), this.plainMat);
    this.plain.position.y = 0.01;
    S.add(this.plain);
    this.glassMat = new THREE.MeshBasicMaterial({ color: 0xbfc4c8, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide });
    this.glass = new THREE.Mesh(this.flat(new THREE.PlaneGeometry(W, H)), this.glassMat);
    S.add(this.glass);
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(this.flat(new THREE.PlaneGeometry(W, H))),
      new THREE.LineBasicMaterial({ color: 0x8a8984 }));
    this.glass.add(edge);
    this.glassEdgeMat = edge.material;

    // height field: one vertex per grid node, depth = the same clipped colour fraction
    const n = this.nx * this.ny;
    this.fieldPos = new Float32Array(n * 3);
    this.fieldCol = new Float32Array(n * 3);
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        const v = j * this.nx + i;
        this.fieldPos[3 * v] = this.X(this.x0 + i * dx);
        this.fieldPos[3 * v + 1] = -FIELD_DEPTH;
        this.fieldPos[3 * v + 2] = this.Z(this.y0 + j * dy);
      }
    }
    const idx = [];
    for (let j = 0; j < this.ny - 1; j++) {
      for (let i = 0; i < this.nx - 1; i++) {
        const a = j * this.nx + i;
        const b = a + 1;
        const c = a + this.nx;
        const d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
    }
    const fg = new THREE.BufferGeometry();
    fg.setAttribute('position', new THREE.BufferAttribute(this.fieldPos, 3));
    fg.setAttribute('color', new THREE.BufferAttribute(this.fieldCol, 3));
    fg.setIndex(idx);
    this.fieldGeo = fg;
    this.field = new THREE.Mesh(fg, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.94 }));
    S.add(this.field);

    // bus waveguide in 9 segments: input -> ring 0 -> ... -> ring 7 -> output
    const busY = g.bus_waveguide_um[0][1];
    const xs = [g.bus_waveguide_um[0][0], ...g.bus_order.map((r) => g.ring_centers_um[r][0]),
      g.bus_waveguide_um[g.bus_waveguide_um.length - 1][0]];
    this.busSegs = [];
    for (let j = 0; j < xs.length - 1; j++) {
      const len = xs[j + 1] - xs[j];
      const m = new THREE.Mesh(new THREE.BoxGeometry(len, 0.7, WG_DRAW_UM), new THREE.MeshBasicMaterial({ color: 0x999999 }));
      m.position.set(this.X((xs[j] + xs[j + 1]) / 2), 0.36, this.Z(busY));
      m.userData = { kind: 'bus', idx: j };
      S.add(m);
      this.busSegs.push(m);
      this.pickables.push(m);
    }
    // a wider invisible hit target along the bus
    for (let j = 0; j < xs.length - 1; j++) {
      const len = xs[j + 1] - xs[j];
      const hit = new THREE.Mesh(new THREE.BoxGeometry(len, 2, 5), new THREE.MeshBasicMaterial({ visible: false }));
      hit.position.copy(this.busSegs[j].position);
      hit.userData = { kind: 'bus', idx: j };
      S.add(hit);
      this.pickables.push(hit);
    }

    // drop waveguides (drawn, neutral)
    this.dropMat = new THREE.MeshBasicMaterial({ color: 0x8a8984 });
    for (const poly of g.drop_waveguides_um || []) {
      for (let i = 0; i < poly.length - 1; i++) {
        const [xa, ya] = poly[i];
        const [xb, yb] = poly[i + 1];
        const len = Math.hypot(xb - xa, yb - ya);
        const m = new THREE.Mesh(new THREE.BoxGeometry(len + WG_DRAW_UM * 0.8, 0.5, WG_DRAW_UM * 0.8), this.dropMat);
        m.position.set(this.X((xa + xb) / 2), 0.26, this.Z((ya + yb) / 2));
        m.rotation.y = Math.atan2(-(this.Z(yb) - this.Z(ya)), this.X(xb) - this.X(xa));
        S.add(m);
      }
    }

    // rings, heaters, heater-power columns, outline and pick targets
    const R = g.ring_radius_um;
    const ha = g.heater_annulus_um;
    this.ringMats = [];
    this.columns = [];
    this.ringLabels = [];
    this.outlineMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    const heaterMat = new THREE.MeshBasicMaterial({ color: 0xc8834f });
    const colMat = new THREE.MeshStandardMaterial({ color: 0xc8834f, roughness: 0.45, metalness: 0.25, transparent: true, opacity: 0.9 });
    g.ring_centers_um.forEach(([x, y], r) => {
      const outline = new THREE.Mesh(this.flat(new THREE.RingGeometry(R - WG_DRAW_UM * 1.1, R + WG_DRAW_UM * 1.1, 80)), this.outlineMat);
      outline.position.set(this.X(x), 0.05, this.Z(y));
      S.add(outline);
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const torus = new THREE.Mesh(new THREE.TorusGeometry(R, WG_DRAW_UM / 2, 10, 80), mat);
      torus.rotation.x = Math.PI / 2;
      torus.position.set(this.X(x), WG_DRAW_UM / 2 + 0.1, this.Z(y));
      S.add(torus);
      this.ringMats.push(mat);
      const heater = new THREE.Mesh(this.flat(new THREE.RingGeometry(ha.inner, ha.outer, 56)), heaterMat);
      heater.position.set(this.X(x), 0.12, this.Z(y));
      S.add(heater);
      const col = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 1.8, 1, 24).translate(0, 0.5, 0), colMat);
      col.position.set(this.X(x), 0.15, this.Z(y));
      col.userData = { kind: 'heater', idx: r };
      S.add(col);
      this.columns.push(col);
      this.pickables.push(col);
      const hit = new THREE.Mesh(this.flat(new THREE.CircleGeometry(R + 2.5, 40)), new THREE.MeshBasicMaterial({ visible: false }));
      hit.position.set(this.X(x), 0.6, this.Z(y));
      hit.userData = { kind: 'ring', idx: r };
      S.add(hit);
      this.pickables.push(hit);
      this.ringLabels.push(this.label(`R${r}`, x, y - 14, 0.5, 'ring'));
    });
    this.haloMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    this.halo = new THREE.Mesh(this.flat(new THREE.RingGeometry(R + 2.2, R + 3.3, 80)), this.haloMat);
    S.add(this.halo);

    // monitor PDs (colour = PD temperature) and drivers
    this.pdMats = [];
    this.edgeMats = [];
    const pdGeo = new THREE.BoxGeometry(PD_SIZE, 1.6, PD_SIZE);
    const pdEdges = new THREE.EdgesGeometry(pdGeo);
    g.monitor_pd_um.forEach(([x, y], r) => {
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const pd = new THREE.Mesh(pdGeo, mat);
      pd.position.set(this.X(x), 0.8, this.Z(y));
      pd.userData = { kind: 'pd', idx: r };
      const em = new THREE.LineBasicMaterial({ color: 0x111111 });
      pd.add(new THREE.LineSegments(pdEdges, em));
      this.edgeMats.push(em);
      S.add(pd);
      this.pdMats.push(mat);
      this.pickables.push(pd);
    });
    const drvGeo = new THREE.BoxGeometry(8, 1.0, 5);
    const drvMat = new THREE.MeshStandardMaterial({ color: 0x8d8b85, roughness: 0.8 });
    const drvEdges = new THREE.EdgesGeometry(drvGeo);
    g.driver_um.forEach(([x, y], r) => {
      const d = new THREE.Mesh(drvGeo, drvMat);
      d.position.set(this.X(x), 0.5, this.Z(y));
      d.userData = { kind: 'driver', idx: r };
      const em = new THREE.LineBasicMaterial({ color: 0x111111 });
      d.add(new THREE.LineSegments(drvEdges, em));
      this.edgeMats.push(em);
      S.add(d);
      this.pickables.push(d);
    });

    // aggressor sites: a marker and a glow whose size and opacity follow the power
    const glowTex = new THREE.CanvasTexture(glowCanvas());
    glowTex.colorSpace = THREE.SRGBColorSpace;
    this.agg = {};
    for (const [site, [x, y]] of Object.entries(g.aggressor_sites_um)) {
      const marker = new THREE.Mesh(this.flat(new THREE.RingGeometry(2.4, 4.0, 40)), new THREE.MeshBasicMaterial({ color: 0xd03b3b }));
      marker.position.set(this.X(x), 1.25, this.Z(y));
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, transparent: true, depthWrite: false, opacity: 0 }));
      glow.position.set(this.X(x), 3, this.Z(y));
      const hit = new THREE.Mesh(this.flat(new THREE.CircleGeometry(6, 24)), new THREE.MeshBasicMaterial({ visible: false }));
      hit.position.set(this.X(x), 1.4, this.Z(y));
      hit.userData = { kind: 'aggressor', idx: site };
      S.add(marker, glow, hit);
      this.pickables.push(hit);
      const lab = this.label(site === 'aggressor' ? 'aggressor' : 'stress site', x, y - 11, 0.5, 'minor');
      this.agg[site] = { marker, glow, hit, lab };
    }

    // a few fixed labels and a 50 um scale bar
    const busX0 = g.bus_waveguide_um[0][0];
    this.label('light in →', busX0 + 22, busY - 7, 0.5, 'minor');
    this.label('monitor PDs', -32, g.monitor_pd_um[0][1], 0.5, 'minor');
    this.label('drivers', -32, g.driver_um[0][1], 0.5, 'minor');
    const sbY = this.y0 + 6;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(50, 0.5, 1.2), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    bar.position.set(this.X(this.x1 - 35), 0.3, this.Z(sbY));
    S.add(bar);
    this.barMat = bar.material;
    this.label('50 µm', this.x1 - 35, sbY + 6, 0.5, 'minor');
  }

  // distance at which the whole die fits the width of the view (with a margin)
  fitDistance() {
    const halfW = (this.x1 - this.x0) / 2 + 12;
    const tanH = Math.tan((this.camera.fov * Math.PI) / 360) * this.camera.aspect;
    // 1.15: the near edge of the tilted die is closer to the camera than the target
    return Math.max(380, (halfW / tanH) * 1.15);
  }

  resetView() {
    const d = this.fitDistance();
    const el = (52 * Math.PI) / 180;
    this.camera.position.set(0, d * Math.sin(el), d * Math.cos(el));
    this.controls.target.set(0, 0, 6);
    this.camera.up.set(0, 1, 0);
    this.controls.update();
    this.userMoved = false;
    this.view = 'reset';
    this.dirty = true;
  }

  topView() {
    this.camera.position.set(0, this.fitDistance(), 0.01);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.userMoved = false;
    this.view = 'top';
    this.dirty = true;
  }

  resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    if (this.S && (w < 700) !== this.compact) requestAnimationFrame(() => this.setFrame(this.frame));
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.labels.setSize(w, h);
    this.compact = w < 700;
    this.host.classList.toggle('compact', this.compact);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    // refit while the viewer has not moved the camera
    if (!this.userMoved) { if (this.view === 'top') this.topView(); else this.resetView(); }
    this.dirty = true;
  }

  setTheme() {
    this.scene.background = new THREE.Color(token('--scene-bg'));
    this.haloMat.color.set(token('--text-primary'));
    this.barMat.color.set(token('--text-primary'));
    this.busUnlit = token('--bus-unlit');
    this.busLit = token('--bus-lit');
    this.key = -1;
    if (this.S) this.setFrame(this.frame);
    this.dirty = true;
  }

  setScenario(S) {
    this.S = S;
    this.key = -1;
    for (const [site, a] of Object.entries(this.agg)) {
      const show = (S.aggMax[site] || 0) > 0 || site === 'aggressor';
      a.marker.visible = show;
      a.hit.visible = show;
      a.lab.obj.visible = show;
      a.lab.textContent = (S.aggMax[site] || 0) > 0 ? (site === 'aggressor' ? 'aggressor' : 'stress site') : 'aggressor (off)';
      a.glow.visible = false;
    }
  }

  setRing(r) {
    this.ring = r;
    const [x, y] = this.meta.geometry.ring_centers_um[r];
    this.halo.position.set(this.X(x), 0.07, this.Z(y));
    this.ringLabels.forEach((el, i) => el.classList.toggle('selected', i === r));
    this.dirty = true;
  }

  setColorMode(m) { this.colorMode = m; if (this.S) this.setFrame(this.frame); }

  setMapMode(m) {
    this.mapMode = m;
    this.mapMesh.visible = m === 'flat';
    this.plain.visible = m === 'off';
    this.slab.visible = m !== 'height';
    this.field.visible = m === 'height';
    this.glass.visible = m === 'height';
    this.dirty = true;
  }

  setFrame(f) {
    const S = this.S;
    if (!S) return;
    this.frame = f;
    const R = S.scn.rings;
    const sc = S.scales;
    for (let r = 0; r < S.nR; r++) {
      let c;
      if (this.colorMode === 'err') c = sc.err.rgb(S.err[f][r]);
      else if (this.colorMode === 'det') c = sc.det.rgb(R.detuning_pm[f][r]);
      else c = sc.temp.rgb(R.t_ring_k[f][r] - K0);
      this.ringMats[r].color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
      const p = sc.temp.rgb(R.t_pd_k[f][r] - K0);
      this.pdMats[r].color.setRGB(p[0] / 255, p[1] / 255, p[2] / 255, THREE.SRGBColorSpace);
      const h = Math.max(0.05, COL_MAX_UM * (R.p_heater_mw[f][r] || 0) / S.pHeatMax);
      this.columns[r].scale.y = h;
      const lo = R.line_on[f][r];
      this.ringLabels[r].textContent = this.compact ? `R${r}` : lo >= 0 ? `R${r} · L${lo}` : `R${r} · off-line`;
    }
    for (const [site, a] of Object.entries(this.agg)) {
      const series = S.scn.drives.aggressor_w[site];
      const mx = S.aggMax[site] || 0;
      if (!series || mx <= 0) { a.glow.visible = false; continue; }
      const t = Math.max(0, series[f] || 0) / mx;
      a.glow.visible = t > 1e-4;
      a.glow.material.opacity = 0.25 + 0.75 * t;
      const s = 8 + 36 * Math.sqrt(t);
      a.glow.scale.set(s, s, 1);
    }
    const k = S.nearestKey[f];
    if (k !== this.key) {
      this.key = k;
      this.updateMap(k);
      this.updateBus(k);
    }
    this.dirty = true;
  }

  updateMap(k) {
    const S = this.S;
    const rise = S.therm.rise_k[k];
    const base = S.therm.t_base_k[k] - K0;
    const temp = S.scales.temp;
    const nx = this.nx;
    for (let j = 0; j < this.ny; j++) {
      const row = rise[j];
      for (let i = 0; i < nx; i++) {
        const v = j * nx + i;
        const T = row[i] == null ? null : row[i] + base;
        const c = temp.rgb(T);
        this.texData[4 * v] = c[0];
        this.texData[4 * v + 1] = c[1];
        this.texData[4 * v + 2] = c[2];
        this.texData[4 * v + 3] = 255;
        const l = temp.lin(T);
        this.fieldCol[3 * v] = l[0];
        this.fieldCol[3 * v + 1] = l[1];
        this.fieldCol[3 * v + 2] = l[2];
        this.fieldPos[3 * v + 1] = -FIELD_DEPTH + (FIELD_DEPTH - 1) * temp.frac(T);
      }
    }
    this.tex.needsUpdate = true;
    this.fieldGeo.attributes.position.needsUpdate = true;
    this.fieldGeo.attributes.color.needsUpdate = true;
    this.fieldGeo.computeBoundingSphere();
  }

  updateBus(k) {
    const S = this.S;
    if (!S.bus) return;
    const seg = S.bus[k];
    this.busSegs.forEach((m, j) => {
      const c = mixHex(this.busUnlit, this.busLit, seg[j] / S.busMax);
      m.material.color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);
    });
  }

  pick(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const live = this.pickables.filter((o) => o.visible);
    const hits = this.raycaster.intersectObjects(live, false);
    if (!hits.length) return null;
    // a ring wins over the heater column and bus that sit inside or beside it
    const order = { ring: 0, aggressor: 1, pd: 2, heater: 3, driver: 4, bus: 5 };
    hits.sort((a, b) => (order[a.object.userData.kind] - order[b.object.userData.kind]) || (a.distance - b.distance));
    return hits[0].object.userData;
  }
}

function glowCanvas() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,240,190,1)');
  g.addColorStop(0.25, 'rgba(255,170,60,0.85)');
  g.addColorStop(0.6, 'rgba(230,70,30,0.35)');
  g.addColorStop(1, 'rgba(200,30,20,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return c;
}
