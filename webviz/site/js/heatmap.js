// "How hot is the die?" Top view of the temperature map at the nearest key frame (the model's point-site
// heat law on a 120 x 50 grid), same colour scale as the 3-D view. The rings are filled with their own
// (engine) temperature, because the point law under-reads inside a ring.

import { plotBase, axis, PLOT_CONFIG, token } from './theme.js';
import { K0 } from './data.js';
import { fmtNum } from './traces.js';

export class HeatmapView {
  constructor(card, meta) {
    this.gd = card.querySelector('.plot');
    this.meta = meta;
    this.wrap = this.gd.parentElement;
    this.width = 0;
    new ResizeObserver(() => {
      const w = this.wrap.clientWidth;
      if (Math.abs(w - this.width) > 4 && this.last) { this.width = w; this.update(...this.last); }
    }).observe(this.wrap);
  }

  update(S, ring, k) {
    this.last = [S, ring, k];
    const g = this.meta.geometry;
    const th = S.therm;
    const b = plotBase();
    const muted = token('--text-muted');
    // the plot area sets how much text fits: below 600 px only the selected ring is labelled (no crammed labels)
    const wrapW = this.wrap.clientWidth || 600;
    const m = { l: 52, r: 70, t: 8, b: 44 };
    const plotW = Math.max(200, wrapW - m.l - m.r);
    const compact = plotW < 600;
    const base = th.t_base_k[k] - K0;
    const z = th.rise_k[k].map((row) => row.map((v) => (v == null ? null : v + base)));
    const temp = S.scales.temp;
    const obs = th.observers;
    const Rr = g.ring_radius_um;
    const ringC = obs.ring_rise_k[k].map((v) => v + base);
    const pdC = obs.pd_rise_k[k].map((v) => v + base);
    const mapRing = obs.map_at_ring_centre_k[k].map((v) => v + base);
    const xs = g.ring_centers_um.map((c) => c[0]);
    const ys = g.ring_centers_um.map((c) => c[1]);
    const data = [
      {
        type: 'heatmap', x: th.grid.x_um, y: th.grid.y_um, z, zmin: temp.lo, zmax: temp.hi, colorscale: temp.plotly(),
        zsmooth: 'best',
        colorbar: { title: { text: '°C', side: 'top', font: { size: 11, color: b.font.color } }, thickness: 12, len: 1, outlinewidth: 0, tickfont: { size: 10, color: b.font.color } },
        hovertemplate: 'x %{x:.0f} µm, y %{y:.1f} µm<br>map %{z:.1f} °C<extra></extra>',
      },
      {
        type: 'scatter', x: xs, y: ys, mode: 'markers', marker: { size: 18, opacity: 0 }, showlegend: false,
        customdata: ringC.map((v, r) => [r, v, mapRing[r]]),
        hovertemplate: 'R%{customdata[0]} own temperature %{customdata[1]:.1f} °C<br>(map at its centre %{customdata[2]:.1f} °C)<extra></extra>',
      },
      {
        type: 'scatter', x: g.monitor_pd_um.map((p) => p[0]), y: g.monitor_pd_um.map((p) => p[1]), mode: 'markers',
        marker: { size: 14, opacity: 0 }, showlegend: false, customdata: pdC.map((v, r) => [r, v]),
        hovertemplate: 'PD of R%{customdata[0]} %{customdata[1]:.2f} °C<extra></extra>',
      },
    ];
    const shapes = [];
    const ann = [];
    // bus (thin line), ring discs filled with their own temperature, PD squares, aggressor site. The map is the
    // same inferno colours in both themes, so the marks on it use fixed light colours, not theme tokens.
    const by = g.bus_waveguide_um[0][1];
    shapes.push({ type: 'line', x0: g.bus_waveguide_um[0][0], x1: g.bus_waveguide_um[1][0], y0: by, y1: by, line: { width: 1, color: hexOr(muted) } });
    g.ring_centers_um.forEach(([x, y], r) => {
      shapes.push({ type: 'circle', x0: x - Rr, x1: x + Rr, y0: y - Rr, y1: y + Rr, fillcolor: temp.css(ringC[r]),
        line: { width: r === ring ? 3 : 1, color: r === ring ? '#fff' : 'rgba(255,255,255,0.45)' } });
      if (!compact || r === ring) {
        ann.push({ x, y: y + Rr + 1.5, yanchor: 'bottom', showarrow: false, text: `R${r}`, font: { size: 11, color: '#fff' } });
      }
    });
    g.monitor_pd_um.forEach(([x, y]) => {
      shapes.push({ type: 'rect', x0: x - 3.5, x1: x + 3.5, y0: y - 3.5, y1: y + 3.5, line: { width: 1.2, color: '#fff' } });
    });
    for (const [site, [x, y]] of Object.entries(g.aggressor_sites_um)) {
      const p = S.aggMax[site] || 0;
      if (!(p > 0) && site !== 'aggressor') continue;
      shapes.push({ type: 'circle', x0: x - 4, x1: x + 4, y0: y - 4, y1: y + 4, line: { width: 1.5, color: '#fff', dash: p > 0 ? 'solid' : 'dot' } });
      const now = S.scn.drives.aggressor_w[site][S.keys[k]];
      if (!compact) {
        ann.push({ x: x - 6, y, xanchor: 'right', showarrow: false, text: p > 0 ? `aggressor<br>${fmtNum(now)} W` : 'aggressor<br>(off)', align: 'right', font: { size: 11, color: '#fff' }, bgcolor: 'rgba(0,0,0,0.6)', borderpad: 3 });
      }
      data.push({
        type: 'scatter', x: [x], y: [y], mode: 'markers', marker: { size: 14, opacity: 0 }, showlegend: false,
        hovertemplate: `${site === 'aggressor' ? 'aggressor site' : 'aggressor stress site'}: ${p > 0 ? `${fmtNum(now)} W` : 'off'}<extra></extra>`,
      });
    }
    const spanX = th.grid.x_um[th.grid.x_um.length - 1] - th.grid.x_um[0];
    const spanY = th.grid.y_um[th.grid.y_um.length - 1] - th.grid.y_um[0];
    const height = Math.round(plotW * spanY / spanX + m.t + m.b);
    const layout = {
      paper_bgcolor: b.paper_bgcolor, plot_bgcolor: b.plot_bgcolor, font: b.font, hoverlabel: b.hoverlabel,
      height, margin: m,
      xaxis: axis(b, { title: { text: 'x along the bus (µm)' }, range: [th.grid.x_um[0], th.grid.x_um[th.grid.x_um.length - 1]], showgrid: false, zeroline: false, constrain: 'domain' }),
      yaxis: axis(b, { title: { text: 'y (µm)' }, range: [th.grid.y_um[0], th.grid.y_um[th.grid.y_um.length - 1]], showgrid: false, zeroline: false, scaleanchor: 'x', scaleratio: 1, constrain: 'domain', dtick: 50 }),
      shapes, annotations: ann,
    };
    Plotly.react(this.gd, data, layout, PLOT_CONFIG);
  }
}

function hexOr(c) { return c || '#888'; }
