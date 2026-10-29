// Small shared helpers of the time plots: short event labels, colour with alpha, number format.

export function shortEvent(ev, ring) {
  const l = ev.label || '';
  if (l.startsWith('metrics start')) return null;
  if (l.startsWith('aggressor power at its peak')) return 'aggressor peak';
  if (l.includes('hot cusp')) return null;
  if (l.includes('laser power on every line')) return l.replace(' on every line', '');
  if (ev.ring != null) return ev.ring === ring ? `R${ring} locked` : null;
  if (ev.t_s === 0) return null;
  return l.length > 24 ? `${l.slice(0, 22)}…` : l;
}

export function hexA(color, a) {
  const c = color.trim();
  if (c.startsWith('#') && c.length === 7) {
    const r = parseInt(c.slice(1, 3), 16);
    const g = parseInt(c.slice(3, 5), 16);
    const b = parseInt(c.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return c;
}

export function fmtNum(v) {
  if (v == null || !Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  if (a >= 1) return v.toFixed(2);
  if (a >= 0.01) return v.toFixed(3);
  if (a === 0) return '0';
  return v.toPrecision(2);
}
