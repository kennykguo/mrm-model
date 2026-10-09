"""Smoke test: gdsfactory - an add-drop ring layout written to GDS. Run:
  uv run --isolated --no-project --python 3.12 --with gdsfactory==9.51.0 python docs/research/05_tools_smoke/t09_gdsfactory.py
"""
import json, os, time, tempfile
t0 = time.time()
import gdsfactory as gf
gf.gpdk.PDK.activate()
c = gf.components.ring_double(radius=7.5, gap=0.2, length_x=0.01, length_y=0.01, cross_section="strip")
p = os.path.join(tempfile.mkdtemp(), "ring_double.gds")
c.write_gds(p)
print(json.dumps(dict(gdsfactory=gf.__version__, ports=[pp.name for pp in c.ports], bbox_um=str(c.dbbox()),
                      gds_bytes=os.path.getsize(p), seconds=round(time.time() - t0, 1))))
