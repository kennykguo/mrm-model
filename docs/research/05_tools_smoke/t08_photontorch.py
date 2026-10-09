"""Smoke test: Photontorch 0.4.1 (2021, AGPL-3.0) on a current PyTorch - import and a time-domain
all-pass ring. Run:
  uv run --isolated --no-project --python 3.12 --with photontorch==0.4.1 --with torch python docs/research/05_tools_smoke/t08_photontorch.py
"""
import json, time
t0 = time.time()
import torch, photontorch as pt
print("photontorch", pt.__version__, "torch", torch.__version__)
class Ring(pt.Network):
    def __init__(self):
        super().__init__()
        self.src = pt.Source(); self.det = pt.Detector()
        self.dc = pt.DirectionalCoupler(coupling=0.1, trainable=False)
        self.wg = pt.Waveguide(length=47.1e-6, loss=1000, neff=2.49, ng=4.34, wl0=1.31e-6, trainable=False)
        self.link("src:0", "0:dc:1", "0:det"); self.link("dc:2", "0:wg:1", "3:dc")
nw = Ring()
env = pt.Environment(wl=1.31e-6, t=None, t_start=0, t_end=2e-11, dt=1e-14, freqdomain=False)
with env:
    det = nw(source=1)
print(json.dumps(dict(shape=list(det.shape), final=float(det[-1].flatten()[0]), seconds=round(time.time() - t0, 1))))
