"""Build pi_ctrl.v with Icarus Verilog and run test_pi.py through the cocotb runner, then emulate the same
PI in fxpmath and apytypes to show the Python-side fixed-point libraries agree with plain integers.
Run with Icarus Verilog 12 (Homebrew, /opt/homebrew/bin/iverilog must be on PATH):
  PATH=/opt/homebrew/bin:$PATH uv run --isolated --no-project --python 3.12 --with cocotb==2.1.0 \
     --with fxpmath==0.4.10 --with apytypes==0.5.1 python docs/research/05_tools_smoke/t12_fixed_point/run_cocotb.py
Run with Verilator 5.052 (conda-forge: micromamba create -p <env> -c conda-forge verilator=5.052):
  SIM=verilator PATH=<env>/bin:/opt/homebrew/bin:$PATH uv run ... (same arguments)
"""
import os, sys, tempfile, json
import numpy as np
from cocotb_tools.runner import get_runner

here = os.path.dirname(os.path.abspath(__file__))
build = tempfile.mkdtemp(prefix="cocotb_pi_")
sys.path.insert(0, here)
SIM = os.environ.get("SIM", "icarus")          # or "verilator"
r = get_runner(SIM)
r.build(sources=[os.path.join(here, "pi_ctrl.v")], hdl_toplevel="pi_ctrl", build_dir=build, always=True)
res = r.test(hdl_toplevel="pi_ctrl", test_module="test_pi", build_dir=build, test_dir=here)
print("results file:", res)
import xml.etree.ElementTree as ET
root = ET.parse(res).getroot()
print("cocotb testcases:", [(t.get("name"), "FAIL" if t.find("failure") is not None else "PASS") for t in root.iter("testcase")])
os.remove(res)

# fixed-point libraries: Q-format quantisation of a float PI gain and a wrap/saturate check
from fxpmath import Fxp
import apytypes
kp = 0.37512
f = Fxp(kp, signed=True, n_word=16, n_frac=8)          # Q7.8
a = apytypes.APyFixed.from_float(kp, int_bits=8, frac_bits=8)
x = Fxp(np.array([1000.0, -1000.0, 40000.0]), signed=True, n_word=16, n_frac=0, overflow="saturate")
print(json.dumps(dict(fxp_kp=float(f), fxp_kp_raw=int(f.val), apy_kp=float(a),
                      kp_int_ref=round(kp * 256) / 256, sat=list(map(int, x.get_val())),
                      apytypes=apytypes.__version__)))
