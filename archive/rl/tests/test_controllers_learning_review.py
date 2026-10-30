# archived learning-specific regression checks from the shared controller review.
import dataclasses
import inspect
import math
import re

import numpy as np
import pytest


def test_fxp_mlp_derives_its_accumulator_proof():
    from archive.rl import fxp_mlp
    w, b = [0.9 * np.ones((1, 4))], [np.zeros(1)]
    narrow = fxp_mlp.quantize_mlp(w, b, ["linear"], acc_bits=20, out_fmt="Q7.8")
    wide = fxp_mlp.quantize_mlp(w, b, ["linear"], out_fmt="Q7.8")
    assert not narrow.proven and wide.proven
    x = np.full(4, narrow.in_fmt.raw_max, np.int64)
    # built by hand from the narrow layers: the proof does not hold, so the declared 20-bit accumulator is emulated
    m2 = fxp_mlp.FxpMlp(in_fmt=narrow.in_fmt, layers=narrow.layers, rounding=narrow.rounding, overflow=narrow.overflow)
    assert not m2.proven and not m2.describe()["proven_no_acc_overflow"]
    assert m2.forward(x).tolist() == narrow.forward(x).tolist() == [0]
    # a layer without the acc_bits_needed record (0 = unknown): the proof is recomputed from the weights
    bare = tuple(dataclasses.replace(l, acc_bits_needed=0) for l in narrow.layers)
    assert not fxp_mlp.FxpMlp(in_fmt=narrow.in_fmt, layers=bare).proven
    assert fxp_mlp.FxpMlp(in_fmt=wide.in_fmt, layers=tuple(dataclasses.replace(l, acc_bits_needed=0)
                                                           for l in wide.layers)).proven
    # a claimed proof that does not hold is refused (also through dataclasses.replace of a proven network)
    with pytest.raises(ValueError, match="accumulator"):
        fxp_mlp.FxpMlp(in_fmt=narrow.in_fmt, layers=narrow.layers, proven=True)
    with pytest.raises(ValueError, match="accumulator"):
        dataclasses.replace(wide, layers=narrow.layers, _packed=None)
    # proven=False (the per-addition path) stays allowed for a proven network
    slow = dataclasses.replace(wide, proven=False, _packed=None)
    assert slow.forward(x).tolist() == wide.forward(x).tolist()
    # the packed structural check refuses a proof flag on an accumulator narrower than its worst case
    p = narrow.pack().copy()
    assert fxp_mlp.fxp_mlp_check(p, 0) == 0
    p[fxp_mlp.H_PROVEN] = 1
    assert fxp_mlp.fxp_mlp_check(p, 0) == 14
    assert fxp_mlp.fxp_mlp_check(wide.pack(), 0) == 0



def test_reward_docstring_values():
    """The reward docstring: every '<value> at <n> s' of the log lock term is -log1p(n^2)."""
    from archive.rl.gym_env import RewardSpec
    pairs = re.findall(r"(-\d+(?:\.\d+)?) at (?:an rms error of )?(\d+(?:\.\d+)?) s\b", inspect.getcomments(RewardSpec))
    assert len(pairs) >= 2, pairs
    for v, n in pairs:
        assert float(v) == pytest.approx(-math.log1p(float(n) ** 2), abs=0.05), (v, n)

