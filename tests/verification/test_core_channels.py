"""Verification of mrm.core.channels (07 P0-7, D4): the v1 channel list, ChannelSpec checks, binding rules and the
flat int64 ABI layout."""
import numpy as np
import pytest

from mrm.core import channels as ch
from mrm.core.channels import BindingError, ChannelError, ChannelHardware, ChannelSpec, TempSensorHw

V1_ACTUATORS = {"tx.heater", "tx.bias", "tx.assignment", "datapath.bitstat_threshold", "monitor.pga_select"}
V1_SENSORS = {"tx.monitor.samples", "tx.monitor.sample_ticks", "tx.monitor.count", "tx.monitor.latest", "temp.die",
              "tx.ones_count", "tx.bits", "datapath.bitstat_above", "rx.error_count", "rx.error_count_valid",
              "monitor_valid", "adc_overrange", "tx.heater.applied", "tx.bias.applied"}


def test_v1_channel_list_defaults():
    c = ch.build_v1_channels(n_lines=8)
    assert {n for n, s in c.items() if s.kind == "actuator"} == V1_ACTUATORS
    assert {n for n, s in c.items() if s.kind == "sensor"} == V1_SENSORS
    h = c["tx.heater"]
    assert (h.shape, h.dtype, h.bits, h.min, h.max, h.unit) == (("B", "N"), "uint16", 16, 0, 65535, "code")
    s = c["tx.monitor.samples"]
    assert (s.shape, s.dtype, s.agg, s.count_channel) == (("B", "N", "M"), "int32", "block", "tx.monitor.count")
    assert c["tx.monitor.sample_ticks"].dtype == "int64"
    assert c["temp.die"].shape == ("B", 1)
    assert (c["tx.assignment"].min, c["tx.assignment"].max, c["tx.assignment"].init) == (-1, 7, -1)
    b = c["tx.bias"]
    assert b.offset == 1.0 and abs(b.offset + b.scale * b.max - 3.75) < 1e-12      # G26 axis 1.0-3.75 V
    assert c["rx.error_count"].rate_hz == 0.5                                       # one count per 2.0 s
    assert c["monitor_valid"].init == 1 and c["monitor_valid"].dtype == "bool"
    assert "heater.readback_v" not in c and "laser.tune" not in c                  # absent hardware


def test_optional_channels_and_temp_sensors():
    hw = ChannelHardware(heater_readback_present=True, laser_tune=True,
                         temp_sensors=(TempSensorHw("bank"), TempSensorHw("die", bits=10)))
    c = ch.build_v1_channels(hw, n_lines=4)
    assert {"heater.readback_v", "heater.readback_i", "laser.tune", "temp.bank", "temp.die"} <= set(c)
    assert c["laser.tune"].shape == ("B", "L") and c["laser.tune"].min == -32768 and c["laser.tune"].optional
    assert c["temp.die"].max == 1023
    rv = c["heater.readback_v"]
    assert rv.max == 4095 and abs(rv.scale * rv.max - hw.readback_v_full_scale_v) < 1e-12


def test_hardware_from_config_mapping():
    hw = ChannelHardware.from_config({
        "ring.heater.dac_bits": 14, "ring.monitor.adc_rate_hz": 250e3, "ring.monitor.adc_full_scale_ua": 20.0,
        "ring.monitor.pga_gains": [1, 4, 16], "ring.heater.readback.present": True,
        "temp_sensors": [{"name": "edge", "sample_rate_hz": 2000.0, "resolution_k": 0.1}]})
    assert hw.heater_dac_bits == 14 and hw.adc_rate_hz == 250e3 and hw.pga_gains == (1.0, 4.0, 16.0)
    assert abs(hw.adc_full_scale_a - 20e-6) < 1e-15 and hw.heater_readback_present
    assert hw.temp_sensors[0].name == "edge" and hw.temp_sensors[0].lsb_k == 0.1
    c = ch.build_v1_channels(hw)
    assert c["tx.heater"].max == 2**14 - 1 and c["monitor.pga_select"].max == 2
    assert abs(c["tx.monitor.latest"].scale * 65535 - 20e-6) < 1e-15


@pytest.mark.parametrize("kw, match", [
    (dict(name="Tx.Heater"), "dotted lower-case"),
    (dict(shape=("N", "B")), "start with 'B'"),
    (dict(shape=("B", "Q")), "one of"),
    (dict(dtype="float64"), "integer or bool"),
    (dict(min=5, max=1), "min 5 > max 1"),
    (dict(max=70000), "does not fit dtype"),
    (dict(agg="block"), "count_channel"),
    (dict(init=-1), "init"),
    (dict(rate_hz=0.0), "rate_hz"),
])
def test_channelspec_validation(kw, match):
    base = dict(name="tx.heater", kind="actuator", shape=("B", "N"), dtype="uint16", bits=16, min=0, max=65535,
                unit="code")
    base.update(kw)
    with pytest.raises(ChannelError, match=match):
        ChannelSpec(**base)


def test_shapes_resolve():
    c = ch.build_v1_channels()
    dims = {"B": 3, "N": 8, "L": 8, "M": 5, "S": 1}
    assert c["tx.monitor.samples"].resolve_shape(dims) == (3, 8, 5)
    assert c["tx.monitor.samples"].element_shape(dims) == (8, 5)
    assert c["tx.monitor.samples"].flag_shape(dims) == (8,)
    assert c["temp.die"].resolve_shape(dims) == (3, 1)
    with pytest.raises(ChannelError, match="'M'"):
        c["tx.monitor.samples"].resolve_shape({"B": 1, "N": 2})


def test_samples_per_step_is_exact():
    """M = ceil(P / A) is the most samples any controller window of P ticks can contain."""
    for p in range(1, 40):
        for a in range(1, 12):
            grid = np.arange(0, 400, a)
            most = max(int(np.sum((grid > k) & (grid <= k + p))) for k in range(0, 100))
            assert most == ch.samples_per_step(p, a) == max(1, -(-p // a))       # tight, for any phase
    with pytest.raises(ChannelError):
        ch.samples_per_step(0, 2)


# ---- binding ----------------------------------------------------------------------------------------------------------

def test_default_bindings():
    b = ch.resolve_bindings(ch.build_v1_channels())
    assert b == {"tx.heater": "controller", "tx.bias": "scenario", "tx.assignment": "scenario",
                 "datapath.bitstat_threshold": "controller", "monitor.pga_select": "controller"}


def test_bias_bound_to_controller_and_scenario_fails():
    """07 P0-7: a config that binds the bias to both the controller and the scenario fails to load."""
    c = ch.build_v1_channels()
    with pytest.raises(BindingError, match="tx.bias is bound to the controller"):
        ch.resolve_bindings(c, {"tx.bias": "controller"}, scenario_drives=["tx.bias"])
    # an additive scenario offset (ripple, droop) stays allowed
    b = ch.resolve_bindings(c, {"tx.bias": "controller"}, scenario_offsets=["tx.bias"])
    assert b["tx.bias"] == "controller"
    # the scenario may drive it when it owns it
    assert ch.resolve_bindings(c, None, scenario_drives=["tx.bias"])["tx.bias"] == "scenario"


@pytest.mark.parametrize("bind, drives, match", [
    ({"tx.heaterr": "controller"}, (), "unknown channel"),
    ({"tx.monitor.latest": "controller"}, (), "is a sensor"),
    ({"tx.bias": "plant"}, (), "must be one of"),
    ({}, ("temp.die",), "not an actuator"),
])
def test_binding_errors(bind, drives, match):
    with pytest.raises(BindingError, match=match):
        ch.resolve_bindings(ch.build_v1_channels(), bind, scenario_drives=drives)


# ---- ABI layout --------------------------------------------------------------------------------------------------------

def _layout(B=2, N=8, L=8, M=5, bind=None, hw=None):
    c = ch.build_v1_channels(hw, n_lines=L)
    b = ch.resolve_bindings(c, bind)
    dims = {"B": B, "N": N, "L": L, "M": M, "S": len((hw or ChannelHardware()).temp_sensors)}
    return c, b, dims, ch.make_abi_layout(c, dims, b)


def test_layout_offsets_are_contiguous_and_disjoint():
    c, b, dims, lay = _layout()
    spans = [(0, ch.OBS_HEADER)] + [(f.offset, f.offset + f.size) for f in lay.obs + lay.new]
    spans.sort()
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0
    assert spans[-1][1] == lay.obs_size
    act_spans = sorted((f.offset, f.offset + f.size) for f in lay.act)
    assert act_spans[0][0] == 0 and act_spans[-1][1] == lay.act_size
    assert lay.obs_field("tx.monitor.samples").size == 8 * 5
    assert lay.new_field("tx.monitor.samples").size == 8
    assert {f.name for f in lay.act} == {n for n, o in b.items() if o == "controller"}
    with pytest.raises(ChannelError):
        lay.act_field("tx.bias")                                        # bound to the scenario


def test_layout_ids_and_tables():
    c, b, dims, lay = _layout(bind={"tx.bias": "controller"})
    t = lay.obs_table()
    assert t[ch.ObsId.TX_MONITOR_LATEST] == lay.obs_field("tx.monitor.latest").offset
    assert t[ch.TEMP_ID_BASE] == lay.obs_field("temp.die").offset
    assert lay.new_table()[ch.ObsId.TX_MONITOR_COUNT] == lay.new_field("tx.monitor.count").offset
    at = lay.act_table()
    assert at[ch.ActId.TX_BIAS] == lay.act_field("tx.bias").offset
    assert at[ch.ActId.TX_ASSIGNMENT] == -1
    assert int(ch.ObsId.TX_MONITOR_SAMPLES) == 0 and int(ch.ActId.TX_HEATER) == 0     # frozen ids


def test_layout_fingerprint_tracks_shape():
    _, _, _, a = _layout(M=5)
    _, _, _, b = _layout(M=5)
    _, _, _, c = _layout(M=6)
    _, _, _, d = _layout(M=5, bind={"tx.bias": "controller"})
    assert a.fingerprint() == b.fingerprint()
    assert len({a.fingerprint(), c.fingerprint(), d.fingerprint()}) == 3


def test_pack_unpack_round_trip():
    c, b, dims, lay = _layout(B=3)
    r = np.random.default_rng(0)
    sensors, new = {}, {}
    for f in lay.obs:
        spec = c[f.name]
        lo, hi = max(spec.min, -10**6), min(spec.max, 10**6)
        sensors[f.name] = r.integers(lo, hi + 1, (3,) + f.shape).astype(spec.dtype)
    for f in lay.new:
        new[f.name] = r.integers(0, 2, (3,) + f.shape).astype(bool)
    obs = lay.pack_obs(77, sensors, new, active_ring=3)
    assert obs.shape == (3, lay.obs_size) and obs.dtype == np.int64
    tick, active, s2, n2 = lay.unpack_obs(obs)
    assert tick == 77 and active == 3
    for k in sensors:
        assert np.array_equal(s2[k], sensors[k]) and s2[k].dtype == sensors[k].dtype
    for k in new:
        assert np.array_equal(n2[k], new[k])
    act = np.zeros((3, lay.act_size), dtype=np.int64)
    heat = r.integers(0, 65536, (3, 8)).astype(np.uint16)
    lay.pack_act({"tx.heater": heat}, act)
    out = lay.unpack_act(act)
    assert np.array_equal(out["tx.heater"], heat) and out["tx.heater"].dtype == np.uint16
    rows = lay.describe()
    assert rows[0]["name"] == "tick" and any(rw["vector"] == "act" for rw in rows)


# ---- review additions -----------------------------------------------------------------------------------------------

def test_ids_of_a_subset_follow_the_full_set():
    """Hiding a channel (controller_interface.expose) must not renumber another: temp.die stays TEMP_ID_BASE + 1
    when temp.bank is hidden, so a compiled controller never reads the wrong sensor."""
    hw = ChannelHardware(temp_sensors=(TempSensorHw("bank"), TempSensorHw("die")))
    full = ch.build_v1_channels(hw)
    b = ch.resolve_bindings(full)
    obs_ids, act_ids = ch.assign_channel_ids(full)
    assert obs_ids["temp.bank"] == ch.TEMP_ID_BASE and obs_ids["temp.die"] == ch.TEMP_ID_BASE + 1
    sub = {n: c for n, c in full.items() if n != "temp.bank"}
    dims = {"B": 1, "N": 2, "L": 2, "M": 3, "S": 2}
    lay = ch.make_abi_layout(sub, dims, b, obs_ids=obs_ids, act_ids=act_ids)
    t = lay.obs_table()
    assert t[ch.TEMP_ID_BASE] == -1 and t[ch.TEMP_ID_BASE + 1] == lay.obs_field("temp.die").offset
    with pytest.raises(ChannelError, match="no channel id"):
        ch.make_abi_layout(sub, dims, b, obs_ids={"tx.monitor.latest": 3}, act_ids=act_ids)
    dup = dict(obs_ids, **{"temp.die": int(ch.ObsId.TX_BITS)})
    with pytest.raises(ChannelError, match="share an id"):
        ch.make_abi_layout(sub, dims, b, obs_ids=dup, act_ids=act_ids)


def test_signed_and_unsigned_code_dtypes():
    """A signed w-bit code fits int32 up to w = 32; an unsigned one up to w = 31 (int32 has 31 value bits)."""
    c = ch.build_v1_channels(ChannelHardware(laser_tune=True, laser_tune_bits=32, adc_bits=31))
    assert c["laser.tune"].dtype == "int32" and c["laser.tune"].min == -2**31
    assert c["tx.monitor.samples"].dtype == "int32" and c["tx.monitor.samples"].max == 2**31 - 1
    c = ch.build_v1_channels(ChannelHardware(adc_bits=32))
    assert c["tx.monitor.samples"].dtype == "int64" and c["tx.monitor.samples"].max == 2**32 - 1
    assert c["datapath.bitstat_threshold"].dtype == "uint32"


def test_unpack_act_cast_wraps_but_int64_does_not():
    c, b, dims, lay = _layout(B=1, N=2)
    act = np.zeros((1, lay.act_size), dtype=np.int64)
    off = lay.act_field("tx.heater").offset
    act[0, off:off + 2] = [70000, -5]
    assert lay.unpack_act(act)["tx.heater"].tolist() == [[4464, 65531]]           # documented hazard of cast=True
    raw = lay.unpack_act(act, cast=False)["tx.heater"]
    assert raw.dtype == np.int64 and raw.tolist() == [[70000, -5]]
