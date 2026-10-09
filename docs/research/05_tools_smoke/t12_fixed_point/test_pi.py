"""cocotb test: drive random setpoint/measurement pairs, compare the DAC code each cycle with a
bit-exact Python golden model (plain integers)."""
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly

KP, KI, SH = 96, 5, 8


def golden(state, sp, me):
    e = sp - me
    acc = max(-2**31, min(2**31 - 1, state["acc"] + KI * e))
    v = (KP * e + acc) >> SH          # Python >> on negative ints is arithmetic, like >>>
    state["acc"] = acc
    return 0 if v < 0 else 4095 if v > 4095 else v


@cocotb.test()
async def pi_matches_golden(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.setpoint.value = 0
    dut.meas.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    st, rnd, mism = {"acc": 0}, random.Random(1), 0
    for _ in range(2000):
        sp, me = rnd.randint(-3000, 3000), rnd.randint(-3000, 3000)
        await FallingEdge(dut.clk)
        dut.setpoint.value, dut.meas.value = sp, me
        await RisingEdge(dut.clk)
        await ReadOnly()
        exp = golden(st, sp, me)
        mism += int(dut.dac.value) != exp
    assert mism == 0, f"{mism} mismatches"
    dut._log.info("2000 cycles bit-exact")
