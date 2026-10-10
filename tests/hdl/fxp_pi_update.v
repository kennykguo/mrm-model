// Verilog twin of mrm.control.fxp.fx_pi_update (07 P1-21): one positional PI update in declared formats.
// Combinational and synthesizable. Bit-exact against the Python/numba and C twins for RND = 0 (floor) and
// RND = 1 (half_up, the FxpPi default); the other rounding modes exist only in the Python and C twins.
//
//   a      = sat_ACC(integ + ki * e)                    exact 64-bit product and sum, one saturation
//   integ' = clamp(a, (lo - u0) << FRAC, (hi - u0) << FRAC)
//   v      = (u0 << FRAC) + kp * e + integ'
//   u      = clamp(round_FRAC(v), lo, hi)
//
// Signedness (05 C12: "a {...} concatenation is unsigned in Verilog, which turned >>> into a logical shift"):
// every operand is widened into a SIGNED 64-bit wire first; the unsigned codes go through $signed({zeros, x}).
module fxp_pi_update #(
    parameter integer FRAC = 20,       // fraction bits of kp, ki and the integrator
    parameter integer ACC_BITS = 40,   // integrator register width (signed)
    parameter integer KW = 24,         // gain register width (signed), e.g. Q3.20
    parameter integer RND = 1          // 0 floor, 1 half_up
) (
    input  wire signed [16:0]         e,          // y - y_set in ADC codes (Q16.0)
    input  wire signed [ACC_BITS-1:0] integ,
    input  wire signed [KW-1:0]       kp,
    input  wire signed [KW-1:0]       ki,
    input  wire        [15:0]         u0,
    input  wire        [15:0]         lo,
    input  wire        [15:0]         hi,
    output wire signed [ACC_BITS-1:0] integ_new,
    output wire        [15:0]         u
);
    wire signed [63:0] e64  = e;
    wire signed [63:0] kp64 = kp;
    wire signed [63:0] ki64 = ki;
    wire signed [63:0] in64 = integ;
    wire signed [63:0] u064 = $signed({48'd0, u0});
    wire signed [63:0] lo64 = $signed({48'd0, lo});
    wire signed [63:0] hi64 = $signed({48'd0, hi});
    localparam signed [63:0] AMAX = (64'sd1 <<< (ACC_BITS - 1)) - 64'sd1;
    localparam signed [63:0] AMIN = -(64'sd1 <<< (ACC_BITS - 1));
    localparam signed [63:0] HALF = (RND == 1 && FRAC > 0) ? (64'sd1 <<< (FRAC - 1)) : 64'sd0;

    wire signed [63:0] sum   = in64 + e64 * ki64;
    wire signed [63:0] a_sat = (sum > AMAX) ? AMAX : ((sum < AMIN) ? AMIN : sum);
    wire signed [63:0] ilo   = (lo64 - u064) <<< FRAC;
    wire signed [63:0] ihi   = (hi64 - u064) <<< FRAC;
    wire signed [63:0] a     = (a_sat < ilo) ? ilo : ((a_sat > ihi) ? ihi : a_sat);
    wire signed [63:0] v     = (u064 <<< FRAC) + e64 * kp64 + a;
    wire signed [63:0] r     = (v + HALF) >>> FRAC;
    wire signed [63:0] uc    = (r < lo64) ? lo64 : ((r > hi64) ? hi64 : r);

    assign integ_new = a[ACC_BITS-1:0];
    assign u = uc[15:0];
endmodule
