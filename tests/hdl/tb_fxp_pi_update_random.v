// Random-operand check of fxp_pi_update: stimulus and expected results from the Python twin, $readmemh.
// Each case: 7 stimulus words {e, integ, kp, ki, u0, lo, hi} and 2 expected words {u, integ_new}.
// Defines: STIM_HEX, EXP_HEX (strings), NCASE, FRAC, ACC, KW, RND.
`timescale 1ns/1ps
module tb_fxp_pi_update_random;
    reg [63:0] stim [0:7*`NCASE-1];
    reg [63:0] expv [0:2*`NCASE-1];
    reg signed [16:0]     e;
    reg signed [`ACC-1:0] integ;
    reg signed [`KW-1:0]  kp, ki;
    reg        [15:0]     u0, lo, hi;
    wire signed [`ACC-1:0] integ_new;
    wire       [15:0]     u;
    fxp_pi_update #(.FRAC(`FRAC), .ACC_BITS(`ACC), .KW(`KW), .RND(`RND)) dut (
        .e(e), .integ(integ), .kp(kp), .ki(ki), .u0(u0), .lo(lo), .hi(hi), .integ_new(integ_new), .u(u));
    integer c, mism;
    reg signed [63:0] want_i;
    initial begin
        $readmemh(`STIM_HEX, stim);
        $readmemh(`EXP_HEX, expv);
        mism = 0;
        for (c = 0; c < `NCASE; c = c + 1) begin
            e = stim[7 * c][16:0];
            integ = stim[7 * c + 1][`ACC-1:0];
            kp = stim[7 * c + 2][`KW-1:0];
            ki = stim[7 * c + 3][`KW-1:0];
            u0 = stim[7 * c + 4][15:0];
            lo = stim[7 * c + 5][15:0];
            hi = stim[7 * c + 6][15:0];
            #1;
            want_i = expv[2 * c + 1];
            if ({48'd0, u} !== expv[2 * c] || integ_new !== want_i[`ACC-1:0]) begin
                mism = mism + 1;
                if (mism <= 10)
                    $display("MISMATCH case %0d: u rtl=%0d py=%0d integ rtl=%0d py=%0d", c, u, expv[2 * c],
                             integ_new, want_i);
            end
        end
        $display("CASES %0d MISMATCHES %0d", `NCASE, mism);
        $finish;
    end
endmodule
