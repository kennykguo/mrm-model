// Replays golden vectors (mrm.sim.vectors, $readmemh) through FxpPi's per-ring sequencing and the Verilog
// fxp_pi_update, and compares every heater action with the recorded one. Defines (from the test): OBS_HEX,
// ACT_HEX (strings), KP, KI, FRAC, ACC, KW, RND, LO, HI. The include file vec.layout.vh comes from record_vectors.
`timescale 1ns/1ps
module tb_fxp_pi_replay;
`include "vec.layout.vh"
    localparam integer N  = MRMV_OBS_TX_MONITOR_LATEST_SIZE;
    localparam integer NW = MRMV_K * MRMV_B * MRMV_OBS_SIZE;
    localparam integer NA = MRMV_K * MRMV_B * MRMV_ACT_SIZE;
    localparam signed [`KW-1:0] KP = `KP;
    localparam signed [`KW-1:0] KI = `KI;

    reg [63:0] obs_mem [0:NW-1];
    reg [63:0] act_mem [0:NA-1];
    reg        [15:0]     u0s   [0:MRMV_B*N-1];
    reg        [15:0]     us    [0:MRMV_B*N-1];
    reg        [15:0]     ysets [0:MRMV_B*N-1];
    reg                   caps  [0:MRMV_B*N-1];
    reg signed [`ACC-1:0] integs[0:MRMV_B*N-1];

    reg signed [16:0]     e;
    reg signed [`ACC-1:0] integ_in;
    reg        [15:0]     u0_in;
    wire signed [`ACC-1:0] integ_out;
    wire       [15:0]     u_out;
    fxp_pi_update #(.FRAC(`FRAC), .ACC_BITS(`ACC), .KW(`KW), .RND(`RND)) dut (
        .e(e), .integ(integ_in), .kp(KP), .ki(KI), .u0(u0_in), .lo(16'd`LO), .hi(16'd`HI),
        .integ_new(integ_out), .u(u_out));

    integer k, b, i, idx, base, abase, mism, checked;
    reg fresh, valid;
    reg [15:0] y;
    reg [63:0] expected;
    initial begin
        $readmemh(`OBS_HEX, obs_mem);
        $readmemh(`ACT_HEX, act_mem);
        mism = 0;
        checked = 0;
        e = 0; integ_in = 0; u0_in = 0;
        for (k = 0; k < MRMV_K; k = k + 1) begin
            for (b = 0; b < MRMV_B; b = b + 1) begin
                base  = (k * MRMV_B + b) * MRMV_OBS_SIZE;
                abase = (k * MRMV_B + b) * MRMV_ACT_SIZE;
                for (i = 0; i < N; i = i + 1) begin
                    idx = b * N + i;
                    if (k == 0) begin
                        u0s[idx] = obs_mem[base + MRMV_OBS_TX_HEATER_APPLIED + i][15:0];
                        us[idx] = u0s[idx];
                        integs[idx] = 0;
                        caps[idx] = 0;
                    end
                    fresh = obs_mem[base + MRMV_NEW_TX_MONITOR_LATEST + i] != 0;
                    valid = obs_mem[base + MRMV_OBS_MONITOR_VALID + i] != 0;
                    y = obs_mem[base + MRMV_OBS_TX_MONITOR_LATEST + i][15:0];
                    if (fresh && valid) begin
                        if (!caps[idx]) begin
                            ysets[idx] = y;
                            caps[idx] = 1;
                        end else begin
                            e = $signed({1'b0, y}) - $signed({1'b0, ysets[idx]});
                            integ_in = integs[idx];
                            u0_in = u0s[idx];
                            #1;
                            integs[idx] = integ_out;
                            us[idx] = u_out;
                        end
                    end
                    expected = act_mem[abase + MRMV_ACT_TX_HEATER + i];
                    checked = checked + 1;
                    if ({48'd0, us[idx]} !== expected) begin
                        mism = mism + 1;
                        if (mism <= 10)
                            $display("MISMATCH k=%0d b=%0d ring=%0d rtl=%0d recorded=%0d", k, b, i, us[idx],
                                     expected);
                    end
                end
            end
        end
        $display("CHECKED %0d MISMATCHES %0d", checked, mism);
        $finish;
    end
endmodule
