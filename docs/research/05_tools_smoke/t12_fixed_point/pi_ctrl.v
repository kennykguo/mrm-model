`timescale 1ns/1ps
// Tiny fixed-point PI heater controller (smoke test for cocotb + Icarus Verilog / Verilator).
// e = setpoint - meas (signed 16 bit), acc += KI*e (signed 32 bit, saturating),
// u = (KP*e + acc) >>> SH, clamped to an unsigned 12-bit heater DAC code.
module pi_ctrl #(parameter signed [15:0] KP = 16'sd96, parameter signed [15:0] KI = 16'sd5,
                 parameter SH = 8)
  (input clk, input rst, input signed [15:0] setpoint, input signed [15:0] meas,
   output reg [11:0] dac);
  reg signed [31:0] acc;
  wire signed [15:0] e = setpoint - meas;
  wire signed [47:0] e48 = {{32{e[15]}}, e};
  wire signed [47:0] kp48 = {{32{KP[15]}}, KP};
  wire signed [47:0] ki48 = {{32{KI[15]}}, KI};
  wire signed [47:0] acc48 = {{16{acc[31]}}, acc};
  wire signed [47:0] acc_next = acc48 + ki48 * e48;
  wire signed [31:0] acc_sat = (acc_next > 48'sh7FFFFFFF) ? 32'sh7FFFFFFF :
                               (acc_next < -48'sh80000000) ? 32'sh80000000 : acc_next[31:0];
  wire signed [47:0] accs48 = {{16{acc_sat[31]}}, acc_sat};  // concatenation is unsigned: keep it in a signed wire
  wire signed [47:0] v = (kp48 * e48 + accs48) >>> SH;       // all-signed operands, so >>> is arithmetic
  always @(posedge clk) begin
    if (rst) begin acc <= 0; dac <= 0; end
    else begin
      acc <= acc_sat;
      dac <= (v < 0) ? 12'd0 : (v > 48'sd4095) ? 12'd4095 : v[11:0];
    end
  end
endmodule
