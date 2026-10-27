module RefModule (
  input clk,
  input reset,
  input [3:0] duty,
  output pwm
);
  reg [3:0] count;
  always @(posedge clk)
    if (reset) count <= 4'd0;
    else count <= count + 4'd1;
  assign pwm = (count < duty);
endmodule
