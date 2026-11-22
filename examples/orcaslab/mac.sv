// mac: a multiply-accumulate, the demo design of orcaslab/circt (README, Quick start).
// PIPE=0: the product and the add share one cycle, a deep path.
// PIPE=1: the product is registered first, so each cycle holds one of them.
module mac #(parameter int W = 16, parameter bit PIPE = 0) (
  input  logic           clk,
  input  logic           rst,
  input  logic           en,
  input  logic [W-1:0]   a,
  input  logic [W-1:0]   b,
  output logic [2*W+7:0] acc
);
  logic [W-1:0]   a_q, b_q;
  logic           en_q, add_en;
  logic [2*W-1:0] add_in;
  always_ff @(posedge clk) begin
    a_q  <= a;
    b_q  <= b;
    en_q <= en;
  end
  if (PIPE) begin : g_pipe
    logic [2*W-1:0] p_q;
    logic           en_p;
    always_ff @(posedge clk) begin
      p_q  <= a_q * b_q;
      en_p <= en_q;
    end
    assign add_in = p_q;
    assign add_en = en_p;
  end else begin : g_flat
    assign add_in = a_q * b_q;
    assign add_en = en_q;
  end
  always_ff @(posedge clk)
    if (rst)         acc <= '0;
    else if (add_en) acc <= acc + add_in;
endmodule
