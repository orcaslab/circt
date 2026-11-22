// RUN: circt-verilog --ir-hw %s | FileCheck %s
// REQUIRES: slang

// A task without timing controls called from a clocked process (here the empty
// task a non-formal `assert` macro expands to, as PicoRV32 writes it, and a
// task that computes) runs within the caller's time step like a function: the
// calls are inlined and the process lowers to a register.

// CHECK-LABEL: hw.module @EmptyTaskCall
// CHECK-NOT: llhd.
// CHECK: seq.firreg
// CHECK-NOT: llhd.
module EmptyTaskCall(input logic clk, input logic [7:0] a, output logic [7:0] q);
  task empty_statement;
    begin end
  endtask
  task automatic twice(input logic [7:0] x, output logic [7:0] y);
    y = x + x;
  endtask
  logic [7:0] t;
  always @(posedge clk) begin
    empty_statement;
    twice(a, t);
    q <= t;
  end
endmodule
