// RUN: circt-opt %s --maxsim-fame5="module=Hold threads=4" | FileCheck %s

// A register that holds its value unless enabled is written only when
// enabled, with the new value, rather than written back every cycle.
// CHECK-LABEL: hw.module @Hold_threaded_Hold(
// CHECK: seq.firmem.write_port %r[%threadIdx] = %d_sink_bits, clock %clk_buffer.O enable %en_sink_bits

// One that holds unless reset or enabled is written when either is.
// CHECK: [[EN:%.+]] = comb.mux %reset_sink_bits, %true{{.*}}, %en_sink_bits : i1
// CHECK: [[DATA:%.+]] = comb.mux %reset_sink_bits, %c0_i8, %d_sink_bits : i8
// CHECK: seq.firmem.write_port %s[%threadIdx] = [[DATA]], clock %clk_buffer.O enable [[EN]]
hw.module @Hold(in %clk : !seq.clock, in %reset : i1, in %en : i1, in %d : i8, out q : i8, out p : i8) {
  %c0_i8 = hw.constant 0 : i8
  %m = comb.mux %en, %d, %r : i8
  %r = seq.firreg %m clock %clk : i8
  %n = comb.mux %en, %d, %s : i8
  %s = seq.firreg %n clock %clk reset sync %reset, %c0_i8 : i8
  hw.output %r, %s : i8, i8
}
