// RUN: circt-opt %s --maxsim-extract-models="module=Parent models=Child" | FileCheck %s

hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %id : i2, in %a : i8, out y : i8, out code : i4) {
  %c3_i3 = hw.constant 3 : i3
  %sum = comb.add %a, %a : i8
  %y = seq.firreg %sum clock %clk : i8
  %bit = comb.extract %a from 0 : (i8) -> i1
  %next = comb.concat %bit, %c3_i3 : i1, i3
  %code = seq.firreg %next clock %clk : i4
  hw.output %y, %code : i8, i4
}

// The low bits of `code` are always 3, so only its top bit is a port. `a.a` is
// driven by the parent's logic and `b.a` by `a`, so both are ports; `c.a` is
// the parent's input, connected to `c` directly. `id` is constant: a port of
// no bits, for the tokens only.
// CHECK-LABEL: hw.module @Parent(
// CHECK-SAME:    in %x : i8,
// CHECK-SAME:    in %a_y : i8, in %a_code_3_ : i1, in %b_y : i8, in %b_code_3_ : i1, in %c_y : i8, in %c_code_3_ : i1,
// CHECK-SAME:    out z : i8, out w : i4,
// CHECK-SAME:    out a_id : i0, out a_a : i8, out b_id : i0, out b_a : i8, out c_id : i0)
// CHECK-NOT:   hw.instance
// CHECK-DAG:   [[C3:%.+]] = hw.constant 3 : i3
// CHECK-DAG:   [[A_CODE:%.+]] = comb.concat %a_code_3_, [[C3]] : i1, i3
// CHECK-DAG:   [[SUM:%.+]] = comb.add %x, %x : i8
// CHECK-DAG:   [[W:%.+]] = comb.xor [[A_CODE]], {{%.+}} : i4
// CHECK:       hw.output %b_y, [[W]], {{%.+}}, [[SUM]], {{%.+}}, %a_y, {{%.+}} : i8, i4, i0, i8, i0, i8, i0
hw.module @Parent(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8, out w : i4) {
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %sum = comb.add %x, %x : i8
  %a.y, %a.code = hw.instance "a" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c1_i2: i2, a: %sum: i8) -> (y: i8, code: i4)
  %b.y, %b.code = hw.instance "b" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %a.y: i8) -> (y: i8, code: i4)
  %c.y, %c.code = hw.instance "c" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %x: i8) -> (y: i8, code: i4)
  %w = comb.xor %a.code, %c.code : i4
  hw.output %b.y, %w : i8, i4
}
