// RUN: circt-opt %s --maxsim-extract-models="module=Parent models=Child" | FileCheck %s

hw.module @Child(in %clk : !seq.clock, in %reset : i1, in %id : i2, in %a : i8, in %unread : i8, out y : i8, out code : i4) {
  %c3_i3 = hw.constant 3 : i3
  %c0_i6 = hw.constant 0 : i6
  %idw = comb.concat %c0_i6, %id : i6, i2
  %twice = comb.add %a, %a : i8
  %sum = comb.add %twice, %idw : i8
  %y = seq.firreg %sum clock %clk : i8
  %bit = comb.extract %a from 0 : (i8) -> i1
  %next = comb.concat %bit, %c3_i3 : i1, i3
  %code = seq.firreg %next clock %clk : i4
  hw.output %y, %code : i8, i4
}

// The low bits of `code` are always 3, so only its top bit is a port. `a.a` is
// driven by the parent's logic, so it is a port; `b.a` is `a`'s output and
// `c.a` the parent's input, so both are connected directly. `id` is constant:
// a port of no bits, for the tokens only. `unread` is not read by `Child`: no
// port. The outputs the parent's logic does not read, `a.y` only wired to `b`,
// are channels of tokens too.
// CHECK-LABEL: hw.module @Parent(
// CHECK-SAME:    in %x : i8,
// CHECK-SAME:    in %a_y : i0, in %a_code_3_ : i1, in %b_y : i8, in %b_code : i0, in %c_y : i0, in %c_code_3_ : i1,
// CHECK-SAME:    out z : i8, out w : i4,
// CHECK-SAME:    out a_id : i0, out a_a : i8, out b_id : i0, out c_id : i0)
// CHECK-SAME:    attributes {maxsim.direct_inputs = ["b_a = a_y", "c_a = x"]}
// CHECK-NOT:   hw.instance
// CHECK-DAG:   [[C3:%.+]] = hw.constant 3 : i3
// CHECK-DAG:   [[A_CODE:%.+]] = comb.concat %a_code_3_, [[C3]] : i1, i3
// CHECK-DAG:   [[SUM:%.+]] = comb.add %x, %x : i8
// CHECK-DAG:   [[W:%.+]] = comb.xor [[A_CODE]], {{%.+}} : i4
// CHECK:       hw.output %b_y, [[W]], {{%.+}}, [[SUM]], {{%.+}}, {{%.+}} : i8, i4, i0, i8, i0, i0
hw.module @Parent(in %clk : !seq.clock, in %reset : i1, in %x : i8, out z : i8, out w : i4) {
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %sum = comb.add %x, %x : i8
  %a.y, %a.code = hw.instance "a" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c1_i2: i2, a: %sum: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %b.y, %b.code = hw.instance "b" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %a.y: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %c.y, %c.code = hw.instance "c" @Child(clk: %clk: !seq.clock, reset: %reset: i1, id: %c2_i2: i2, a: %x: i8, unread: %sum: i8) -> (y: i8, code: i4)
  %w = comb.xor %a.code, %c.code : i4
  hw.output %b.y, %w : i8, i4
}

// RUN: circt-opt %s --maxsim-extract-models="module=StructParent models=StructChild" | FileCheck %s --check-prefix=STRUCT

// A constant field of a struct is a constant bit too: only `v` is a port.
// STRUCT-LABEL: hw.module @StructParent(
// STRUCT-SAME:    in %c_o_3_0_ : i4,
// STRUCT-SAME:    out q : i8)
hw.module @StructChild(in %clk : !seq.clock, in %a : i4, out o : i8) {
  %c5_i4 = hw.constant 5 : i4
  %s = hw.struct_create (%c5_i4, %a) : !hw.struct<k: i4, v: i4>
  %b = hw.bitcast %s : (!hw.struct<k: i4, v: i4>) -> i8
  hw.output %b : i8
}
hw.module @StructParent(in %clk : !seq.clock, in %x : i4, out q : i8) {
  %c.o = hw.instance "c" @StructChild(clk: %clk: !seq.clock, a: %x: i4) -> (o: i8)
  hw.output %c.o : i8
}

// RUN: circt-opt %s --maxsim-extract-models="module=AggParent models=AggChild input-bits=c_o:3:2,c_o:1:0" | FileCheck %s --check-prefix=AGG
// RUN: not circt-opt %s --maxsim-extract-models="module=AggParent models=AggChild input-bits=c_o:1:0" 2>&1 | FileCheck %s --check-prefix=AGGERR

// The inputs are given: the bits they leave out are an aggregate constant.
// AGG-LABEL: hw.module @AggParent(
// AGG-SAME:    in %c_o_1_0_ : i2, in %c_o_3_2_ : i2,
// AGG:         [[O:%.+]] = comb.concat %c{{.*}}, %c_o_3_2_, %c_o_1_0_ : i4, i2, i2
// AGG:         hw.output [[O]] : i8
// The bits left out must be constants.
// AGGERR: error: bit 2 of 'c_o' is not a constant and is not in an input
hw.module @AggChild(in %clk : !seq.clock, in %a : i4, out o : i8) {
  %k = hw.aggregate_constant [2 : i2, 1 : i2] : !hw.struct<x: i2, y: i2>
  %s = hw.struct_create (%k, %a) : !hw.struct<k: !hw.struct<x: i2, y: i2>, v: i4>
  %b = hw.bitcast %s : (!hw.struct<k: !hw.struct<x: i2, y: i2>, v: i4>) -> i8
  hw.output %b : i8
}
hw.module @AggParent(in %clk : !seq.clock, in %x : i4, out q : i8) {
  %c.o = hw.instance "c" @AggChild(clk: %clk: !seq.clock, a: %x: i4) -> (o: i8)
  hw.output %c.o : i8
}
