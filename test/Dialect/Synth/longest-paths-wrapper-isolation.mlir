// RUN: circt-opt %s --pass-pipeline='builtin.module(synth-print-longest-path-analysis{test=true})' -o /dev/null 2>&1 | FileCheck %s

// The analysis times a word-level operation by lowering a copy of it in a
// wrapper module of its own. Each wrapper is lowered alone: an operation the
// lowering cannot handle (here comb.reverse, which convert-comb-to-synth
// leaves) is a start point, and the compare analyzed after it is still timed
// from its operands rather than failing with it.
// CHECK-LABEL: root=afterUnlowerable
// CHECK-DAG: endPoint=Object($root.x[0]), startPoint=Object($root.c[0], delay={{[1-9][0-9]*}}
// CHECK-DAG: endPoint=Object($root.x[0]), startPoint=Object($root.a[7], delay={{[1-9][0-9]*}}
hw.module @afterUnlowerable(in %a : i8, in %c : i8, out q : i8, out x : i1) {
  %q = comb.reverse %a : i8
  %x = comb.icmp ult %c, %a : i8
  hw.output %q, %x : i8, i1
}
