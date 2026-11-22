// RUN: circt-opt --llhd-inline-calls %s | FileCheck %s
// RUN: circt-opt --llhd-inline-calls --symbol-dce %s | FileCheck %s --check-prefixes=CHECK,CHECK-DCE

// CHECK-LABEL: @Basic
hw.module @Basic(in %a: i42, in %b: i42, out u: i42, out v: i42) {
  // CHECK: llhd.combinational
  %0:2 = llhd.combinational -> i42, i42 {
    // CHECK-NOT: call @foo
    %1:2 = func.call @foo(%a, %b) : (i42, i42) -> (i42, i42)
    // CHECK-NEXT:   cf.br [[BB1:\^.+]]
    // CHECK-NEXT: [[BB1]]:
    // CHECK-NEXT:   [[TMP1:%.+]] = comb.add %a, %b :
    // CHECK-NOT:    call @bar
    // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
    // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor %a, [[TMP2]] :
    // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3]], %b :
    // CHECK-NEXT:   cf.br [[BB2:\^.+]]
    // CHECK-NEXT: [[BB2]]:
    // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
    // CHECK-NEXT: [[BB3]]([[CALLRES0:%.+]]: i42, [[CALLRES1:%.+]]: i42):

    // CHECK-NEXT: scf.execute_region
    %2:2 = scf.execute_region -> (i42, i42) {
      // CHECK-NOT: call @foo
      %3:2 = func.call @foo(%1#0, %1#1) : (i42, i42) -> (i42, i42)
      // CHECK-NEXT:   cf.br [[BB1:\^.+]]
      // CHECK-NEXT: [[BB1]]:
      // CHECK-NEXT:   [[TMP1:%.+]] = comb.add [[CALLRES0]], [[CALLRES1]] :
      // CHECK-NOT:    call @bar
      // CHECK-NEXT:   [[TMP2:%.+]] = hw.constant 42 :
      // CHECK-NEXT:   [[TMP3:%.+]] = comb.xor [[CALLRES0]], [[TMP2]] :
      // CHECK-NEXT:   [[TMP4:%.+]] = comb.mul [[TMP3]], [[CALLRES1]] :
      // CHECK-NEXT:   cf.br [[BB2:\^.+]]
      // CHECK-NEXT: [[BB2]]:
      // CHECK-NEXT:   cf.br [[BB3:\^.+]]([[TMP1]], [[TMP4]] : i42, i42)
      // CHECK-NEXT: [[BB3]]([[CALLRES0:%.+]]: i42, [[CALLRES1:%.+]]: i42):

      // CHECK-NEXT: scf.yield [[CALLRES0]], [[CALLRES1]]
      scf.yield %3#0, %3#1 : i42, i42
    }

    llhd.yield %2#0, %2#1 : i42, i42
  }
  hw.output %0#0, %0#1 : i42, i42
}

// CHECK-DCE-NOT: @foo
func.func private @foo(%arg0: i42, %arg1: i42) -> (i42, i42) {
  cf.br ^bb1
^bb1:
  %0 = comb.add %arg0, %arg1 : i42
  %1 = call @bar_wrapper(%arg0) : (i42) -> i42
  %2 = comb.mul %1, %arg1 : i42
  cf.br ^bb2
^bb2:
  return %0, %2 : i42, i42
}

// CHECK-DCE-NOT: @bar
func.func private @bar(%arg0: i42) -> i42 {
  %0 = hw.constant 42 : i42
  %1 = comb.xor %arg0, %0 : i42
  return %1 : i42
}

// CHECK-DCE-NOT: @bar_wrapper
func.func private @bar_wrapper(%arg0: i42) -> i42 {
  %0 = call @bar(%arg0) : (i42) -> i42
  return %0 : i42
}

// Inlining a multi-block function with an `ub.unreachable` terminator must not
// crash and must keep the terminator in place.
// CHECK-LABEL: @Unreachable
hw.module @Unreachable(in %a: i1, out u: i42) {
  // CHECK: llhd.combinational
  %0 = llhd.combinational -> i42 {
    // CHECK-NOT: call @maybeUnreachable
    %1 = func.call @maybeUnreachable(%a) : (i1) -> i42
    // CHECK:      cf.cond_br %a, [[BB1:\^.+]], [[BB2:\^.+]]
    // CHECK-NEXT: [[BB1]]:
    // CHECK-NEXT:   ub.unreachable
    // CHECK-NEXT: [[BB2]]:
    llhd.yield %1 : i42
  }
  hw.output %0 : i42
}

// CHECK-DCE-NOT: @maybeUnreachable
func.func private @maybeUnreachable(%arg0: i1) -> i42 {
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  ub.unreachable
^bb2:
  %0 = hw.constant 42 : i42
  return %0 : i42
}

// Mutually recursive calls cannot be fully inlined without expanding
// infinitely in the IR. The call that closes the cycle is left in place
// on a best-effort basis instead of failing the pass.
// CHECK-LABEL: @Recursive
hw.module @Recursive() {
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK-NOT: call @recBar
    func.call @recFoo() : () -> ()
    // CHECK: call @recFoo
    llhd.yield
  }
}

func.func private @recFoo() {
  call @recBar() : () -> ()
  return
}

func.func private @recBar() {
  call @recFoo() : () -> ()
  return
}

// A directly self-recursive function cannot be inlined into itself either.
// CHECK-LABEL: @SelfRecursive
hw.module @SelfRecursive() {
  // CHECK: llhd.combinational
  llhd.combinational {
    // CHECK: call @recSelf
    func.call @recSelf() : () -> ()
    llhd.yield
  }
}

func.func private @recSelf() {
  call @recSelf() : () -> ()
  return
}

// A call of a coroutine that never suspends (a task without timing controls,
// such as an empty task an `assert` macro expands to) is inlined like a
// function call; its `llhd.return` continues after the call with its values.
// CHECK-LABEL: @NonSuspendingCoroutine
hw.module @NonSuspendingCoroutine(in %a: i42, in %b: i42) {
  %t = llhd.constant_time <0ns, 1d, 0e>
  %sig = llhd.sig %a : <i42>
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.call_coroutine @emptyTask
    llhd.call_coroutine @emptyTask() : () -> ()
    // CHECK-NOT: llhd.call_coroutine @addTask
    // CHECK: [[SUM:%.+]] = comb.add %a, %b
    %0 = llhd.call_coroutine @addTask(%a, %b) : (i42, i42) -> i42
    // CHECK-NOT: llhd.call_coroutine @outerTask
    // CHECK: comb.xor
    %1 = llhd.call_coroutine @outerTask(%0) : (i42) -> i42
    // A coroutine with control flow: its returns branch to the continuation.
    // CHECK-NOT: llhd.call_coroutine @branchTask
    // CHECK: cf.cond_br
    // CHECK: cf.br [[CONT:\^.+]]([[X:%.+]] : i42)
    // CHECK: cf.br [[CONT]]
    // CHECK: [[CONT]]([[R:%.+]]: i42):
    %2 = llhd.call_coroutine @branchTask(%1) : (i42) -> i42
    // A coroutine that waits, or calls one that does, stays a call.
    // CHECK: llhd.call_coroutine @waitTask()
    llhd.call_coroutine @waitTask() : () -> ()
    // CHECK: llhd.call_coroutine @callsWaitTask()
    llhd.call_coroutine @callsWaitTask() : () -> ()
    llhd.drv %sig, %2 after %t : i42
    llhd.halt
  }
}

// CHECK-DCE-NOT: @emptyTask
llhd.coroutine private @emptyTask() {
  llhd.return
}

// CHECK-DCE-NOT: @addTask
llhd.coroutine private @addTask(%x: i42, %y: i42) -> i42 {
  %0 = comb.add %x, %y : i42
  llhd.return %0 : i42
}

llhd.coroutine private @outerTask(%x: i42) -> i42 {
  %c = hw.constant 7 : i42
  %0 = comb.xor %x, %c : i42
  llhd.call_coroutine @emptyTask() : () -> ()
  llhd.return %0 : i42
}

// CHECK-DCE-NOT: @branchTask
llhd.coroutine private @branchTask(%x: i42) -> i42 {
  %c0 = hw.constant 0 : i42
  %c = comb.icmp eq %x, %c0 : i42
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  llhd.return %x : i42
^bb2:
  %y = comb.sub %c0, %x : i42
  llhd.return %y : i42
}

// CHECK-DCE: llhd.coroutine private @waitTask
llhd.coroutine private @waitTask() {
  llhd.wait ^bb1
^bb1:
  llhd.return
}

llhd.coroutine private @callsWaitTask() {
  llhd.call_coroutine @waitTask() : () -> ()
  llhd.return
}
