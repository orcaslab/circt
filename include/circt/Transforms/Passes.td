//===-- Passes.td - Transforms pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes within the Transforms/ directory.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSFORMS_PASSES
#define CIRCT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def FlattenMemRef : Pass<"flatten-memref", "::mlir::ModuleOp"> {
  let summary = "Flatten memrefs";
  let description = [{
    Flattens multidimensional memories and accesses to them into
    single-dimensional memories.}];
  let constructor = "circt::createFlattenMemRefPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def FlattenMemRefCalls : Pass<"flatten-memref-calls", "::mlir::ModuleOp"> {
  let summary = "Flatten memref calls";
  let description = [{
    Flattens calls to functions which have multidimensional memrefs as arguments.
    This is done by casting memref arguments through memref.subview operations.
    Any called functions which had their type signatures changes will be replaced
    by a private function definition with the new signature.
    It is up to users of this pass to define how these rewritten functions are
    to be implemented.}];
  let constructor = "circt::createFlattenMemRefCallsPass()";
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def StripDebugInfoWithPred : Pass<"strip-debuginfo-with-pred", "::mlir::ModuleOp"> {
  let summary = "Selectively strip debug info from all operations";

  let description = [{
    This pass extends mlir::StripDebugInfoPass to selectively strip locations with a
    given predicate.
  }];
  let constructor = "circt::createStripDebugInfoWithPredPass({})";
  let options = [
    Option<"dropSuffix", "drop-suffix", "std::string",
           /*default=*/"",
           "Drop file location info with the specified suffix. This option is"
           "intended to be used for testing."> ];
}

def MapArithToCombPass : Pass<"map-arith-to-comb"> {
  let summary = "Map arith ops to combinational logic";
  let description = [{
    A pass which does a simple `arith` to `comb` mapping wherever possible.
    This pass will not convert:  
    * floating point operations
    * operations using `vector`-typed values

    This **does not** intend to be the definitive lowering/HLS pass of `arith`
    operations in CIRCT (hence the name "map" instead of e.g. "lower").
    Rather, it provides a simple way (mostly for testing purposes) to map
    `arith` operations.
  }];
  let options = [
    Option<"enableBestEffortLowering", "enable-best-effort-lowering",
           "bool", /*default=*/"false",
           "Enable best effort lowering of operations"
           "(eg, existing arith operations that have no lowering stay as is)">
  ];
  
  let constructor = "circt::createMapArithToCombPass(/*enableBestEffortLowering=*/false)";
  let dependentDialects = ["circt::hw::HWDialect, mlir::arith::ArithDialect, circt::comb::CombDialect"];
}

def ConvertIndexToUInt : Pass<"convert-index-to-uint", "::mlir::ModuleOp"> {
  let summary = "Rewrite index-based switch comparisons into unsigned integer ops.";
  let description = [{
    Replace `arith.cmpi` operations whose operands are `index` values (often
    produced when lowering `scf.index_switch`) with comparisons over the
    original integer type so that downstream hardware mapping passes (e.g.
    `--map-arith-to-comb`) do not encounter unsupported index-typed arithmetic.
    The pass converts any associated index constants and erases the redundant
    casts that become dead afterwards. For example, it rewrites:

      `arith.cmpi eq, (arith.index_cast %v : i4 to index),
                      (arith.constant 5 : index)`

    into:

      `arith.cmpi eq, %v, (arith.constant 5 : i4)`
  }];
  let constructor = "circt::createConvertIndexToUIntPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def InsertMergeBlocks : Pass<"insert-merge-blocks", "::mlir::ModuleOp"> {
  let summary = "Insert explicit merge blocks";
  let description = [{
    This pass inserts additional merge blocks for each block with more than
    two successors. A merge block is a block that only contains one operation,
    a terminator, and has two predecessors.
    The order successors are merged together mirrors the order different control
    paths are created. Thus, each block with two successors will have a corresponding
    merge block.

    This pass brings the CFG into a canonical form for further transformations.

    Treats loops and sub-CFGs with irregular control flow like single blocks.
  }];
  let constructor = "circt::createInsertMergeBlocksPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"];
}

def MaximizeSSA : Pass<"maximize-ssa", "::mlir::ModuleOp"> {
  let summary = "Convert every function in the module into maximal SSA form";
  let description = [{
    Convert the region within every function into maximal SSA form. This
    ensures that every value used within a block is also defined within the
    block, making dataflow explicit and removing block dominance-based dataflow
    semantics. The pass achieves this by adding block arguments wherever
    necessary to forward values to the block(s) where they are used.
  }];
  let constructor = "circt::createMaximizeSSAPass()";
}

def PrintOpCount : Pass<"print-op-count", "::mlir::ModuleOp"> {
  let summary = "Print operation count analysis results";
  let description = [{
    This pass prints data on operation counts in a builtin.module.
  }];
  let constructor = "circt::createPrintOpCountPass()";
  let options = [
    Option<"emissionFormat", "emission-format",
           "circt::OpCountEmissionFormat",
           "circt::OpCountEmissionFormat::Readable",
           "Specify the format to emit op count info in",
           [{::llvm::cl::values(
            clEnumValN(circt::OpCountEmissionFormat::Readable,
              "readable", "Emit the data in human-readable form"),
            clEnumValN(circt::OpCountEmissionFormat::ReadableSorted,
              "readable-sorted", "Emit the data in human-readable form, sorted for output determinism"),
            clEnumValN(circt::OpCountEmissionFormat::JSON,
              "json", "Emit the data in JSON form")
          )}]
           >
  ];
}

def MemoryBanking : Pass<"memory-banking", "::mlir::func::FuncOp"> {
  let summary = "Partition the memories used in affine parallel loops into banks";
  let constructor = "circt::createMemoryBankingPass()";
  let options = [
    ListOption<"bankingFactorsList", "factors", "unsigned",
           "Use banking factors to partition all memories that don't have banking attributes."
           "The elements specified in banking factors should be greater than 1;"
           "The elements specified in banking factors will be paired with the ones specified in banking dimensions."
           "In principle, the number of elements in banking factors should be equal to banking dimensions',"
           "with a single exception case: there is one banking factor with zero banking dimensions.">,
    ListOption<"bankingDimensionsList", "dimensions", "unsigned",
           "The dimensions along which to bank the memory. If unspecified and"
           "there is only one factor, the innermost dimension with size > 1 is used.">
  ];
  let dependentDialects = ["mlir::memref::MemRefDialect, mlir::scf::SCFDialect, mlir::affine::AffineDialect"];
}

def IndexSwitchToIf : Pass<"switch-to-if", "::mlir::ModuleOp"> {
  let summary = "Index switch to if";
  let description = [{
      Transform `scf.index_switch` to a series of `scf.if` operations.
      This is necessary for dialects that don't support switch statements, e.g., Calyx.
      An example:
      ```
        %0 = scf.index_switch %cond -> i32
          case 0 { ... }
          case 1 { ... }
          ...

        =>

        %c0 = arith.cmpi eq %0, 0
        %c1 = arith.cmpi eq %0, 1
        %0 = scf.if %c0 {
         ...
        } else {
          %1 = scf.if %c1 {
            ...
          } else {
            ...
          }
        }
      ```
  }];
  let constructor = "circt::createIndexSwitchToIfPass()";
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def HierarchicalRunner : Pass<"hierarchical-runner", "::mlir::ModuleOp"> {
  let summary = "Run passes under hierarchy";
  let description = [{
    This pass runs a specified pipeline of passes on the hierarchy of modules
    starting from a given top-level module. It allows for hierarchical
    application of transformations, which can be useful for targeting specific
    parts of a design or for applying different optimizations at different
    levels of the module hierarchy.
  }];
  let options = [
    Option<"pipelineStr", "pipeline", "std::string", "",
           "The pipeline to run under hierarchy">,
    Option<"topName", "top-name", "std::string", "",
           "The name of the top-level module to run the pass on">,
    Option<"includeBoundInstances", "include-bound-instances", "bool", "false",
           "Whether to include bound instances in the hierarchy">
  ];
}

def CellIFTInstrument : Pass<"cellift-instrument", "::mlir::ModuleOp"> {
  let summary = "CellIFT: cell-level information flow tracking instrumentation";
  let description = [{
    This pass instruments HW/Comb/Seq IR with parallel taint-tracking logic
    using the CellIFT methodology (cell-level dynamic information flow
    tracking). For every data value `v : iN`, the pass creates a parallel
    taint value `v_t : iN`. Module signatures are extended with taint ports,
    instances are updated accordingly, and combinational/sequential operations
    are instrumented with taint propagation rules.

    CellIFT operates at the macrocell level (add, mux, shift, compare, etc.)
    rather than at the gate level, which provides better scalability than
    gate-level IFT (GLIFT).

    The pass supports:
      - Combinational operations: All comb.* operations except comb.truth_table
      - Sequential operations: compreg, firreg
      - Hierarchy: module signature rewriting and instance updating

    Options:
      --taint-constants: If set, constants are assigned all-ones taint instead
                         of zero taint.
      --taint-suffix: The suffix appended to taint port/value names
                      (default: "_t").
  }];
  let options = [
    Option<"taintConstants", "taint-constants", "bool", /*default=*/"false",
           "Assign all-ones taint to constants instead of zero">,
    Option<"taintSuffix", "taint-suffix", "std::string",
           /*default=*/"\"_t\"",
           "Suffix appended to port names for taint signals">
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

def MaxsimExtractModels : Pass<"maxsim-extract-models", "::mlir::ModuleOp"> {
  let summary = "Cut the instances of separately simulated models out of a "
                "parent";
  let description = [{
    FireSim's Golden Gate simulates each partitioned class as its own model,
    connected to the others by channels. This pass removes the instances of
    the `models` modules from the flat `module`, leaving it with ports for
    what it exchanges with them, named `<instance>_<port>`:

    - An input of an instance becomes an output of the parent, unless inputs
      of the parent drive it, with constants: the instance is then connected
      to those directly. One that only constants drive becomes an output of
      no bits, a channel of tokens only.
    - An output of an instance becomes an input of the parent, but for the
      bits that are always the same constant in the model, which the parent
      uses instead. Those are found through wiring and through register bits
      whose next and reset values are that constant. The inputs for the other
      bits are named `<instance>_<port>_<hi>_<lo>_`, or
      `<instance>_<port>_<bit>_` for one, when they are not all of them.

    Clock ports are left out, as models take their clock from the simulator.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The parent to cut the models out of">,
    ListOption<"models", "models", "std::string",
               "The modules simulated as separate models">,
    ListOption<"inputBits", "input-bits", "std::string",
               "The inputs to take for the outputs of the models, as "
               "<instance>_<port>:<hi>:<lo>, to match an existing interface; "
               "the bits of such a port they leave out must be constants">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
}

def MaxsimFAME5 : Pass<"maxsim-fame5", "::mlir::ModuleOp"> {
  let summary = "Thread a flat model's state for FireSim's FAME-5 simulation";
  let description = [{
    Turns a flat `hw.module`, the model of one class of identical instances,
    into the FAME-5 multithreaded model FireSim's Golden Gate builds for it:
    one module simulating all `threads` instances, one host cycle per thread
    in turn, with each target port a latency-insensitive channel.

    - Every register becomes a memory of one entry per thread, read
      combinationally at the current thread and written on the target clock.
    - Each target port `P` becomes a decoupled channel: an input becomes
      `P_sink_{ready,valid,bits}`, an output `P_source_{ready,valid,bits}`. A
      thread finishes its target cycle once every input is valid and every
      output has fired; outputs that do not depend combinationally on an input
      fire eagerly, with the next cycle's value once the thread has finished.
      An input the model does not read has no channel.
    - The target clock is `hostClock` through a `FireSimFabricClockGate`,
      enabled when the current thread finishes.

    The module must be flat, with one clock input, and registers without
    asynchronous reset. A model of one instance keeps its state in registers;
    its target clock may come in on a channel (`clock-channel`), as from
    FireSim's clock bridge: then a cycle whose edge is not enabled has no
    tokens, and the first token only says which inputs the first cycle takes.
  }];
  let options = [
    Option<"moduleName", "module", "std::string", "",
           "The model to thread">,
    Option<"threads", "threads", "unsigned", "0",
           "The number of instances the model simulates">,
    Option<"threadedName", "threaded-name", "std::string", "",
           "The name of the threaded module (default: the model's name with "
           "`_threaded_<name>`)">,
    ListOption<"portSlices", "port-slices", "std::string",
               "Ports split into channels of some of their bits, as "
               "<port>:<hi>:<lo>; a channel is named <port>_<hi>_<lo>_, or "
               "<port>_<hi>_ for one bit">,
    ListOption<"channelFields", "channel-fields", "std::string",
               "Ports carried together as the fields of one channel, as "
               "<channel>.<field>=<port>, first field first; each field has "
               "its bits port <channel>_{sink,source}_<field>">,
    Option<"clockChannel", "clock-channel", "std::string", "",
           "For a model of one instance: the channel its target clock comes "
           "in on, whose tokens say whether each cycle's edge is enabled">,
  ];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect",
    "circt::seq::SeqDialect"
  ];
}

#endif // CIRCT_TRANSFORMS_PASSES
