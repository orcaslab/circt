<p align="center"><img src="docs/includes/img/circt-logo.svg"/></p>

[![](https://github.com/llvm/circt/actions/workflows/buildAndTest.yml/badge.svg?event=push)](https://github.com/llvm/circt/actions?query=workflow%3A%22Build+and+Test%22)
[![Nightly integration tests](https://github.com/llvm/circt/workflows/Nightly%20integration%20tests/badge.svg)](https://github.com/llvm/circt/actions?query=workflow%3A%22Nightly+integration+tests%22)

# orcaslab/circt: CIRCT for FoldSim and Beacon

This is a fork of [llvm/circt](https://github.com/llvm/circt), kept on top of
its `main` with full history. It carries the CIRCT changes that
[FoldSim](https://github.com/orcaslab/foldsim) (hierarchical FAME-5 folding
of RTL onto FPGAs) and [Beacon](https://github.com/orcaslab/beacon) (logic
depth, area and timing screens of RTL before synthesis) need. They are general
CIRCT fixes and features, kept here until they land upstream:

- **Synth:** `LongestPathAnalysis` that scales to a full GPU (each wrapper
  module lowered alone, no stale cached paths after an erase, distributed-RAM
  reads and memory addresses timed), the generic LUT mapper with arrival times
  at module boundaries and carry blocks, `LowerWordToBits` concat lookup by
  binary search, `SOPBalancing` of constant functions.
- **ImportVerilog / circt-verilog:** black-box modules (ports and parameters
  only), memories detected for synthesis, hierarchical names, signedness of
  propagated operands, `--max-unroll-trip-count`.
- **LLHD:** `Sig2Reg`, `Mem2Reg`, `Deseq`, `HoistSignals`, `CombineDrives` and
  `UnrollLoops` changes that lower the Vortex GPU's RTL to HW/Comb/Seq
  (aggregate presets, part-of-signal drives, `keep` registers), and
  `InlineCalls` inlining calls of coroutines that never suspend (a task with no
  timing control, such as PicoRV32's empty `assert` task).
- **ExportVerilog:** `PrepareForEmission` keeps declaration initializers of
  spilled constants.
- **HW / Seq:** `hw-expand-array-writes`, aggregates lowered to integers
  (`HWAggregateToComb`), `seq-reg-const-prop`, `seq-retime-firmem-reads`,
  `RegOfVecToMem` and `HWMemSimImpl` memory lowering for FPGA RAM inference.

Every merge from upstream follows one rule: one design per overlap, chosen on
evidence (tests, generality, measured QoR); the other is deleted, never kept as
a fallback.

## Quick start: build and run a demo

Tested on Ubuntu 24.04 (x86-64), clang 18, CMake 3.31, Ninja 1.11. Plan for
about 60 GB of disk, 2 GB of RAM per build job, and roughly an hour on 16
cores for LLVM/MLIR and CIRCT together.

**1. Prerequisites.**

```sh
sudo apt-get install -y git cmake ninja-build clang lld python3 ccache
```

CMake 3.20 or newer, and clang 16+ or GCC 11+. `lld` is optional: if it is
not installed, set `LINKER=gold` (or leave `LLVM_USE_LINKER` out) in step 3.

**2. Clone CIRCT and its LLVM.** The `llvm` submodule pins the LLVM commit this
CIRCT builds with.

```sh
mkdir -p ~/orcaslab && cd ~/orcaslab
git clone https://github.com/orcaslab/circt.git
cd circt
git submodule update --init --depth 1 llvm
```

**3. Build LLVM and MLIR** into `llvm/build`. `J` is the number of build jobs.

```sh
J=$(nproc); LINKER=lld
cmake -G Ninja -S llvm/llvm -B llvm/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=$LINKER \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_CCACHE_BUILD=ON
ninja -C llvm/build -j$J
```

**4. Build CIRCT** into `build`, with the SystemVerilog front end
(`circt-verilog`, slang is fetched at configure time), and run its tests.

```sh
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=$LINKER \
  -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=$PWD/llvm/build/bin/llvm-lit \
  -DCIRCT_SLANG_FRONTEND_ENABLED=ON -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF   # slang's C++20 module scan needs clang-scan-deps
ninja -C build -j$J
ninja -C build check-circt          # lit tests
export PATH=$PWD/build/bin:$PATH    # circt-verilog, circt-opt, circt-synth, firtool
```

**5. Run the demo.** [`examples/orcaslab`](examples/orcaslab) holds `mac.sv`, a
multiply-accumulate whose product and add share one cycle (`PIPE=0`), or take
one cycle each (`PIPE=1`). `demo.sh` runs the flow FoldSim and Beacon build on:

1. import SystemVerilog to the HW/Comb/Seq dialects (`circt-verilog`);
2. map to 6-input LUTs (`circt-synth --lower-to-k-lut=6`);
3. report the longest register-to-register paths
   (`synth-print-longest-path-analysis`);
4. write Verilog back out (`firtool`).

```sh
examples/orcaslab/demo.sh            # writes examples/orcaslab/out/
```

Expected output:

```
PIPE=0: Maximum path delay: 11, Verilog in out/mac0_out.sv
PIPE=1: Maximum path delay: 8, Verilog in out/mac1_out.sv
```

`examples/orcaslab/out/mac0_paths.txt` lists the deepest end points, `acc` from
`a_q` and `b_q` through the multiplier and the adder. The same steps by hand:

```sh
cd examples/orcaslab
circt-verilog --ir-hw --top=mac -G PIPE=0 mac.sv -o mac.mlir
circt-synth mac.mlir --top mac --lower-to-k-lut=6 -o mac_lut.mlir
circt-opt mac_lut.mlir -o /dev/null \
  --pass-pipeline='builtin.module(synth-print-longest-path-analysis{top-module-name=mac})'
firtool mac.mlir --format=mlir -o mac_out.sv
```

**Next.** [Beacon](https://github.com/orcaslab/beacon) counts the same paths
as Vivado does (carry chains, DSPs, distributed RAM), and
[FoldSim](https://github.com/orcaslab/foldsim) folds a design into an FPGA
simulator. Both build against this CIRCT with
`-DCIRCT_DIR=$PWD/build/lib/cmake/circt -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm`.

**Related repositories:** [orcaslab/foldsim](https://github.com/orcaslab/foldsim),
[orcaslab/beacon](https://github.com/orcaslab/beacon),
[orcaslab/firesim](https://github.com/orcaslab/firesim),
[orcaslab/slash](https://github.com/orcaslab/slash), and the
[Vortex GPGPU](https://github.com/vortexgpgpu/vortex) that uses them.

The rest of this file is upstream CIRCT's README.

---

# ⚡️ "CIRCT" / Circuit IR Compilers and Tools

"CIRCT" stands for "Circuit Intermediate Representations (IR) Compilers and Tools".  One might also interpret
it as the recursively as "CIRCT IR Compiler and Tools".  The T can be
selectively expanded as Tool, Translator, Team, Technology, Target, Tree, Type,
... we're ok with the ambiguity.

The CIRCT community is an open and welcoming community.  If you'd like to
participate, you can do so in a number of different ways:

1) Join our [Discourse Forum](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/) 
on the LLVM Discourse server.  To get a "mailing list" like experience click the 
bell icon in the upper right and switch to "Watching".  It is also helpful to go 
to your Discourse profile, then the "emails" tab, and check "Enable mailing list 
mode".  You can also do chat with us on [CIRCT channel](https://discord.com/channels/636084430946959380/742572728787402763) 
of LLVM discord server.

2) Join our weekly video chat.  Please see the
[meeting notes document](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#)
for more information.

3) Contribute code.  CIRCT follows all of the LLVM Policies: you can create pull
requests for the CIRCT repository, and gain commit access using the [standard LLVM policies](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access). See our [Developer Policy](docs/DeveloperPolicy.md) and [AI Tool Use Policy](docs/AIToolPolicy.md) for more details.

## Motivation

The EDA industry has well-known and widely used proprietary and open source
tools.  However, these tools are inconsistent, have usability concerns, and were
not designed together into a common platform.  Furthermore
these tools are generally built with
[Verilog](https://en.wikipedia.org/wiki/Verilog) (also
[VHDL](https://en.wikipedia.org/wiki/VHDL)) as the IRs that they
interchange.  Verilog has well known design issues, and limitations, e.g.
suffering from poor location tracking support.

The CIRCT project is an (experimental!) effort looking to apply MLIR and
the LLVM development methodology to the domain of hardware design tools.  Many
of us dream of having reusable infrastructure that is modular, uses
library-based design techniques, is more consistent, and builds on the best
practices in compiler infrastructure and compiler design techniques.

By working together, we hope that we can build a new center of gravity to draw
contributions from the small (but enthusiastic!) community of people who work
on open hardware tooling.  In turn we hope this will propel open tools forward,
enables new higher-level abstractions for hardware design, and
perhaps some pieces may even be adopted by proprietary tools in time.

For more information, please see our longer [charter document](docs/Charter.md).

## Getting Started

To get started hacking on CIRCT quickly, run the following commands. If you want to include `circt-verilog` in the build, add `-DCIRCT_SLANG_FRONTEND_ENABLED=ON` to the cmake call:

```sh
# Clone the repository and its submodules
git clone git@github.com:llvm/circt.git --recursive
cd circt

# Configure the build
cmake -G Ninja llvm/llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
    -DLLVM_ENABLE_LLD=ON
```

If you want to build everything about the CIRCT tools and libraries, run below command(also runs all tests):
```
ninja -C build check-circt
```

If you want to only build a specific part, for example the `circt-opt` tool:
```sh
ninja -C build bin/circt-opt
```

or the `firtool` tool:
```sh
ninja -C build bin/firtool
```

This will only build the necessary parts of LLVM, MLIR, and CIRCT, which can be a lot quicker than building everything.

### Dependencies

If you have git, ninja, python3, cmake, and a C++ toolchain installed, you should be able to build CIRCT.
For a more detailed description of dependencies, take a look at:

- [Getting Started with MLIR](https://mlir.llvm.org/getting_started/)
- [LLVM Requirements](https://llvm.org/docs/GettingStarted.html#requirements)

### Useful Options

The `-DCMAKE_BUILD_TYPE=Debug` flag enables debug information, which makes the whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.
The `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` flag generates a `compile_commands.json` file, which can be used by editors and language servers for autocomplete and other IDE-like features.

To get something that runs faster but is still easy to debug, use the `-DCMAKE_BUILD_TYPE=RelWithDebInfo` flag to do a release build with debug info.

To do a release build that runs very fast, use the `-DCMAKE_BUILD_TYPE=Release` flag.
Release mode makes a very large difference in performance.

Consult the [Getting Started](docs/GettingStarted.md) page for detailed information on configuring and compiling CIRCT.

Consult the [Python Bindings](docs/PythonBindings.md) page if you are mainly interested in using CIRCT from a Python prompt or script.

### Submodules

CIRCT contains LLVM as a git submodule.
The LLVM repository here includes staged changes to MLIR which may be necessary to support CIRCT.
It also represents the version of LLVM that has been tested.
MLIR is still changing relatively rapidly, so feel free to use the current version of LLVM, but APIs may have changed.

Whenever you checkout a new CIRCT branch that points to a different version of LLVM, run the following command to update the submodule:
```sh
git submodule update
```

The repository is set up to perform a shallow clone of the submodules, meaning it downloads just enough of the LLVM repository to check out the currently specified commit.
If you wish to work with the full history of the LLVM repository, you can manually "unshallow" the submodule:
```sh
cd llvm
git fetch --unshallow
```

### Building LLVM/MLIR Separately

You can also build LLVM/MLIR in isolation first, and then build CIRCT using that first build.
This allows you to pick different compiler options for the two builds, such as building CIRCT in debug mode but LLVM/MLIR in release mode.

First, build and test *LLVM/MLIR*:
```sh
cd llvm
cmake -G Ninja llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host
ninja -C build
ninja -C build check-mlir
cd ..
```

Then build and test *CIRCT*:
```sh
cmake -G Ninja . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm
ninja -C build
ninja -C build check-circt
ninja -C build check-circt-integration
```
