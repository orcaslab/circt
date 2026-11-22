#!/bin/bash
# demo.sh: the orcaslab/circt quick-start demo (README). Needs circt-verilog, circt-synth,
# circt-opt and firtool on PATH. For PIPE=0 and PIPE=1 it imports mac.sv to HW/Comb/Seq,
# maps it to 6-input LUTs, prints the longest path and writes the design back as Verilog.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-out}
mkdir -p "$OUT"
for p in 0 1; do
  circt-verilog --ir-hw --top=mac -G PIPE=$p mac.sv -o "$OUT/mac$p.mlir"
  circt-synth "$OUT/mac$p.mlir" --top mac --lower-to-k-lut=6 -o "$OUT/mac${p}_lut.mlir"
  circt-opt "$OUT/mac${p}_lut.mlir" -o /dev/null --pass-pipeline='builtin.module(synth-print-longest-path-analysis{top-module-name=mac show-top-k-percent=2 output-file='"$OUT/mac${p}_paths.txt"'})'
  firtool "$OUT/mac$p.mlir" --format=mlir -o "$OUT/mac${p}_out.sv"
  echo "PIPE=$p: $(grep -m1 'Maximum path delay' "$OUT/mac${p}_paths.txt"), Verilog in $OUT/mac${p}_out.sv"
done
