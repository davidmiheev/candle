#!/usr/bin/env bash
# Thread-count sweep for candle CPU quantized inference.
# Usage: GGUF=... ./sweep.sh [llm|qmatmul|quantize] > results.csv
# Variants (binaries must be built first, see README.md):
#   v0102-base   candle 0.10.2 from crates.io, default RUSTFLAGS  (what ignite pins today)
#   v0102-native candle 0.10.2, -C target-cpu=native
#   fork-base    candle fork (0.11), -C target-cpu=x86-64 (overrides candle's .cargo/config native)
#   fork-native  candle fork (0.11), -C target-cpu=native
#   fork-ctx     fork-native + Device::with_context around generation
#   anth-ctx     fork-ctx built against Anthropic's rayon-core 1.13.5000+anthropic
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-llm}
THREADS=${THREADS:-"1 2 4 8"}
REPS=${REPS:-2}
declare -A BIN=(
  [v0102-base]=v0102/target/release
  [v0102-native]=v0102/target-native/release
  [fork-base]=target-truebase/release
  [fork-native]=target-native/release
  [fork-ctx]=target-native/release
  [anth-ctx]=anth/target/release
)
VARIANTS=${VARIANTS:-"v0102-base v0102-native fork-base fork-native fork-ctx anth-ctx"}
for v in $VARIANTS; do
  ctx=0; [[ $v == *-ctx ]] && ctx=1
  for t in $THREADS; do
    for r in $(seq "$REPS"); do
      out=$(RAYON_NUM_THREADS=$t CANDLE_NUM_THREADS=$t WITH_CONTEXT=$ctx \
            "${BIN[$v]}/${MODE}_sweep" 2>/dev/null)
      while IFS= read -r line; do echo "$v,$t,$r,$line"; done <<<"$out"
    done
  done
done
