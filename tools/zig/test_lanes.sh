#!/usr/bin/env bash
# The lane core's tests, plus record_lanes.py's vectors and traces where TF_LANES_FIXTURES names them; ZIG picks the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
lanes="$root/zig/src/core/lanes/lanes.zig"
grammar="$root/zig/src/core/grammar/grammar.zig"
"$zig" test -lc --dep grammar "-Mroot=$lanes" "-Mgrammar=$grammar"
for name in lanes_units lanes_replay; do
  "$zig" test -lc --dep lanes "-Mroot=$root/zig/tests/$name.zig" --dep grammar "-Mlanes=$lanes" "-Mgrammar=$grammar"
done
