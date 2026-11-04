# Native DeepSeek required and named tool calls

Base: `bertholomus/TensorFold:deepseek-v41-zig-tp4` at
`b8b115e7ed6c86d81b7575caa6f3d4ade52b323c`.
Tested source: `253c38ca4a30fca8e0fe242d5599844e84cc187c`, Zig tree
`12ad2fa7f1d21e85fa47129c2a5d97b8dee315a0`. The receipt amendment changes no code.

DeepSeek's tool-call opener spans a text head, a DSML token and a text suffix.
This port recognizes that split opener and enables a request-local call gate
for native CUDA `dsv41`. It constrains the first tool-call opener and an offered
function name across serial, speculative and concurrent generation. Named
choices use the narrowed tool list; thinking stays unconstrained until it closes.
`auto` and `none` do not install a gate. Upstream grammar handling is preserved.

The gate checks tokens before forwarding or drafting, cuts speculative output
at the first invalid token, and observes only committed tokens. Quoted names
remain constrained until the closing quote. Leading whitespace before the
opener is permitted; trailing whitespace inside its text head is rejected.
It does not enforce argument JSON/schema conformance, later calls, or completion
when a token limit, stop condition or cancellation ends generation.

## Validation (2026-10-21 UTC)

Repository-pinned Zig 0.17.0, macOS ARM64 host checks:

| Command | Result |
| --- | --- |
| `zig test -lc zig/src/core/lanes/lanes.zig` | 132 passed |
| `ZIG=zig bash zig/tests/server/zig_test.sh` | 375 server, five JSON and 170 engine API tests passed |
| `zig build native -Dtarget=aarch64-linux-gnu -Dcpu=generic -j4 --summary all` | All seven steps passed |
| `zig build test test-golden -Dcpu=apple_m1 -j4 --summary all` | 208/212 steps succeeded; 2,291 tests passed, three skipped; overall failure at inherited macOS `SO.BUSY_POLL` in `dsv41/link.zig:23` |
| Golden corpus from that run | 309/319 equal, ten known differences, zero unexpected |
| `python3 tools/zig/lean_check.py` | 1,615 inherited findings, unchanged from base |
| `git diff --check` | Passed |

Coverage includes required/named preflight, auto/none, unsupported-backend refusal,
split and merged tokens, quoted-name delimiters, opener whitespace, thinking
boundaries, request isolation, committed correction history and output limits.
Astra reviewed the implementation and independently passed all seven focused
gate tests. Host compilation is separate from the GPU
checks below; the full host suite is not represented as a clean pass.

## Matched GPU benchmark (2026-10-21 UTC)

Four NVIDIA GB10 hosts running Ubuntu 24.04, native Zig CUDA TP4 DSpark.
EXL3 checkpoint `bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw` at
`a2f91e2794923d0beebebe0d267fb3468eaf0b65`, Engram at
`1bae0e1088993b9b82de08f59de5be5a34e42170`, patched NCCL 2.30.7.
Both profiles use Zig 0.17.0, identical CUDA kernel assets and
`-Dtarget=aarch64-linux-gnu.2.28 -Doptimize=fast -Dsm=121`.
The same external build adapter exposes the existing production lane executable;
production transport source does not differ between profiles.

Effective runtime: `--context 1048576 --parallel 16 --max-tokens 32768`,
window 2097152, round rows48, served depth solo, neighbor RDMA enabled, GID3.
Measurements use the head-loopback API as `deepseek-v4.1-flash`.
[Benchmark data](benchmark.json) retains source/binary hashes, per-load timings,
OS/kernel details, request settings and the benchmark client hash.

Order: upstream, patched, upstream, patched; fresh load and identical warmup
each time. Four streams and 256 output tokens: set-a one warmup and six measured
bursts, set-a 30-second steady admission, set-b 90-second sustained admission.
Ordinary requests do not install a call gate. Results average two loads per build.

| Workload | Upstream tok/s | Patched tok/s | Change |
| --- | ---: | ---: | ---: |
| Four-stream burst | 132.40 | 132.81 | +0.311% |
| 30-second steady | 147.97 | 147.73 | -0.162% |
| 90-second sustained | 169.84 | 169.59 | -0.150% |

No meaningful performance reduction was observed in this benchmark. All 96
measured burst replies match UTF-8 across builds; all 36 qualification replies
match token IDs and text across builds and serial/drafted/concurrent paths.
All workload stages report zero worker errors. Fourteen serial and four
concurrent tool-choice feature cases pass on the patched build.

The supplementary cooperative-tool workload compares auto/required/named on the
patched build: four streams, alternating choice order, one warmup and six timed
bursts per choice. All 84 calls return the expected function and arguments;
output token IDs match across choices. Full-response aggregate throughput:

| Choice | tok/s | Change from auto |
| --- | ---: | ---: |
| auto | 201.57 | +0.000% |
| required | 200.91 | -0.327% |
| named | 201.29 | -0.138% |

Required/named throughput is within 0.33% of auto. These samples are not a
statistical upper bound on overhead. A forced choice that changes the generated
answer can change latency. The active-call workload is separate from the upstream
benchmark; these checks do not establish argument-schema enforcement or
long-context quality.

Public prompts and client: [pinned `kit_bench.py`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/blob/9af1f4567ee49ec2a31f68975982093277e7090d/tools/kit_bench.py).
Its SHA-256 matches `workloads.client_sha256` in the data. Equivalent invocations
(the measurement wrapper additionally retained raw responses):

```sh
python3 kit_bench.py --base http://127.0.0.1:40053 --model deepseek-v4.1-flash concurrent --set a --streams 4 --tokens 256 --reps 1
python3 kit_bench.py --base http://127.0.0.1:40053 --model deepseek-v4.1-flash concurrent --set a --streams 4 --tokens 256 --reps 6
python3 kit_bench.py --base http://127.0.0.1:40053 --model deepseek-v4.1-flash sustained --set a --streams 4 --tokens 256 --seconds 30
python3 kit_bench.py --base http://127.0.0.1:40053 --model deepseek-v4.1-flash sustained --set b --streams 4 --tokens 256 --seconds 90
```

## Provenance

The gate and initial fixtures were ported from an earlier native adaptation of
[Elevatormusic/TensorFold PR #1](https://github.com/Elevatormusic/TensorFold/pull/1),
commit `003dd66273313d7d69d5675097e873e3af39ffd0`, under Apache-2.0.
See [ATTRIBUTION.md](../../../ATTRIBUTION.md). The scheduler integration was
adapted to this branch's `Engine.addStream` and existing round paths. This
change contains no transport optimization or SGLang code.
