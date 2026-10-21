# Decode-sized replay position alignment

`replayCut` formerly kept `pos0 + first * sizeof(i64)`. With an odd cut and
1–15 surviving rows, that view is 8 modulo 16. Decode-sized split sparse
attention uses POS as its default RBASE/CBASE too; the served aligned AOT
variants cannot accept those pointers. The fix uploads the exact surviving
host positions into the existing aligned `pos0`, in the existing stream order.
Even cuts and chunks larger than `tri.decode_rows` keep the view. No kernel
math, replay geometry, capture metadata or alignment checks change.

Since v0.3.0 this branch carries the served line's `replayCut`, which uploads
the surviving positions into the aligned `pos0` for every cut, not only the
small odd ones: the same positions, always in the chunk's own buffer. The CPU
test below follows it (one upload and `pos == pos0` for every cut).

## CPU regression (no model or CUDA required)

Use Zig 0.17.0, as pinned in `.zig-version`:

```sh
python3 tools/dsv41-zig/test_replay_positions.py
```

The test extracts and compiles the actual `replayCut` function, stubbing only
GPU collaborators. It checks exact positions (including noncontiguous sentinel
values), unchanged host data, alignment, upload count, unchanged even/larger
views, and no-cut/encoder-only chunks. It does not test CUDA execution or stream
ordering. To reproduce the red result without changing the checkout:

```sh
git show b65c8b92d2403dfdefccf15a3b23ba609a4553e2:zig/src/families/dsv41/prompt.zig > baseline-prompt.zig
python3 tools/dsv41-zig/test_replay_positions.py --source baseline-prompt.zig
```

Baseline: two tests fail (`expected 0, found 8`); candidate: all three pass.
Remove or keep the generated baseline file outside your publication tree.

## Opt-in server reproduction

`repro_replay_positions.py` uses only synthetic comment padding and a synthetic
`check_code` tool. The URL is a native server base URL without `/v1`. Sizing
calls `/tokenize`; `--run` additionally performs inference and a tool continuation.
It never starts, stops, rebuilds, or reconfigures a server. Use an explicitly
approved test server and preserve each request file between baseline/candidate:

```sh
python3 tools/dsv41-zig/repro_replay_positions.py --url http://127.0.0.1:8080 \
  --tokens 39039 --request request-39039.json
python3 tools/dsv41-zig/repro_replay_positions.py --url http://127.0.0.1:8080 \
  --tokens 39039 --request request-39039.json --run --output baseline-39039
# After your separately approved candidate rebuild, send those same saved bytes:
python3 tools/dsv41-zig/repro_replay_positions.py --url http://127.0.0.1:8080 \
  --tokens 39039 --request request-39039.json --run --output candidate-39039
```

Use a new output directory per run. Baseline SSE errors are retained in
`first.sse` before the script exits nonzero. Candidate success requires actual
`response.completed`, exact tool arguments, actual input-token usage matching
the target, and `ALIGNMENT_OK` from a completed tool continuation. HTTP 200 alone
is not success. The restricted chat translation used for sizing is checked
against Responses usage; different templates may require adapting it.

For the cached-prefix cases, first run the same fixture at 38783 and 122747
respectively, then 39039 and 123003. Inspect returned cached-token usage: cache
state and scheduling matter, and token residue alone is not a universal trigger.
Check adjacent lengths 39038/39040 and 123002/123004 separately.

## Existing four-Spark qualification

These are earlier real GPU results on the recipe's engine base
`59402086381a42c74d53053443224eec2790b4c5` plus its four patches, before/after
this narrow repair, **not** a GPU qualification of the newer PR base.
The recipe is `tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`,
`runs/2026-10-17-1017-tensorfold` (DeepSeek-V4.1-Flash 1017, TP4, four GB10 Sparks).

| Rendered tokens | Baseline | Same first request after repair | Cached tokens after repair |
|---:|---|---|---:|
| 10353 | MissingTritonVariant, zero generated tokens | completed + tool continuation | 0 |
| 10363 | MissingTritonVariant, zero generated tokens | completed + tool continuation | 0 |
| 39039 | MissingTritonVariant, zero generated tokens | completed + tool continuation | 36864 |
| 123003 | MissingTritonVariant, zero generated tokens | completed + tool continuation | 120832 |

All four first requests were byte-identical before/after. The neighbors
39038/39040 and 123002/123004 and prefix warmers also completed with the exact
tool call and continuation. With chunk size 2048 and window 128, the failing
lengths yield cuts at 2033/2043/2047/2043, leaving 15/5/1/5 rows respectively.
The failing sparse-attention launch had ring_size=144 and n_idx=512;
POS/RBASE/CBASE were all 8 modulo 16. Existing H16/H32 captures require alignment.

Only four-Spark TP4 was GPU-tested. No TP2, other accelerator, benchmark, soak,
private workload replay, broad long-job stability, or token-for-token whole-model
quality claim is made. This PR does not change any running deployment.
