# DESIGN-CP4: one 1,048,576-token Mistral Large 4 stream on four GB10 nodes

Dims: 72 layers, d 6144, 64 heads, kv_lora 512, nope 64 + rope 64, v 128, indexer 32 x 128,
top 2048, MoE 128 experts top-4 + shared. NoPE layers 3,7,..,71. Layers 39,43,..,71 read layer 35's rows through their
own kv_b (`kv_source_layers` names 35 for all nine readers, and layer 3..31 NoPE writers keep their own rows). 44 full-indexer layers, 28 shared.

## 1. What has to exist, and why the eager path cannot get there

`model.py attend()` builds K and V for all T rows and, with a selection, a `[S, 2048, T]` bool. `_layer` starts the
indexer key plane only once `past + S > 2048`, so chunked prefill past 2048 tokens scores against a key plane with holes.
So eager is a valid reference for one-shot prefill plus decode only, it runs out of memory past a few thousand tokens,
and the 128K DP case hits the same wall. CP4 therefore brings, besides the merge math: a paged fp8 plane, a real
indexer kernel, a real sparse MLA kernel, and a batched all-to-all. The DP path gets them too (section 8).

## 2. Striping

Fixed at admission: block `B` rows per rank per round (power of two, divides 262,144; default 1024, `TF_ML4_CP_BLOCK`),
world `W = 4`, round width `WB`.

    rank(p)  = (p // B) % W
    row(p)   = (p // (W*B)) * B + p % B              local row on rank(p), increasing in p
    vis(k,p) = ((p+1) // (W*B)) * B + clamp((p+1) % (W*B) - k*B, 0, B)   rows of rank k visible to a query at p

Local rows are in position order, so what a query sees on a rank is always a prefix of length `vis`: no mask planes,
no position plane (`p` comes back from a row with one formula). Position blocks keep work level at every depth: in
each round every rank owns `B` new rows and holds the same prefix, which is why this beats four contiguous quarters
(there rank 3 would own all the late, expensive queries and rank 0 would idle). N <= 1,048,576 gives at most 262,144
rows per rank. A short last round fills rank 0 first; ranks without rows still join the collectives.

All planes of all layers use the same map, so a local row number means the same position in every layer. That single
fact is what makes readers and shared-indexer layers free (section 5).

## 3. One round

A round is a set of rows at consecutive positions, each row owned by `rank(p)`. Prefill rounds carry up to `WB` rows;
a decode step is a round with one row. There is one code path: the row's owner does everything that is per-token
(norms, projections, KV store, o_proj, router, residual), every rank does what is per-key.

Per layer, on the owner: `xn`; writers compute `kv_a -> latent/k_pe`, encode the fp8 row and store it at `row(p)`
(the query attends through the plane, so it sees exactly what later queries will see); full layers compute indexer
keys and store them fp16; `q`, and on full layers the indexer `q` and `weights_proj`.

Cross-rank, full layer (shared layers skip 2-4):

1. All-gather raw `q` (fp16, 16 KiB/token), and on full layers indexer `q` (8 KiB/token) and weights (128 B/token).
   What goes on the wire is the projection's own fp16 output, not post-ssmax values: q is fp16 out of the EXL3 kernel, so
   this wire is lossless, and the receiver applies ssmax and rope in fp32 from `p`.
2. Every rank scores all round queries against its local key prefix and takes its local top-K as unique int64 keys
   `(ordered(score) << 32) | (0xFFFFFFFF - p)`: larger is better, ties go to the lower position, exactly the eager rule
   (`select_topk`: stable sort, ties to the lower index). Padding is `INT64_MIN`.
3. All-to-all of candidate keys to each query's owner (K x 8 B per query per peer). The owner takes the K-th largest of
   the W*K: threshold `T`. If fewer than K rows are visible, `T` is the pad and all visible rows are selected, which is
   the eager short-context rule and needs no special case.
4. All-gather `T` (8 B/query). Each rank keeps its candidates with `key >= T`. Keys are unique, so the union over ranks
   is exactly the global top-K with no further exchange. The result per rank is `(sel_rows[Q, K] int32, cnt[Q])`,
   ascending row order, kept for the following shared layers.
5. Each rank runs sparse MLA over its selected rows, absorbed form: `q_lat = q_nope @ W_UK` per head (a bmm on a bf16
   view of `kv_b`, no fp32 folds), scores `q_lat . latent + q_pe . k_pe`, online softmax, `o_lat = sum p * latent`, then
   `W_UV` at the holder (it commutes with the merge, and it shrinks the wire from 512 to 128 per head). Output per
   query: normalized partial `o[64,128]` plus `lse[64]`.
6. All-to-all of `(o fp16, lse fp32)` to the owner. The owner merges in rank order, in fp32:
   `m = max_k lse_k; a_k = exp(lse_k - m); o = sum_k a_k o_k / sum_k a_k`. A rank with no selected rows has `lse = -inf`, `o = 0`.

Then on the owner: `o_proj`, residual, `ffn_norm`, router (fp32). MoE: dedupe picks by destination rank, one grouped
all-to-all out (fp16 rows), `routed()` over the received rows and the own rows in one call, one grouped all-to-all back
(fp16 partials), sum in destination-rank order. Shared expert on the owner. Counts ride one small all-gather.

Collectives per layer: full 7 (q, cand, T, o, MoE counts, MoE out, MoE back), shared 5. Each is one NCCL group.
`NCCL` gains `alltoall(sends, recvs)` built from grouped send/recv; `EmuComm` gets the same call for CPU tests.

The last prompt row's owner applies the final norm and the head and samples, the token rides the next op. Decode row
`p` is owned by `rank(p)`, so ownership moves one rank every `B` tokens and the head stays replicated. There is no
replicated row and so no way for ranks to drift apart: each number exists on one rank and is shipped from there.

## 4. Exactness of the merge

Softmax over a union of disjoint sets, written as weighted partials, is an identity: the only difference from single
rank attention is rounding. Top-K over a union equals top-K of the union of per-set top-Ks, and the unique keys make
the cut exact. So in fp32 CP4 must reproduce single-rank attention up to reduction order. Gates, fixed before any code
(fp32 CPU, `W = 4`, `EmuComm`, seeds 0-2, fp16 row store off so that only the merge is under test):

| gate | input | pass |
|---|---|---|
| M1 attention merge | random (acc, lse) splits, real dims | cosine >= 0.9999999, max-abs <= 1e-5 x output rms |
| M2 selection | tiny model, `B` in {2,4,8}, N > 8 x top-K | selected sets equal for every (full layer, query); a swap is allowed only between keys whose fp32 scores differ by <= 1e-6 relative |
| M3 layer stream | tiny model, full forward + 16 decode steps | every layer's residual: cosine >= 0.999999, max-abs <= 1e-4 x rms |
| M4 logits | same | cosine >= 0.999999, max-abs <= 1e-3, greedy tokens identical |
| M5 real dims | real-size attention layers (rope writer, NoPE writer, reader, shared), N = 8192, K = 2048 | selection as M2; output cosine >= 0.99999, max-abs <= 1e-3 x rms |
| M6 kernels vs torch | CUDA node, each Triton kernel against its torch reference | selection identical except fp16-level score ties; cosine >= 0.9995 for fp8 attention rows |

Two gates moved after they met real numbers, both tightened in meaning and both on the record here:

- M1's reference is now fp64. Against fp32 single-rank attention the merge missed 1e-5 (1.77e-5), but fp32 single-rank
  itself misses fp64 truth by up to 1.66e-5, and the merge stays at 6.85e-6. Same number, honest reference.
- M5 first ran as written and failed: 7 of 16,384 selections differed and streams reached max-abs/rms 1.64e-2. One
  flipped key at the K-th place moves a row by about 3 % of its attention output, and fp32 sums in another order do flip
  keys whose scores agree to the last bit or two. So M5 now holds three things apart: (a) the merged selection equals
  the top-K of the same scores, exactly; (b) selection against the reference differs only at near ties, gap <= 1e-5 of
  the K-th score (M2's 1e-6 is below fp32's rounding of a 4,096-term score at real dims: worst seen 2.3e-6); (c) with
  the reference's selection forced, streams meet the table's numbers.

Results: M1 cosine 1.000000000, max-abs/rms 6.85e-06. M2-M4 on three seeds and block sizes: no selection differs, layer
cosine 1.00000000, max-abs/rms <= 2.2e-6, logits max-abs <= 3.9e-6, greedy tokens identical. Short streams and a decode
that crosses the budget pass against eager one-shot (eager's own key plane starts only when the budget binds, so eager
is not a reference for a short prefill that decodes past K). M5: (a) 0 of 16,384 differ, (b) 5 flips, worst 2.34e-06,
(c) cosine 1.000000000, max-abs/rms 5.13e-06. M6 on GB10 at real dims: scores cosine 1.000000000, max-abs/rms <= 1.9e-6;
sparse MLA cosine 1.0000000, lse within 1.2e-6.

## 5. Readers, shared indexers, NoPE

Readers (39,43,..,71): no store; they read layer 35's plane at the same local rows with their own `W_UK`/`W_UV` from their
own `kv_b`. Under striping nothing else changes, because layer 35's plane is striped identically.
Shared-indexer layers: reuse `(sel_rows, cnt)` of the nearest preceding full layer from the same round. The lists are local
row numbers, and those mean the same position in every plane.
NoPE layers: `nope_w = 128`, `W_UK` 128 wide, row 528 B, no `k_pe`; the kernel takes `rope_w = 0`, the indexer does not
rotate on these layers (model.py header). A full-indexer NoPE layer stores keys as usual.
ssmax, LeRoPE and YaRN are functions of `p` alone, so they work from the row's true position on any rank.

## 6. fp8-latent row

One row per writer layer: 512 e4m3 + 16 B scales + 64 fp16 `k_pe` = 656 B (NoPE 528 B); keys stay fp16 (selection
is already fp16-borderline). 54 x 656 + 9 x 528 + 44 x 256 = 51,440 B/token; 262,144 positions = 13.485 GB = 12.56 GiB.
Rope and keys are not quantized. Scale layout: 4 x fp32 per 128 dims. The other two 16-byte candidates (8 x fp16 per 64,
16 x e8m0 per 32) gave the same accuracy within noise on the pack's own tensors (section 11), so the simplest kernel wins.
Readers-parity results are in section 11.

## 7. Prefill schedule and cost

`WB` tokens per round, default 4096: a 1M prompt is 256 rounds, each round all layers. Rounds run back to back; prefill
is chunked by the round, nothing else. Per rank (roofline-style estimates made before the runs; section 12 has what came out):

| part | work per rank | estimate |
|---|---|---|
| indexer scores | 5.0e16 FLOP (44 layers, causal) | 1000-1700 s |
| local top-K | 6e12 scores, about 5 passes | 400-600 s |
| sparse MLA | 25 TB gathered, 5.4e15 FLOP | 300-600 s |
| MoE + linear | 2.6e16 FLOP through EXL3 decode | 1100-2000 s |
| fabric | 3.7 TB received per rank | 200-400 s |
| total | | 1-2 hours |

Heat is a real limit at this length: one node reset at 85-89 C, another hung above 82 C. A round boundary is a natural place to wait
if a node runs hot (`TF_ML4_CP_MAX_TEMP`, default off until measured).

## 8. Coexisting with DP4

One fp8 pool per rank replaces the naive cache on CUDA, extent based. A DP stream is one extent on its owner rank. A CP
stream is an extent of `ceil(total / WB) * B` rows on all four ranks. Admission: `total <= dp_limit` goes DP, as today;
`total > dp_limit` goes CP, one CP stream at a time, refused with the free runs in the message when the extent does not fit.
Ledgers count rows, so the least-loaded placement is unchanged. A round is CP or DP, never both; decode rows of DP streams run between CP rounds.
The DP forward is the same code with the group of one: local rows, no steps 1-4 and 6, MoE all-to-all as ever. This is
what makes 128K DP possible at all, so "still serves 128K" is a real check here.

Ops: `admit` carries the extents and no tokens; `["cp", seq, sid, start, n, tokens, last]` is a round, with that round's
tokens, so a 1M prompt crosses the link once, 4096 ids at a time. Decode is the same op with `n = 1` and the fed token, and
`last` names the row whose owner answers. The digest check stays as it is.

## 9. Fabric bytes

Received bytes summed over all ranks, per token (W = 4):

| item | full layer | shared layer |
|---|---|---|
| q | 49,152 | 49,152 |
| indexer q + weights | 24,960 | 0 |
| candidates + T | 49,176 | 0 |
| o + lse | 49,920 | 49,920 |
| MoE, 2.05 remote ranks a token, out and back | 50,400 | 50,400 |
| total | 223,608 | 149,472 |

Whole forward 14.0 MB per token, so 3.5 MB per rank: 3.7 TB per rank for 1M tokens, about 185 s at 20 GB/s. Measured on one node
during the 1M prefill, from the two ports' RDMA counters over 20,480 tokens: 3.66 MB sent and 3.66 MB received per token, 1.4
GB/s each way, far below what the links carry, so the fabric is not what limits a round. Decode: the same bytes for one token,
and latency bound: 7 x 44 + 5 x 28 = 448 collectives. The lane must run NCCL over RoCE: the
Phase B containers mount no `/dev/infiniband`, so they run NCCL over TCP. CP launches add the RDMA devices and the
`NCCL_IB_*`/`NCCL_SOCKET_IFNAME` settings of the DS lanes: `--ipc host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add
IPC_LOCK --device /dev/infiniband`, `NCCL_SOCKET_IFNAME=<the fabric interface>`, `NCCL_IB_HCA=<the RDMA devices>`,
`NCCL_IB_GID_INDEX=5`, `TF_RDMA_DEVICES` the same pair, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`.

## 10. Memory (measured, per rank, GiB)

The run `mem5`, on the four nodes, `--rows 262144`, one round, then one full-depth round at position 1,040,384 over a pool filled by
copying the first round's rows, then 8 decode steps. Ranges are the four ranks. "Host" is RssAnon: on GB10 host pages and GPU
pages come from the same memory.

| step | MemAvailable | torch reserved | host |
|---|---|---|---|
| idle node | 118.1 | | |
| NCCL up, 4 channels | 116.9 | 0 | 0.71 |
| weights loaded | 20.8 | 91.20 | 2.19 |
| `prepare`: kv_b fp16, embedding mapped, wq_b fp8, malloc_trim | 24.5-24.8 | 88.25 | 1.13-1.44 |
| fp8 pool, 262,144 rows, 12.56 GiB | 11.6-12.0 | 100.81 | 1.13-1.44 |
| first round | 8.1-8.4 | 102.31-102.62 | 2.70-3.02 |
| full-depth round and decode | **7.95-8.22** | 102.48-102.62 | 2.72-3.03 |

What each step gave, against the numbers of the first run on this tree (5.0-5.3 GiB left, 3.3 GiB short) and against Phase B:

| change | GiB | cost |
|---|---|---|
| no fp32 `kv_b` folds: views of fp16 `kv_b`, absorbed matmuls read it per layer | +1.8 | none |
| embedding read through the page cache (`torch.from_file`, checked on 4 rows at load) | +2.3 | a round's rows come from the CPU side |
| `NCCL_MAX_NCHANNELS=4` | +1.15 | none seen: round 9.7 s, decode 4.9 tok/s, both unchanged |
| `expandable_segments:True` | +0.8 | none seen |
| `wq_b` read from its fp8 bytes, each layer checked bit for bit against the fp16 at load | +1.03 | decode layer 0.15 ms faster, prompt layer 1.0 ms slower (0.5 % of a round) |
| score tiles and the dense layer's 288 MB decode buffer left to the allocator, so the experts' transients reuse them | +0.45 | none |
| `malloc_trim` after load and after the first round | +0.4 | none |

Where the rest goes, from a round's start to its peak: torch +1.8 (expert scratch 0.42, EXL3 workspace 0.14, transients), host +1.6
(first-use loads of cuBLAS, torch and Triton kernels: `ops.mat` +0.38, `torch.topk` +0.29, the MoE glue +0.42; `malloc_trim` does
not bring it back, it is live). `CP4_TRACE=1` in tools/ml4/cp4_run.py prints who grows host memory and torch's peak.

The floor holds at the 8 GiB line with no margin to speak of: 7.95 on the worst rank in one run, 8.1-8.2 on the others. The
lane's env is `TF_ML4_KV_TOKENS=262144 TF_ML4_MEM_FLOOR_GIB=8 TF_ML4_HEADROOM_GIB=3 TF_ML4_POOL_SAFETY_GIB=0`, which the ceiling
turns into 104.9 GiB (resident 88.3 + growth 16.6). A lane that wants less than 1M a stream gets the margin back at 51,440 B a
position: 8 GiB more is 167,000 positions a rank.

## 11. fp8 parity results

Pack tensors (kv_a, kv_a_norm, kv_b of L02, L03, L35, L39, L71), N = 8192 rows from hidden-state-shaped draws, K = 2048,
queries at chosen sharpness (q_proj is EXL3, so the host cannot draw them from the real projection). fp8 rows against fp16
rows, per query and head; "eff rows" is the softmax's effective support:

| layer, sharpness (eff rows) | layout | output cosine | worst head cosine | rel error |
|---|---|---|---|---|
| L35 NoPE source, 2.0 (1388) | f32x128 | 0.999978 | 0.999920 | 0.00665 |
| | f16x64 | 0.999979 | 0.999899 | 0.00648 |
| | e8m0x32 | 0.999977 | 0.999898 | 0.00690 |
| L35, 8.0 (466) | f32x128 | 0.999906 | 0.996449 | 0.01374 |
| | f16x64 | 0.999902 | 0.993445 | 0.01400 |
| | e8m0x32 | 0.999884 | 0.992361 | 0.01528 |
| L39 reader, 0.5 (1995) | f32x128 | 0.999982 | 0.999951 | 0.00603 |
| L39, 2.0 | f32x128 | 0.999979 | 0.999824 | 0.00655 |
| L39, 8.0 (476) | f32x128 | 0.999891 | 0.994015 | 0.01480 |
| | f16x64 | 0.999905 | 0.996456 | 0.01376 |
| | e8m0x32 | 0.999890 | 0.994806 | 0.01483 |
| L71 reader, 8.0 (482) | f32x128 | 0.999894 | 0.992016 | 0.01455 |
| | f16x64 | 0.999898 | 0.988518 | 0.01430 |
| | e8m0x32 | 0.999892 | 0.995185 | 0.01473 |

Row error is about 2.5 % in all three layouts; the reader layers behave like their source. Full table: `run/fp8-real.log`.

Tiny model, readers live, the same stream on fp16 rows and fp8 rows, 3 seeds, 75 steps (test_gate2_fp8_against_fp16): worst
KL 2.0e-4 nats, greedy token the same on 75 of 75. Logit cosine fell to 0.972, which says little: the random model's logits
sit near zero, so the gate is KL and greedy.

On the real pack end to end, fp8 rows only: the 60K needle finds all three codes (section 12).

## 12. Measured

All on the four nodes, real pack, fp8 rows, through `tensorfold serve` unless stated. The estimate in section 7 was 1-2 hours; the run took 61 minutes.

| run | tokens | prefill | decode | answer |
|---|---|---|---|---|
| short chat, DP | 28 | ttft 4.59 s | 5.96 tok/s | "The capital of France is Paris." |
| 8K needle, striped | 8,000 | ttft 23.5 s | 5.21 tok/s | 3 of 3 codes |
| 60K needle, striped (cp4_run) | 60,000 | 148 s, 405 tok/s | 5.22 tok/s | 3 of 3 codes |
| 120K needle, DP on one rank | 120,163 | 584.0 s, 206 tok/s | 4.87 tok/s | 3 of 3 codes |
| 1M needle, striped | 1,044,249 | 3,659.8 s, 285 tok/s | 4.74 tok/s | 3 of 3 codes |

Rounds of the 1M prompt (4,096 tokens each): 13.0 s (the first: kernels compile), 10.7 s at 131K, 11.7 s at 262K, 13.9 s at 524K, 16.0 s at 786K, 17.3 s at 918K, 19.1 s at 1,044K: about 8.4 s of fixed work plus 10 s a million positions of depth. One node reached the 80 °C limit in rounds 59-121 and paused 10 times for 10 s each. Power 50-56 W mean, 64 W max.

## 13. Deep prompts: what changed for the serving goal

Sections 1-12 describe the first build, where a 1M prompt took 61 minutes. Serving needs about 15 minutes. This section keeps the design decisions of that work.

**Three kinds of prompt.** Up to `TF_ML4_CP_COPY` (163,840) rows every rank keeps a position-ordered copy of the layer's rows and every query runs on one rank in one piece: exact e4m3 scores, the same bits as the brief's rounds. Past that a prompt runs on **slots**: every rank holds ONE layer of all rows, the next layer's old rows are all-gathered on a second NCCL communicator (`comm.fork`, stream `Slots.stream`) behind this layer's experts, and `vis = own + 1` stripes. A second deep prompt while slots are busy falls back to striped rounds.

**Two rounds in flight.** Per layer: a.front, b.front, slots.done(), slots.fetch(L+1), a.back, b.back, pace. `TF_ML4_CP_PAIR=8192` under `TF_ML4_CP_ROUND=16384`. A flight is the 16,384 rows of one such pair.

**FP4 index scores, deep prompts only.** `mma...kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` on sm_121a, rows of 80 bytes, least-error block scale. 236 TF against 141. Selections move, the loss does not: +0.0005 +- 0.0017 over 295K tokens of mixed text.

**Shared front selections, deep prompts only.** The model itself reuses selections from layer 17 on (`ssFF`). `TF_ML4_CP_REUSE` extends that to the front layers on slots rounds. Late layers carry the retrieval and must keep their own; layers 1, 3, 4 and 16 keep theirs too. Reusing layers still write their keys, because decode scores with every layer's own selection. Their keys are not gathered during the prompt.

**Heat is the limit, not arithmetic.** A node's sustained temperature follows its mean power, so a prompt's time is at least its energy over the cooling a node has: about 58-65 W GPU in steady state. Thermal memory is about 2 minutes. So only changes that do less work count at 1M; a kernel that does the same work faster mostly moves time into pauses. FP4 scores and shared selections do less work, the wider GEMM tile only partly.

**Threads.** The cores are 10 little (cpu0-4, 10-14, 2.8 GHz) and 10 big (cpu5-9, 15-19, 3.9 GHz). The threads that only wait (two NCCL proxies, pace, the RDMA nap) go to the little cores; moving them off the big cores was worth 90 s on the 1M prompt because the big cores share a power budget with the GPU.
