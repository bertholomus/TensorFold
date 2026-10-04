# REPORT — DeepSeek-V4.1-Flash on our own TensorFold TP2 engine (two NVIDIA GB10 nodes)

Date: 2026-10-03, release build 2026-10-04. Branch `deepseek-v41-tp2` of our TensorFold fork (on upstream TensorFold v0.6.3),
family `src/tensorfold/families/deepseek_v41`, tools `tools/dsv41/`. Raw results:
`<RESULTS_DIR>/` (kit: `kit-20261003T081806Z`; ours: `ours-final-20261003T123858Z`,
`ours-20261003T102235Z`; reference: `ref/`). Design: `DESIGN.md`. Attribution: `ATTRIBUTION.md` (both in this
folder).

Written independently: clean-room with respect to the kit (below) and the public jayleaton recipe, whose code was
never opened; the kit was only run and measured. The model math was re-implemented from DeepSeek's own MIT inference
code (`inference/` on their HF repo) and tech report, which we read; no code was copied. Also used: upstream
TensorFold 0.6.3, our TensorFold fork for GLM-5.3 (github.com/bertholomus/TensorFold, branch `glm-dsa-tp4`) (RDMA gather, prompt expert kernels, the concurrent-decoding design) and
the checkpoint's file format.

"The kit" below is the MiaAI-Lab vLLM kit, commit 6f7d1590ad49 (AGPL-3.0), run as a black box on the same two nodes
and measured with the same client as our engine. It ran with its own start script and the configuration deployed on
these nodes: max model length 262,144 and a 1024-token batch budget, where its configuration file names 600k and
1536 as defaults. Its prefill may be higher at its default batch budget; we measured only the deployed configuration
(DSpark k=3, at most 2 sequences).

## 1. Release build

The served configuration (the launch line in section 7): `--vision --parallel 4 --mtp-drafts 5`, a 262,144-token
window shared by up to four streams, bounded replay prefill, kept prompts, the per-rank weight cache. Measured on the
release build with the same public client (`kit_bench.py`), `needle.py`, `vision_http.py`, `parallel_check.py` and
the gates; raw files in `release-<commit>/` of the results folder. Greedy decode, DSpark depth from the confidence
head (up to the drafter's block of five).

| | Release build (TP2) |
|---|---|
| Start to ready, restart with the weight cache | **65 s** (the start that writes the cache: 330 s) |
| Greedy decode, 512 tok: code / prose / structured | **60.5 / 38.0 / 74.5 tok/s** |
| Greedy decode, 384 tok (prompt set b): code / prose / structured | **69.9 / 42.3 / 96.5 tok/s** |
| Greedy output identical run to run | **yes** (all 6 prompts) |
| 2 streams aggregate, 256 tok / 384 tok (set b) | 54.4, 53.7 / 55.7, 58.0 tok/s |
| 4 streams aggregate, 256 tok / 384 tok (set b) | **73.2, 77.5 / 79.7, 80.5 tok/s** |
| Cold prefill 8K / 32K / 64K / 128K, bounded replay | 1349-1365 / 1247 / 1342 / 1262 tok/s |
| Decode after a 128K prompt | 61.8 tok/s (the 128K prompt took 103.5 s) |
| Needle recall, 8K / 32K / 128K / 250K x 3 depths | **12/12** |
| Needle at a 1M-token context (separate one-stream server start) | not re-run on the release build; an earlier build found it at 1,039,833 tokens (one trial) |
| Image input checks: colour, text reading, two images, image + tool call, image in a tool result, image in an assistant turn refused | **6/6** |
| Concurrent replies equal to their solo replies (greedy and seeded T=0.6, burst and staggered) | **12/12** burst, **12/12** staggered |
| Solo replies equal to the previous served build | 12/12 |
| Drafted == serial | **8/8** at T=0 (the gates) |

Kept prompts (measured on the commit that added them, same configuration): an agent session with a 17.9K-token system
prompt and 24 tool schemas over 8 tool-call turns went from 12.7-19.9 s to 1.2-2.4 s to the first token, with replies
equal to fresh prefill 10/10; resumed == fresh token ids 13/13 with replay prefill and 13/13 with exact prefill.

The kit was not re-run for the release; its numbers in section 2 stand as measured (black box, commit cited there).

## 2. First build vs the kit (same client, `tools/dsv41/kit_bench.py`, same nodes)

| | Ours (TP2, DSpark k=3) | Kit (vLLM TP2, DSpark k=3) |
|---|---|---|
| Start to ready | **134 s** | 265 s |
| Greedy decode, 512 tok: code / prose / structured | **53.0 / 37.5 / 60.5 tok/s** | 38.9 / 26.5 / 48.9 |
| Greedy output identical run to run | **yes** | no (the 3 replies were not all identical for any of the 3 prompts) |
| 2 streams aggregate | 35.8 / 39.6 | 37.7 / 37.0 |
| 4 streams aggregate | 45.4 / 45.5 | 39.4 / 41.5 (admits 2) |
| Cold prefill 8K / 32K / 128K, bounded replay, 262K context, one run (8K: 1246 then 1354) | 1246-1354 / **1396 / 1344 tok/s** | 1154 / 1165 / 1099 |
| Same, 5 earlier runs, context setting not recorded (8K: first request) | 705-1141 / 1183-1255 / 1164-1239 | — |
| Cold prefill 8K / 32K, exact (every layer, every token) | 789-835 / 824 tok/s | — |
| Decode after a 128K prompt | 32.6 tok/s | not measured |
| Context | **1M** (packed FP4 KV) | 262K configured |
| Needle recall (replay) | 8K/32K/128K x 3 depths 9/9 (before the packed FP4 caches); 256K, 512K, **1,039,833** found (one trial each) | not measured |

Prefill at a 1M-token context setting (instead of 262K) is ~10% slower (memory pressure: ~5-8 GiB free a rank). A 1M
prompt prefills in 1154 s (1.9 s a 2048-token chunk at the start, 2.5 s near 1M), on a separate server start.

The speed rows were measured 12:39-13:27 UTC on uncommitted development states, before the development commit of
13:29 UTC; the working tree may have changed between rows. They were not re-run on the release build. Ours had a warm
kernel cache at start; page-cache state was not recorded for either side.

## 3. Exactness and the agreement bar

- **Drafted == serial, bit for bit**: 8/8 prompts at T=0 (every run since exact DSpark drafting landed, eager and
  CUDA graphs) and 8/8 at T=0.6 (keyed sampling). Decode-path kernels are designed to be row-independent and are
  tested for it (`kernel_test.py`, `rowinv_test.py`); every drafted == serial check we ran passed.
- **The 0.99 teacher-forced top-1 target is below the model's own noise floor.** Our deterministic fp32 reference flips
  2% of its top-1 choices when only the number of sequences sharing a matmul changes (p90 logprob gap 0.10, max 1.12):
  FP4/FP8 KV rounding and MoE routing ties amplify last-bit differences. The kit agrees with itself (its decode vs its
  prefill) 0.980.
- Teacher-forced top-1 on 24 prompts / 1823 positions: reference vs kit 0.949 (0.996 where the kit's margin >= 2
  nats); engine vs reference 0.956; engine vs kit 0.948. Exact-fp32 engine mode 0.957; bf16 KV on both sides 0.970.
  The engine figures are teacher-forced in exact mode on the first kernel build (not replay mode, not the release
  build); the engine's agreement by margin was not kept.
- Bar we hold: drafted == serial exactly; agreement on confident positions (shown for reference vs kit; no kept file
  backs an engine figure by margin); kit agreement reported.
- Bounded replay vs exact prefill (tech report text, 2K/8K/32K, 2 offsets): first token equal 6/6, top-1 logprob gap
  <= 0.26; greedy continuations diverged in 3 of 6 cases (after 6, 11 and 25 tokens) and matched all 64 measured
  tokens in the other 3.

## 4. Design (ours; DESIGN.md has the detail)

- TP2 split: routed + shared experts by intermediate halves (1152 a rank); attention heads 32 a rank (wq_b columns,
  wo_a groups, wo_b rows); wq_a, wkv, compressor, indexer and KV caches replicated; Engram hash columns 12 a rank; head
  by vocabulary. ~99 GiB a rank with DSpark.
- Reductions: all-gather of fp32 partials + rank-order sum (ranks bit-equal, no broadcast). Decode gathers over our
  RDMA-write gather (~14 us), prompt chunks over NCCL/RoCE (Simple protocol, 4 channels).
- KV in DeepSeek's trained formats, packed: compressed main KV E2M1 + E4M3/16 (288 B), indexer K E2M1 + E8M0/32
  (68 B), SWA FP8 in 144-slot rings; rollback after a partial accept by length alone.
- Decode/verify windows as CUDA graphs per (rows, context bucket); DSpark drafter + Markov loop as one graph (5.2 ms).
- Prompt chunks of 2048 rows; prompt expert kernels from our earlier GLM-5.3 fork; optional CED bounded replay
  (decoder over the last 128 prompt tokens, as DeepSeek deploys it), `TF_DS_REPLAY=1`.
- Engram: hashes on the host, rows by parallel preads (C++) from the original FP8 shards on local NVMe.
- Serial decode ~30 tok/s is weight-bandwidth-bound: ~21 ms of ~31 ms a token streams EXL3 weights.

## 5. What failed / what it cost

- NGC image forces TF32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`): reference batch-dependent until disabled.
- First DSpark verify broke drafted == serial: torch reductions over [rows, 4, 5120] (Engram gate) depend on row count.
- CUDA graphs alone gained ~1 tok/s (GPU-bound). Python thread-pool preads for Engram cost 5 ms a token (C++ fixed).
- The first packed-FP4 attention kernel was 6x slower; rewritten (split-K, bit-built E2M1): 22 us a decode row.
- Adaptive verify length from the confidence head measured worse (40.1 vs 43.6 tok/s): off by default.
- An exact (non-replay) 128K prefill at a 262K context with DSpark loaded ran the nodes out of memory: the kernel's OOM
  killer ended our rank-1 process and a system service on the rank-0 node (restarted within a second). Smaller indexer
  row blocks and an allocator release after each prompt went in afterwards; exact 128K not re-run.

## 6. Open

- Done since the first build: concurrent decoding (up to four streams, each reply its solo reply), image input,
  kept prompts (prompt-end reuse), the per-rank weight cache.
- Verify windows read more experts (a 4-row window costs ~1.60x a one-row step); one-stream code and counting are bound
  by the round's cost and the drafter's block of five.
- Exact prefill is slower than the kit (789-835 vs 1154 at 8K); replay is faster. The engine's code default is exact
  prefill; the deployed configuration (the launch line below) uses replay, DeepSeek's deployment mode.
- Prefill compute/communication overlap (gathers are ~14% of a chunk); the start-up warm-ups (~29 s of the 65 s).
- Memory headroom: ~99 GiB weights a rank leaves ~5-8 GiB; 1M context fits but costs ~10% prefill speed.

## 7. Reproduce

```
# on the worker node then the head node (rank 1 first), each in a throwaway nvcr.io/nvidia/pytorch:26.07-py3 container
# with this branch installed. <MODEL_DIR>: the EXL3 checkpoint; <HEAD_IP>: rank 0's address on the inter-node link.
# Engram tables: TF_DS_ENGRAM=<ENGRAM_DIR>, or a folder next to <MODEL_DIR> whose name contains "Engram" (found
# automatically). NCCL_SOCKET_IFNAME / NCCL_IB_HCA / NCCL_IB_GID_INDEX select the RoCE link (the RDMA gather reads
# NCCL_IB_HCA and NCCL_IB_GID_INDEX too, GID index 5 by default). Optional: TF_API_KEY_FILE=<API_KEY_FILE> makes every
# route but /health require that key (pass the same file to kit_bench.py with --key-file).
# the served configuration: TF_DS_RANK_CACHE=<CACHE_DIR> keeps each rank's weights in one file (~106 GB a rank,
# written on the first start)
TF_DS_REPLAY=1 TF_DS_PREFILL_CHUNK=2048 TF_DS_RANK_CACHE=<CACHE_DIR> tensorfold serve <MODEL_DIR> \
  --tp 2 --rank R --master <HEAD_IP> --host 127.0.0.1 --port 18891 --context 262144 --vision --parallel 4 \
  --mtp-drafts 5 --temperature 0
python3 tools/dsv41/kit_bench.py --base http://127.0.0.1:18891 --model <name> decode|concurrent|prefill|depth

# engine-harness checks (no server; stop it first): the same command on each rank, R = 0 on the head, 1 on the worker
python3 tools/dsv41/spec_check.py --rank R --master <HEAD_IP> --model <MODEL_DIR> --engram <ENGRAM_DIR> \
  --oracle oracle.jsonl --tokens 128 --drafts 3 --out spec.json
python3 tools/dsv41/replay_check.py --rank R --master <HEAD_IP> --model <MODEL_DIR> --engram <ENGRAM_DIR> \
  --text <TEXT_FILE> --out replay_check.json
python3 tools/dsv41/engine_check.py --rank R --world 2 --master <HEAD_IP> --model <MODEL_DIR> --engram <ENGRAM_DIR> \
  --oracle oracle.jsonl --ref-top ref_top.jsonl --out eng.json
```
