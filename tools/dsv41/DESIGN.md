# DESIGN — DeepSeek-V4.1-Flash on our TensorFold TP2 engine (two NVIDIA GB10 nodes)

Status: v1, 2026-10-03, written before any engine code. Written independently: clean-room with respect to the kit
and the public jayleaton recipe, which are not read ("the kit": the MiaAI-Lab vLLM kit, commit 6f7d1590ad49,
AGPL-3.0, only run as a black box for baseline numbers). The model math is re-implemented from DeepSeek's own MIT
`inference/` (model.py, engram.py, kernel.py) and tech report, read but not copied. Also used: upstream TensorFold
(Apache-2.0), our TensorFold fork for GLM-5.3 (github.com/bertholomus/TensorFold, branch `glm-dsa-tp4`), and the checkpoint's own file format.
Upstream vLLM's (Apache-2.0) `deepseek_v41` is a cross-check for math only. Sources and licenses: `ATTRIBUTION.md`
in this folder.

## 1. What the model needs (numbers from config.json and the safetensors headers)

- 40 layers, d=5120, residual carried as hc=4 streams (single-pass mHC: a block's input mix comes from the
  previous sublayer). Layers 0–1 SWA only; 2–19 CSA2 ratio 2 (Full at 2/8/14, Reuse otherwise); 20–39 ratio 1
  (Full at 20, Reindex at 24/28/32/36, Reuse otherwise). Layer 20's indexer also builds the candidate pool
  (2048 blocks x 8) that 24–36 search inside.
- Attention: q_lora 1280 -> 64 heads x 512; one shared 512-wide KV latent (key = value, RoPE on the last 64);
  SWA window 128; attention sink per head; output de-RoPE, 8 groups x (4096 -> 1024), then 8192 -> 5120.
- MoE every layer: 384 routed (top-6, sqrt-softplus, bias only for selection, scale 1.5, SwiGLU clamp 10) + 1 shared.
- Engram before layers 1 and 14: 3 n-gram orders x 8 heads = 24 rows of 256 (FP8 + scale) from ~384M-row tables,
  `wkv` 6144 -> 5x5120 (4 keys, one per hc stream, plus a value), gate per stream.
- DSpark: 3 blocks (window-only attention, 128 experts top-3) over `main_proj` of the mean-over-hc inputs of
  layers 37/38/39; block of 5 drafts, Markov head (rank 256) and a confidence head.
- Checkpoint (EXL3 mul1, 196.1 GiB): routed experts 182.6 GiB (3-bit; 2-bit in 18–22), DSpark experts 6.3,
  attention 2.9 (wq_a/wkv 6-bit, wq_b/wo_a/wo_b 5-bit), embed bf16 1.2, head 6-bit 0.5, shared experts 0.75,
  Engram wkv 0.17 (5/4-bit), indexer wk 8-bit. Engram tables: 2 x ~95 GiB FP8 on each node's NVMe.

## 2. TP2 split (ours)

| Part | Split | Why |
|---|---|---|
| Routed + shared experts | intermediate 2304 -> 1152 a rank (w1/w3 by output, w2 by input) | every token's 6 experts load both ranks equally; 1152 = 9 x 128 keeps EXL3's 128-wide Hadamard blocks whole |
| Attention heads | 32 a rank: wq_b by output, wo_a groups 0–3 / 4–7, wo_b by input, sinks by head | heads are independent until wo_b |
| wq_a, wkv, q/kv norms, compressor, indexer (wq_b, weights_proj, wk) | replicated | small; every rank needs the full latent and indices, so no gather |
| KV caches (window, compressed, indexer K) | replicated, written identically on both ranks | KV is one latent shared by all heads; 890 B/token native means a full copy each is cheap |
| Router, mHC, norms, Engram gate | replicated | per-token vectors |
| Engram | hash columns 0–11 on rank 0, 12–23 on rank 1: each rank reads its 12 rows and holds wkv's matching input half | halves NVMe reads and wkv bytes a rank; one 25,600-wide reduction |
| Embedding, head, Markov head | vocab halves | 0.6 + 0.25 GiB back a rank; head gathers top candidates, not full logits |
| DSpark blocks | as the backbone (experts split, heads split) | same kernels |

Per rank: ~91.3 GiB routed experts + 3.2 DSpark experts + ~2.6 everything else ≈ 97 GiB; KV under 1 GiB at 1M
tokens; workspace and graphs ~3–4 GiB. The kit allocates ~99.5 GiB weights a rank; we must stay at or under it.

## 3. Collectives

- Every reduction is an all-gather of partials then a rank-order sum (TensorFold's rule), so both ranks hold
  bit-identical streams and sample identically without a broadcast.
- Per decode row: 2 a layer (after wo_b, after the MoE combine, shared expert folded into the same partial) = 80,
  + 2 Engram, + embedding, + head candidates ≈ 84. Payload 5120 fp32 a row (20 KiB), Engram 25,600.
- Transport: the RDMA-write gather from our earlier GLM-5.3 fork (`cuda/rdma.py`, measured ~23 µs at one row on
  CX7) for decode-sized payloads, NCCL over RoCE (the CX7 HCA, `NCCL_IB_HCA=<HCA>`, GID index 5) for prompt chunks.
  Budget ≈ 2 ms a token at 84 gathers.
- Prompt chunks overlap the gather of one micro-batch with the next one's compute (as in our GLM-5.3 fork).

## 4. KV cache

- Native formats, as trained (QAT): compressed main KV in FP4 E2M1 with an E4M3 scale per 16 (288 B a token for
  ratio 1); indexer K in FP4 with a UE8M0 scale per 32 (68 B); SWA KV FP8 E4M3 with a UE8M0 scale per 32.
  Total global KV ≈ 890 B a token (decoder 288 + encoder 3 x 144 + indexer 170): 1M tokens = 0.89 GB a rank.
  A bf16 cache mode stays as a switch for A/B against the oracle.
- Storage: per sequence, append-only position-indexed arrays for the 4 compressed latents and 4 indexer-K planes;
  SWA rings of 128 + 8 slots per layer (the extra 8 so a rejected verify window never clobbers live entries);
  ratio-2 compressors keep their half-filled group (kv, score) as state that a rejected window restores.
- Rollback after a partial accept = reset lengths + restore the compressor tails and the Engram hash tail.
- 1M context: the cache fits easily; the cost is layer 20's full-range index scan (and 2/8/14 at half range). Phase 1
  replicates the indexer on both ranks; phase 2 splits the scanned positions between ranks above a threshold,
  merging local top-512 / top-2048-block lists exactly (ties by lower index).

## 5. Prefill

- Exact mode: all 40 layers over every prompt token, in chunks (2,048–8,192 rows), as the reference does.
- Replay mode (CED's design, trained for): encoder (0–19) over all tokens, which also produces layer 20's global
  KV and index K; decoder (20–39) only over the last 128 tokens with SWA truncated to that segment. Roughly halves
  prefill. Off by default until its oracle agreement and needle recall are measured; within one engine both serial
  and drafted runs use the same prefill, so exactness is unaffected.
- A continued conversation keeps its decoder SWA rings in memory and never needs replay.

## 6. Exactness

- Greedy drafted == greedy serial, always: every kernel is row-invariant (a row's arithmetic does not depend on
  how many rows share the call: fixed reduction order, no M-dependent split-K, expert combine in top-k slot order),
  attention and indexer per row, top-k ties by lower index.
- Sampling keyed by (seed or prompt, position, token), the TensorFold rule, so T>0 drafted == serial too.
- Gate on every change: teacher-forced top-1 vs the kit oracle (target >= 0.99) and a drafted-vs-serial diff on a
  fixed prompt set.

## 7. DSpark speculative decoding

- After each target pass, the taps (mean over hc of the inputs of layers 37/38/39) of the newly accepted tokens go
  through main_proj + main_norm into each draft block's window ring (positions of those tokens).
- Draft: block [last token, noise x 4] -> 5 base logits in one pass; the Markov head adds its bias row by row
  (greedy argmax, or keyed sampling at T>0) -> d1..d5 and confidences.
- Verify k of them (start fixed k=3, then pick k from the confidence head's survival estimate and our measured
  throughput curve) in one target pass of k+1 rows; accept the matching prefix plus the target's next token.
- The drafter only proposes; the target's serial arithmetic decides every token.

## 8. Engram

- Hashes from the token stream (DeepSeek's `engram.py` math: compressed-token map built once from the tokenizer,
  per-layer odd multipliers, XOR rolling hash, prime buckets). Known as soon as a token is.
- Rows read from the original FP8 shards on local NVMe by offset (pread / io_uring, a small hot-row cache), prefetched
  while layer 0 runs (layer 1's) and while layers 1–13 run (layer 14's). Each rank reads only its 12 hash columns.

## 9. Reference oracle (step 2)

- PyTorch, one file, DeepSeek's model.py math with the quant simulation of the kernels (FP8 window KV, FP4 compressed
  KV and indexer, attention sink) re-implemented in plain torch; EXL3 weights dequantized on the fly per tensor
  with TensorFold's EXL3 decoder, layers streamed from disk, only the touched experts decoded. One GB10, short
  prompts, fp32 accumulation. Matched to the kit's teacher-forced top-k on the fixed set, then used as the oracle
  for every engine kernel (per-layer dumps).

## 10. Order of work and gates

1. Kit baseline. 2. Reference matches kit (top-1 >= 0.99). 3. Engine single-rank
pieces vs reference per layer. 4. TP2 serial decode end to end, oracle agreement. 5. DSpark, drafted == serial.
6. Server, then speed (profile first), then context growth with needle recall.

## 11. Risks

- Memory: 97 GiB a rank leaves ~10 GiB; Engram I/O must use no pinned table memory.
- EXL3 2/3-bit expert kernels on GB10 decide decode speed (~3.5 GB of weights read a token a rank, ~78 tok/s ceiling).
- The kit may use bounded decoder replay or a different KV format; the oracle comparison will show it.
