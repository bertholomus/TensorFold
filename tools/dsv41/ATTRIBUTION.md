# Attribution: the `deepseek_v41` family

The `deepseek_v41` family of this TensorFold fork (`src/tensorfold/families/deepseek_v41/`, the RDMA gather in
`src/tensorfold/cuda/rdma.py` and `rdma_gather.cu`, and `tools/dsv41/`) was written independently. It is clean-room
with respect to the MiaAI-Lab vLLM kit (AGPL-3.0, only run as a black box) and the jayleaton recipe (never opened).
The model math was re-implemented from DeepSeek's MIT inference code and tech report, which we did read; no code was
copied. This file lists every outside source we used, what we took from it, how, and under which license. Ideas and
common techniques we built ourselves are not listed. `NOTICE` at the repository root lists the upstream files this
fork changes; `THIRD_PARTY_NOTICES.md` has a section for this family.

## Rules we kept

- The model math comes from reading DeepSeek's own MIT inference code and the DeepSeek-V4.1-Flash tech report. We
  re-implemented it. No code was copied.
- The tensor-parallel split, caches, collectives, kernels and server integration are ours (`DESIGN.md` in this
  folder).
- The MiaAI-Lab vLLM kit (AGPL-3.0) was only run as a black box, to measure baseline numbers through its HTTP API. We
  did not read, copy or adapt any of its code. We read the configuration file it was deployed with (settings and
  comments, not code) only to describe its settings in `REPORT.md`.
- A public recipe by jayleaton for running this model on TensorFold exists. We never opened it, and we did not tune
  toward its numbers.

## Sources

| Source | License | What we used | How | Where it shows up |
|---|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference code (`inference/model.py`, `engram.py`, `kernel.py`) and `encoding/` | MIT, Copyright (c) 2023 DeepSeek | The model math: attention with compressed KV and the lightning indexer, the compressor, mHC with Sinkhorn, Engram hashing (per-layer seeds, odd multipliers, prime bucket counts, the compressed-token map), the MoE gate, the DSpark forward, the FP8/FP4 quantization rules | Read and re-implemented. No code copied. The DSML tool-call format follows DeepSeek's encoding and chat template | `src/tensorfold/families/deepseek_v41/`; the reference model `tools/dsv41/ref/`; DSML parsing in `src/tensorfold/cuda/reply_text.py` |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) vision code (`inference/vision.py`, `inference/image_processor.py`, the image paths of `inference/model.py`, the image blocks of `encoding/encoding.py`) | MIT, Copyright (c) 2023 DeepSeek | Image preprocessing, the ViT, the aligner, the image-span layout, the gates' VL bias inside image spans, Engram shut inside them, how image blocks join a message | Read and re-implemented, ops and their order kept (the tower's rows are the reference's bit for bit when it runs the reference's attention call). No code copied | `src/tensorfold/families/deepseek_v41/cuda/vision.py`, image handling in `model.py`, `engine.py`, `app.py` |
| DeepSeek-V4.1-Flash weights, the `ffn.gate.bias_vl` tensors | MIT, Copyright (c) 2023 DeepSeek | The gates' bias for image-span tokens, which the EXL3 checkpoint leaves out | Loaded at run time from a small file of those tensors (read from the original shards). Not included | `attach_vl_bias` in the weight loader |
| DeepSeek-V4.1-Flash tech report | DeepSeek's publication | Architecture: encoder/decoder halves and bounded replay, CSA2 attention modes, the hierarchical indexer, single-pass mHC, Engram, DSpark, FP4 KV | Read | Design; replay prefill mode (`TF_DS_REPLAY`); the input text of `replay_check.py` (not included) |
| DeepSeek-V4.1-Flash weights, original shards 47 and 48 | MIT, Copyright (c) 2023 DeepSeek | The Engram tables, and the file-format facts of them (FP8 rows of 256 with an E8M0 scale per 32) | Loaded at run time. Not included | The Engram reader |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights the family serves, and file-format facts from their headers (EXL3 tensor groups, `wo_a` stored as 8 slices) | Loaded at run time. Not included | The weight loader |
| [TensorFold](https://github.com/ashhart/TensorFold) v0.6.3 | Apache-2.0, Copyright 2026 TensorFold contributors | The engine the family plugs into: the server and OpenAI-compatible API, the EXL3 module, the grouped expert kernel, the NCCL wrapper, keyed exact sampling | Forked. TensorFold's LICENSE, LICENSES/ and the text of its NOTICE and THIRD_PARTY_NOTICES.md are kept | This repository |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format (trellis layout, codebooks). TensorFold's EXL3 module implements it | Read as a format reference. This fork extends TensorFold's EXL3 expert kernels (`experts.cu`, and `experts_grouped.cuh`, whose header credits ExLlamaV3, MIT); that credit and the MIT text in `THIRD_PARTY_NOTICES.md` are kept | Weight decoding in the family and the reference model; the extended grouped expert kernel |
| [vLLM](https://github.com/vllm-project/vllm), upstream `deepseek_v41` model code | Apache-2.0, Copyright contributors to the vLLM project | A math cross-check, for example that its Engram hashing at the start of a sequence matches DeepSeek's | Read. No code copied | Nothing copied; it confirmed our reading of DeepSeek's code |
| Our TensorFold fork for GLM-5.3, [github.com/bertholomus/TensorFold, branch `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4) | Apache-2.0, same authors | The RDMA-write all-gather for decode partials, the prompt-chunk expert kernels, and the concurrent-decoding design (slot pool, rounds over every stream, the rank step link, step digests, watchdog) | Ported by the same authors | `src/tensorfold/cuda/rdma.py`, `rdma_gather.cu`; the prompt-chunk expert kernels in `src/tensorfold/cuda/exl3/` |
| MiaAI-Lab vLLM kit, commit 6f7d1590ad49 | AGPL-3.0 | Baseline numbers, measured by our own client (`kit_bench.py`) on the same nodes; the facts about its settings | Run as a black box through its HTTP API. No code read or copied. Its deployed configuration file read for its settings, paraphrased, not quoted | `REPORT.md` |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL) | NVIDIA's container license and the components' own licenses | The run-time environment | Used as supplied. Not included | Not part of this repository |

## Names

- "DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this family runs and on what hardware.
- This fork is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.
