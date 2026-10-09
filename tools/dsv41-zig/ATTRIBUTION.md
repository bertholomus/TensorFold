# Attribution: the `dsv41` family (Zig)

The `dsv41` family of this TensorFold fork (`zig/src/families/dsv41/`, `zig/kernels/cuda/dsv41/`, the DSML tool-call
parser in `zig/src/server/tool_parse.zig`, and `tools/dsv41-zig/`) is Bertholomus AI's port, to TensorFold 1.0's
native Zig engine, of our own DeepSeek-V4.1 family of the Python line
([github.com/bertholomus/TensorFold, branch `deepseek-v41-tp2`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2)).
It was written independently and is clean-room: upstream TensorFold's own DeepSeek code (its pull request #300 and
any `deepseek` family upstream) was never opened, nor was the jayleaton recipe; the MiaAI-Lab vLLM kit (AGPL-3.0) was
only ever run as a black box. The model math was re-implemented from DeepSeek's MIT inference code and tech report,
which we did read; no code was copied. This file lists every outside source we used, what we took from it, how, and
under which license. `NOTICE` at the repository root lists the upstream files this fork changes;
`THIRD_PARTY_NOTICES.md` has a section for this family.

## How the port was checked

The port's replies were compared token for token against our served Python family on the same two nodes: its
prompts' logits, every layer's outputs on recorded requests, greedy and sampled replies, image prompts and the
server's refusals. The recording tools in `tools/dsv41-zig/rec/` capture our own Python family's run (its tensors,
kernel launches and replies) for those comparisons.

## Sources

| Source | License | What we used | How | Where it shows up |
|---|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference code (`inference/model.py`, `engram.py`, `kernel.py`) and `encoding/` | MIT, Copyright (c) 2023 DeepSeek | The model math: attention with compressed KV and the lightning indexer, the compressor, mHC with Sinkhorn, Engram hashing, the MoE gate, the DSpark forward, the quantization rules | Read and re-implemented (first in our Python family, then ported). No code copied. The DSML tool-call format follows DeepSeek's encoding and chat template | `zig/src/families/dsv41/`; `zig/src/server/tool_parse.zig` |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) vision code (`inference/vision.py`, `inference/image_processor.py`, the image paths of `inference/model.py`, the image blocks of `encoding/encoding.py`) | MIT, Copyright (c) 2023 DeepSeek | Image preprocessing, the ViT, the aligner, the image-span layout, the gates' VL bias inside image spans | Read and re-implemented. No code copied | `picture.zig`, `vit.zig`, `vision.zig`, image spans in `prompt.zig` |
| DeepSeek-V4.1-Flash weights, the `ffn.gate.bias_vl` tensors | MIT, Copyright (c) 2023 DeepSeek | The gates' bias for image-span tokens, which the EXL3 checkpoint leaves out | Loaded at run time from a small file of those tensors (read from the original shards). Not included | the kit's `vision/gate_bias_vl.safetensors` |
| DeepSeek-V4.1-Flash tech report | DeepSeek's publication | Architecture: bounded replay, the attention modes, the indexer, mHC, Engram, DSpark | Read | Design |
| DeepSeek-V4.1-Flash weights, original shards 47 and 48 | MIT, Copyright (c) 2023 DeepSeek | The Engram tables and their file format | Loaded at run time. Not included | The Engram reader |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights the family serves, and file-format facts from their headers | Loaded at run time. Not included | The weight loader |
| [TensorFold](https://github.com/ashhart/TensorFold) v1.0.2 | Apache-2.0, Copyright 2026 TensorFold contributors | The engine the family plugs into: the native server and OpenAI-compatible API, the lane core, the EXL3 module, the CUDA driver, NCCL and Triton AOT layers, keyed exact sampling | Forked. TensorFold's LICENSE, its LICENSES/ files and the text of its NOTICE and THIRD_PARTY_NOTICES.md are kept | This repository |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format, through TensorFold's EXL3 module; token frequencies of its standard calibration text (only the resulting ranking of token ids is stored) | Read as a format reference; calibration text tokenized and counted | Weight decoding; `markov_tokens.zig` |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (`LICENSES/Pillow-MIT-CMU.txt`) | `convert("RGB")` and 8-bit BICUBIC `resize` arithmetic; formats other than PNG decoded by the runtime's own Pillow in a child process | Re-implemented for PNG without including Pillow's source; called, not bundled, for other formats | `picture.zig`, `pil.zig` |
| PyTorch memory-efficient attention kernel (sm_120 binary from `libtorch_cuda.so`), built on NVIDIA CUTLASS | BSD-3-Clause (`LICENSES/PyTorch-BSD-3-Clause.txt`, `LICENSES/CUTLASS-BSD-3-Clause.txt`) | The vision tower's attention kernel | Extracted from the container into a deployment's kernel kit and launched. Not included in this repository | `fmha.zig` |
| [vLLM](https://github.com/vllm-project/vllm), upstream `deepseek_v41` model code | Apache-2.0, Copyright contributors to the vLLM project | A math cross-check of our Python family (for example its Engram hashing at the start of a sequence) | Read. No code copied | Nothing copied |
| Our TensorFold fork for GLM-5.3, [github.com/bertholomus/TensorFold, branch `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4) | Apache-2.0, same authors | The RDMA-write all-gather and the concurrent-decoding design, through our Python family | Ported by the same authors | `zig/src/families/dsv41/` (the RDMA ring, the rank link) |
| MiaAI-Lab vLLM kit | AGPL-3.0 | Nothing in this family; early baseline measurements of the Python line | Run as a black box through its HTTP API. No code read or copied | Not part of this repository |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL, CUDA) | NVIDIA's container license and the components' own licenses | The build and run-time environment; Triton kernels of our Python family captured as binaries | Used as supplied. Not included | Not part of this repository |

## Names

- "DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this family runs and on what hardware.
- This fork is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.
