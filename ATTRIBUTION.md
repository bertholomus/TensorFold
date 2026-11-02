# Attribution: DeepSeek-V4.1-Flash on TensorFold's native engine, four nodes (branch `deepseek-v41-zig-tp4`)

This branch adds a DeepSeek-V4.1-Flash family (`zig/src/families/dsv41/`, its kernels in `zig/kernels/cuda/`, its
tools in `tools/dsv41-zig/`) to TensorFold 1.0.2's native Zig engine, and runs it on two or four NVIDIA GB10 nodes. It ports our own DeepSeek-V4.1 family of TensorFold's Python line (the TP2 recipe
[bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10))
and extends it to an exact two-dimensional split over four nodes. It was written independently and is clean-room with
respect to other people's DeepSeek-V4.1 work: the jayleaton recipe, upstream TensorFold's DeepSeek-V4.1 pull requests
(#299, #300 and later) and the DeepSeek-named files of upstream's tree were never opened, and the MiaAI-Lab vLLM kit
(AGPL-3.0) was only ever run as a black box, by the TP2 work, for baseline numbers. This file lists every outside source the branch uses, what it took and how. Ideas and techniques we built
ourselves are not listed.

## Rules we kept

- The model math (compressed attention and its indexer, the compressor, mHC with Sinkhorn, Engram, the MoE gate and its
  image-span bias, the DSpark draft blocks, the vision tower and aligner) comes from reading DeepSeek's own MIT
  inference code and the DeepSeek-V4.1-Flash tech report, re-implemented. No DeepSeek code is included.
- The four-node split, its exchanges, the kept-prompt placement and the server integration are ours.
- Copied code is marked where it lives and listed below (one kernel's device code, from TensorFold's own EXL3 module).

## Sources

| Source | License | What we used | How | Where it shows up |
|---|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference code (`inference/model.py`, `engram.py`, `kernel.py`, `vision.py`, `image_processor.py`) and `encoding/` | MIT, Copyright (c) 2023 DeepSeek | The model math, the image processor, vision tower and aligner, the image-span layout, the chat encoding and DSML tool calls | Read and re-implemented. No code copied | `zig/src/families/dsv41/` |
| DeepSeek-V4.1-Flash tech report | DeepSeek's publication | The architecture: encoder / decoder halves and bounded replay, compressed attention modes, the hierarchical indexer, mHC, Engram, DSpark | Read | Design of the family |
| DeepSeek-V4.1-Flash weights: the original shards 47 and 48, and the 43 `ffn.gate.bias_vl` tensors | MIT, Copyright (c) 2023 DeepSeek | The Engram tables (loaded at run time) and the gates' bias for image-span tokens (`gate_bias_vl.safetensors` in a deployment's kernel kit, read byte for byte from the original shards) | Loaded at run time. Not redistributed | The Engram reader; the image routing |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights this branch serves, and file-format facts from their headers | Loaded at run time. Not redistributed | The weight loader |
| DeepSeek-V3 Technical Report (arXiv 2412.19437), "redundant experts" in its inference deployment | Public paper (DeepSeek models MIT) | The idea of holding experts on both replicas and dealing each round's experts between them | Idea only | The balanced experts split of the four-node decode (`TF_DS_2D_GU=parity`) |
| [TensorFold](https://github.com/ashhart/TensorFold) 1.0.2 (native engine: CUDA runtime, Triton AOT loader, lane core, server, build; the EXL3 module) | Apache-2.0, Copyright 2026 TensorFold contributors | The engine the family plugs into | Forked. LICENSE and the upstream files in LICENSES/ are unchanged; NOTICE lists every upstream file the branch changes; THIRD_PARTY_NOTICES.md gains this family's section | This branch |
| TensorFold's EXL3 grouped expert kernel (`experts_grouped.cuh` of its Python line, whose header credits ExLlamaV3) | Apache-2.0 (TensorFold); MIT, Copyright (c) 2025 Turboderp (ExLlamaV3) | The device code of `grouped_cp_kernel` with its gate / up epilogue and helpers | Copied, arithmetic unchanged, into a one-launch gate / up for the four-node split; credited in the file's header | `zig/kernels/cuda/dsv41/experts_par.cu` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format, which TensorFold's EXL3 module implements; the calibration text a token ranking was generated from | Format reference; a generated frequency list (`markov_tokens.zig`) | Weight decoding; the Markov draft rows |
| [Triton](https://github.com/triton-lang/triton) | MIT | The Python family's kernels, compiled by Triton | Launched from their captured binaries (a deployment's kernel kit, not in this repository) | The CUDA launches of the family |
| PyTorch's memory-efficient attention kernel (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`) | BSD-3-Clause (`LICENSES/PyTorch-BSD-3-Clause.txt`) | The vision tower's attention | Launched from its sm_120 binary, extracted from the container's `libtorch_cuda.so` into a deployment's kit (not in this repository) | `zig/src/families/dsv41/fmha.zig` |
| NVIDIA CUTLASS (the PyTorch kernel above is built on it) | BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES (`LICENSES/CUTLASS-BSD-3-Clause.txt`) | As above | As above | As above |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (`LICENSES/Pillow-MIT-CMU.txt`) | How pictures decode and resample (`convert("RGB")`, 8-bit BICUBIC `resize`), written for TensorFold without its source; the runtime's own Pillow decodes formats other than PNG in a `python3` child | Re-implemented for PNG; called as a separate process for other formats; not bundled | `picture.zig`, `pil.zig` |
| Zig 0.17.0 | MIT | The compiler | Build time only | `build.zig`, `zig/build/` |
| Our TensorFold fork for GLM-5.3, [bertholomus/TensorFold `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4) | Apache-2.0, same authors | The RDMA-write all-gather for decode partials (through our TP2 family) | Ported by the same authors | `zig/src/families/dsv41/rdma.zig`, `zig/kernels/cuda/dsv41/rdma_gather.cu` |
| Our DeepSeek-V4.1 TP2 family of TensorFold's Python line ([recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10), engine branch `deepseek-v41-tp2`) | Apache-2.0, same authors | The family this branch ports: its kernels (as captured binaries), its exactness gates and recorded references | Ported by the same authors | The whole family; `tools/dsv41-zig/rec/` |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL) | NVIDIA's container license and the components' own licenses | The run-time and build environment | Used as supplied. Not redistributed | The serving and build containers |

## Names

- "DeepSeek" and "DeepSeek-V4.1-Flash" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this branch runs and on what hardware.
- This branch is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.
