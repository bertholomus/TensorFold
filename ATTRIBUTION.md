# Attribution: DeepSeek-V4.1-Flash on TensorFold's native engine, four nodes (branch `deepseek-v41-zig-tp4`)

This branch adds a DeepSeek-V4.1-Flash family (`zig/src/families/dsv41/`, its kernels in `zig/kernels/cuda/`, its
tools in `tools/dsv41-zig/`) to TensorFold 1.0.12's native Zig engine (1.0.2 up to the branch's v0.1.1), and runs it on two or four NVIDIA GB10 nodes. It ports our own DeepSeek-V4.1 family of TensorFold's Python line (the TP2 recipe
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
- The four-node split, its exchanges, the kept-prompt placement and the server integration are ours, but for the
  contributions to this fork listed below (each under its author's name, or credited in the commits that port it).
- Copied code is marked where it lives and listed below (one kernel's device code, from TensorFold's own EXL3 module).

## Sources

| Source | License | What we used | How | Where it shows up |
|---|---|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference code (`inference/model.py`, `engram.py`, `kernel.py`, `vision.py`, `image_processor.py`) and `encoding/` | MIT, Copyright (c) 2023 DeepSeek | The model math, the image processor, vision tower and aligner, the image-span layout, the chat encoding and DSML tool calls | Read and re-implemented. No code copied | `zig/src/families/dsv41/` |
| DeepSeek-V4.1-Flash tech report | DeepSeek's publication | The architecture: encoder / decoder halves and bounded replay, compressed attention modes, the hierarchical indexer, mHC, Engram, DSpark | Read | Design of the family |
| DeepSeek-V4.1-Flash weights and [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)'s: the original shards 47 and 48 of each, and their 43 `ffn.gate.bias_vl` tensors | MIT, Copyright (c) 2023 DeepSeek | The Engram tables (loaded at run time) and the gates' bias for image-span tokens (`gate_bias_vl.safetensors` in a deployment's kernel kit, read byte for byte from the original shards of the checkpoint served) | Loaded at run time. Not redistributed | The Engram reader; the image routing |
| [bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw), our EXL3 quant of DeepSeek-V4.1-Flash-1017 | MIT, inherited from the base model | The weights this branch serves since v0.2.0 | Made by us with ExLlamaV3's converter; loaded at run time; published on its own | The weight loader |
| [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) | MIT, inherited from the base model | The weights the branch served up to v0.1.1, and file-format facts from their headers | Loaded at run time. Not redistributed | The weight loader |
| DeepSeek-V3 Technical Report (arXiv 2412.19437), "redundant experts" in its inference deployment | Public paper (DeepSeek models MIT) | The idea of holding experts on both replicas and dealing each round's experts between them | Idea only | The balanced experts split of the four-node decode (`TF_DS_2D_GU=parity`) |
| [TensorFold](https://github.com/ashhart/TensorFold) 1.0.12 (1.0.2 up to v0.1.1; native engine: CUDA runtime, Triton AOT loader, lane core, server, build; the EXL3 module) | Apache-2.0 from TensorFold 0.6.0, Copyright 2026 TensorFold contributors; code written before 0.6.0 keeps its MIT notice (`LICENSES/MIT.txt`) | The engine the family plugs into; 1.0.2's Triton kit packer (`aot_pack.py`), which 1.0.12 replaced | Forked, with 1.0.12 merged. LICENSE and the upstream files in LICENSES/ are unchanged; NOTICE lists every upstream file the branch changes; THIRD_PARTY_NOTICES.md gains this family's section; the packer is kept as 1.0.2 shipped it, with a two-line header | This branch; `tools/dsv41-zig/kit/aot_pack.py` |
| TensorFold's EXL3 grouped expert kernel (`experts_grouped.cuh` of its Python line, whose header credits ExLlamaV3) | MIT, Copyright (c) 2026 TensorFold contributors (written before TensorFold 0.6.0, it keeps its MIT notice: `LICENSES/MIT.txt`); MIT, Copyright (c) 2025 Turboderp (ExLlamaV3) | The device code of `grouped_cp_kernel` with its gate / up epilogue and helpers | Copied, arithmetic unchanged, into a one-launch gate / up for the four-node split; credited in the file's header | `zig/kernels/cuda/dsv41/experts_par.cu` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format, which TensorFold's EXL3 module implements; the calibration text a token ranking was generated from | Format reference; a generated frequency list (`markov_tokens.zig`) | Weight decoding; the Markov draft rows |
| [Triton](https://github.com/triton-lang/triton) | MIT | The Python family's kernels, compiled by Triton | Launched from their captured binaries (a deployment's kernel kit, not in this repository) | The CUDA launches of the family |
| PyTorch's memory-efficient attention kernel (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`) | BSD-3-Clause (`LICENSES/PyTorch-BSD-3-Clause.txt`) | The vision tower's attention | Launched from its sm_120 binary, extracted from the container's `libtorch_cuda.so` into a deployment's kit (not in this repository) | `zig/src/families/dsv41/fmha.zig` |
| NVIDIA CUTLASS (the PyTorch kernel above is built on it) | BSD-3-Clause, Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES (`LICENSES/CUTLASS-BSD-3-Clause.txt`) | As above | As above | As above |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU (`LICENSES/Pillow-MIT-CMU.txt`) | How pictures decode and resample (`convert("RGB")`, 8-bit BICUBIC `resize`), written for TensorFold without its source; the runtime's own Pillow decodes formats other than PNG in a `python3` child | Re-implemented for PNG; called as a separate process for other formats; not bundled | `picture.zig`, `pil.zig` |
| Zig 0.17.0 | MIT | The compiler | Build time only | `build.zig`, `zig/build/` |
| Our TensorFold fork for GLM-5.3, [bertholomus/TensorFold `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4) | Apache-2.0, same authors | The RDMA-write all-gather for decode partials (through our TP2 family) | Ported by the same authors | `zig/src/families/dsv41/rdma.zig`, `zig/kernels/cuda/dsv41/rdma_gather.cu` |
| Our DeepSeek-V4.1 TP2 family of TensorFold's Python line ([recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10), engine branch `deepseek-v41-tp2`) | Apache-2.0, same authors | The family this branch ports: its kernels (as captured binaries), its exactness gates and recorded references | Ported by the same authors | The whole family; `tools/dsv41-zig/rec/` |
| Lucas Peters ([@lucas-peters](https://github.com/lucas-peters)), [pull request #4](https://github.com/bertholomus/TensorFold/pull/4) to this fork | Apache-2.0, contributed to this fork | The prompt indexer passes each row block's `vis` 16-byte aligned (a copy when its view is not), so prompts whose last chunk ends in a short keyed block at an odd row answer | Merged; the commit keeps its author | `zig/src/families/dsv41/` (the prompt indexer) |
| lepetitprince716-prog ([@lepetitprince716-prog](https://github.com/lepetitprince716-prog)), [pull request #18](https://github.com/bertholomus/TensorFold/pull/18) to this fork | Apache-2.0, contributed to this fork | The loader reads each layer's expert run from the rank cache whole, with O_DIRECT, instead of nine strided sweeps (a faster load) | Merged; the commit keeps its author | `zig/src/families/dsv41/` (the loader) |
| Manrique ([@hashd1ve](https://github.com/hashd1ve)), [pull request #26](https://github.com/bertholomus/TensorFold/pull/26) to this fork's two-node branch | Apache-2.0, contributed to this fork | A prefix two conversations share no longer stales the first one's kept prompt states | Cherry-picked; the commit keeps its author | `zig/src/families/dsv41/lanes.zig` |
| Manrique ([@hashd1ve](https://github.com/hashd1ve)), pull requests [#28](https://github.com/bertholomus/TensorFold/pull/28) and [#29](https://github.com/bertholomus/TensorFold/pull/29) to this fork's two-node branch | Apache-2.0, contributed to this fork | Prompts with images keep and resume their prompt states, matched by the images' digests (#28); a prompt pass in pieces between the other streams' rounds, off unless `TF_FILL_CHUNKS` is set (#29) | Ported to the four-node family; the commits credit them | `zig/src/server/prompt.zig`, `zig/src/core/lane_host.zig`, `zig/src/families/dsv41/lanes.zig` |
| Scott Woolsey Biggart ([@swbiggart](https://github.com/swbiggart)), [pull request #30](https://github.com/bertholomus/TensorFold/pull/30) to this fork | Apache-2.0, contributed to this fork | The optional neighbor transport: adjacent-edge RDMA exchanges on a physical four-node ring (`TF_DS_NEIGHBOR_RDMA`, off unless set), its recipe and receipts | Merged; the commits keep their author | `zig/src/families/dsv41/neighbor.zig`, `docs/recipes/deepseek-v41-neighbor.md` |
| mrsergiopascal ([@mrsergiopascal](https://github.com/mrsergiopascal)), [pull request #31](https://github.com/bertholomus/TensorFold/pull/31) to this fork | Apache-2.0, contributed to this fork | A CPU test and a server reproducer for the replay cut's position alignment, with their write-up | Merged; the commit keeps its author. The branch keeps its own replay cut (every cut's positions uploaded), which the test checks | `tools/dsv41-zig/test_replay_positions.py`, `repro_replay_positions.py`, `replay_positions.md` |
| [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) @ `774ce244` | Apache-2.0 | Ideas: sampling with top_k off (top_p alone), its default; special-token text that a user, a tool result or a tool call's arguments carry, encoded as ordinary tokens; a reply's end-of-reasoning and DSML markers read at their token ids; grammar masks on every verify row, with held drafts walked through the grammar; draft accept counts by draft position and a periodic stats line | Read; ideas only, no code copied | `zig/src/families/dsv41/sampling.zig`; `zig/src/server/specials.zig` (`TENSORFOLD_PLAIN_SPECIALS`); `zig/src/server/reply_text.zig` (`TENSORFOLD_REPLY_IDS`); the structured rounds in `zig/src/families/dsv41/lanes.zig`; `zig/src/core/lanes/stream.zig` and `zig/src/server/metrics.zig` (`TENSORFOLD_STATS_EVERY`) |
| [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (`97787376`) | Apache-2.0, Copyright (c) 2024 by XGrammar Contributors | The idea of holding `tool_choice` "required" or a named function to one DSML calls block by a structural tag in DeepSeek-V4.1's call style; the grammar library itself (libtfgrammar, see THIRD_PARTY_NOTICES.md) | The call grammar is ours, written from the idea; the library is built from xgrammar's source with TensorFold's shim and loaded at run time from a deployment's kit (not in this repository) | `zig/src/server/tool_grammar.zig`; `TENSORFOLD_GRAMMAR_LIB` |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL) | NVIDIA's container license and the components' own licenses | The run-time and build environment | Used as supplied. Not redistributed | The serving and build containers |

## Reports

- [@Capicua25x](https://github.com/Capicua25x) ([two-node recipe issue #18](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/18))
  found the loop guard's exact-cycle check missing templated reasoning loops in production traffic; the guard's
  novelty signal (`zig/src/core/lanes/novelty.zig`, our Python engine's `ThinkLoop`) and its 10-window default follow
  their report and the runaways they captured.
- [@den13501](https://github.com/den13501) ([two-node recipe issue #12](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/issues/12))
  found that long prefills look dead to clients with an idle timer; `TENSORFOLD_SSE_KEEPALIVE` (an SSE comment while
  a streamed reply has nothing to send) follows their report.

## Co-authors named in the commits

The commits that port #28 and #29 name Manrique in a `Co-Authored-By` trailer. Many commits, #18's among them, also
name Claude (Anthropic's AI assistant), which helped write them. The export removed no trailer.

## Names

- "DeepSeek", "DeepSeek-V4.1-Flash" and "DeepSeek-V4.1-Flash-1017" belong to DeepSeek. "DGX Spark" and "GB10" belong to NVIDIA. We use the names
  only to say what this branch runs and on what hardware.
- This branch is not affiliated with or endorsed by DeepSeek, NVIDIA, the TensorFold authors, Mia-AiLab or MiaAI-Lab.
