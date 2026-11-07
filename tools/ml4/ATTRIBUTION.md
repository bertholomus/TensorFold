# Attribution: the `mistral_large4` family

The `mistral_large4` family of this TensorFold fork (`src/tensorfold/families/mistral_large4/`, `tools/ml4/`, the
`tests/test_ml4_*.py` tests) was written for this fork by BertholomusAI (Albert Lee, https://github.com/bertholomus),
Copyright 2026, under the repository's Apache-2.0 license. It re-implements the model's math; no code was copied from the
sources below. This file lists every outside source we used, what we took from it, how, and under which license. Ideas
and common techniques we built ourselves are not listed. `NOTICE` at the repository root lists the upstream files this
fork changes. Beyond these sources, no outside contributor's patch or idea is part of this family.

## Sources

| Source | License | What we used | How | Where it shows up |
|---|---|---|---|---|
| [mistralai/Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) (Mistral AI): weights, configuration, tokenizer, chat template, the draft head `mtp_dspark.*` | Mistral Ecosystem Open-Weight License, a modified MIT license. Attribution notice: 2026 - Mistral. The notice must be included in all copies | The base model of the served weights; the configuration and chat format | Quantized by us; the served pack carries the license. Weights are not in this repository | The served pack; `cuda/weights.py`, `config.py`, `tools/ml4_step4_chat.py` |
| [TensorFold](https://github.com/ashhart/TensorFold) v0.6.6 (ashhart) | Apache-2.0, Copyright 2026 TensorFold contributors | The engine the family plugs into: the server and OpenAI-compatible API, the EXL3 module and its grouped expert kernels, NCCL communication | This fork | `NOTICE` lists every changed upstream file |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp) | MIT, Copyright (c) 2025 Turboderp | The EXL3 format and its converter: the served pack was quantized with it. TensorFold's EXL3 module implements the format | Used to quantize; read as a format reference | The served pack; `src/tensorfold/cuda/exl3/` |
| Our ExLlamaV3 branch `ml4-loader` (Mistral Large 4 support in ExLlamaV3, same authors as this fork) | MIT | The reference forward the parity tools and the loss rows compare against; the configuration checks `config.py` mirrors | Run and read. Not published with this release | `tools/parity/`, `config.py` |
| [Hugging Face Transformers](https://github.com/huggingface/transformers), `models/mistral_large4/` (`modeling_mistral_large4.py`, `configuration_mistral_large4.py`) | Apache-2.0, Copyright 2026 The HuggingFace Inc. team | The definition of each operation: KV sharing, per-head scaling, the indexer, LeRoPE, NoPE layers, routing, logit softcap | Read as the reference. No code copied | `cuda/model.py`, `cuda/cp4.py` |
| [vLLM pull request 64534](https://github.com/vllm-project/vllm/pull/64534) "[Model] Add Mistral Large 4" (NickLucche, open when we read it at commit 28a4495d: `vllm/models/mistral/mistral_large4/nvidia/dspark.py`, `vllm/v1/worker/gpu/spec_decode/dspark/`, `vllm/transformers_utils/configs/mistral.py`) | Apache-2.0, Copyright contributors to the vLLM project | How the draft head's tensors fit together: the taps, the context rows, the block of anchor and mask rows, the recurrent head | Read as the reference. No code copied; the drafter's kernels, its ring and its place in a step are ours | `cuda/dspark.py`, `cuda/dspark_cuda.py` |
| [SpecForge](https://github.com/sgl-project/SpecForge) `specforge/modeling/draft/dspark.py` at commit d80daf31 (MIT) and [SGLang](https://github.com/sgl-project/sglang) `dspark_mla.py` from a pull request branch at commit eb80b0b6 (Apache-2.0) | As listed | A second reading of the same structure | Read only. No code copied | `cuda/dspark.py` |
| Our DeepSeek-V4.1 family in this fork (`src/tensorfold/families/deepseek_v41/`, `src/tensorfold/cuda/rdma.py`) | Apache-2.0, same authors | The RDMA-write all-gather, the per-rank weight cache, the grouped expert kernels, the way a drafter sits beside verification | Reused | `tools/dsv41/ATTRIBUTION.md` lists that family's sources |
| NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3` (PyTorch BSD-3-Clause, Triton MIT, NCCL) | NVIDIA's container license and the components' own licenses | The run-time environment | Used as supplied. Not included | Not part of this repository |

## Names

- "Mistral" and "Mistral Large" belong to Mistral AI. "DGX Spark" and "GB10" belong to NVIDIA. We use the names only to
  say what this family runs and on what hardware.
- This fork is not affiliated with or endorsed by Mistral AI, NVIDIA or the TensorFold authors.
