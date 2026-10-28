"""Beam-1.0 (model_type ``beam``, BeamForCausalLM): sandwich-norm decoder layers with depth-scaled residuals, gated
GQA attention on window-512 RoPE layers and full NoPE layers with SSMax-scaled queries, grouped sigmoid routing over
256 experts, a soft-capped head; EXL3 conversions on the CUDA engine over TP ranks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("beam",)
TITLE = "Beam-1.0"
LANES = False
MODELS = ()
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"
KERNEL_DEPENDENCIES = ("tensorfold.cuda.exl3.experts", "tensorfold.cuda.exl3.linear", "tensorfold.cuda.exl3.prefill")
KERNEL_VERSION = "v1"
DRAFTER = ""


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read: a checkpoint that is not EXL3, or a switch it does not implement."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config
    from tensorfold.families.beam.config import Config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"Beam-1.0's CUDA engine reads ExLlamaV3 (EXL3) conversions. {OWN_MODEL_HELP}")
    Config.read(model_dir)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 2, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """Beam-1.0 over two GPUs (``--tp 2``), its prompts and replies in one scheduler on rank 0: ``--parallel N``
    sequences decode together (4 by default), ``--context`` sizes the KV pool (else what memory leaves, at most the
    checkpoint's 1,048,576), ``--kv-dtype`` picks the full layers' store (fp8 by default), ``--drafter DIR`` loads the
    DFlash2 drafter (TF_BEAM_SPEC_K drafts a step, 3 by default; TF_BEAM_DRAFTER=lookup: prompt-lookup drafts)."""

    import os

    from tensorfold.families.beam.cuda.engine import DEFAULT_CHUNK, DEFAULT_SLOTS, DEFAULT_SPEC_K, BeamEngine

    if int(tp) != 2:
        raise ValueError(f"Beam-1.0's CUDA engine splits over 2 GPUs, not {tp}: run the same `tensorfold serve` "
                         "command with --tp 2 --rank R --master ADDRESS on both machines")
    if not master:
        raise ValueError("--tp 2 needs --master: rank 0's address on the link between the machines")
    os.environ["TF_TP_WORLD"] = "2"
    # Eight channels carry these gathers as fast as NCCL's 64 (8K prompt 954 tok/s, decode 20.3 tok/s) and keep 16
    # connection buffers instead of 128: 1.4 GiB more for the KV pool on each node.
    os.environ.setdefault("NCCL_MAX_NCHANNELS", "8")
    kv = str(options.get("kv_dtype") or os.environ.get("TF_BEAM_KV", "fp8"))
    slots = int(options.get("parallel") or os.environ.get("TF_BEAM_SLOTS", DEFAULT_SLOTS))
    chunk = int(os.environ.get("TF_BEAM_CHUNK", DEFAULT_CHUNK))
    spec_k = int(os.environ.get("TF_BEAM_SPEC_K", DEFAULT_SPEC_K))
    return BeamEngine(Path(model_dir), rank=int(rank), world=2, master=master, port=int(master_port),
                      context=options.get("context"), slots=slots, chunk=chunk, kv_dtype=kv,
                      drafter="" if no_drafts else str(drafter or ""), spec_k=spec_k)


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":                # imported on first use, so discovery never loads the CUDA server
        from .cuda.app import BeamApp

        return BeamApp
    raise AttributeError(name)
