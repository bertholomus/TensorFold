"""Ling-3.1-flash (model_type ``bailing_hybrid``, BailingMoeV3ForCausalLM): KDA and RoPE-MLA layers with a
grouped sigmoid router, drafted by the checkpoint's own MTP layer, on the CUDA engine over TP ranks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("bailing_hybrid",)
TITLE = "Ling-3.1-flash"
LANES = False
# the EXL3 conversion is a local build (the FP8 checkpoint is its source and the reference forward's);
# no hub repo until one is published and verified
MODELS = ()
# one source per kernel: the KDA chain, replay and conv shift are glm5_next's, the EXL3 linear, the rope-key
# latent attention and the rank collectives glm_moe_dsa's, the routed experts tensorfold.cuda.exl3's; bailing_v3's
# own kernels (the grouped router, the reference-rounded rope, the shared expert's clamped SwiGLU) live in this
# package, which the fingerprint hashes
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"
KERNEL_DEPENDENCIES = ("tensorfold.families.glm5_next.cuda.kda", "tensorfold.families.glm5_next.cuda.qmm",
                       "tensorfold.families.glm5_next.cuda.glue", "tensorfold.families.glm5_next.cuda.latent",
                       "tensorfold.families.glm5_next.cuda.forward", "tensorfold.families.glm_moe_dsa.cuda.x3",
                       "tensorfold.families.glm_moe_dsa.cuda.mla_pe", "tensorfold.families.glm_moe_dsa.cuda.glue",
                       "tensorfold.families.glm_moe_dsa.cuda.forward", "tensorfold.cuda.exl3.experts")
KERNEL_VERSION = "v1"
# the CUDA engine's storage: EXL3 groups wherever the conversion made them (routed experts at any width and
# codebook), stored weights elsewhere (the router, expert_bias, b_proj, kv_b, the conv weights, the norms)
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"
# no draft model yet: the engine drafts with the checkpoint's own MTP layer (a DSpark-class drafter is designed,
# not built)
DRAFTER = ""


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read: a checkpoint that is not EXL3, or states a switch it does not implement."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"Ling-3.1-flash's CUDA engine reads ExLlamaV3 (EXL3) conversions; this checkpoint is "
                         f"not EXL3-quantized (the FP8 original is the conversion's source). "
                         f"{OWN_MODEL_HELP}")
    from tensorfold.families.bailing_v3.config import Config

    cfg = Config.from_dict(config)                     # the architecture checks (rope, KDA, routing)
    from tensorfold.cuda.exl3 import format as exl3_format

    exl3_format.require_config(config, where="NVIDIA GPUs (CUDA)", tested="our 3.03 bpw Ling-3.1-flash conversion",
                               help=OWN_MODEL_HELP)
    missing = sorted(set(cfg.missing_tensors({"_model_dir": str(model_dir)})))
    if missing:
        raise ValueError(f"this checkpoint lacks {len(missing)} tensor(s) the architecture needs, {missing[0]} "
                         f"first. {OWN_MODEL_HELP}")
    print(f"[tensorfold] Ling-3.1-flash runs on NVIDIA GPUs with world-parameterized tensor parallelism "
          f"(--tp 2 or --tp 4, one GPU per machine): pull the checkpoint on every rank and serve with the same "
          f"--tp, --rank and --master on each (TF_TP_WORLD selects the split)", flush=True)


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed experts (everything a rank-folder split keeps on disk)."""

    import json

    from tensorfold.streaming.checkpoint import tensor_bytes

    def routed(name: str) -> bool:
        return ".mlp.experts." in name

    return tensor_bytes(Path(model_dir), routed)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 2, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """Build the world-parameterized engine (TF_TP_WORLD ranks of 2 or 4) with MTP drafting or serial decoding."""

    from tensorfold.families.bailing_v3.cuda.engine import DEFAULT_POLICY, MAX_ROWS, BailingEngine

    if int(tp) not in (2, 4):
        raise ValueError(f"Ling-3.1-flash splits over 2 or 4 GPUs, not {tp}: run the same `tensorfold serve` command "
                         "with --tp W --rank R --master ADDRESS on every rank (the others start first)")
    if not master:
        raise ValueError(f"--tp {tp} needs --master: rank 0's address on the link between the machines")
    if drafter and not no_drafts:
        raise ValueError("Ling-3.1-flash drafts with its own MTP layer and reads no draft model yet: serve without "
                         "--drafter (or with --drafter none)")
    if int(options.get("parallel") or 1) > 1:
        raise ValueError("Ling-3.1-flash's engine decodes one request at a time (the others wait their turn): serve "
                         "without --parallel")
    if mtp_drafts is not None and not 0 <= int(mtp_drafts) < MAX_ROWS:
        raise ValueError(f"--mtp-drafts {mtp_drafts}: a round verifies at most {MAX_ROWS} rows, so 0 to "
                         f"{MAX_ROWS - 1} drafts")
    import os

    os.environ["TF_TP_WORLD"] = str(int(tp))           # the split, weight loaders and engine read the world here
    if mtp_drafts is None:
        policy = DEFAULT_POLICY                        # adaptive MTP depth from the costs measured at start
    elif int(mtp_drafts) == 0:
        policy = "0"                                   # every round decodes one token (the serial reference)
    else:
        policy = str(int(mtp_drafts))
    return BailingEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                         context=options.get("context"), context_explicit=options.get("context_explicit"),
                         serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":                # imported on first use, so discovery never loads the CUDA server
        from .cuda.app import LingApp

        return LingApp
    raise AttributeError(name)
