"""GLM-5.3 (model_type ``glm_moe_dsa``): the full 355B MoE, served by the CUDA engine over TP ranks 1/2/4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("glm_moe_dsa",)
TITLE = "GLM-5.3"
LANES = False
# TensorFold's only EXL3 read path: 4-bit mcg-codebook routed experts, BF16 attention, dense MLPs,
# shared experts, router and vocabulary head. Modeled on Mia's GLM-5.3-Flash EXL3 conversion.
MODELS = ("Mia-AiLab/GLM-5.3-EXL3-TR3-4bpw-routed",)
KERNEL_PACKAGE = "tensorfold.kernels.glm.flash.v1"   # the shared GLM CUDA kernels (qmm, glue, experts)
KERNEL_DEPENDENCIES = ("tensorfold.families.glm5_next.cuda.qmm", "tensorfold.families.glm5_next.cuda.glue")
KERNEL_VERSION = "v1"
QUANT_METHODS = {"cuda": ("exl3",)}
# 4-bit trellis, the "mcg" codebook, routed experts only (the same variant the Flash family's CUDA engine reads)
EXL3_VARIANT = {"bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}
# the draft model is optional: without it the engine drafts with the checkpoint's own MTP head
DRAFTER = ""


def check(model_dir: str | Path) -> None:
    """Refuse what the engine does not read: anything but EXL3 4-bit mcg routed experts with BF16 elsewhere."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"GLM-5.3's CUDA engine reads EXL3 checkpoints with 4-bit mcg-codebook routed experts "
                         f"and BF16 elsewhere ({MODELS[0]}); this checkpoint is not EXL3-quantized. {OWN_MODEL_HELP}")
    from tensorfold.families.glm_moe_dsa.config import Config

    cfg = Config.from_dict(config)                     # the architecture checks (rope, indexer, routing)
    from tensorfold.cuda.exl3 import format as exl3_format

    fields = exl3_format.require_config(config, where="NVIDIA GPUs (CUDA)", tested=", ".join(MODELS),
                                        help=OWN_MODEL_HELP)
    if int(fields.get("bits") or 4) != EXL3_VARIANT["bits"] \
            or str(fields.get("codebook") or "mcg").lower() != EXL3_VARIANT["codebook"]:
        raise ValueError(f"GLM-5.3's CUDA engine reads EXL3 checkpoints with 4-bit mcg-codebook routed experts "
                         f"and BF16 elsewhere ({MODELS[0]}); this one states "
                         f"{fields.get('bits')} bits with the {fields.get('codebook')!r} codebook. {OWN_MODEL_HELP}")
    missing = sorted(set(cfg.missing_tensors({"_model_dir": str(model_dir)})))
    if missing:
        raise ValueError(f"this checkpoint lacks {len(missing)} tensor(s) the architecture needs, {missing[0]} "
                         f"first. {OWN_MODEL_HELP}")
    print(f"[tensorfold] GLM-5.3 runs on NVIDIA GPUs with world-parameterized tensor parallelism "
          f"(--tp 2 or --tp 4, one GPU per machine): pull the checkpoint on every rank and serve with the "
          f"same --tp, --rank and --master on each (TF_TP_WORLD selects the split)", flush=True)


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed experts (everything a rank-folder split keeps on disk)."""

    import json

    from tensorfold.streaming.checkpoint import tensor_bytes

    def routed(name: str) -> bool:
        return ".mlp.experts." in name

    return tensor_bytes(Path(model_dir), routed)


def engine_settings(model: Any) -> dict[str, Any]:
    """Rows a round verifies at most: the widest window captured as a graph at startup."""

    return {"max_rows": int(getattr(model, "max_rows", 8)), "max_draft": int(getattr(model, "max_rows", 8)) - 1}


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 2, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    """Build the world-parameterized engine (TF_TP_WORLD ranks of 1, 2 or 4) with MTP drafting or serial decoding."""

    from tensorfold.families.glm_moe_dsa.cuda.engine import GlmEngine

    if int(tp) not in (1, 2, 4):
        raise ValueError(f"GLM-5.3 splits over 1, 2 or 4 GPUs, not {tp}: run the same `tensorfold serve` command "
                         "with --tp W --rank R --master ADDRESS on every rank (the others start first)")
    import os

    if int(tp) != int(os.environ.get("TF_TP_WORLD", str(tp))):
        os.environ["TF_TP_WORLD"] = str(int(tp))       # the split and weight loaders read the world from here
    if int(tp) > 1 and not master:
        raise ValueError(f"--tp {tp} needs --master: rank 0's address on the link between the machines")
    if mtp_drafts is None:
        policy = "auto"
    elif int(mtp_drafts) == 0:
        policy = "0"                                   # every round decodes one token (the serial reference)
    else:
        policy = str(int(mtp_drafts))
    return GlmEngine(Path(model_dir), rank=int(rank), master=master, port=int(master_port), policy=policy,
                     drafter=Path(drafter) if drafter and not no_drafts else None,
                     context=options.get("context"), context_explicit=options.get("context_explicit"),
                     serial_only=bool(no_drafts))


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":                # imported on first use, so discovery never loads the CUDA server
        from .cuda.app import GlmApp

        return GlmApp
    raise AttributeError(name)
