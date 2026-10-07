"""DeepSeek-V4.1-Flash (model_type ``deepseek_v41``): our TP2 CUDA engine for the EXL3 checkpoint on two GB10s.

Clean-room: the math follows DeepSeek's own MIT inference code (model.py, engram.py) and tech report; the TP split,
caches, collectives and kernels are ours (see tools/dsv41/DESIGN.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("deepseek_v41",)
TITLE = "DeepSeek-V4.1-Flash"
CUDA_VISION = True          # --vision: the tower and image spans are this family's own (cuda/vision.py)
LANES = False
MODELS = ("Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw",)
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"
DRAFTER = ""                    # DSpark blocks ship inside the checkpoint (mtp.*)
KERNEL_PACKAGE = "tensorfold.families.deepseek_v41.cuda"
KERNEL_VERSION = "v1"
KERNEL_DEPENDENCIES = ()


def check(model_dir: str | Path) -> None:
    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"{TITLE}'s CUDA engine reads the EXL3 conversion ({MODELS[0]}). {OWN_MODEL_HELP}")
    from .config import Cfg

    Cfg.from_dict(config)


def engine_settings(model: Any) -> dict[str, Any]:
    return {"max_rows": int(getattr(model, "max_rows", 8)), "max_draft": int(getattr(model, "max_rows", 8)) - 1}


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 2, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, mtp_drafts: int | None = None, **options: Any):
    import os

    # fp32 GEMMs stay fp32 (GitHub #6): NVIDIA's PyTorch containers set TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which runs
    # every fp32 cuBLAS GEMM in TF32, here the prompt chunks' router logits, indexer weights and mHC mixes: other prompt
    # bits than the reference and the gates, and so other replies for a seed. torch reads the switch at its first cuBLAS
    # handle, after this; fp32_gemms() checks that it took.
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0") not in ("", "0"):
        print("[tensorfold] TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (NVIDIA's PyTorch container default) set to 0: "
              f"{TITLE}'s fp32 GEMMs stay fp32", flush=True)
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    from .cuda.engine import DsEngine

    if int(tp) not in (1, 2):
        raise ValueError(f"{TITLE} runs on 1 or 2 ranks, not {tp}")
    os.environ["TF_TP_WORLD"] = str(int(tp))
    if int(tp) > 1 and not master:
        raise ValueError(f"--tp {tp} needs --master: rank 0's address on the link between the machines")
    drafts = 0 if no_drafts else (3 if mtp_drafts is None else int(mtp_drafts))
    engine = DsEngine(Path(model_dir), rank=int(rank), world=int(tp), master=master, port=int(master_port),
                      drafts=drafts, context=options.get("context"), engram_dir=os.environ.get("TF_DS_ENGRAM") or None,
                      vision=bool(options.get("vision")), vision_urls=bool(options.get("vision_urls")),
                      parallel=int(options.get("parallel") or 1))
    fp32_gemms()
    return engine


def fp32_gemms() -> None:
    """Raise when this process's fp32 cuBLAS GEMMs run in TF32 (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 was read before
    cuda_engine set it to 0): 4096 terms of (1 + 2^-12)^2 sum to 4098 in fp32; TF32 rounds 1 + 2^-12 to 1 (4096)."""

    import torch

    a = torch.full((256, 4096), 1 + 2 ** -12, dtype=torch.float32, device="cuda")
    got = float((a @ a[:128].t())[0, 0])
    if got != 4098.0:
        raise RuntimeError(f"fp32 GEMMs run in TF32 in this process (the check summed to {got}, fp32 gives 4098.0): "
                           "start the server with TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0")


def __getattr__(name: str) -> Any:
    if name == "CUDA_APP":
        from .cuda.app import DsApp

        return DsApp
    raise AttributeError(name)
