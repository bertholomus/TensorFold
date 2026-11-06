"""Mistral-Large-4 (model_type ``mistral_large4_text`` / ``mistral_large4``): the EXL3 pack read by this repo's
own CUDA engine on four Spark ranks (EP4 experts + DP4 attention, notes D2 and D3).

Phase 1 was the family registration and the pack loader (``cuda/weights.py``, D3 §1-§4). Phase 2a added the
eager forward (``cuda/model.py``, D1 §1-§10). This branch's phase 2b is the serving lane: ``cuda/engine.py``
(rank wiring, KV pool sizing), ``cuda/a2a.py`` (the two-phase MoE all-to-all), ``cuda/dp4.py`` (owner
placement, per-rank step payloads, the digest agreement and the rank-0/follower link, D2 §1-§5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("mistral_large4_text", "mistral_large4")
TITLE = "Mistral-Large-4"
LANES = False
MODELS = ()                        # a local pack path; no download name listed here
QUANT_METHODS = {"cuda": ("exl3",)}
EXL3_VARIANT = "any"               # one recipe: mul1 codebook, 2.5/5.0/6.0 bpw (D3 §4 checks the manifest)
DRAFTER = ""                       # the DSpark draft is not part of the text pack (PACK-LAYOUT.md)
KERNEL_PACKAGE = "tensorfold.families.mistral_large4.cuda"
KERNEL_VERSION = "v0"
KERNEL_DEPENDENCIES = ()
CUDA_MAX_RANKS = 4                 # the EP4 lane (D2 §1): cli gates --tp above this for this family

#: the phase-1 refusal, kept importable for the loader test's history check (cuda_engine serves since phase 2b)
PHASE1_MESSAGE = (
    "Mistral-Large-4 phase 1 is the family skeleton and the pack loader only: the CUDA engine (EP4 expert "
    "all-to-all + DP-attention, note D2) and the MLA/ssmax/NoPE/indexer math (note D1) land in phase 2 on this "
    "branch. Load the pack with tensorfold.families.mistral_large4.cuda.weights.load() to check it.")


def check(model_dir: str | Path) -> None:
    """The checkpoint is an EXL3 pack whose config passes ``Cfg`` (all reference asserts, D1 §12)."""

    from tensorfold.families import OWN_MODEL_HELP, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) != "exl3":
        raise ValueError(f"{TITLE}'s CUDA engine reads the ML4 EXL3 pack (quant/pack_ml4.py output). {OWN_MODEL_HELP}")
    from .config import Cfg

    Cfg.from_dict(config)


def cuda_engine(model_dir: str | Path, *, drafter: str = "", tp: int = 1, rank: int = 0, master: str = "",
                master_port: int = 29551, no_drafts: bool = False, **options: Any):
    """The family-owned rank dispatch (D1 §Extension points): the cli passes tp/rank/master through; the
    engine builds its own comm, loads this rank's pack slots, and rank>0 serves only through ``follow()``."""

    if int(tp) not in (1, 2, 3, 4):
        raise ValueError(f"{TITLE} runs on 1 to 4 ranks (the EP4 lane, note D2 §1), not {tp}")
    if int(tp) > 1 and not master:
        raise ValueError(f"--tp {tp} needs --master: rank 0's address on the link between the machines")
    if drafter:
        raise ValueError(f"{TITLE} has no drafter (the DSpark draft is not part of the text pack, PACK-LAYOUT.md)")
    from .cuda.engine import Ml4Engine

    return Ml4Engine(Path(model_dir), rank=int(rank), world=int(tp), master=master, port=int(master_port),
                     context=options.get("context"))
