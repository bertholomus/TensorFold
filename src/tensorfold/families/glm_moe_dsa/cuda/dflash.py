"""A DFlash2 drafter the operator brings (``--drafter DIR``) on GLM-5.3: Flash's drafter over this family's taps,
embedding and head.

The drafter reads the residual after its ``target_layer_ids`` (forward.tap) for every committed row and drafts a block
in one pass; the engine verifies the block like MTP drafts, so replies stay the serial reference's. Nothing here ships
or picks a drafter: any checkpoint in the DFlash2 layout trained for GLM-5.3 (hidden 6,144, the 154,880-token
vocabulary) attaches; its licence is the operator's to keep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tensorfold.families.glm5_next.cuda.dflash2 import Drafter

# the drafter's linears: 4-bit copies (4, Flash's default) or BF16 as stored (0); some drafters lose acceptance in 4 bits
BITS = int(os.environ.get("TF_GLM_DRAFTER_BITS") or 4)
# drafts a round: at most this many of the block's (block - 1 when 0), cut once their chained probability falls under
# TF_GLM_DFLASH_CONF (default 0.3; 0: never). On TP4 (DCP4 + Q5, ~8-9k-token prompts, modal-labs' drafter in 4 bits)
# 7 drafts cut at 0.3 decoded code / prose at 35.1 / 33.4 tok/s, uncut 28.2 / 26.1 (MTP-3 on the same lane 34.3 / 31.2)
DEPTH = int(os.environ.get("TF_GLM_DFLASH_DEPTH") or 0)
CONF = float(os.environ.get("TF_GLM_DFLASH_CONF") or 0.3)


def check(draft_dir: str | Path, cfg, world: int) -> dict:
    """The drafter's config.json, refused with the reason when it cannot draft for this checkpoint on ``world`` ranks."""

    path = Path(draft_dir)
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise ValueError(f"--drafter {draft_dir}: a DFlash2 drafter directory holds config.json and model.safetensors")
    t = json.loads((path / "config.json").read_text())
    dc = t.get("dflash_config") or {}

    def no(why: str) -> ValueError:
        return ValueError(f"--drafter {draft_dir} cannot draft for GLM-5.3 here: {why}")

    if "DFlash2DraftModel" not in (t.get("architectures") or []) or "selector_top_k" not in dc:
        raise no("only the DFlash2 layout (DFlash2DraftModel: dynamic convolutions and a candidate selector) is read, "
                 f"not {t.get('architectures')}")
    if int(t["hidden_size"]) != cfg.hidden:
        raise no(f"its hidden size {t['hidden_size']} is not the target's {cfg.hidden}")
    if int(t.get("vocab_size") or cfg.vocab) != cfg.vocab:
        raise no(f"its vocabulary {t['vocab_size']} is not the target's {cfg.vocab}")
    layers = [int(i) for i in dc.get("target_layer_ids") or []]
    if not layers or min(layers) < 0 or max(layers) >= cfg.layers:
        raise no(f"its target layers {layers} are not all among the target's {cfg.layers}")
    if int(dc.get("conv_kernel_size", 2)) != 2:
        raise no("only two-tap convolutions are ported")
    if not 0 <= int(dc["mask_token_id"]) < cfg.vocab:
        raise no(f"its mask token {dc['mask_token_id']} is outside the vocabulary")
    for key in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if int(t[key]) % world:
            raise no(f"{key} {t[key]} does not split over {world} ranks")
    if int(t.get("sliding_window") or 0) <= 0:
        raise no("a drafter without a sliding window would read the whole context; only windowed drafters are ported")
    if float(dc.get("output_multiplier", 1.0)) != 1.0:
        raise no(f"an output multiplier of {dc['output_multiplier']} is not ported (1.0 is)")
    if not 2 <= int(dc["block_size"]) <= 8:
        raise no(f"its block of {dc['block_size']} is past the 8-row verify windows")
    return t


def need_rows(t: dict) -> int:
    """Prompt rows whose taps the drafter's first block can read: its window and block."""

    return int(t["sliding_window"]) + int(t["dflash_config"]["block_size"])


def prompt_from(drafter: Drafter, length: int) -> int:
    """The first prompt position a block pass after a ``length``-token prompt reads (its window before the block)."""

    return max(0, length - drafter.window - 1)


def weights_estimate(draft_dir: str | Path, world: int, bits: int = BITS):
    """The drafter's bytes a rank (Flash's 4-bit estimate, or with BF16 linears 2 bytes a weight)."""

    from tensorfold.cuda.capacity import Weights, headers
    from tensorfold.cuda.geometry import dflash2_weights

    q4 = dflash2_weights(draft_dir, world)
    if bits == 4:
        return q4
    total = 0
    for name, info in headers(draft_dir).items():
        n = 1
        for x in info["shape"]:
            n *= int(x)
        split = name.startswith("layers.") and name.endswith("_proj.weight")     # attention and MLP: a rank's share
        twice = name.endswith(("k_proj.weight", "v_proj.weight"))               # in qkv, and kv for context rows
        total += (n // world if split else n) * (4 if name.endswith("_codebook") else 2) * (2 if twice else 1)
    return Weights(total, q4.staging, 0)


class GlmDrafter(Drafter):
    """Flash's DFlash2 drafter with GLM-5.3's embedding (forward.embed: each rank's span, gathered) and EXL3 head.

    ``buf``: the decode window buffers whose gather and EXL3 scratch the block pass borrows (the block runs between
    forwards, never during one)."""

    DIRECT_READ = True

    def __init__(self, draft_dir: str | Path, w, buf, *, capacity: int, bits: int = BITS) -> None:
        super().__init__(draft_dir, w, capacity=capacity, ring=True, bits=bits)     # (short windows: a flat buffer)
        self.buf = buf
        self.logits = torch.empty((self.block - 1, w.head.n), dtype=torch.bfloat16, device=self.dev)
        self.depth = min(DEPTH or self.block - 1, self.block - 1)
        self.confidence = CONF

    def _embed(self, x: torch.Tensor) -> None:
        from . import forward as fwd

        fwd.embed(self.w, self.buf, self.ids, x)

    def _head(self, h: torch.Tensor, hs: torch.Tensor) -> tuple[torch.Tensor, int]:
        from . import forward as fwd

        return fwd.mm(self.buf, h, self.w.head, hs, self.logits), self.w.vocab_offset
