"""DeepSeek-V4.1's image input: preprocessing, the ViT and the aligner on rank 0, and the image spans of a prompt.

Re-implemented from reading DeepSeek's MIT reference (``inference/vision.py``, ``inference/image_processor.py`` and the
image paths of ``inference/model.py``); the ops and their order are the reference's, so the tower's output is its
output. An image becomes an ``n_vit_h x n_vit_w`` patch grid (14 px) and, after the aligner's 3 x 3 merge, an
``n_llm_h x n_llm_w`` token grid, laid out in the prompt as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every span position carries ``image_token_id``; the delimiters take learned rows, the IMAGE slots the aligner's rows in
reading order. Inside a span the MoE gate selects experts with its VL bias and Engram is shut (see model.py).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
PLACEHOLDER = "<｜deepseek_image｜>"


@dataclass(frozen=True)
class VisionConfig:
    layers: int
    dim: int
    heads: int
    inter: int
    patch: int
    theta: float
    ratio: int
    max_tokens: int
    min_pixels: int
    max_wh_ratio: float | None
    image_token_id: int
    model_dim: int

    @staticmethod
    def read(config: dict) -> "VisionConfig | None":
        v = config.get("vision_config")
        if not v or not int(v.get("num_hidden_layers") or 0):
            return None
        return VisionConfig(int(v["num_hidden_layers"]), int(v["hidden_size"]), int(v["num_attention_heads"]),
                            int(v["intermediate_size"]), int(v["patch_size"]), float(v.get("rope_theta", 10000.0)),
                            int(v["downsample_ratio"]), int(v["max_image_tokens"]), int(v["min_pixels"]),
                            v.get("max_wh_ratio"), int(config["image_token_id"]),
                            int((config.get("text_config") or config)["hidden_size"]))


# -- preprocessing (DeepSeek's image_processor) -----------------------------------------------------------------------
def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def _llm_grid(best_h: int, best_w: int, p: int, r: int) -> tuple[int, int]:
    return math.ceil((best_h // p) / r), math.ceil((best_w // p) / r)


def _solve_resize(h: float, w: float, p: int, r: int, max_n: int) -> tuple[int, int]:
    ratio = h / w
    max_w = math.sqrt((max_n - 2) / ratio + 0.25) - 0.5
    max_h = max_w * ratio
    cell = p * r
    if max_w < 1.0:
        return (max_n - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_n - 3) * cell
    beta = min(math.floor(max_w) * cell / w, math.floor(max_h) * cell / h)
    return math.floor(h * beta / p) * p, math.floor(w * beta / p) * p


def plan_grid(width: int, height: int, cfg: VisionConfig) -> tuple[int, int, int, int]:
    """(n_llm_h, n_llm_w, best_height, best_width) for an image of this size: a pure function of its arguments."""

    p, r = cfg.patch, cfg.ratio
    if cfg.max_wh_ratio is not None and width > height * cfg.max_wh_ratio:
        width = height * cfg.max_wh_ratio
    if 0 < width * height < cfg.min_pixels:
        scale = (cfg.min_pixels / (width * height)) ** 0.5
        width, height = int(width * scale), int(height * scale)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_h, n_w = _llm_grid(best_h, best_w, p, r)
    if num_image_tokens(n_h, n_w) > cfg.max_tokens:
        best_h, best_w = _solve_resize(height, width, p, r, cfg.max_tokens)
        n_h, n_w = _llm_grid(best_h, best_w, p, r)
    return n_h, n_w, best_h, best_w


@dataclass
class Picture:
    """One decoded image: its ViT patches and both grids."""

    patches: torch.Tensor          # [n_vit_h * n_vit_w, 3, p, p] bf16 (CPU)
    n_vit_h: int
    n_vit_w: int
    n_llm_h: int
    n_llm_w: int

    @property
    def tokens(self) -> int:
        return num_image_tokens(self.n_llm_h, self.n_llm_w)


def decode(data: bytes, cfg: VisionConfig, max_pixels: int = 64 << 20) -> Picture:
    import numpy as np
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = max_pixels
    with Image.open(io.BytesIO(data)) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_h, best_w = plan_grid(image.width, image.height, cfg)
    p = cfg.patch
    n_vit_h, n_vit_w = best_h // p, best_w // p
    if cfg.max_wh_ratio is not None and image.width >= cfg.max_wh_ratio * image.height:
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return Picture(patches.contiguous(), n_vit_h, n_vit_w, n_llm_h, n_llm_w)


def span_types(n_llm_h: int, n_llm_w: int) -> list[int]:
    return [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]


@dataclass
class VisionPrompt:
    """A prompt's image spans: ``token_ids`` (every span position ``image_token_id``), each span's start and picture.
    The server hands it to the engine as ``vision``; a continuation keeps the spans (they lie in the prompt)."""

    token_ids: list[int]
    spans: list[tuple[int, Picture]] = field(default_factory=list)

    def positions(self) -> list[int]:
        return [s + i for s, pic in self.spans for i in range(pic.tokens)]


# -- the tower (DeepSeek's vision.py) ---------------------------------------------------------------------------------
@lru_cache(8)
def _cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: str):
    # on the device, as the reference builds them (its default device is the GPU)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (w * x).to(dtype)


class Tower:
    """The ViT and the aligner with the checkpoint's bf16 weights (norm weights fp32, as the reference keeps them)."""

    def __init__(self, cfg: VisionConfig, model_dir: str | Path, device: str = "cuda", attention: str | None = None):
        from .weights import Shards

        # attention over an image's patches. "fp32": the memory-efficient kernel on fp32 q, k, v (the reference's
        # values: its call on 3-D bf16 tensors runs PyTorch's math path, which computes in fp32 but holds the whole
        # score matrix, ~10 GiB at 994 image tokens); "math": that very call (checks only); "bf16": the kernel on bf16
        # (its probabilities in bf16: 2-13 % from the reference's rows on our test images)
        import os

        self.cfg, self.device = cfg, device
        self.attention = attention or os.environ.get("TF_DS_VISION_ATTENTION") or "fp32"
        if self.attention not in ("fp32", "math", "bf16"):
            raise ValueError(f"TF_DS_VISION_ATTENTION is fp32, math or bf16, not {self.attention!r}")
        sh = Shards(model_dir)
        names = [n for n in sh.entries if n.startswith(("vision.", "aligner.")) or n in
                 ("image_start", "image_end", "image_newline")]
        if not names:
            raise ValueError("this checkpoint holds no vision tower (vision.* tensors)")
        # a fresh host copy before the device copy: a copy straight out of the read buffer crawled (64 KiB pages)
        self.w = {}
        for n in names:
            t = sh.get(n).clone()
            if n.endswith(("norm1.weight", "norm2.weight")) or n == "vision.norm.weight":
                t = t.float()
            self.w[n] = t.to(device)
        sh.close()
        self.bytes = sum(t.numel() * t.element_size() for t in self.w.values())

    def _block(self, i: int, x: torch.Tensor, cos, sin) -> torch.Tensor:
        w, c = self.w, self.cfg
        p = f"vision.blocks.{i}."
        n = x.size(0)
        hd = c.dim // c.heads
        h = _rms(x, w[p + "norm1.weight"])
        q, k, v = (t.view(n, c.heads, hd) for t in F.linear(h, w[p + "attn.wqkv.weight"],
                                                              w[p + "attn.wqkv.bias"]).chunk(3, dim=-1))
        q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
        if self.attention == "math":
            o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            dt = torch.float32 if self.attention == "fp32" else q.dtype
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                o = F.scaled_dot_product_attention(q.transpose(0, 1)[None].to(dt), k.transpose(0, 1)[None].to(dt),
                                                   v.transpose(0, 1)[None].to(dt))[0].to(q.dtype)
        x = x + F.linear(o.transpose(0, 1).reshape(n, -1), w[p + "attn.wo.weight"], w[p + "attn.wo.bias"])
        h = _rms(x, w[p + "norm2.weight"])
        gate, up = F.linear(h, w[p + "mlp.w1.weight"]).chunk(2, dim=-1)
        return x + F.linear(F.silu(gate) * up, w[p + "mlp.w2.weight"])

    @torch.inference_mode()
    def encode(self, pic: Picture) -> torch.Tensor:
        """The aligner's rows [n_llm_h * n_llm_w, model_dim] bf16, in reading order."""

        c, w = self.cfg, self.w
        x = F.linear(pic.patches.to(self.device).flatten(1), w["vision.patch_embed.proj.weight"],
                     w["vision.patch_embed.proj.bias"])
        cos, sin = _cos_sin(pic.n_vit_h, pic.n_vit_w, c.dim // c.heads // 2, c.theta, self.device)
        for i in range(c.layers):
            x = self._block(i, x, cos, sin)
        x = _rms(x, w["vision.norm.weight"])
        r = c.ratio
        x = x.view(pic.n_vit_h, pic.n_vit_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -pic.n_vit_w % r, 0, -pic.n_vit_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return F.linear(F.gelu(F.linear(x, w["aligner.w1.weight"], w["aligner.w1.bias"])),
                        w["aligner.w2.weight"], w["aligner.w2.bias"])

    @torch.inference_mode()
    def span_rows(self, pic: Picture) -> torch.Tensor:
        """Rows [pic.tokens, model_dim] bf16 for one image span: the learned delimiters and the aligner's rows."""

        types = torch.tensor(span_types(pic.n_llm_h, pic.n_llm_w), device=self.device)
        rows = torch.empty((types.numel(), self.cfg.model_dim), dtype=torch.bfloat16, device=self.device)
        rows[types == IMAGE_START] = self.w["image_start"].to(rows.dtype)
        rows[types == IMAGE_END] = self.w["image_end"].to(rows.dtype)
        rows[types == IMAGE_NEW_LINE] = self.w["image_newline"].to(rows.dtype)
        rows[types == IMAGE] = self.encode(pic).to(rows.dtype)
        return rows


def load_image_bytes(url: str, allow_urls: bool, max_bytes: int) -> bytes:
    """An image_url's bytes: a base64 data URL, or (``allow_urls``) an http(s) URL."""

    import base64
    import urllib.request

    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        if ";base64" not in header:
            raise ValueError("image data URLs must be base64")
        data = base64.b64decode(payload, validate=False)
    elif url.startswith(("http://", "https://")):
        if not allow_urls:
            raise ValueError("image URLs are off on this server (send a base64 data URL)")
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read(max_bytes + 1)
    else:
        raise ValueError("image_url must be a data: URL" + (" or an http(s) URL" if allow_urls else ""))
    if len(data) > max_bytes:
        raise ValueError(f"an image is larger than {max_bytes} bytes")
    return data


# -- the server's side: image_url parts -> DeepSeek's prompt layout -------------------------------------------------
class DsVision:
    """The engine's ``vision``: turns a chat's image_url parts (user turns and tool results) into DeepSeek's encoding
    (each image a placeholder text block, a message's blocks joined by blank lines), and the rendered prompt's
    placeholders into image spans."""

    def __init__(self, cfg: VisionConfig, *, allow_urls: bool = False, max_images: int = 8,
                 max_bytes: int = 20 << 20):
        self.cfg, self.allow_urls, self.max_images, self.max_bytes = cfg, allow_urls, max_images, max_bytes
        self.videos = False

    @staticmethod
    def has_images(messages) -> bool:
        return any(isinstance(m, dict) and isinstance(m.get("content"), list)
                   and any(isinstance(p, dict) and p.get("type") in ("image_url", "image") for p in m["content"])
                   for m in messages or [])

    def split(self, messages: list) -> tuple[list, list[Picture]]:
        """Messages with every image part replaced by the placeholder (content as one string, DeepSeek's join) and
        the decoded pictures in prompt order. ValueError for anything DeepSeek's encoding refuses."""

        out, pictures = [], []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, str) and PLACEHOLDER in content:
                raise ValueError("message text must not contain the image placeholder token")
            if not isinstance(content, list):
                out.append(m)
                continue
            role = m.get("role")
            texts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind in ("image_url", "image"):
                    if role not in ("user", "tool"):
                        raise ValueError("images are accepted only in user messages and tool results")
                    if len(pictures) >= self.max_images:
                        raise ValueError(f"a request may carry at most {self.max_images} images")
                    url = part.get("image_url")
                    url = url.get("url") if isinstance(url, dict) else url
                    if not isinstance(url, str) or not url:
                        raise ValueError("an image_url part needs a url")
                    pictures.append(decode(load_image_bytes(url, self.allow_urls, self.max_bytes), self.cfg))
                    texts.append(PLACEHOLDER)
                elif kind == "text":
                    text = part.get("text")
                    if not isinstance(text, str):
                        raise ValueError("a text part needs a text string")
                    if PLACEHOLDER in text:
                        raise ValueError("message text must not contain the image placeholder token")
                    texts.append(text)
                else:
                    raise ValueError("content parts must be text or image_url")
            out.append({**m, "content": "\n\n".join(texts)})
        return out, pictures

    def expand(self, ids: list[int], pictures: list[Picture]) -> VisionPrompt:
        """The rendered prompt's placeholder tokens, each grown into its image's span."""

        tid = self.cfg.image_token_id
        if sum(1 for t in ids if t == tid) != len(pictures):
            raise ValueError("the rendered prompt does not hold one placeholder per image")
        tokens, spans, it = [], [], iter(pictures)
        for t in ids:
            if t != tid:
                tokens.append(t)
                continue
            pic = next(it)
            spans.append((len(tokens), pic))
            tokens += [tid] * pic.tokens
        return VisionPrompt(tokens, spans)

    def continued(self, prompt: VisionPrompt, ids: list[int]) -> VisionPrompt:
        """A continuation of ``prompt`` (the same spans; ``ids`` extends its tokens)."""

        if list(ids[:len(prompt.token_ids)]) != list(prompt.token_ids):
            raise ValueError("a continuation must extend the image prompt")
        return VisionPrompt(list(ids), prompt.spans)
