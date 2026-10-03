"""The engine's vision tower against DeepSeek's reference (inference/vision.py + image_processor.py), one GPU: the same
checkpoint weights in both, the same test images; patches, grids and the image-span rows compared bit for bit.

  python3 vision_check.py --model M --ref <folder holding DeepSeek's vision.py and image_processor.py> --out F
Also writes the test images (PNG) next to F, for the HTTP checks.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch


def test_images():
    from PIL import Image, ImageDraw, ImageFont

    out = {}
    out["red"] = Image.new("RGB", (400, 300), (220, 20, 20))
    img = Image.new("RGB", (640, 240), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default(size=72)
    d.text((30, 70), "OPEN 24 HOURS", fill=(0, 0, 0), font=font)
    out["text"] = img
    g = Image.new("RGB", (1500, 1000))
    px = g.load()
    for y in range(1000):
        for x in range(0, 1500, 3):
            c = (x * 255 // 1500, y * 255 // 1000, 128)
            px[x, y] = c
            if x + 1 < 1500:
                px[x + 1, y] = c
            if x + 2 < 1500:
                px[x + 2, y] = c
    d = ImageDraw.Draw(g)
    d.ellipse((500, 300, 900, 700), fill=(30, 160, 40))
    out["gradient_circle"] = g
    out["tall"] = Image.new("RGB", (120, 2000), (10, 60, 200))
    out["wide"] = Image.new("RGB", (3000, 90), (250, 200, 0))
    out["tiny"] = Image.new("RGB", (16, 16), (0, 0, 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda.vision import Tower, VisionConfig, decode, span_types

    cfgj = json.loads((Path(a.model) / "config.json").read_text())
    cfg = VisionConfig.read(cfgj)
    tower = Tower(cfg, a.model, attention="math")         # the reference's attention call: bit for bit
    fast = Tower(cfg, a.model)                            # the served tower (memory-efficient kernel, fp32)
    truth = Tower(cfg, a.model, attention="bf16")         # the same kernel on bf16 (how far that would be)
    sys.path.insert(0, a.ref)
    import image_processor as ip
    import vision as rv

    class Args:
        vision_n_layers, vision_dim, vision_n_heads = cfg.layers, cfg.dim, cfg.heads
        vision_inter_dim, vision_patch_size, vision_rope_theta = cfg.inter, cfg.patch, cfg.theta
        vision_downsample_ratio, vision_max_n_token, vision_min_pixels = cfg.ratio, cfg.max_tokens, cfg.min_pixels
        vision_max_wh_ratio, dim, vision_enabled = cfg.max_wh_ratio, cfg.model_dim, True

    out_dir = Path(a.out).parent
    ours = {}
    for name, img in test_images().items():          # the engine's path first, in the engine's defaults
        buf = io.BytesIO()
        img.save(buf, "PNG")
        data = buf.getvalue()
        (out_dir / f"vision_{name}.png").write_bytes(data)
        pic = decode(data, cfg)
        ours[name] = (data, img.size, pic, tower.span_rows(pic), fast.span_rows(pic), truth.span_rows(pic))
    torch.set_default_dtype(torch.bfloat16)            # the reference's generate.py runs this way
    torch.set_default_device("cuda")
    vit, aligner = rv.ViT(Args), rv.Aligner(Args)
    sd_v = {k[len("vision."):]: v for k, v in tower.w.items() if k.startswith("vision.")}
    sd_a = {k[len("aligner."):]: v for k, v in tower.w.items() if k.startswith("aligner.")}
    missing = vit.load_state_dict(sd_v, strict=True), aligner.load_state_dict(sd_a, strict=True)
    vit.eval(), aligner.eval()
    rows = []
    for name, (data, size, pic, mine, served, exact) in ours.items():
        r_patches, r_vh, r_vw, r_lh, r_lw = ip.load_image({"data": data}, Args)
        same_prep = (bool(torch.equal(pic.patches, r_patches.cpu())) and
                     (pic.n_vit_h, pic.n_vit_w, pic.n_llm_h, pic.n_llm_w) == (r_vh, r_vw, r_lh, r_lw))
        with torch.inference_mode():
            feats = aligner(vit(r_patches.cuda(), r_vh, r_vw), r_vh, r_vw)
            types = torch.tensor(span_types(r_lh, r_lw), device="cuda")
            ref = torch.empty_like(mine)
            ref[types == 0] = tower.w["image_start"]
            ref[types == 3] = tower.w["image_end"]
            ref[types == 2] = tower.w["image_newline"]
            ref[types == 1] = feats.to(ref.dtype)
        diff = (mine.float() - ref.float()).abs().max().item()
        sf, rf = served.float(), ref.float()
        cos = torch.nn.functional.cosine_similarity(sf, rf, dim=-1)
        served_vs_ref = {"max_abs_diff": round((sf - rf).abs().max().item(), 5),
                         "rel_l2": round(((sf - rf).norm() / rf.norm()).item(), 6),
                         "min_row_cosine": round(cos.min().item(), 6),
                         "deterministic": bool(torch.equal(served, fast.span_rows(pic)))}
        tf = exact.float()
        vs_fp32 = {"bf16_kernel_vs_ref_rel_l2": round(((tf - rf).norm() / rf.norm()).item(), 6)}
        rows.append({"image": name, "size": list(size), "vit_grid": [pic.n_vit_h, pic.n_vit_w],
                     "llm_grid": [pic.n_llm_h, pic.n_llm_w], "span_tokens": pic.tokens, "preprocessing_equal": same_prep,
                     "span_rows_bit_equal": bool(torch.equal(mine, ref)), "max_abs_diff": diff,
                     "served_kernel_vs_ref": served_vs_ref, "bf16_kernel": vs_fp32})
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"load": str(missing), "rows": rows, "all_bit_equal": all(r["span_rows_bit_equal"] for r in rows)},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
