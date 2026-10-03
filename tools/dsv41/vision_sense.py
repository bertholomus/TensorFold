"""How sensitive the vision tower is: the reference path's rows against itself with a 1-ulp input change, and per-layer
divergence between the reference attention call and the fp32 fused kernel (one GPU)."""
import io, json, sys
from pathlib import Path
import torch
from tensorfold.families.deepseek_v41.cuda.vision import Tower, VisionConfig, decode, _cos_sin, _rms

model = sys.argv[1]; images = Path(sys.argv[2])
cfg = VisionConfig.read(json.loads((Path(model) / "config.json").read_text()))
A, B = Tower(cfg, model, attention="math"), Tower(cfg, model, attention="fp32")
B.w = A.w
out = {}
for name in ("red", "text", "gradient_circle"):
    pic = decode((images / f"vision_{name}.png").read_bytes(), cfg)
    base = A.span_rows(pic).float()
    p2 = pic.patches.clone(); flat = p2.view(-1); flat[12345] = torch.nextafter(flat[12345].float(), torch.tensor(9.0)).to(flat.dtype)
    if torch.equal(flat[12345], pic.patches.view(-1)[12345]):
        flat[12345] = (flat[12345].float() + 0.01).to(flat.dtype)
    pic2 = type(pic)(p2, pic.n_vit_h, pic.n_vit_w, pic.n_llm_h, pic.n_llm_w)
    pert = A.span_rows(pic2).float()
    # per-layer divergence of the two attention paths from the same input
    w = A.w
    x = torch.nn.functional.linear(pic.patches.cuda().flatten(1), w["vision.patch_embed.proj.weight"], w["vision.patch_embed.proj.bias"])
    cos, sin = _cos_sin(pic.n_vit_h, pic.n_vit_w, cfg.dim // cfg.heads // 2, cfg.theta, "cuda")
    xa, xb, per = x, x, []
    with torch.inference_mode():
        for i in range(cfg.layers):
            xa, xb = A._block(i, xa, cos, sin), B._block(i, xb, cos, sin)
            one = B._block(i, xa, cos, sin)                      # the fused kernel from the reference's own input
            per.append({"layer": i, "rel_l2_paths": round(((xa.float() - xb.float()).norm() / xa.float().norm()).item(), 6),
                        "rel_l2_one_step": round(((xa.float() - one.float()).norm() / xa.float().norm()).item(), 7),
                        "max_abs_x": round(xa.float().abs().max().item(), 1)})
    out[name] = {"one_ulp_input_change_rel_l2": round(((pert - base).norm() / base.norm()).item(), 6), "layers": per}
    print(name, out[name]["one_ulp_input_change_rel_l2"], [(p["layer"], p["rel_l2_one_step"], p["rel_l2_paths"], p["max_abs_x"]) for p in per[::4]], flush=True)
json.dump(out, open(images / "vision_sensitivity.json", "w"), indent=1)
