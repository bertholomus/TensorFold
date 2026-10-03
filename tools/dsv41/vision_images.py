"""More test images (PIL-drawn, ours): a bar chart with labels, coloured shapes, a receipt; copies the small ones too."""
import shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for n in ("red", "text", "tall", "wide", "tiny"):
    shutil.copy(src / f"vision_{n}.png", dst / f"vision_{n}.png")


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


img = Image.new("RGB", (560, 400), "white"); d = ImageDraw.Draw(img)
for i, (label, h, c) in enumerate([("Mon", 120, "steelblue"), ("Tue", 260, "orange"), ("Wed", 180, "seagreen")]):
    x = 80 + i * 150
    d.rectangle((x, 340 - h, x + 90, 340), fill=c); d.text((x + 20, 350), label, fill="black", font=font(28))
d.text((150, 20), "Sales per day", fill="black", font=font(32))
img.save(dst / "vision_chart.png")
img = Image.new("RGB", (500, 380), (235, 235, 235)); d = ImageDraw.Draw(img)
d.ellipse((40, 60, 200, 220), fill="red"); d.rectangle((260, 80, 440, 240), fill="blue")
d.polygon([(150, 360), (250, 250), (350, 360)], fill="yellow")
img.save(dst / "vision_shapes.png")
img = Image.new("RGB", (420, 520), "white"); d = ImageDraw.Draw(img)
lines = ["CORNER CAFE", "", "Latte        4.50", "Croissant    3.25", "Water        1.00", "", "TOTAL        8.75"]
for i, t in enumerate(lines):
    d.text((40, 40 + i * 60), t, fill="black", font=font(30))
img.save(dst / "vision_receipt.png")
print(sorted(p.name for p in dst.glob("*.png")))
