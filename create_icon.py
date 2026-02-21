"""
Generate the Agent Debate System icon.
Two glowing AI orbs (Astra=cyan, Nova=magenta) facing each other
with a lightning arc between them — on a deep-space dark background.

Outputs: resources/icon.ico  (multi-size: 16/32/48/64/128/256)
         resources/icon_256.png (for the shortcut preview)
"""
from __future__ import annotations

import math
import struct
import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError:
    raise SystemExit("Install Pillow first:  pip install Pillow")


# ── Palette ───────────────────────────────────────────────────────────────────
BG      = (8,   8,  20)          # near-black deep space
ASTRA   = (0,   229, 255)        # cyan  (Astra)
NOVA    = (213,  0,  249)        # magenta (Nova)
SPARK   = (255, 220,  60)        # yellow lightning
WHITE   = (255, 255, 255)
# ──────────────────────────────────────────────────────────────────────────────


def _lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _glow(draw: ImageDraw.ImageDraw, cx, cy, r, color, layers=8):
    """Paint concentric transparent circles to fake a glow bloom."""
    for i in range(layers, 0, -1):
        alpha = int(180 * (i / layers) ** 2)
        radius = int(r * (1 + (layers - i) * 0.28))
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        rgba = color + (alpha,)
        draw.ellipse(bbox, fill=rgba)


def _hard_circle(draw: ImageDraw.ImageDraw, cx, cy, r, color):
    """Solid bright core circle."""
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (255,))


def _lightning(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, w=3):
    """Jagged zig-zag lightning bolt between two points."""
    pts = [(x1, y1)]
    steps = 6
    for i in range(1, steps):
        t = i / steps
        mx = int(x1 + (x2 - x1) * t)
        my = int(y1 + (y2 - y1) * t)
        # alternate offset
        off = int((16 if i % 2 == 0 else -16) * (1 - abs(t - 0.5) * 2))
        pts.append((mx, my + off))
    pts.append((x2, y2))
    # Draw with glow layers
    for thickness, alpha in [(w + 8, 40), (w + 4, 100), (w + 2, 180), (w, 255)]:
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=SPARK + (alpha,), width=thickness)


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gfx = ImageDraw.Draw(img, "RGBA")

    S = size
    cx, cy = S // 2, S // 2

    # ── Background circle (dark navy with subtle radial) ──────────────────────
    for r in range(cx, 0, -1):
        t = r / cx
        c = _lerp_color((30, 30, 60), BG, t)
        gfx.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c + (255,))

    if size < 32:
        # Simplified for tiny sizes
        _hard_circle(gfx, cx, cy, cx - 1, ASTRA[:3] if True else NOVA[:3])
        return img

    # ── Grid/nebula dots (very dim) ────────────────────────────────────────────
    if size >= 128:
        import random
        rng = random.Random(42)
        for _ in range(size * 2):
            dx = rng.randint(0, S - 1)
            dy = rng.randint(0, S - 1)
            dist = math.hypot(dx - cx, dy - cy)
            if dist < cx * 0.92:
                br = rng.randint(60, 140)
                r_dot = 1 if size < 200 else rng.choice([1, 1, 1, 2])
                gfx.ellipse((dx-r_dot, dy-r_dot, dx+r_dot, dy+r_dot), fill=(br,br,br,rng.randint(30, 90)))

    # ── Orb positions ─────────────────────────────────────────────────────────
    orb_r    = int(S * 0.195)
    offset_x = int(S * 0.265)
    astra_cx = cx - offset_x
    nova_cx  = cx + offset_x
    orb_cy   = cy

    # ── Glow halos ────────────────────────────────────────────────────────────
    _glow(gfx, astra_cx, orb_cy, orb_r, ASTRA, layers=10)
    _glow(gfx, nova_cx,  orb_cy, orb_r, NOVA,  layers=10)

    # ── Orb cores ─────────────────────────────────────────────────────────────
    _hard_circle(gfx, astra_cx, orb_cy, orb_r, ASTRA)
    # inner bright highlight
    hl = max(3, orb_r // 4)
    _hard_circle(gfx, astra_cx - orb_r // 4, orb_cy - orb_r // 4, hl, WHITE)

    _hard_circle(gfx, nova_cx, orb_cy, orb_r, NOVA)
    _hard_circle(gfx, nova_cx - orb_r // 4, orb_cy - orb_r // 4, hl, WHITE)

    # ── Lightning arc between orbs ────────────────────────────────────────────
    lx1 = astra_cx + orb_r + 2
    lx2 = nova_cx  - orb_r - 2
    lw  = max(2, S // 64)
    _lightning(gfx, lx1, orb_cy, lx2, orb_cy, w=lw)

    # ── Labels "A" and "N" inside orbs ────────────────────────────────────────
    if size >= 64:
        font_size = max(10, orb_r)
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        def draw_label(x, y, text, bg):
            # Shadow
            gfx.text((x+1, y+1), text, fill=(0,0,0,180), font=font, anchor="mm")
            gfx.text((x, y), text, fill=WHITE+(255,), font=font, anchor="mm")

        draw_label(astra_cx, orb_cy, "A", ASTRA)
        draw_label(nova_cx,  orb_cy, "N", NOVA)

    # ── "AGENT DEBATE" text arc along bottom ─────────────────────────────────
    if size >= 128:
        label = "AGENT  DEBATE"
        fs = max(8, S // 22)
        try:
            tfont = ImageFont.truetype("arialbd.ttf", fs)
        except Exception:
            tfont = ImageFont.load_default()
        tw = gfx.textlength(label, font=tfont)
        tx = cx - tw // 2
        ty = int(S * 0.82)
        # subtle glow
        gfx.text((tx+1, ty+1), label, fill=(0,0,0,120), font=tfont)
        gfx.text((tx, ty), label, fill=(200, 220, 255, 210), font=tfont)

    # ── Thin outer ring ────────────────────────────────────────────────────────
    ring_r = cx - 2
    gfx.ellipse(
        (cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
        outline=(80, 80, 130, 140),
        width=max(1, S // 120),
    )

    return img


def main():
    out_dir = Path(__file__).parent / "resources"
    out_dir.mkdir(exist_ok=True)

    sizes  = [16, 32, 48, 64, 128, 256]
    frames = [draw_icon(s) for s in sizes]

    ico_path = out_dir / "icon.ico"
    frames[-1].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[:-1],
    )
    print(f"Saved {ico_path}")

    png_path = out_dir / "icon_256.png"
    frames[-1].save(png_path, format="PNG")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
