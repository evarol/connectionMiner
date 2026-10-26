"""Build the BernFac slide deck (16:9 PPTX) from the study figures.

Slides are declared in SLIDES as (title, image path | None, bullets). The PPTX is
then uploaded to Google Drive (converted to Google Slides).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "outputs/bernfac_20261020"
W, H = Inches(13.333), Inches(7.5)
DARK, ACCENT, GREY = RGBColor(0x22, 0x22, 0x22), RGBColor(0x0F, 0x4D, 0x92), RGBColor(0x66, 0x66, 0x66)
FONT = "Helvetica Neue"


def add_text(slide, x, y, w, h, lines, size=16, color=DARK, bold_first=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.font.bold = bold_first and i == 0
        p.font.name = FONT
        p.space_after = Pt(6)
    return tb


def add_rule(slide, y):
    """Thin accent rule under the slide title."""
    from pptx.enum.shapes import MSO_SHAPE
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), y, Inches(1.2), Inches(0.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


JPEG = {"max_w": 2000, "quality": 85}


def compress(path):
    """Downscale to <= max_w px and save as JPEG so the deck stays small to upload."""
    out = STUDY / "deck/img" / (Path(path).parent.parent.name + "_" + Path(path).stem + ".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(path).convert("RGB")
    # trim near-white margins (BioRender exports and LaTeX panels carry wide borders)
    ink = np.asarray(im.convert("L")) < 245
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    if rows.size and cols.size:
        pad = 12
        im = im.crop((max(cols[0] - pad, 0), max(rows[0] - pad, 0),
                      min(cols[-1] + pad, im.width), min(rows[-1] + pad, im.height)))
    max_w = JPEG["max_w"]
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    im.save(out, quality=JPEG["quality"])
    return out


def add_image_fit(slide, path, x, y, w, h):
    path = compress(path)
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = int(iw * s), int(ih * s)
    slide.shapes.add_picture(str(path), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)


def build(slides, out, start=1):
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    for n, sl in enumerate(slides, start):
        s = prs.slides.add_slide(blank)
        if sl.get("kind") == "title":
            add_text(s, Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.4), [sl["title"]], 40, DARK)
            add_text(s, Inches(0.8), Inches(3.8), Inches(11.7), Inches(2.5), sl["bullets"], 20, GREY)
            continue
        if sl.get("kind") == "section":
            add_text(s, Inches(0.8), Inches(0.8), Inches(4), Inches(0.6), [sl.get("part", "")], 18, ACCENT, True)
            add_text(s, Inches(0.8), Inches(2.4), Inches(11.7), Inches(1.2), [sl["title"]], 36, DARK, True)
            add_rule(s, Inches(3.55))
            add_text(s, Inches(0.8), Inches(3.8), Inches(11.7), Inches(3.2), sl.get("bullets", []), 18, GREY)
            add_text(s, W - Inches(1.0), H - Inches(0.45), Inches(0.8), Inches(0.35), [str(n)], 10, GREY)
            continue
        add_text(s, Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.8), [sl["title"]], 28, DARK, True)
        add_rule(s, Inches(0.95))
        img, bullets = sl.get("image"), sl.get("bullets", [])
        top = Inches(1.15)
        if sl.get("images"):
            # two (or more) images side by side, bullets underneath
            imgs = sl["images"]
            bh = Inches(0.45) * len(bullets) + (Inches(0.2) if bullets else 0)
            cw = (W - Inches(0.6)) // len(imgs)
            for i, im in enumerate(imgs):
                add_image_fit(s, im, Inches(0.3) + i * cw, top, cw - Inches(0.1), H - top - bh - Inches(0.35))
            if bullets:
                add_text(s, Inches(0.5), H - bh - Inches(0.3), Inches(12.3), bh, bullets, 14)
        elif img and bullets and sl.get("layout") == "side":
            add_image_fit(s, img, Inches(0.4), top, Inches(8.6), H - top - Inches(0.4))
            add_text(s, Inches(9.2), top, Inches(3.8), H - top - Inches(0.4), bullets, 14)
        elif img:
            bh = Inches(0.45) * len(bullets) + (Inches(0.2) if bullets else 0)
            add_image_fit(s, img, Inches(0.3), top, W - Inches(0.6), H - top - bh - Inches(0.35))
            if bullets:
                add_text(s, Inches(0.5), H - bh - Inches(0.3), Inches(12.3), bh, bullets, 14)
        else:
            add_text(s, Inches(0.6), top, Inches(12.1), H - top - Inches(0.4), bullets, 18)
        if sl.get("notes"):
            s.notes_slide.notes_text_frame.text = sl["notes"]
        add_text(s, W - Inches(1.0), H - Inches(0.45), Inches(0.8), Inches(0.35), [str(n)], 10, GREY)
    prs.save(out)
    print(f"wrote {out} ({len(slides)} slides)")


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text())
    for sl in spec["slides"]:
        if sl.get("image"):
            sl["image"] = str(STUDY / sl["image"])
        if sl.get("images"):
            sl["images"] = [str(STUDY / i) for i in sl["images"]]
    JPEG.update(spec.get("jpeg", {}))
    build(spec["slides"], spec["out"], spec.get("start", 1))
