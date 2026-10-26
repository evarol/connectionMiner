"""Patch spelling errors that BioRender's AI image generator baked into the flat
figure exports (the text is raster, so it cannot be edited in BioRender itself).

Word boxes come from `tesseract <png> - tsv`. Each box is filled with the local
background colour and the corrected word is redrawn at the same height.
Originals are kept as *_raw.png; patched files overwrite the deck inputs.
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIR = Path(__file__).resolve().parents[3] / "outputs/bernfac_20261020/biorender"
FONT = "/System/Library/Fonts/Helvetica.ttc"

# file -> list of (x, y, w, h, corrected text, vertical?)  [tesseract boxes]
FIXES = {
    "bernfac_2_pgem_inference": [(827, 1070, 114, 33, "solved", False),
                                 (2194, 1070, 114, 33, "solved", False)],
    "bernfac_3_realdata_design": [(2219, 811, 205, 38, "log-likelihood", False),
                                  (2236, 1091, 187, 30, "Hemilineage", False),
                                  (89, 1049, 30, 97, "libraries", True)],
    "bernfac_4_synthetic_cv_design": [(425, 691, 126, 26, "Bernoulli", False),
                                      (425, 1160, 126, 26, "Bernoulli", False),
                                      (425, 1633, 126, 26, "Bernoulli", False),
                                      (1226, 1175, 206, 28, "Bernoulli draw", False)],
}


def background(arr, x, y, w, h, pad=6):
    """Median colour of a thin frame around the box (the panel fill)."""
    y0, y1, x0, x1 = max(y - pad, 0), y + h + pad, max(x - pad, 0), x + w + pad
    frame = np.concatenate([arr[y0:y0 + 3, x0:x1].reshape(-1, 3), arr[y1 - 3:y1, x0:x1].reshape(-1, 3),
                            arr[y0:y1, x0:x0 + 3].reshape(-1, 3), arr[y0:y1, x1 - 3:x1].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(frame, axis=0))


def text_tile(text, height, fg, bg, width):
    """Render `text` to fill the old word's box: width scaled by the length change, ink height capped."""
    target_w = width
    size = int(height * 2)
    for _ in range(80):
        font = ImageFont.truetype(FONT, size)
        l, t, r, b = font.getbbox(text)
        if r - l <= target_w * 1.04 and b - t <= height * 1.2:
            break
        size -= 1
    tile = Image.new("RGB", (r - l + 4, b - t + 4), bg)
    ImageDraw.Draw(tile).text((2 - l, 2 - t), text, font=font, fill=fg)
    return tile


OLD_LEN = {"solved": 6, "log-likelihood": 14, "Hemilineage": 11, "libraries": 8, "Bernoulli": 8, "Bernoulli draw": 13}


def patch(name, fixes):
    src = DIR / f"{name}.png"
    raw = DIR / f"{name}_raw.png"
    if not raw.exists():
        shutil.copy(src, raw)
    im = Image.open(raw).convert("RGB")
    arr = np.asarray(im)
    for x, y, w, h, text, vertical in fixes:
        bg = background(arr, x, y, w, h)
        box = arr[y:y + h, x:x + w].reshape(-1, 3)
        fg = tuple(int(v) for v in box[np.argmin(box.sum(1))])        # darkest ink pixel
        ImageDraw.Draw(im).rectangle([x - 3, y - 3, x + w + 3, y + h + 3], fill=bg)
        if vertical:                                                   # reads bottom-to-top
            tile = text_tile(text, w, fg, bg, h * len(text) / OLD_LEN[text]).rotate(90, expand=True)
            im.paste(tile, (x + (w - tile.width) // 2, y + h - tile.height))
        else:
            tile = text_tile(text, h, fg, bg, w * len(text) / OLD_LEN[text])
            im.paste(tile, (x - 2, y + (h - tile.height) // 2))
    im.save(src)
    print("patched", name)


if __name__ == "__main__":
    for name, fixes in FIXES.items():
        patch(name, fixes)
