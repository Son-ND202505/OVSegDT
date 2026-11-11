# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_SERIF_BI = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf'

def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

def _noise(rng, h, w, amp):
    return rng.normal(0.0, amp, (h, w, 1))

def _save(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)

def floor(path, rng):
    s = 512
    img = np.full((s, s, 3), (176, 170, 160), float) + _noise(rng, s, s, 4)
    for i in range(2):
        for j in range(2):
            img[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] += rng.normal(0, 5)
    for k in (0, 256):
        img[k:k + 3, :] = (110, 106, 100)
        img[:, k:k + 3] = (110, 106, 100)
    _save(img, path)

def wall(path, rng):
    s = 256
    img = np.full((s, s, 3), (222, 218, 208), float) + _noise(rng, s, s, 3)
    _save(img, path)

def wood(path, rng):
    s = 512
    x = np.arange(s)[None, :, None]
    grain = 12 * np.sin(x / 7.0 + 3 * np.sin(np.arange(s)[:, None, None] / 60.0)) + _noise(rng, s, s, 5)
    img = np.array((150, 104, 62), float)[None, None, :] + grain
    _save(img, path)

def _can(path, bg, band, text, text_col, font_path, accent=None, globe=False):
    w, h = (512, 256)
    im = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 22], fill=band)
    d.rectangle([0, h - 22, w, h], fill=band)
    for k in range(2):
        cx = k * (w // 2) + w // 4
        if accent is not None:
            d.polygon([(cx - 110, 150), (cx - 40, 120), (cx + 40, 150), (cx + 110, 118), (cx + 110, 138), (cx + 40, 172), (cx - 40, 140), (cx - 110, 170)], fill=accent)
        if globe:
            r = 46
            d.pieslice([cx - r, 60 - 10, cx + r, 60 + 2 * r - 10], 180, 360, fill=(215, 30, 45))
            d.pieslice([cx - r, 60 - 10, cx + r, 60 + 2 * r - 10], 0, 180, fill=(20, 60, 160))
            d.rectangle([cx - r, 60 - 10 + r - 9, cx + r, 60 - 10 + r + 9], fill=(245, 245, 245))
        f = _font(font_path, 44 if not globe else 34)
        tw = d.textlength(text, font=f)
        d.text((cx - tw / 2, 168 if globe else 60), text, font=f, fill=text_col)
    im.save(path)

def coca(path, rng):
    _can(path, (200, 16, 30), (205, 205, 210), 'Coca-Cola', (250, 250, 250), FONT_SERIF_BI, accent=(250, 250, 250))

def pepsi(path, rng):
    _can(path, (14, 50, 150), (205, 205, 210), 'PEPSI', (250, 250, 250), FONT_BOLD, globe=True)

def orange(path, rng):
    s = 256
    img = np.array((238, 132, 20), float)[None, None, :] + _noise(rng, s, s, 10)
    pores = rng.random((s, s)) > 0.985
    img[pores] *= 0.75
    _save(img, path)

def basket(path, rng):
    s = 256
    img = np.full((s, s, 3), (232, 190, 30), float) + _noise(rng, s, s, 5)
    for k in range(0, s, 16):
        img[k:k + 3, :] *= 0.72
        img[:, k:k + 3] *= 0.72
    for k in range(8, s, 32):
        img[k:k + 8, ::2] *= 0.88
    _save(img, path)

def ensure_textures(out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    paths = {}
    for name, fn in [('floor', floor), ('wall', wall), ('wood', wood), ('coca', coca), ('pepsi', pepsi), ('orange', orange), ('basket', basket)]:
        p = out_dir / f'{name}.png'
        if not p.exists():
            fn(p, rng)
        paths[name] = str(p)
    return paths
