# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import cv2
import numpy as np
CORNERS = ('tl', 'tr', 'bl', 'br')
TINT = np.array([40, 220, 80], np.float32)
ACCENT = {None: (255, 255, 255), 'correct': (60, 220, 90), 'near': (240, 200, 60), 'fail': (235, 70, 70)}
ACTIONS = {0: 'STOP', 1: 'forward', 2: 'left', 3: 'right', 4: 'up', 5: 'down'}

def panel_rect(win_w: int, win_h: int, aspect: float, size: float=0.45, corner: str='tr', margin: int=10, left0: int=0, bottom0: int=0):
    if corner not in CORNERS:
        raise ValueError(f'corner must be one of {CORNERS}, got {corner!r}')
    h = max(int(round(float(np.clip(size, 0.1, 0.95)) * win_h)), 48)
    w = max(int(round(h * aspect)), 27)
    fit = min(1.0, (win_w - 2 * margin) / w, (win_h - 2 * margin) / h)
    if fit < 1.0:
        w, h = (max(int(w * fit), 27), max(int(h * fit), 48))
    left = margin if corner[1] == 'l' else win_w - w - margin
    bottom = margin if corner[0] == 'b' else win_h - h - margin
    return (left0 + left, bottom0 + bottom, w, h)

def _fit_text(text: str, max_px: int, font, scale: float, thick: int) -> str:
    if cv2.getTextSize(text, font, scale, thick)[0][0] <= max_px:
        return text
    while len(text) > 1 and cv2.getTextSize(text + '...', font, scale, thick)[0][0] > max_px:
        text = text[:-1]
    return text + '...'

def _bar(img, y0: int, y1: int, alpha: float=0.6):
    roi = img[y0:y1]
    roi[:] = (roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

def panel_image(rgb: np.ndarray, w: int, h: int, mask=None, title: str='ROBOT CAMERA', lines=(), accent=(255, 255, 255)) -> np.ndarray:
    interp = cv2.INTER_AREA if w < rgb.shape[1] else cv2.INTER_LINEAR
    img = np.ascontiguousarray(cv2.resize(rgb, (w, h), interpolation=interp))
    if mask is not None and mask.any():
        m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        img[m] = (0.55 * img[m] + 0.45 * TINT).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = float(np.clip(w / 640.0, 0.34, 0.8))
    thick = 1 if scale < 0.6 else 2
    line_h = cv2.getTextSize('Ag', font, scale, thick)[0][1] + 7
    _bar(img, 0, min(h, line_h + 4))
    cv2.putText(img, _fit_text(title, w - 10, font, scale, thick), (6, line_h - 1), font, scale, accent, thick, cv2.LINE_AA)
    lines = [ln for ln in lines if ln][:3]
    if lines:
        y0 = max(h - len(lines) * line_h - 6, 0)
        _bar(img, y0, h)
        for i, ln in enumerate(lines):
            cv2.putText(img, _fit_text(ln, w - 10, font, scale, thick), (6, y0 + (i + 1) * line_h), font, scale, accent if i == 1 and len(lines) > 1 else (255, 255, 255), thick, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), accent, 2)
    return np.ascontiguousarray(img)

def hud_lines(hud: dict) -> list:
    if not hud or 'cat' not in hud:
        return ['type a furniture word in the terminal']
    goal = f"goal: {str(hud['cat']).replace('_', ' ')}   step {hud.get('t', 0)}"
    if hud.get('verdict'):
        return [goal, hud['verdict'], f"nearest {hud['final_d']:.2f} m" if hud.get('final_d') is not None else '']
    act = hud.get('action')
    act = ACTIONS.get(act, str(act)) if isinstance(act, (int, np.integer)) else act or ''
    d = hud.get('d')
    dist = '-' if d is None or d != d else f'{d:.2f} m'
    return [goal, f"{hud.get('driver', 'policy')}: {act}   p(stop) {hud.get('p_stop', 0.0):.2f}", f"nearest {dist}   mask {hud.get('px', 0)} px"]
