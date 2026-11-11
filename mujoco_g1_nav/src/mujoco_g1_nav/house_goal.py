# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from .furniture import CATEGORIES, Item, Layout
from .house_check import _frontal_surface
R_LOOSE_M, R_CLOSE_M = (1.2, 0.9)
WALL_R_M = (1.5, 1.2)
FLAT_R_M = (0.3, 0.05)
YAW_TOL_DEG = 25.0
MIN_PX = 300
FRONT_PHI_SURFACE_DEG = 30.0
MIN_START_PATH_M = 3.0

def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def nearest_point(it: Item, x: float, y: float) -> tuple:
    lx, ly = (float(v) for v in it.local(x, y))
    if it.shape == 'round':
        r, n = (it.depth / 2, math.hypot(lx, ly))
        px, py = (lx, ly) if n <= r else (lx * r / n, ly * r / n)
    else:
        px, py = (min(max(lx, -it.depth / 2), it.depth / 2), min(max(ly, -it.width / 2), it.width / 2))
    return it.to_local_point(px, py)

def instance_metrics(layout: Layout, it: Item, x: float, y: float, yaw: float) -> dict:
    cat = CATEGORIES[it.cat]
    if it.kind in ('surface', 'wall'):
        tx, ty = (it.x, it.y)
        d = math.hypot(tx - x, ty - y)
    else:
        tx, ty = nearest_point(it, x, y)
        d = float(it.distance(x, y))
    bearing = math.atan2(ty - y, tx - x) if d > 1e-06 else yaw
    if it.kind == 'surface' and it.parent in layout.by_id and (cat.front_phi is None):
        front = bool(_frontal_surface(layout.by_id[it.parent], it, np.array([x]), np.array([y]), FRONT_PHI_SURFACE_DEG)[0])
    elif cat.front_phi is not None:
        vx, vy = (x - it.x, y - it.y)
        ang = math.degrees(abs(math.atan2(-vx * it.front[1] + vy * it.front[0], vx * it.front[0] + vy * it.front[1])))
        front = ang <= cat.front_phi
    else:
        front = True
    return dict(item=it.id, kind=it.kind, d=d, yaw_err=math.degrees(abs(wrap(yaw - bearing))), front=front)

def region_flags(m: dict, px: int) -> dict:
    if m['kind'] == 'flat':
        loc = m['d'] <= FLAT_R_M[0]
        return dict(loc=loc, loose=loc, close=m['d'] <= FLAT_R_M[1], front=m['d'] <= FLAT_R_M[1])
    r_loose, r_close = WALL_R_M if m['kind'] == 'wall' else (R_LOOSE_M, R_CLOSE_M)
    look = m['yaw_err'] <= YAW_TOL_DEG and px >= MIN_PX
    close = m['d'] <= r_close and look
    return dict(loc=m['d'] <= r_loose, loose=m['d'] <= r_loose and look, close=close, front=close and m['front'])

def goal_metrics(layout: Layout, cat: str, x: float, y: float, yaw: float, px: int) -> dict:
    per = [instance_metrics(layout, it, x, y, yaw) for it in layout.of_cat(cat)]
    best = min(per, key=lambda m: m['d'])
    fl = [region_flags(m, px) for m in per]
    return dict(item=best['item'], d=best['d'], yaw_err=best['yaw_err'], px=px, **{k: any((f[k] for f in fl)) for k in ('loc', 'loose', 'close', 'front')})

def success_flags(m: dict, called_stop: bool) -> dict:
    return dict(success=bool(called_stop and m['loose']), success_close=bool(called_stop and m['close']), success_front=bool(called_stop and m['front']), success_loc=bool(called_stop and m['loc']))

def category_geoms(house, layout: Layout, cat: str) -> list:
    out = []
    for it in layout.of_cat(cat):
        out += list(house.visual_geoms.get(it.id, []))
    return out

def category_lookup(house, layout: Layout, cat: str, ngeom: int) -> np.ndarray:
    lut = np.zeros(ngeom + 1, dtype=bool)
    lut[category_geoms(house, layout, cat)] = True
    return lut

def goal_cells(report, layout: Layout, cat: str, max_cells: int=400) -> list:
    pts = [report.stand_xy[it.id] for it in layout.of_cat(cat) if it.id in report.stand_xy and len(report.stand_xy[it.id])]
    if not pts:
        return []
    allp = np.concatenate(pts)
    step = max(1, len(allp) // max_cells)
    return [(float(x), float(y)) for x, y in allp[::step]]
