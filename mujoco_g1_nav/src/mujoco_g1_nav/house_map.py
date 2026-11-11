# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import weakref
import numpy as np
from scipy import ndimage
from .world_map import WorldMap
OPAQUE_MIN_TOP_M = 1.2
_CACHE = weakref.WeakKeyDictionary()

def furniture_arrays(layout, X, Y, res: float) -> tuple:
    key = (round(float(res), 4), X.shape, round(float(X[0, 0]), 4), round(float(Y[0, 0]), 4))
    per = _CACHE.setdefault(layout, {})
    if key not in per:
        occ = np.zeros(X.shape, dtype=bool)
        tall = np.zeros(X.shape, dtype=bool)
        for it in layout.items:
            if not it.blocks or (it.fixed and it.cat == 'table'):
                continue
            x0, y0, x1, y1 = it.bbox()
            j0, j1 = (max(int((x0 - X[0, 0]) // res) - 1, 0), min(int((x1 - X[0, 0]) // res) + 3, X.shape[1]))
            i0, i1 = (max(int((y0 - Y[0, 0]) // res) - 1, 0), min(int((y1 - Y[0, 0]) // res) + 3, X.shape[0]))
            m = it.contains(X[i0:i1, j0:j1], Y[i0:i1, j0:j1])
            occ[i0:i1, j0:j1] |= m
            if it.top >= OPAQUE_MIN_TOP_M:
                tall[i0:i1, j0:j1] |= m
        dist = np.full(X.shape, 99.0) if not occ.any() else np.maximum(ndimage.distance_transform_edt(~occ) * res - res / 2, 0.0)
        per[key] = (dist, tall)
    return per[key]

def furniture_distance(layout, X, Y, res: float) -> np.ndarray:
    return furniture_arrays(layout, X, Y, res)[0]

def height_map(layout, X, Y, res: float) -> np.ndarray:
    key = ('height', round(float(res), 4), X.shape, round(float(X[0, 0]), 4), round(float(Y[0, 0]), 4))
    per = _CACHE.setdefault(layout, {})
    if key not in per:
        h = np.zeros(X.shape, dtype=np.float32)
        for it in layout.items:
            if it.kind not in ('floor', 'surface'):
                continue
            x0, y0, x1, y1 = it.bbox()
            j0, j1 = (max(int((x0 - X[0, 0]) // res) - 1, 0), min(int((x1 - X[0, 0]) // res) + 3, X.shape[1]))
            i0, i1 = (max(int((y0 - Y[0, 0]) // res) - 1, 0), min(int((y1 - Y[0, 0]) // res) + 3, X.shape[0]))
            m = it.contains(X[i0:i1, j0:j1], Y[i0:i1, j0:j1])
            h[i0:i1, j0:j1] = np.maximum(h[i0:i1, j0:j1], np.where(m, np.float32(it.top), np.float32(0.0)))
        per[key] = h
    return per[key]

class HouseMap(WorldMap):

    def __init__(self, building, cfg, layout, with_table: bool=True, res: float=None):
        self.layout = layout
        super().__init__(building, cfg, with_table, res)
        self.opaque = furniture_arrays(layout, self.X, self.Y, self.res)[1]

    def _furniture_distance(self) -> np.ndarray:
        return furniture_distance(self.layout, self.X, self.Y, self.res)

    def wall_clearance(self, x, y) -> float:
        base = super().wall_clearance(x, y)
        if not self.inside(x, y):
            return base
        e = self.cfg['expert']
        return min(base, float(self.furn_d[self.cell(x, y)]) + (e['wall_margin_m'] - e['table_inflation_m']))
