# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from scipy import ndimage
DEFAULT_RANGE_M = 10.0
MIN_UNEXPLORED_M2 = 1.5
NEAR_FIELD_M = 0.6

class Fog:

    def __init__(self, wmap, cfg, range_m: float=DEFAULT_RANGE_M, rays: int=80):
        self.wm = wmap
        self.space = wmap.space
        self.opaque = getattr(wmap, 'opaque', None)
        self.explored = np.zeros_like(self.space)
        cam = cfg['camera']
        self.half = math.radians(cam['hfov_deg'] / 2)
        self.fwd = cam['forward_offset_m']
        self.range = range_m
        step = wmap.res / 2
        self.d = np.arange(step, range_m + 1e-09, step)
        self.rel = np.linspace(-self.half, self.half, rays)
        self._near = max(1, int(round(NEAR_FIELD_M / wmap.res)))

    def reset(self):
        self.explored[:] = False

    def update(self, x: float, y: float, yaw: float) -> int:
        wm = self.wm
        cx, cy = (x + self.fwd * math.cos(yaw), y + self.fwd * math.sin(yaw))
        ang = yaw + self.rel[:, None]
        px = cx + self.d[None, :] * np.cos(ang)
        py = cy + self.d[None, :] * np.sin(ang)
        j = np.clip(((px - wm.x0) // wm.res).astype(np.int64), 0, wm.nx - 1)
        i = np.clip(((py - wm.y0) // wm.res).astype(np.int64), 0, wm.ny - 1)
        blocked = ~self.space[i, j]
        if self.opaque is not None:
            blocked |= self.opaque[i, j]
        first = np.where(blocked.any(1), blocked.argmax(1), len(self.d))
        seen = np.arange(len(self.d))[None, :] < first[:, None]
        before = int(self.explored.sum())
        self.explored[i[seen], j[seen]] = True
        ci, cj = wm.cell(x, y)
        r = self._near
        sub = (slice(max(0, ci - r), ci + r + 1), slice(max(0, cj - r), cj + r + 1))
        self.explored[sub] |= self.space[sub]
        return int(self.explored.sum()) - before

    def fraction(self) -> float:
        return float(self.explored.sum()) / max(1, int(self.space.sum()))

    def frontier_cells(self, reachable: np.ndarray, min_unexplored_m2: float=MIN_UNEXPLORED_M2):
        unexplored = self.space & ~self.explored
        lab, n = ndimage.label(unexplored)
        if n == 0:
            return np.zeros_like(self.space)
        sizes = ndimage.sum(unexplored, lab, np.arange(1, n + 1)) * self.wm.res ** 2
        big = np.isin(lab, 1 + np.nonzero(sizes >= min_unexplored_m2)[0])
        near = ndimage.binary_dilation(big, structure=np.ones((3, 3), bool))
        return self.explored & near & reachable
