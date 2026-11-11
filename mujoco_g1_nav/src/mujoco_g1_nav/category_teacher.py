# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from .house_check import CAMERA_H
from .visibility import Fog

class HeightFog(Fog):

    def __init__(self, wmap, cfg, hmap, target_h: float, range_m: float, rays: int=80):
        super().__init__(wmap, cfg, range_m=range_m, rays=rays)
        h = np.where(wmap.space, hmap, 9.0).astype(np.float32)
        self.h = h
        self.slope_target = (CAMERA_H - target_h) / self.d

    def update(self, x: float, y: float, yaw: float) -> int:
        wm = self.wm
        cx, cy = (x + self.fwd * math.cos(yaw), y + self.fwd * math.sin(yaw))
        ang = yaw + self.rel[:, None]
        px = cx + self.d[None, :] * np.cos(ang)
        py = cy + self.d[None, :] * np.sin(ang)
        j = np.clip(((px - wm.x0) // wm.res).astype(np.int64), 0, wm.nx - 1)
        i = np.clip(((py - wm.y0) // wm.res).astype(np.int64), 0, wm.ny - 1)
        hh = self.h[i, j]
        slope = np.where(hh > 0, (CAMERA_H - hh) / self.d[None, :], np.inf)
        smin = np.minimum.accumulate(slope, axis=1)
        before = np.concatenate([np.full((slope.shape[0], 1), np.inf), smin[:, :-1]], axis=1)
        seen = self.slope_target[None, :] <= before
        before_n = int(self.explored.sum())
        self.explored[i[seen], j[seen]] = True
        ci, cj = wm.cell(x, y)
        r = self._near
        sub = (slice(max(0, ci - r), ci + r + 1), slice(max(0, cj - r), cj + r + 1))
        self.explored[sub] |= self.space[sub]
        return int(self.explored.sum()) - before_n

def instance_pixels(ctx, seg, items) -> dict:
    gi = ctx.house.geom_item
    idx = gi[np.where(seg < 0, len(gi) - 1, seg)]
    counts = np.bincount(idx.ravel(), minlength=len(ctx.house.layout.items) + 1)
    pos = {it.id: k + 1 for k, it in enumerate(ctx.house.layout.items)}
    return {it.id: int(counts[pos[it.id]]) for it in items}
