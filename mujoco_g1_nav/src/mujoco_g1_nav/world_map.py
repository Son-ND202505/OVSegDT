# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from .nav import Field, rect_distance
BUILDING_RES_M = 0.1
_STATIC = {}

def _static(building, res):
    key = (building.name, res, building.footprint.tuple())
    if key not in _STATIC:
        fp = building.footprint
        nx, ny = (int(round(fp.w / res)), int(round(fp.h / res)))
        xs = fp.x0 + (np.arange(nx) + 0.5) * res
        ys = fp.y0 + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        space = building.is_free(X, Y)
        wall_d = np.maximum(ndimage.distance_transform_edt(space) * res - res / 2, 0.0)
        _STATIC[key] = (xs, ys, X, Y, space, wall_d)
    return _STATIC[key]

class WorldMap:

    def __init__(self, building, cfg, with_table: bool=True, res: float=None):
        self.building, self.cfg = (building, cfg)
        e = cfg['expert']
        self.res = float(res or BUILDING_RES_M)
        self.xs, self.ys, self.X, self.Y, self.space, self.wall_d = _static(building, self.res)
        self.nx, self.ny = (len(self.xs), len(self.ys))
        self.x0, self.y0 = (float(self.xs[0] - self.res / 2), float(self.ys[0] - self.res / 2))
        self.with_table = with_table
        self.clearance = rect_distance(self.X, self.Y, cfg) if with_table else np.full(self.X.shape, 99.0)
        self.furn_d = self._furniture_distance()
        self.free = self.space & (self.clearance >= e['table_inflation_m']) & (self.furn_d >= e['table_inflation_m']) & (self.wall_d >= e['wall_margin_m'])
        self.cost = 1.0 + e['path_margin_weight'] * np.maximum(0.0, e['path_margin_m'] - self.clearance) + 3.0 * np.maximum(0.0, 0.8 - self.wall_d) + e['path_margin_weight'] * np.maximum(0.0, e['path_margin_m'] - self.furn_d)
        self._build_graph()

    def _furniture_distance(self) -> np.ndarray:
        return np.full(self.X.shape, 99.0)

    def _build_graph(self):
        ny, nx, res = (self.ny, self.nx, self.res)
        idx = -np.ones((ny, nx), dtype=np.int64)
        idx[self.free] = np.arange(int(self.free.sum()))
        rows, cols, w = ([], [], [])
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ys0, ys1 = (0, ny - dy)
            xs0, xs1 = (max(0, -dx), nx - max(0, dx))
            ok = self.free[ys0:ys1, xs0:xs1] & self.free[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            ia = idx[ys0:ys1, xs0:xs1][ok]
            ib = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok]
            rows += [ia, ib]
            cols += [ib, ia]
            cf = 0.5 * (self.cost[ys0:ys1, xs0:xs1][ok] + self.cost[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok])
            wt = res * math.hypot(dx, dy) * cf
            w += [wt, wt]
        m = int(self.free.sum())
        self._edges = (np.concatenate(rows), np.concatenate(cols), np.concatenate(w))
        self.graph = coo_matrix((self._edges[2], (self._edges[0], self._edges[1])), shape=(m, m)).tocsr()
        self.m, self.idx = (m, idx)
        self.cells = np.argwhere(self.free)

    def inside(self, x, y) -> bool:
        return self.x0 <= x < self.x0 + self.nx * self.res and self.y0 <= y < self.y0 + self.ny * self.res

    def cell(self, x, y):
        i = int(np.clip((y - self.y0) // self.res, 0, self.ny - 1))
        j = int(np.clip((x - self.x0) // self.res, 0, self.nx - 1))
        return (i, j)

    def is_free(self, x, y) -> bool:
        return self.inside(x, y) and bool(self.free[self.cell(x, y)])

    def wall_clearance(self, x, y) -> float:
        return float(self.wall_d[self.cell(x, y)]) if self.inside(x, y) else 0.0

    def field(self, goal_xy, goal_cost=None) -> Field:
        m = self.m
        costs = {}
        for n, (gx, gy) in enumerate(goal_xy):
            i, j = self.cell(gx, gy)
            r = int(0.3 / self.res)
            sub = self.free[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
            if not sub.any():
                continue
            ii, jj = np.nonzero(sub)
            k = np.argmin((ii - min(r, i)) ** 2 + (jj - min(r, j)) ** 2)
            c = int(self.idx[max(0, i - r) + ii[k], max(0, j - r) + jj[k]])
            costs[c] = min(costs.get(c, math.inf), 0.0 if goal_cost is None else float(goal_cost[n]))
        if not costs:
            return Field(np.full((self.ny, self.nx), np.inf), np.full(m, -1, dtype=np.int64))
        src = np.array(list(costs))
        w0 = np.array([costs[c] for c in costs]) + 1e-06
        rows, cols, w = self._edges
        g = coo_matrix((np.concatenate([w, w0, w0]), (np.concatenate([rows, np.full(src.size, m), src]), np.concatenate([cols, src, np.full(src.size, m)]))), shape=(m + 1, m + 1)).tocsr()
        d, pred = dijkstra(g, directed=False, indices=m, return_predecessors=True)
        out = np.full((self.ny, self.nx), np.inf)
        out[self.free] = d[:m]
        return Field(out, pred[:m].astype(np.int64))

    def value(self, field: Field, x, y) -> float:
        if not self.inside(x, y):
            return math.inf
        return float(field.dist[self.cell(x, y)])

    def nearest_free(self, x, y, radius=0.6):
        i0, j0 = self.cell(x, y)
        r = int(radius / self.res)
        sub = self.free[max(0, i0 - r):i0 + r + 1, max(0, j0 - r):j0 + r + 1]
        if not sub.any():
            return None
        ii, jj = np.nonzero(sub)
        ii, jj = (ii + max(0, i0 - r), jj + max(0, j0 - r))
        k = np.argmin((ii - i0) ** 2 + (jj - j0) ** 2)
        return (int(ii[k]), int(jj[k]))

    def lookahead(self, field: Field, x, y, dist_m: float):
        start = self.nearest_free(x, y)
        if start is None or not np.isfinite(field.dist[start]):
            return None
        k = int(self.idx[start])
        traveled, last = (0.0, start)
        while True:
            nxt = int(field.pred[k])
            if nxt < 0 or nxt >= self.m:
                break
            ci = tuple(self.cells[nxt])
            traveled += self.res * math.hypot(ci[0] - last[0], ci[1] - last[1])
            k, last = (nxt, ci)
            if traveled >= dist_m:
                break
        return (float(self.xs[last[1]]), float(self.ys[last[0]]))

    def chosen_goal(self, field: Field, x, y):
        return self.lookahead(field, x, y, math.inf)

    def segment_free(self, x0, y0, x1, y1) -> bool:
        L = math.hypot(x1 - x0, y1 - y0)
        for t in np.linspace(0.0, 1.0, max(2, int(L / (self.res * 0.5)))):
            if not self.is_free(x0 + t * (x1 - x0), y0 + t * (y1 - y0)):
                return False
        return True

    def ring_field(self, clearance: float) -> Field:
        cache = self.__dict__.setdefault('_ring_cache', {})
        if clearance not in cache:
            mask = self.free & (np.abs(self.clearance - clearance) < 0.06)
            ys, xs = np.nonzero(mask[::2, ::2])
            cache[clearance] = self.field([(self.xs[2 * j], self.ys[2 * i]) for i, j in zip(ys, xs)])
        return cache[clearance]

    def path_length(self, field: Field, x, y) -> float:
        start = self.nearest_free(x, y)
        if start is None or not np.isfinite(field.dist[start]):
            return math.inf
        k, last = (int(self.idx[start]), start)
        total = math.hypot(self.xs[start[1]] - x, self.ys[start[0]] - y)
        while True:
            nxt = int(field.pred[k])
            if nxt < 0 or nxt >= self.m:
                return total
            ci = tuple(self.cells[nxt])
            total += self.res * math.hypot(ci[0] - last[0], ci[1] - last[1])
            k, last = (nxt, ci)

    def path(self, field: Field, x, y, step_m: float=0.5) -> list:
        pts, d = ([], 0.0)
        start = self.nearest_free(x, y)
        if start is None or not np.isfinite(field.dist[start]):
            return pts
        k, last = (int(self.idx[start]), start)
        pts.append((float(self.xs[last[1]]), float(self.ys[last[0]])))
        while True:
            nxt = int(field.pred[k])
            if nxt < 0 or nxt >= self.m:
                return pts
            ci = tuple(self.cells[nxt])
            d += self.res * math.hypot(ci[0] - last[0], ci[1] - last[1])
            k, last = (nxt, ci)
            if d >= step_m:
                pts.append((float(self.xs[last[1]]), float(self.ys[last[0]])))
                d = 0.0
