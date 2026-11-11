# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rect_distance(x, y, cfg):
    cx, cy = cfg['table']['center_xy']
    hx, hy = (cfg['table']['size_xy'][0] / 2, cfg['table']['size_xy'][1] / 2)
    dx = np.maximum(np.abs(np.asarray(x) - cx) - hx, 0.0)
    dy = np.maximum(np.abs(np.asarray(y) - cy) - hy, 0.0)
    return np.hypot(dx, dy)

@dataclass
class Field:
    dist: np.ndarray
    pred: np.ndarray

class NavMap:

    def __init__(self, cfg):
        self.cfg = cfg
        e = cfg['expert']
        self.res = e['grid_res_m']
        self.half = cfg['room']['size_xy'][0] / 2
        self.n = int(round(2 * self.half / self.res))
        c = (np.arange(self.n) + 0.5) * self.res - self.half
        self.xs, self.ys = (c, c)
        X, Y = np.meshgrid(c, c)
        self.X, self.Y = (X, Y)
        self.clearance = rect_distance(X, Y, cfg)
        wall_d = self.half - np.maximum(np.abs(X), np.abs(Y))
        self.free = (self.clearance >= e['table_inflation_m']) & (wall_d >= e['wall_margin_m'])
        self.cost = 1.0 + e['path_margin_weight'] * np.maximum(0.0, e['path_margin_m'] - self.clearance) + 3.0 * np.maximum(0.0, 0.8 - wall_d)
        self._build_graph()

    def _build_graph(self):
        n, res = (self.n, self.res)
        idx = -np.ones((n, n), dtype=np.int64)
        idx[self.free] = np.arange(int(self.free.sum()))
        rows, cols, w = ([], [], [])
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            ys0, ys1 = (0, n - dy)
            xs0, xs1 = (max(0, -dx), n - max(0, dx))
            A = self.free[ys0:ys1, xs0:xs1]
            B = self.free[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            ok = A & B
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
        self.m = m
        self.idx = idx
        self.cells = np.argwhere(self.free)

    def cell(self, x, y):
        i = int(np.clip((y + self.half) // self.res, 0, self.n - 1))
        j = int(np.clip((x + self.half) // self.res, 0, self.n - 1))
        return (i, j)

    def is_free(self, x, y) -> bool:
        if abs(x) >= self.half or abs(y) >= self.half:
            return False
        return bool(self.free[self.cell(x, y)])

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
            return Field(np.full((self.n, self.n), np.inf), np.full(m, -1, dtype=np.int64))
        src = np.array(list(costs))
        w0 = np.array([costs[c] for c in costs]) + 1e-06
        rows, cols, w = self._edges
        g = coo_matrix((np.concatenate([w, w0, w0]), (np.concatenate([rows, np.full(src.size, m), src]), np.concatenate([cols, src, np.full(src.size, m)]))), shape=(m + 1, m + 1)).tocsr()
        d, pred = dijkstra(g, directed=False, indices=m, return_predecessors=True)
        out = np.full((self.n, self.n), np.inf)
        out[self.free] = d[:m]
        return Field(out, pred[:m].astype(np.int64))

    def value(self, field: Field, x, y) -> float:
        if abs(x) >= self.half or abs(y) >= self.half:
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
            pts = [(self.xs[2 * j], self.ys[2 * i]) for i, j in zip(ys, xs)]
            cache[clearance] = self.field(pts)
        return cache[clearance]

    def path_length(self, field: Field, x, y) -> float:
        start = self.nearest_free(x, y)
        if start is None or not np.isfinite(field.dist[start]):
            return math.inf
        k, last, total = (int(self.idx[start]), start, 0.0)
        total += math.hypot(self.xs[start[1]] - x, self.ys[start[0]] - y)
        while True:
            nxt = int(field.pred[k])
            if nxt < 0 or nxt >= self.m:
                return total
            ci = tuple(self.cells[nxt])
            total += self.res * math.hypot(ci[0] - last[0], ci[1] - last[1])
            k, last = (nxt, ci)
