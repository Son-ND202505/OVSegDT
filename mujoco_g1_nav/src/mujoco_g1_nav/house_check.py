# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage
from .building import DOOR0, NORMAL, WALL, Building
from .config import load_config
from .furniture import CATEGORIES, Item, Layout
NEIGH8 = np.ones((3, 3), dtype=bool)
CAMERA_H = 1.31
REACH_CLOSE_M = 0.9
REACH_WALL_M = 1.5
REACH_FLAT_M = 0.3
FRONT_PHI_SURFACE = 30.0
CORNER_M = 0.15
ARC_BIN_DEG = 15.0
T_LEVELS = [round(0.8 - 0.05 * k, 2) for k in range(11)]

class Grid:

    def __init__(self, building: Building, res: float):
        fp = building.footprint
        self.res, self.x0, self.y0 = (res, fp.x0, fp.y0)
        self.nx, self.ny = (int(round(fp.w / res)), int(round(fp.h / res)))
        self.xs = fp.x0 + (np.arange(self.nx) + 0.5) * res
        self.ys = fp.y0 + (np.arange(self.ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        codes = building.codes(self.X, self.Y)
        self.space = codes != WALL
        self.door = codes <= DOOR0
        self.wall_d = np.maximum(ndimage.distance_transform_edt(self.space) * res - res / 2, 0.0)

    def cell(self, x: float, y: float) -> tuple:
        return (int((y - self.y0) // self.res), int((x - self.x0) // self.res))

    def window(self, x0, y0, x1, y1) -> tuple:
        j0, j1 = (int(math.floor((x0 - self.x0) / self.res)), int(math.ceil((x1 - self.x0) / self.res)) + 1)
        i0, i1 = (int(math.floor((y0 - self.y0) / self.res)), int(math.ceil((y1 - self.y0) / self.res)) + 1)
        return (max(i0, 0), min(i1, self.ny), max(j0, 0), min(j1, self.nx))

    def mask_of(self, item: Item, margin: float=0.0) -> np.ndarray:
        m = np.zeros((self.ny, self.nx), dtype=bool)
        x0, y0, x1, y1 = item.bbox()
        i0, i1, j0, j1 = self.window(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        m[i0:i1, j0:j1] = item.contains(self.X[i0:i1, j0:j1], self.Y[i0:i1, j0:j1], margin)
        return m

    def xy(self, ij) -> np.ndarray:
        ij = np.asarray(ij)
        return np.stack([self.xs[ij[:, 1]], self.ys[ij[:, 0]]], axis=1)

def _edt_from(mask_blocked: np.ndarray, res: float) -> np.ndarray:
    if not mask_blocked.any():
        return np.full(mask_blocked.shape, 99.0)
    return np.maximum(ndimage.distance_transform_edt(~mask_blocked) * res - res / 2, 0.0)

@dataclass
class ItemReport:
    id: str
    cat: str
    region: str
    kind: str
    issues: list = field(default_factory=list)
    warns: list = field(default_factory=list)
    n_stand: int = 0
    n_seen: int = 0
    n_front: int = 0
    arc_deg: float = 0.0
    aisle_m: float = None
    stand: tuple = None

    @property
    def ok(self) -> bool:
        return not self.issues

@dataclass
class HouseReport:
    house: str
    items: list
    issues: list = field(default_factory=list)
    warns: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    stand_xy: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues and all((i.ok for i in self.items))

    def failures(self) -> list:
        return [f'house: {m}' for m in self.issues] + [f'{i.id}: {m}' for i in self.items for m in i.issues]

    def warnings(self) -> list:
        return [f'house: {m}' for m in self.warns] + [f'{i.id}: {m}' for i in self.items for m in i.warns]

    def format(self, verbose: bool=False) -> str:
        s = self.stats
        lines = [f"house {self.house}: {len(self.items)} items -> {('OK' if self.ok else 'FAIL')}   ({len(self.failures())} failures, {len(self.warnings())} warnings)", f"  circulation (door to door) aisle {s.get('circulation_aisle_m', float('nan')):.2f} m   narrowest item route {s.get('min_item_aisle_m', float('nan')):.2f} m   mean {s.get('mean_item_aisle_m', float('nan')):.2f} m   items with an aisle < {s.get('aisle_good_m', 1.2):.1f} m: {s.get('n_narrow', 0)}", f"  reachable free space {s.get('main_area_m2', 0):.0f} m2 of {s.get('free_area_m2', 0):.0f} m2 inflated-free; pockets cut off: {s.get('n_pockets', 0)}"]
        for f in self.failures():
            lines.append('  FAIL ' + f)
        for w in self.warnings():
            lines.append('  warn ' + w)
        if verbose:
            lines.append(f"  {'id':16s} {'cat':13s} {'reg':3s} {'stand':>6s} {'seen':>6s} {'front':>6s} {'arc':>5s} {'aisle':>6s}")
            for i in self.items:
                a = '' if i.aisle_m is None else f'{i.aisle_m:.2f}'
                lines.append(f'  {i.id:16s} {i.cat:13s} {i.region:3s} {i.n_stand:6d} {i.n_seen:6d} {i.n_front:6d} {i.arc_deg:5.0f} {a:>6s}')
        return '\n'.join(lines)

def _los_clear(G: Grid, hmap: np.ndarray, own: np.ndarray, sx, sy, tx, ty, tz, end_skip: float=0.12, step: float=0.05) -> np.ndarray:
    dx, dy = (tx - sx, ty - sy)
    L = np.hypot(dx, dy)
    n = max(int(math.ceil(float(L.max()) / step)), 2)
    t = np.linspace(0.0, 1.0, n)[None, :]
    px, py = (sx[:, None] + dx[:, None] * t, sy[:, None] + dy[:, None] * t)
    ray_h = CAMERA_H + (tz - CAMERA_H) * t
    i = np.clip(np.floor((py - G.y0) / G.res).astype(int), 0, G.ny - 1)
    j = np.clip(np.floor((px - G.x0) / G.res).astype(int), 0, G.nx - 1)
    tested = t * L[:, None] < L[:, None] - end_skip
    blocked = (hmap[i, j] > ray_h) & ~own[i, j] & tested
    return ~blocked.any(axis=1)

def _frontal_surface(parent: Item, child: Item, sx, sy, phi_deg: float) -> np.ndarray:
    ox, oy = (child.x, child.y)
    if parent.shape == 'round':
        a = math.atan2(oy - parent.y, ox - parent.x)
        normals = [(math.cos(a), math.sin(a))] if math.hypot(ox - parent.x, oy - parent.y) > 1e-06 else [None]
        out = np.zeros(len(sx), dtype=bool)
        for nrm in normals:
            if nrm is None:
                return np.ones(len(sx), dtype=bool)
            ang = np.degrees(np.abs(np.arctan2(-(sx - ox) * nrm[1] + (sy - oy) * nrm[0], (sx - ox) * nrm[0] + (sy - oy) * nrm[1])))
            out |= ang <= phi_deg
        return out
    cu, cv = parent.local(ox, oy)
    cu, cv = (float(cu), float(cv))
    hu, hv = (parent.depth / 2, parent.width / 2)
    c, s = (math.cos(parent.yaw), math.sin(parent.yaw))
    edges = [(hu - cu, (c, s)), (hu + cu, (-c, -s)), (hv - cv, (-s, c)), (hv + cv, (s, -c))]
    dmin = min((e[0] for e in edges))
    out = np.zeros(len(sx), dtype=bool)
    for dist, (nx, ny) in edges:
        if dist <= dmin + CORNER_M:
            ang = np.degrees(np.abs(np.arctan2(-(sx - ox) * ny + (sy - oy) * nx, (sx - ox) * nx + (sy - oy) * ny)))
            out |= ang <= phi_deg
    return out

def check_house(building: Building, layout: Layout, cfg: dict=None, res: float=0.05, aisle_min: float=1.0, aisle_good: float=1.2, apron_m: float=1.5, ring_m: float=1.6, hub: tuple=None, items_to_check: list=None) -> HouseReport:
    cfg = cfg or load_config()
    e = cfg['expert']
    keep_f, keep_w = (float(e['table_inflation_m']), float(e['wall_margin_m']))
    G = Grid(building, res)
    items = layout.items
    reports = {it.id: ItemReport(it.id, it.cat, it.region, it.kind) for it in items}
    rep = HouseReport(layout.name, [reports[it.id] for it in items])
    masks = {it.id: G.mask_of(it) for it in items if it.kind in ('floor', 'flat')}
    blockers = [it for it in items if it.blocks]
    occ_count = np.zeros((G.ny, G.nx), dtype=np.int8)
    hmap = np.zeros((G.ny, G.nx), dtype=np.float32)
    hmap[~G.space] = 9.0
    for it in blockers:
        occ_count += masks[it.id]
        np.maximum(hmap, np.where(masks[it.id], np.float32(it.top), np.float32(0)), out=hmap)
    occ = occ_count > 0
    furn_d = _edt_from(occ, res)
    clearance = np.minimum(G.wall_d, furn_d)
    inflated = G.space & (G.wall_d >= keep_w) & (furn_d >= keep_f)
    for it in items:
        r = reports[it.id]
        if it.kind in ('floor', 'flat'):
            m = masks[it.id]
            if not m.any():
                r.issues.append('footprint smaller than a grid cell')
                continue
            if not G.space[m].all():
                r.issues.append('sticks into a wall / outside the building')
            if G.door[m].any():
                r.issues.append('stands in a doorway')
            if it.blocks and (occ_count[m] > 1).any():
                others = sorted({o.id for o in blockers if o.id != it.id and (masks[o.id] & m).any()})
                r.issues.append(f"overlaps {', '.join(others)}")
        if it.wall is not None:
            qx, qy = (it.x - it.front[0] * (it.depth / 2 + it.gap + 0.05), it.y - it.front[1] * (it.depth / 2 + it.gap + 0.05))
            if int(building.codes(qx, qy)) != WALL:
                r.issues.append(f'no wall behind it (anchored to the {it.wall} side of {it.region})')
        if it.kind == 'surface':
            par = layout.by_id.get(it.parent)
            if par is None:
                r.issues.append('surface item without a parent')
            else:
                if not par.contains(it.corners()[:, 0], it.corners()[:, 1], margin=1e-06).all():
                    r.issues.append(f'overhangs its parent {par.id}')
                if it.z0 + 1e-06 < par.top:
                    r.warns.append(f'bottom {it.z0:.2f} m below the top of {par.id} ({par.top:.2f} m)')
        if it.kind == 'wall':
            fx, fy = it.front
            lx_, ly_ = (-fy, fx)
            pts = [(it.x + fx * d + lx_ * s_, it.y + fy * d + ly_ * s_) for d in (0.12, 0.3) for s_ in np.linspace(-it.width / 2 + 0.05, it.width / 2 - 0.05, 7)]
            hidden = {}
            for o in blockers:
                if o.top > it.z0 + 0.1:
                    n_hit = sum((bool(o.contains(px, py)) for px, py in pts))
                    if n_hit:
                        hidden[o.id] = n_hit / len(pts)
            for oid, frac in hidden.items():
                (r.issues if frac >= 0.5 else r.warns).append(f"{('hidden behind' if frac >= 0.5 else 'partly hidden by')} {oid}")
        reg = building.region_at(it.x, it.y)
        if reg is not None and reg != it.region and (it.kind != 'surface'):
            r.warns.append(f'centre is in region {reg}, listed under {it.region}')
    for d in building.doors.values():
        nx, ny = NORMAL[d.wall]
        tx, ty = (-ny, nx)
        along = (G.X - d.center[0]) * tx + (G.Y - d.center[1]) * ty
        across = (G.X - d.center[0]) * nx + (G.Y - d.center[1]) * ny
        zone = (np.abs(along) <= d.width / 2 + 0.2) & (np.abs(across) <= apron_m + building.wall_t / 2)
        hit = sorted({it.id for it in blockers if (masks[it.id] & zone).any()})
        if hit:
            rep.issues.append(f"{d.id}: furniture in the {apron_m:.1f} m apron: {', '.join(hit)}")
    for t in [it for it in items if it.fixed and it.cat == 'table']:
        ring = t.distance(G.X, G.Y) <= ring_m
        hit = sorted({it.id for it in blockers if it.id != t.id and (masks[it.id] & ring).any()})
        if hit:
            rep.issues.append(f"the {ring_m:.1f} m ring round {t.id} holds {', '.join(hit)}")
    if hub is None:
        hub = building.regions['X'].center if 'X' in building.regions else max((r for r in building.regions.values() if not r.is_room), key=lambda r: r.rect.area).center
    hi, hj = G.cell(*hub)
    lab, n_lab = ndimage.label(inflated, NEIGH8)
    if lab[hi, hj] == 0:
        rep.issues.append(f'the hub {hub} is not free')
        main = np.zeros_like(inflated)
    else:
        main = lab == lab[hi, hj]
    sizes = ndimage.sum(inflated, lab, range(1, n_lab + 1)) * res * res if n_lab else []
    pockets = [(k + 1, float(sz)) for k, sz in enumerate(sizes) if k + 1 != lab[hi, hj] and sz >= 0.5]
    for k, sz in pockets:
        ii, jj = np.argwhere(lab == k).mean(axis=0)
        rep.warns.append(f'cut-off pocket of {sz:.1f} m2 around ({G.xs[int(jj)]:.1f}, {G.ys[int(ii)]:.1f})')
    for rid, region in building.regions.items():
        i0, i1, j0, j1 = G.window(*region.rect.tuple())
        if not main[i0:i1, j0:j1].any():
            rep.issues.append(f'region {rid} cannot be reached from the hub')
    for d in building.doors.values():
        for pt in (d.inside, d.outside):
            if not main[G.cell(*pt)]:
                rep.issues.append(f'{d.id}: the point {tuple((round(v, 1) for v in pt))} in front of the door is not reachable')
    door_cells = [G.cell(*pt) for d in building.doors.values() for pt in (d.inside, d.outside)] + [(hi, hj)]
    circ = 0.0
    for t in T_LEVELS:
        lab_t, _ = ndimage.label(G.space & (clearance >= t), NEIGH8)
        labs = {int(lab_t[c]) for c in door_cells}
        if len(labs) == 1 and 0 not in labs:
            circ = t
            break
    rep.stats['circulation_aisle_m'] = 2 * circ
    if 2 * circ < aisle_min:
        rep.issues.append(f'the door-to-door circulation narrows to {2 * circ:.2f} m (< {aisle_min:.2f} m)')
    elif 2 * circ < aisle_good:
        rep.warns.append(f'the door-to-door circulation narrows to {2 * circ:.2f} m (< {aisle_good:.2f} m)')
    check_ids = set(items_to_check) if items_to_check else None
    aisles = []
    for it in items:
        if check_ids is not None and it.id not in check_ids:
            continue
        r = reports[it.id]
        cat = CATEGORIES[it.cat]
        par = layout.by_id.get(it.parent) if it.parent else None
        reach = {'floor': REACH_CLOSE_M, 'surface': REACH_CLOSE_M, 'wall': REACH_WALL_M, 'flat': REACH_FLAT_M}[it.kind]
        x0, y0, x1, y1 = it.bbox()
        i0, i1, j0, j1 = G.window(x0 - reach - 0.3, y0 - reach - 0.3, x1 + reach + 0.3, y1 + reach + 0.3)
        Xw, Yw = (G.X[i0:i1, j0:j1], G.Y[i0:i1, j0:j1])
        if it.kind in ('surface', 'wall'):
            dist = np.hypot(Xw - it.x, Yw - it.y)
            if it.kind == 'wall':
                dist = np.where(dist >= keep_w, dist, np.inf)
        else:
            dist = it.distance(Xw, Yw)
        cand = main[i0:i1, j0:j1] & (dist <= reach)
        ij = np.argwhere(cand)
        r.n_stand = int(len(ij))
        if r.n_stand == 0:
            r.issues.append(f'no reachable place to stand within {reach:.1f} m')
            continue
        sx, sy = (Xw[cand], Yw[cand])
        if it.kind == 'flat':
            seen = np.ones(len(sx), dtype=bool)
        else:
            own = np.zeros((G.ny, G.nx), dtype=bool)
            for o in (it, par):
                if o is not None and o.id in masks:
                    own |= masks[o.id]
            if it.kind == 'wall':
                tz = it.z0 + it.height / 2
            elif it.kind == 'surface':
                tz = it.z0 + it.height / 2
            else:
                tz = float(np.clip(it.z0 + it.height / 2, 0.05, 1.3))
            seen = _los_clear(G, hmap, own, sx, sy, it.x, it.y, tz)
        r.n_seen = int(seen.sum())
        if r.n_seen == 0:
            r.issues.append('it cannot be seen from any reachable place within reach (line of sight blocked)')
            continue
        if it.kind == 'surface' and par is not None and (cat.front_phi is None):
            front = _frontal_surface(par, it, sx, sy, FRONT_PHI_SURFACE)
        elif cat.front_phi is not None:
            ang = np.degrees(np.abs(np.arctan2(-(sx - it.x) * it.front[1] + (sy - it.y) * it.front[0], (sx - it.x) * it.front[0] + (sy - it.y) * it.front[1])))
            front = ang <= cat.front_phi
        else:
            front = np.ones(len(sx), dtype=bool)
        good = seen & front
        r.n_front = int(good.sum())
        if r.n_front == 0:
            r.issues.append(f'visible stand cells exist but none in the front sector (phi <= {cat.front_phi or FRONT_PHI_SURFACE:.0f} deg)')
        use = good if good.any() else seen
        rep.stand_xy[it.id] = np.stack([sx[use], sy[use]], axis=1)
        ang_c = np.degrees(np.arctan2(sy[use] - it.y, sx[use] - it.x)) % 360.0
        r.arc_deg = ARC_BIN_DEG * len(set((ang_c // ARC_BIN_DEG).astype(int).tolist()))
        dd = dist[cand][use]
        k = int(np.argmin(dd))
        bx, by = (float(sx[use][k]), float(sy[use][k]))
        r.stand = (bx, by, math.atan2(it.y - by, it.x - bx))
        drop = {m for m in (it.id, it.parent) if m in masks and any((o.id == m for o in blockers))}
        if drop:
            occ_x = np.zeros((G.ny, G.nx), dtype=bool)
            for o in blockers:
                if o.id not in drop:
                    occ_x |= masks[o.id]
            clr_x = np.minimum(G.wall_d, _edt_from(occ_x, res))
        else:
            clr_x = clearance
        stand_mask = np.zeros((G.ny, G.nx), dtype=bool)
        stand_mask[i0:i1, j0:j1][cand] = True
        best_t = 0.0
        for t in T_LEVELS:
            lab_t, _ = ndimage.label(G.space & (clr_x >= t), NEIGH8)
            h = lab_t[hi, hj]
            if h and (lab_t[stand_mask] == h).any():
                best_t = t
                break
        r.aisle_m = 2 * best_t
        aisles.append(2 * best_t)
        if 2 * best_t < aisle_min:
            r.issues.append(f'the route to it narrows to {2 * best_t:.2f} m (< {aisle_min:.2f} m)')
        elif 2 * best_t < aisle_good:
            r.warns.append(f'the route to it narrows to {2 * best_t:.2f} m (< {aisle_good:.2f} m)')
        if r.arc_deg < 60 and it.kind == 'floor' and (cat.front_phi is None):
            r.warns.append(f'visible stand cells cover only {r.arc_deg:.0f} deg around it')
    inflated_area = float(inflated.sum() * res * res)
    rep.stats.update(aisle_good_m=aisle_good, aisle_min_m=aisle_min, free_area_m2=inflated_area, main_area_m2=float(main.sum() * res * res), n_pockets=len(pockets), min_item_aisle_m=float(min(aisles)) if aisles else float('nan'), mean_item_aisle_m=float(np.mean(aisles)) if aisles else float('nan'), n_narrow=int(sum((a < aisle_good for a in aisles))))
    rep.arrays = dict(res=res, x0=G.x0, y0=G.y0, inflated=inflated, main=main, clearance=clearance, occupied=occ, space=G.space)
    return rep

def sample_valid_layout(layout: Layout, rng, tries: int=40, sigma: float=0.3, cfg: dict=None, res: float=0.05, **check_kw):
    last = None
    for k in range(1, tries + 1):
        cand = layout.perturbed(rng, sigma=sigma)
        rep = check_house(layout.building, cand, cfg, res=res, **check_kw)
        if rep.ok:
            return (cand, rep, k)
        last = rep
    raise RuntimeError(f'no valid perturbation in {tries} tries; last failures: {last.failures()[:5]}')
