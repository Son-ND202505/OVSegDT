# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from .config import PKG_ROOT
MM = 1000
SOLID = '#'
ROOM, CORRIDOR, SPUR = ('room', 'corridor', 'spur')
SIDES = ('north', 'south', 'east', 'west')
NORMAL = {'north': (0.0, 1.0), 'south': (0.0, -1.0), 'east': (1.0, 0.0), 'west': (-1.0, 0.0)}
DOOR_WIDTH_M = 2.0
DOOR_MARGIN_M = 0.5
WALL, DOOR0 = (-1, -2)
BUILDINGS_DIR = PKG_ROOT / 'configs' / 'buildings'

class PlanError(ValueError):
    pass

def mm(v) -> int:
    return int(round(float(v) * MM))

@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def area(self) -> float:
        return self.w * self.h

    def contains(self, x, y, margin=0.0) -> bool:
        return self.x0 + margin <= x <= self.x1 - margin and self.y0 + margin <= y <= self.y1 - margin

    def distance(self, x, y) -> float:
        return float(np.hypot(max(self.x0 - x, 0.0, x - self.x1), max(self.y0 - y, 0.0, y - self.y1)))

    def tuple(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)

@dataclass
class Door:
    id: str
    room: str
    wall: str
    to: str
    center: tuple
    width: float
    rect: Rect
    inside: tuple
    outside: tuple

@dataclass
class Region:
    id: str
    kind: str
    index: int
    rect: Rect
    wall_rgb: tuple = None
    floor_rgb: tuple = None
    door: Door = None
    doors: list = field(default_factory=list)

    @property
    def is_room(self) -> bool:
        return self.kind == ROOM

    @property
    def center(self) -> tuple:
        return self.rect.center

@dataclass
class WallBox:
    rect: Rect
    rgb: tuple = None

@dataclass
class Building:
    name: str
    description: str
    wall_t: float
    origin: str
    regions: dict
    doors: dict
    walls: list
    footprint: Rect
    graph: dict
    _xe: np.ndarray = field(repr=False, default=None)
    _ye: np.ndarray = field(repr=False, default=None)
    _codes: np.ndarray = field(repr=False, default=None)
    _by_index: list = field(repr=False, default_factory=list)
    _door_by_code: list = field(repr=False, default_factory=list)

    @property
    def room_ids(self) -> list:
        return [r.id for r in self.regions.values() if r.is_room]

    @property
    def open_ids(self) -> list:
        return [r.id for r in self.regions.values() if not r.is_room]

    @property
    def interior(self) -> Rect:
        return Rect(float(self._xe[1]), float(self._ye[1]), float(self._xe[-2]), float(self._ye[-2]))

    def codes(self, x, y) -> np.ndarray:
        x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        j = np.searchsorted(self._xe, x, side='right') - 1
        i = np.searchsorted(self._ye, y, side='right') - 1
        inside = (j >= 0) & (j < len(self._xe) - 1) & (i >= 0) & (i < len(self._ye) - 1)
        return np.where(inside, self._codes[np.clip(i, 0, len(self._ye) - 2), np.clip(j, 0, len(self._xe) - 2)], WALL)

    def region_at(self, x: float, y: float):
        c = int(self.codes(x, y))
        if c >= 0:
            return self._by_index[c]
        if c <= DOOR0:
            return self._door_by_code[DOOR0 - c]
        return None

    def room_at(self, x: float, y: float):
        rid = self.region_at(x, y)
        if rid in self.doors:
            rid = self.doors[rid].room
        return rid if rid in self.regions and self.regions[rid].is_room else None

    def is_free(self, x, y):
        return self.codes(x, y) != WALL

    def free_mask(self, xs, ys) -> np.ndarray:
        X, Y = np.meshgrid(xs, ys)
        return self.is_free(X, Y)

    def wall_distance(self, x: float, y: float) -> float:
        return min((w.rect.distance(x, y) for w in self.walls))

    def door_of(self, room: str) -> Door:
        return self.regions[room].door

    def doors_of(self, room: str) -> list:
        return self.regions[room].doors

    def path_regions(self, a: str, b: str) -> list:
        prev, queue = ({a: None}, deque([a]))
        while queue:
            r = queue.popleft()
            if r == b:
                break
            for n in self.graph[r]:
                if n not in prev:
                    prev[n] = r
                    queue.append(n)
        if b not in prev:
            return []
        out = [b]
        while prev[out[-1]] is not None:
            out.append(prev[out[-1]])
        return out[::-1]

    def summary(self) -> str:
        fp, it = (self.footprint, self.interior)
        lines = [f'{self.name}: {self.description}', f'  footprint (outer walls) {fp.w:.2f} x {fp.h:.2f} m, free space {it.w:.2f} x {it.h:.2f} m, {len(self.room_ids)} rooms, {len(self.doors)} doors, {len(self.walls)} wall boxes']
        for r in self.regions.values():
            door = ', doors ' + ' / '.join((f'{d.wall} (to {d.to})' for d in r.doors)) if r.doors else ''
            lines.append(f'  {r.id:3s} {r.kind:8s} x[{r.rect.x0:7.2f},{r.rect.x1:7.2f}] y[{r.rect.y0:7.2f},{r.rect.y1:7.2f}] {r.rect.w:5.2f} x {r.rect.h:5.2f} m{door}')
        return '\n'.join(lines)

def _axis(lengths, band_after, t, breaks):
    n, segs, pos = (len(lengths), [(-t, 0, 'band', -1)], 0)
    for j, length in enumerate(lengths):
        segs.append((pos, pos + length, 'strip', j))
        pos += length
        b = t if j == n - 1 or band_after[j] else 0
        if b:
            segs.append((pos, pos + b, 'band', j))
            pos += b
    out = []
    for lo, hi, kind, j in segs:
        edges = [lo] + sorted({c for c in breaks if lo < c < hi}) + [hi]
        out += [(a, b, kind, j) for a, b in zip(edges[:-1], edges[1:])]
    return out

def _eset(kind, j):
    return (j,) if kind == 'strip' else (j, j + 1)

def load_plan(name_or_path) -> dict:
    p = Path(name_or_path)
    if p.suffix == '':
        p = BUILDINGS_DIR / f'{name_or_path}.yaml'
    with open(p) as f:
        return yaml.safe_load(f)

def load_building(name_or_path) -> Building:
    return compile_plan(load_plan(name_or_path))

def compile_plan(plan: dict) -> Building:
    t = mm(plan.get('wall_thickness', 0.1))
    cols, rows = ([mm(c) for c in plan['cols']], [mm(r) for r in plan['rows']])
    nc, nr = (len(cols), len(rows))
    grid, specs = (plan['grid'], plan['regions'])
    if len(grid) != nr or any((len(r) != nc for r in grid)):
        raise PlanError(f'grid must be {nr} rows x {nc} columns (rows/cols lists)')
    order = list(specs)
    kind = {k: specs[k].get('kind', ROOM) for k in order}
    if (bad := [k for k, v in kind.items() if v not in (ROOM, CORRIDOR, SPUR)]):
        raise PlanError(f'unknown region kind for {bad}')
    if SOLID in specs:
        raise PlanError('# denotes exterior cells and must not be listed as a region')
    if set(order) != {c for row in grid for c in row if c != SOLID}:
        raise PlanError('the regions listed and the labels used in the grid differ')
    span = {}
    for k in order:
        cells = [(i, j) for i in range(nr) for j in range(nc) if grid[i][j] == k]
        i0, i1, j0, j1 = (min((c[0] for c in cells)), max((c[0] for c in cells)), min((c[1] for c in cells)), max((c[1] for c in cells)))
        if len(cells) != (i1 - i0 + 1) * (j1 - j0 + 1):
            raise PlanError(f'region {k} is not a rectangle of grid cells')
        span[k] = (i0, i1, j0, j1)
    if plan['origin'] not in span or kind[plan['origin']] != ROOM:
        raise PlanError('`origin` must name a room')

    def ext(i, j):
        return grid[i][j] if 0 <= i < nr and 0 <= j < nc else SOLID

    def wall_between(a, b):
        return a != b and (a == SOLID or b == SOLID or kind[a] == ROOM or (kind[b] == ROOM))
    band_u = [any((wall_between(ext(i, j), ext(i, j + 1)) for i in range(nr))) for j in range(nc - 1)]
    band_v = [any((wall_between(ext(i, j), ext(i + 1, j)) for j in range(nc))) for i in range(nr - 1)]
    U0, V0 = (_axis(cols, band_u, t, ()), _axis(rows, band_v, t, ()))
    strip_u = {j: (lo, hi) for lo, hi, kd, j in U0 if kd == 'strip'}
    strip_v = {j: (lo, hi) for lo, hi, kd, j in V0 if kd == 'strip'}

    def plan_rect(k):
        i0, i1, j0, j1 = span[k]
        return (strip_u[j0][0], strip_v[i0][0], strip_u[j1][1], strip_v[i1][1])
    windows, breaks_u, breaks_v = ([], set(), set())
    for k in order:
        ds = specs[k].get('doors')
        if ds is None:
            ds = [specs[k]['door']] if specs[k].get('door') else []
        if ds and kind[k] != ROOM:
            raise PlanError(f'{k}: only rooms have doors')
        for d in ds:
            if d['wall'] not in SIDES:
                raise PlanError(f'{k}: door wall must be one of {SIDES}')
            u0, v0, u1, v1 = plan_rect(k)
            w, off = (mm(d.get('width', DOOR_WIDTH_M)), mm(d['offset']))
            along0, size = (v0, v1 - v0) if d['wall'] in ('east', 'west') else (u0, u1 - u0)
            if off - w // 2 < mm(DOOR_MARGIN_M) or off + w // 2 > size - mm(DOOR_MARGIN_M):
                raise PlanError(f'{k}: the door ({w / MM:.1f} m at offset {off / MM:.2f} m) is closer than {DOOR_MARGIN_M} m to a corner of its {size / MM:.1f} m wall')
            a, b = (along0 + off - w // 2, along0 + off + w // 2)
            if any((w0[0] == k and w0[1] == d['wall'] and (not (b + mm(DOOR_MARGIN_M) <= w0[2] or a >= w0[3] + mm(DOOR_MARGIN_M))) for w0 in windows)):
                raise PlanError(f"{k}: two doors on the {d['wall']} wall are closer than {DOOR_MARGIN_M} m")
            (breaks_v if d['wall'] in ('east', 'west') else breaks_u).update((a, b))
            windows.append((k, d['wall'], a, b, w))
    per_room, door_ids = ({}, [])
    for w0 in windows:
        per_room[w0[0]] = per_room.get(w0[0], 0) + 1
    for k, wall, *_ in windows:
        did = f'door_{k}' if per_room[k] == 1 else f'door_{k}_{wall}'
        n = 2
        while did in door_ids:
            did, n = (f'door_{k}_{wall}_{n}', n + 1)
        door_ids.append(did)
    U, V = (_axis(cols, band_u, t, breaks_u), _axis(rows, band_v, t, breaks_v))
    nU, nV = (len(U), len(V))

    def size_of(label):
        i0, i1, j0, j1 = span[label]
        return (sum(cols[j0:j1 + 1]), sum(rows[i0:i1 + 1]))
    idx = {k: n for n, k in enumerate(order)}
    code = np.zeros((nV, nU), dtype=np.int32)
    exterior = np.zeros((nV, nU), dtype=bool)
    owner = -np.ones((nV, nU), dtype=np.int32)
    for a, (_, _, vk, vj) in enumerate(V):
        for b, (_, _, uk, uj) in enumerate(U):
            labs = {ext(i, j) for i in _eset(vk, vj) for j in _eset(uk, uj)}
            if labs == {SOLID}:
                code[a, b] = WALL
                exterior[a, b] = True
                continue
            if len(labs) == 1:
                code[a, b] = idx[next(iter(labs))]
                continue
            rooms = [k for k in order if k in labs and kind[k] == ROOM]
            if SOLID in labs or rooms:
                code[a, b] = WALL
                owner[a, b] = idx[rooms[0]] if rooms else -1
                continue
            cand = [k for k in order if k in labs]
            if vk == 'band' and uk == 'strip':
                key = lambda k: (size_of(k)[0], -size_of(k)[0] * size_of(k)[1])
            elif uk == 'band' and vk == 'strip':
                key = lambda k: (size_of(k)[1], -size_of(k)[0] * size_of(k)[1])
            else:
                key = lambda k: (size_of(k)[0] * size_of(k)[1], 0)
            code[a, b] = idx[min(cand, key=key)]
    door_info = []
    for n_win, (k, wall, lo, hi, w) in enumerate(windows):
        i0, i1, j0, j1 = span[k]
        along_v = wall in ('east', 'west')
        band_j = {'east': j1, 'west': j0 - 1}[wall] if along_v else {'north': i0 - 1, 'south': i1}[wall]
        bands = [n for n, (_, _, kd, j) in enumerate(U if along_v else V) if kd == 'band' and j == band_j]
        along = [n for n, (a, b, _, _) in enumerate(V if along_v else U) if a >= lo and b <= hi]
        if not bands or not along:
            raise PlanError(f'{k}: no wall to put a door in on its {wall} side')
        for n_band in bands:
            for n_al in along:
                a, b = (n_al, n_band) if along_v else (n_band, n_al)
                if code[a, b] != WALL:
                    raise PlanError(f'{k}: the {wall} door does not lie inside a wall')
                code[a, b] = DOOR0 - len(door_info)
        centre = (lo + hi) // 2
        if along_v:
            strip = min(strip_v, key=lambda j: 0 if strip_v[j][0] <= centre <= strip_v[j][1] else min(abs(centre - strip_v[j][0]), abs(centre - strip_v[j][1])))
            other = ext(strip, j1 + 1) if wall == 'east' else ext(strip, j0 - 1)
        else:
            strip = min(strip_u, key=lambda j: 0 if strip_u[j][0] <= centre <= strip_u[j][1] else min(abs(centre - strip_u[j][0]), abs(centre - strip_u[j][1])))
            other = ext(i0 - 1, strip) if wall == 'north' else ext(i1 + 1, strip)
        if other == SOLID:
            raise PlanError(f'{k}: the {wall} door would open onto the outside of the building')
        door_info.append((k, wall, other, w, door_ids[n_win]))
    ou0, ov0, ou1, ov1 = plan_rect(plan['origin'])
    cu, cv = ((ou0 + ou1) / 2, (ov0 + ov1) / 2)
    ux = lambda u: float(u - cu) / MM
    vy = lambda v: float(cv - v) / MM
    ulo, uhi = (np.array([s[0] for s in U]), np.array([s[1] for s in U]))
    vlo, vhi = (np.array([s[0] for s in V]), np.array([s[1] for s in V]))

    def bbox(mask):
        rr, cc = np.nonzero(mask)
        area = int(((vhi[rr] - vlo[rr]) * (uhi[cc] - ulo[cc])).sum())
        r0, r1, c0, c1 = (rr.min(), rr.max(), cc.min(), cc.max())
        if area != int((vhi[r1] - vlo[r0]) * (uhi[c1] - ulo[c0])):
            return None
        return Rect(ux(ulo[c0]), vy(vhi[r1]), ux(uhi[c1]), vy(vlo[r0]))
    regions = {}
    for k in order:
        r = bbox(code == idx[k])
        if r is None:
            raise PlanError(f'free space of region {k} is not one rectangle')
        regions[k] = Region(k, kind[k], idx[k], r, wall_rgb=_rgb(specs[k].get('wall_rgb')), floor_rgb=_rgb(specs[k].get('floor_rgb')))
    doors = {}
    for n, (k, wall, other, w, did) in enumerate(door_info):
        r = bbox(code == DOOR0 - n)
        nx, ny = NORMAL[wall]
        cx, cy = r.center
        d = Door(id=did, room=k, wall=wall, to=other, center=(cx, cy), width=w / MM, rect=r, inside=(cx - nx * (t / MM / 2 + 1.0), cy - ny * (t / MM / 2 + 1.0)), outside=(cx + nx * (t / MM / 2 + 1.0), cy + ny * (t / MM / 2 + 1.0)))
        doors[d.id] = d
        regions[k].doors.append(d)
        if regions[k].door is None:
            regions[k].door = d

    def wall_key(o):
        return None if o < 0 else regions[order[o]].wall_rgb
    runs = []
    for a in range(nV):
        b = 0
        while b < nU:
            if code[a, b] != WALL or exterior[a, b]:
                b += 1
                continue
            key, b0 = (wall_key(owner[a, b]), b)
            while b < nU and code[a, b] == WALL and (not exterior[a, b]) and (wall_key(owner[a, b]) == key):
                b += 1
            runs.append([int(ulo[b0]), int(uhi[b - 1]), int(vlo[a]), int(vhi[a]), key])
    merged = []
    for run in sorted(runs, key=lambda r: (r[0], r[1], str(r[4]), r[2])):
        if merged and merged[-1][:2] == run[:2] and (merged[-1][4] == run[4]) and (merged[-1][3] == run[2]):
            merged[-1][3] = run[3]
        else:
            merged.append(run)
    merged.sort(key=lambda r: (r[2], r[0]))
    walls = [WallBox(Rect(ux(u0), vy(v1), ux(u1), vy(v0)), key) for u0, u1, v0, v1, key in merged]
    xe = np.array([ux(u) for u in [int(ulo[0])] + [int(u) for u in uhi]])
    ye = np.array([vy(v) for v in [int(vlo[0])] + [int(v) for v in vhi]][::-1])
    codes = code[::-1].astype(np.int16)
    graph = {k: {} for k in order}
    for a in range(nV):
        for b in range(nU):
            c0 = code[a, b]
            if c0 == WALL:
                continue
            for da, db in ((0, 1), (1, 0)):
                if a + da < nV and b + db < nU:
                    c1 = code[a + da, b + db]
                    if c1 != WALL and c1 != c0:
                        if c0 <= DOOR0 or c1 <= DOOR0:
                            dk = c0 if c0 <= DOOR0 else c1
                            other = c1 if c0 <= DOOR0 else c0
                            if other >= 0:
                                room, to, did = (door_info[DOOR0 - dk][0], order[other], door_info[DOOR0 - dk][4])
                                if to != room:
                                    graph[room].setdefault(to, did)
                                    graph[to].setdefault(room, did)
                        else:
                            ra, rb = (order[c0], order[c1])
                            graph[ra].setdefault(rb, None)
                            graph[rb].setdefault(ra, None)
    seen, queue = ({plan['origin']}, deque([plan['origin']]))
    while queue:
        for n in graph[queue.popleft()]:
            if n not in seen:
                seen.add(n)
                queue.append(n)
    if seen != set(order):
        raise PlanError(f"regions not connected to {plan['origin']}: {sorted(set(order) - seen)}")
    footprint = Rect(ux(ulo[0]), vy(vhi[-1]), ux(uhi[-1]), vy(vlo[0]))
    b = Building(name=plan['name'], description=plan.get('description', ''), wall_t=t / MM, origin=plan['origin'], regions=regions, doors=doors, walls=walls, footprint=footprint, graph=graph, _xe=xe, _ye=ye, _codes=codes, _by_index=list(order), _door_by_code=[d[4] for d in door_info])
    return b

def _rgb(v):
    return None if v is None else tuple((float(c) for c in v))
