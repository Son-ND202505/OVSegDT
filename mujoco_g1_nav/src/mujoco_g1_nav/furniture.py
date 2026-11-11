# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml
from .building import NORMAL, Building, load_building
from .config import PKG_ROOT
HOUSES_DIR = PKG_ROOT / 'configs' / 'houses'
KINDS = ('floor', 'surface', 'wall', 'flat')
DEFAULT_GAP_M = 0.03

@dataclass(frozen=True)
class Category:
    name: str
    size: tuple
    kind: str = 'floor'
    shape: str = 'box'
    group: str = 'storage'
    z0: float = 0.0
    front_phi: float = None
    prompts: tuple = ()

def _cats(*rows) -> dict:
    out = {}
    for name, size, kind, shape, group, z0, phi, prompts in rows:
        out[name] = Category(name, tuple(size), kind, shape, group, z0, phi, tuple(prompts))
    return out
CATEGORIES = _cats(('table', (0.8, 1.4, 0.75), 'floor', 'box', 'table', 0.0, None, ('table', 'dining table')), ('refrigerator', (0.75, 0.8, 1.8), 'floor', 'box', 'appliance', 0.0, 60.0, ('refrigerator', 'fridge')), ('cabinet', (0.6, 0.9, 2.0), 'floor', 'box', 'storage', 0.0, 60.0, ('cabinet', 'kitchen cabinet', 'cupboard')), ('counter', (0.65, 1.2, 0.9), 'floor', 'box', 'appliance', 0.0, 60.0, ('kitchen counter', 'counter')), ('sink', (0.65, 1.6, 0.9), 'floor', 'box', 'appliance', 0.0, 60.0, ('sink', 'kitchen sink')), ('stove', (0.65, 0.75, 0.9), 'floor', 'box', 'appliance', 0.0, 60.0, ('stove', 'oven', 'cooker')), ('trash_can', (0.35, 0.35, 0.6), 'floor', 'round', 'decor', 0.0, None, ('trash can', 'bin')), ('chair', (0.45, 0.45, 0.9), 'floor', 'box', 'seating', 0.0, 90.0, ('chair',)), ('potted_plant', (0.5, 0.5, 1.1), 'floor', 'round', 'decor', 0.0, None, ('potted plant', 'plant')), ('tv_stand', (0.4, 1.6, 0.5), 'floor', 'box', 'storage', 0.0, 60.0, ('tv stand', 'tv cabinet')), ('tv', (0.08, 1.2, 0.7), 'surface', 'box', 'electronics', 0.0, 60.0, ('tv', 'television')), ('sofa', (0.9, 2.1, 0.85), 'floor', 'box', 'seating', 0.0, 60.0, ('sofa', 'couch')), ('coffee_table', (0.6, 1.1, 0.42), 'floor', 'box', 'table', 0.0, None, ('coffee table',)), ('glasses', (0.25, 0.35, 0.12), 'surface', 'box', 'decor', 0.0, None, ('set of glasses', 'drinking glasses', 'glass cups')), ('armchair', (0.85, 0.85, 0.9), 'floor', 'box', 'seating', 0.0, 60.0, ('armchair',)), ('piano', (0.6, 1.5, 1.2), 'floor', 'box', 'electronics', 0.0, 60.0, ('piano',)), ('stool', (0.35, 0.35, 0.45), 'floor', 'round', 'seating', 0.0, None, ('stool',)), ('bookshelf', (0.35, 1.0, 1.8), 'floor', 'box', 'storage', 0.0, 60.0, ('bookshelf', 'bookcase')), ('lamp', (0.35, 0.35, 1.6), 'floor', 'round', 'decor', 0.0, None, ('lamp', 'floor lamp')), ('rug', (2.2, 3.0, 0.015), 'flat', 'box', 'decor', 0.0, None, ('rug', 'carpet')), ('picture', (0.04, 0.9, 0.6), 'wall', 'box', 'decor', 0.95, 45.0, ('picture', 'painting', 'picture frame')), ('bed', (2.1, 1.6, 0.55), 'floor', 'box', 'bed', 0.0, 90.0, ('bed',)), ('nightstand', (0.4, 0.45, 0.55), 'floor', 'box', 'storage', 0.0, 60.0, ('nightstand', 'bedside table')), ('wardrobe', (0.6, 1.8, 2.1), 'floor', 'box', 'storage', 0.0, 60.0, ('wardrobe', 'closet')), ('desk', (0.6, 1.2, 0.75), 'floor', 'box', 'table', 0.0, 90.0, ('desk',)), ('mirror', (0.05, 0.55, 1.6), 'wall', 'box', 'decor', 0.1, 45.0, ('mirror',)), ('toy_chest', (0.4, 0.8, 0.45), 'floor', 'box', 'storage', 0.0, 60.0, ('toy chest', 'toy box')), ('round_table', (0.9, 0.9, 0.7), 'floor', 'round', 'table', 0.0, None, ('round table', 'small table')), ('bench', (0.4, 1.2, 0.45), 'floor', 'box', 'seating', 0.0, 60.0, ('bench',)), ('console_table', (0.35, 1.1, 0.8), 'floor', 'box', 'table', 0.0, 60.0, ('console table', 'side table')), ('shoe_rack', (0.35, 0.9, 0.45), 'floor', 'box', 'storage', 0.0, 60.0, ('shoe rack', 'shoe shelf')))

class LayoutError(ValueError):
    pass

@dataclass
class Item:
    id: str
    cat: str
    region: str
    x: float
    y: float
    yaw: float
    depth: float
    width: float
    height: float
    z0: float
    kind: str
    shape: str
    parent: str = None
    wall: str = None
    gap: float = None
    group: str = None
    fixed: bool = False

    @property
    def front(self) -> tuple:
        return (math.cos(self.yaw), math.sin(self.yaw))

    @property
    def top(self) -> float:
        return self.z0 + self.height

    @property
    def blocks(self) -> bool:
        return self.kind == 'floor'

    def local(self, X, Y):
        c, s = (math.cos(self.yaw), math.sin(self.yaw))
        dx, dy = (np.asarray(X, dtype=float) - self.x, np.asarray(Y, dtype=float) - self.y)
        return (c * dx + s * dy, -s * dx + c * dy)

    def contains(self, X, Y, margin: float=0.0):
        lx, ly = self.local(X, Y)
        if self.shape == 'round':
            return lx * lx + ly * ly <= (self.depth / 2 + margin) ** 2
        return (np.abs(lx) <= self.depth / 2 + margin) & (np.abs(ly) <= self.width / 2 + margin)

    def distance(self, X, Y):
        lx, ly = self.local(X, Y)
        if self.shape == 'round':
            return np.maximum(np.hypot(lx, ly) - self.depth / 2, 0.0)
        return np.hypot(np.maximum(np.abs(lx) - self.depth / 2, 0.0), np.maximum(np.abs(ly) - self.width / 2, 0.0))

    def corners(self) -> np.ndarray:
        c, s = (math.cos(self.yaw), math.sin(self.yaw))
        hd, hw = (self.depth / 2, self.width / 2)
        loc = np.array([[-hd, -hw], [hd, -hw], [hd, hw], [-hd, hw]])
        return np.stack([self.x + c * loc[:, 0] - s * loc[:, 1], self.y + s * loc[:, 0] + c * loc[:, 1]], axis=1)

    def bbox(self) -> tuple:
        if self.shape == 'round':
            r = self.depth / 2
            return (self.x - r, self.y - r, self.x + r, self.y + r)
        k = self.corners()
        return (float(k[:, 0].min()), float(k[:, 1].min()), float(k[:, 0].max()), float(k[:, 1].max()))

    def to_local_point(self, dx, dy) -> tuple:
        c, s = (math.cos(self.yaw), math.sin(self.yaw))
        return (self.x + c * dx - s * dy, self.y + s * dx + c * dy)

    def label(self) -> str:
        return self.cat.replace('_', ' ')

def _place(spec: dict, cat: Category, building: Building, placed: dict):
    size = spec.get('size', cat.size)
    depth, width, height = (float(v) for v in size)
    kind, shape = (spec.get('kind', 'surface' if 'on_top' in spec else cat.kind), spec.get('shape', cat.shape))
    if kind not in KINDS:
        raise LayoutError(f"{spec.get('id', cat.name)}: unknown kind {kind!r}")
    if shape == 'round':
        width = depth
    z0 = float(spec.get('z0', cat.z0))
    region = building.regions.get(spec['room'])
    if region is None:
        raise LayoutError(f"{spec.get('id', cat.name)}: unknown room {spec['room']!r}")
    r = region.rect
    cx, cy = r.center
    wall = gap = None
    if 'wall' in spec:
        wall = spec['wall']
        if wall not in NORMAL:
            raise LayoutError(f"{spec.get('id', cat.name)}: wall must be one of {tuple(NORMAL)}")
        gap, along = (float(spec.get('gap', 0.0 if kind == 'wall' else DEFAULT_GAP_M)), float(spec.get('along', 0.0)))
        yaw = math.atan2(-NORMAL[wall][1], -NORMAL[wall][0])
        if wall == 'north':
            x, y = (cx + along, r.y1 - gap - depth / 2)
        elif wall == 'south':
            x, y = (cx + along, r.y0 + gap + depth / 2)
        elif wall == 'east':
            x, y = (r.x1 - gap - depth / 2, cy + along)
        else:
            x, y = (r.x0 + gap + depth / 2, cy + along)
    elif 'on_top' in spec or 'rel' in spec:
        ref = spec.get('on_top') or spec['rel']
        if ref not in placed:
            raise LayoutError(f"{spec.get('id', cat.name)}: {ref!r} must be listed before it")
        p = placed[ref]
        dx, dy = spec.get('at', (0.0, 0.0))
        x, y = p.to_local_point(float(dx), float(dy))
        yaw = p.yaw + math.radians(float(spec.get('yaw', 0.0)))
        if 'z0' not in spec and 'on_top' in spec:
            z0 = p.top
    elif 'world' in spec:
        x, y = (float(v) for v in spec['world'])
        yaw = math.radians(float(spec.get('yaw', 0.0)))
    else:
        dx, dy = spec.get('at', (0.0, 0.0))
        x, y = (cx + float(dx), cy + float(dy))
        yaw = math.radians(float(spec.get('yaw', 0.0)))
    return (x, y, yaw, depth, width, height, z0, kind, shape, wall, gap)

class Layout:

    def __init__(self, building: Building, specs: list, name: str='', rooms: dict=None, source: str=''):
        self.building, self.name, self.rooms, self.source = (building, name, dict(rooms or {}), source)
        self.specs = []
        counts = {}
        for s in specs:
            s = copy.deepcopy(dict(s))
            if s.get('cat') not in CATEGORIES:
                raise LayoutError(f"unknown category {s.get('cat')!r} (known: {sorted(CATEGORIES)})")
            if 'id' not in s:
                k = counts.get(s['cat'], 0)
                s['id'] = f"{s['cat']}_{k}"
            counts[s['cat']] = counts.get(s['cat'], 0) + 1
            self.specs.append(s)
        if len({s['id'] for s in self.specs}) != len(self.specs):
            dup = sorted({s['id'] for s in self.specs if sum((t['id'] == s['id'] for t in self.specs)) > 1})
            raise LayoutError(f'duplicate item ids {dup}')
        anchors = {}
        for s in self.specs:
            if s.get('group') and 'on_top' not in s and ('rel' not in s):
                anchors.setdefault(s['group'], set()).add(s.get('wall', 'free'))
        if (bad := {g: sorted(a) for g, a in anchors.items() if len(a) > 1}):
            raise LayoutError(f'groups mixing different walls / free-standing items: {bad}')
        self.items = []
        self.by_id = {}
        for s in self.specs:
            cat = CATEGORIES[s['cat']]
            x, y, yaw, d, w, h, z0, kind, shape, wall, gap = _place(s, cat, building, self.by_id)
            it = Item(id=s['id'], cat=cat.name, region=s['room'], x=x, y=y, yaw=yaw, depth=d, width=w, height=h, z0=z0, kind=kind, shape=shape, parent=s.get('on_top') or s.get('rel'), wall=wall, gap=gap, group=s.get('group'), fixed=bool(s.get('fixed', False)))
            self.items.append(it)
            self.by_id[it.id] = it

    def __len__(self) -> int:
        return len(self.items)

    def of_region(self, rid: str) -> list:
        return [i for i in self.items if i.region == rid]

    def of_cat(self, cat: str) -> list:
        return [i for i in self.items if i.cat == cat]

    def categories(self) -> list:
        return sorted({i.cat for i in self.items})

    def blockers(self) -> list:
        return [i for i in self.items if i.blocks]

    def summary(self) -> str:
        lines = [f'{self.name}: {len(self)} items, {len(self.categories())} categories']
        for rid, reg in self.building.regions.items():
            its = self.of_region(rid)
            if its:
                lines.append(f'  {rid} ({self.rooms.get(rid, reg.kind)}): {len(its)} items: ' + ', '.join(sorted((i.cat for i in its))))
        return '\n'.join(lines)

    def perturbed(self, rng, sigma: float=0.3, yaw_deg: float=8.0, gap: tuple=(0.02, 0.1), rel_sigma: float=0.12, rel_yaw_deg: float=10.0) -> 'Layout':
        shifts = {}

        def shift(gid):
            if gid not in shifts:
                shifts[gid] = (rng.uniform(-sigma, sigma), rng.uniform(-sigma, sigma), rng.normal(0.0, yaw_deg), rng.uniform(*gap))
            return shifts[gid]
        specs = copy.deepcopy(self.specs)
        for s in specs:
            cat = CATEGORIES[s['cat']]
            kind, shape = (s.get('kind', cat.kind), s.get('shape', cat.shape))
            if s.get('fixed') or 'on_top' in s:
                continue
            if 'rel' in s:
                x, y = s.get('at', (0.0, 0.0))
                s['at'] = [round(float(x) + rng.uniform(-rel_sigma, rel_sigma), 3), round(float(y) + rng.uniform(-rel_sigma, rel_sigma), 3)]
                if shape != 'round':
                    s['yaw'] = round(float(s.get('yaw', 0.0)) + rng.normal(0.0, rel_yaw_deg), 2)
                continue
            du, dv, dyaw, g = shift(s.get('group') or s['id'])
            if 'wall' in s:
                s['along'] = round(float(s.get('along', 0.0)) + du, 3)
                if kind != 'wall':
                    s['gap'] = round(g, 3)
            else:
                key = 'world' if 'world' in s else 'at'
                x, y = s.get(key, (0.0, 0.0))
                s[key] = [round(float(x) + du, 3), round(float(y) + dv, 3)]
                if not s.get('group') and shape != 'round':
                    s['yaw'] = round(float(s.get('yaw', 0.0)) + dyaw, 2)
        return Layout(self.building, specs, name=self.name, rooms=self.rooms, source=self.source)

    def to_dict(self) -> dict:
        return {'name': self.name, 'building': self.building.name, 'rooms': self.rooms, 'items': copy.deepcopy(self.specs)}

    def dump(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=None, width=160)

def load_layout(name_or_path, building: Building=None) -> Layout:
    p = Path(name_or_path)
    if p.suffix == '':
        p = HOUSES_DIR / f'{name_or_path}.yaml'
    with open(p) as f:
        d = yaml.safe_load(f)
    if str(d.get('name', p.stem)).endswith('_plus'):
        enable_plus_catalogue()
    if building is None:
        building = load_building(d['building'])
    elif building.name != d.get('building', building.name):
        raise LayoutError(f"{p.name} is a layout for {d['building']!r}, not {building.name!r}")
    return Layout(building, d['items'], name=d.get('name', p.stem), rooms=d.get('rooms'), source=str(p))

def load_house(name: str) -> tuple:
    lay = load_layout(name)
    return (lay.building, lay)

def enable_plus_catalogue() -> None:
    from .furniture_plus import PLUS_ROWS
    CATEGORIES.update(_cats(*PLUS_ROWS))
import os as _os
if _os.environ.get('G1_CATALOGUE') == 'plus':
    enable_plus_catalogue()
