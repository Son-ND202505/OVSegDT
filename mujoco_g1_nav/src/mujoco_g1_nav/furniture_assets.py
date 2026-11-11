# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass, field
import mujoco
import numpy as np
from .config import PKG_ROOT
from .furniture import CATEGORIES, Item
KENNEY_DIR = PKG_ROOT / 'assets' / 'furniture' / 'kenney'
COLLIDER_GROUP = 4
G = mujoco.mjtGeom
FRONTS = {'+z': ((0, 0, 1), (1, 0, 0)), '-z': ((0, 0, -1), (-1, 0, 0)), '+x': ((1, 0, 0), (0, 0, -1)), '-x': ((-1, 0, 0), (0, 0, 1))}

@dataclass
class Part:
    kind: str
    rgba: tuple
    verts: np.ndarray = None
    size: tuple = None
    pos: tuple = (0.0, 0.0, 0.0)
    name: str = ''
    shiny: float = 0.1

def _parse_mtl(path) -> dict:
    out, cur = ({}, None)
    with open(path) as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            if w[0] == 'newmtl':
                cur = w[1]
            elif w[0] == 'Kd' and cur:
                out[cur] = tuple((float(x) for x in w[1:4]))
    return out
_OBJ_CACHE = {}

def load_obj_parts(model: str) -> list:
    if model in _OBJ_CACHE:
        return _OBJ_CACHE[model]
    path = KENNEY_DIR / f'{model}.obj'
    if not path.exists():
        raise FileNotFoundError(f'{path}: copy it from the Kenney Furniture Kit (Models/OBJ format)')
    cols = _parse_mtl(KENNEY_DIR / f'{model}.mtl')
    verts, faces, cur = ([], {}, 'default')
    with open(path) as f:
        for line in f:
            w = line.split()
            if not w:
                continue
            if w[0] == 'v':
                verts.append([float(x) for x in w[1:4]])
            elif w[0] == 'usemtl':
                cur = w[1]
            elif w[0] == 'f':
                idx = [int(t.split('/')[0]) for t in w[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.setdefault(cur, []).append((idx[0], idx[k], idx[k + 1]))
    V = np.array(verts, dtype=float)
    parts = [(m, cols.get(m, (0.7, 0.7, 0.7)), V[np.array(f).reshape(-1)]) for m, f in faces.items()]
    _OBJ_CACHE[model] = parts
    return parts

def kenney_parts(model: str, size: tuple, front: str='-z') -> list:
    f, l = (np.array(v, dtype=float) for v in FRONTS[front])
    basis = np.stack([f, l, np.array([0.0, 1.0, 0.0])])
    raw = [(m, rgb, v @ basis.T) for m, rgb, v in load_obj_parts(model)]
    allv = np.concatenate([p[2] for p in raw])
    lo, hi = (allv.min(axis=0), allv.max(axis=0))
    ext = np.maximum(hi - lo, 1e-06)
    sc = np.array(size, dtype=float) / ext
    cen = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return [Part('mesh', (*rgb, 1.0), verts=(v - cen) * sc, name=m) for m, rgb, v in raw]

def _box(x, y, z, dx, dy, dz, rgba, name='', shiny=0.1):
    return Part('box', rgba, size=(dx, dy, dz), pos=(x, y, z), name=name, shiny=shiny)

def _cyl(x, y, z, r, h, rgba, name='', shiny=0.1):
    return Part('cyl', rgba, size=(r, h), pos=(x, y, z), name=name, shiny=shiny)
WOOD, WOOD_L, WOOD_D = ((0.58, 0.4, 0.24, 1), (0.68, 0.49, 0.31, 1), (0.33, 0.22, 0.13, 1))
BLACK, WHITE, STEEL = ((0.07, 0.07, 0.08, 1), (0.95, 0.95, 0.93, 1), (0.72, 0.74, 0.76, 1))

def prim_wardrobe(d, w, h):
    t = 0.012
    return [_box(0, 0, 0.0, d, w, h, WOOD, 'body'), _box(0, 0, h - 0.045, d + 0.03, w + 0.03, 0.045, WOOD_D, 'cornice'), _box(0, 0, 0.0, d + 0.03, w + 0.02, 0.07, WOOD_D, 'plinth'), _box(d / 2 + t / 2, -w / 4 - 0.005, 0.1, t, w / 2 - 0.03, h - 0.2, WOOD_L, 'door_r'), _box(d / 2 + t / 2, w / 4 + 0.005, 0.1, t, w / 2 - 0.03, h - 0.2, WOOD_L, 'door_l'), _box(d / 2 + t, -0.05, h * 0.46, 0.02, 0.02, 0.22, STEEL, 'handle_r', 0.6), _box(d / 2 + t, 0.05, h * 0.46, 0.02, 0.02, 0.22, STEEL, 'handle_l', 0.6)]

def prim_piano(d, w, h):
    parts = [_box(-0.06, 0, 0.08, d - 0.12, w, h - 0.08, BLACK, 'body', 0.5), _box(-0.01, 0, 0.0, d - 0.02, w - 0.06, 0.08, BLACK, 'base', 0.4), _box(d / 2 - 0.1, 0, 0.66, 0.2, w - 0.04, 0.05, BLACK, 'key_shelf', 0.5), _box(d / 2 - 0.09, 0, 0.71, 0.15, w - 0.2, 0.025, WHITE, 'keys_white', 0.3)]
    for s in (-1, 1):
        parts.append(_box(d / 2 - 0.1, s * (w / 2 - 0.05), 0.08, 0.2, 0.07, 0.62, BLACK, f'cheek{s}', 0.5))
    nk = 22
    span = w - 0.3
    for k in range(nk):
        if k % 7 in (2, 6):
            continue
        y = -span / 2 + (k + 0.5) * span / nk * 1.0
        parts.append(_box(d / 2 - 0.05, y, 0.735, 0.08, 0.014, 0.022, BLACK, f'bk{k}', 0.3))
    parts.append(_box(d / 2 - 0.2, 0, 0.88, 0.04, w - 0.4, 0.28, BLACK, 'music_stand', 0.4))
    for s in (-0.12, 0, 0.12):
        parts.append(_cyl(d / 2 - 0.12, s, 0.0, 0.02, 0.05, (0.8, 0.65, 0.2, 1), f'pedal{s}', 0.7))
    return parts

def prim_toy_chest(d, w, h):
    blue, yellow = ((0.2, 0.45, 0.8, 1), (0.95, 0.75, 0.15, 1))
    return [_box(0, 0, 0.0, d, w, h - 0.06, blue, 'body'), _box(0, 0, h - 0.06, d + 0.02, w + 0.02, 0.06, yellow, 'lid'), _box(d / 2 + 0.005, 0, h * 0.45, 0.012, 0.09, 0.07, STEEL, 'latch', 0.6), _box(0, -w / 2 - 0.005, 0.1, d - 0.06, 0.012, 0.05, yellow, 'stripe_r'), _box(0, w / 2 + 0.005, 0.1, d - 0.06, 0.012, 0.05, yellow, 'stripe_l')]
GLASS = (0.72, 0.88, 1.0, 0.55)

def prim_glasses(d, w, h):
    r = 0.033
    parts = [_box(0, 0, 0.0, d, w, 0.012, (0.55, 0.38, 0.22, 1), 'tray')]
    for k, (x, y) in enumerate([(-0.055, -0.115), (0.055, -0.115), (-0.055, 0.0), (0.055, 0.0), (0.0, 0.115)] if w >= 0.3 else [(0, -0.05), (0, 0.05)]):
        parts.append(_cyl(x, y, 0.012, r, h - 0.012, GLASS, f'glass{k}', 0.9))
        parts.append(_cyl(x, y, h - 0.004, r * 1.02, 0.004, (0.2, 0.25, 0.3, 1), f'rim{k}', 0.5))
    return parts
_PICTURE_THEMES = ([((0.55, 0.78, 0.95), 0.0, 1.0), ((0.35, 0.6, 0.3), 0.0, 0.45), ((0.98, 0.85, 0.2), 0.55, 0.18)], [((0.15, 0.25, 0.55), 0.0, 1.0), ((0.9, 0.35, 0.25), 0.15, 0.4), ((0.95, 0.9, 0.8), 0.35, 0.3)], [((0.92, 0.88, 0.78), 0.0, 1.0), ((0.25, 0.45, 0.4), 0.2, 0.55), ((0.8, 0.55, 0.25), 0.62, 0.25)], [((0.2, 0.2, 0.25), 0.0, 1.0), ((0.95, 0.95, 0.95), 0.2, 0.6), ((0.85, 0.2, 0.3), 0.4, 0.3)])

def prim_picture(d, w, h, theme=0):
    m = 0.05
    parts = [_box(0, 0, 0.0, d, w, h, WOOD_D, 'frame')]
    cw, ch = (w - 2 * m, h - 2 * m)
    for k, (rgb, z0, hf) in enumerate(_PICTURE_THEMES[theme % len(_PICTURE_THEMES)]):
        if k == 2:
            s = min(cw, ch) * hf
            parts.append(_box(d / 2 + 0.003 * (k + 1), cw * 0.25, m + ch * z0, 0.002, s, s, (*rgb, 1), f'canvas{k}', 0.05))
        else:
            parts.append(_box(d / 2 + 0.003 * (k + 1), 0.0, m + ch * z0, 0.002, cw, ch * hf if k else ch, (*rgb, 1), f'canvas{k}', 0.05))
    return parts

def prim_mirror(d, w, h):
    gold = (0.8, 0.66, 0.3, 1)
    return [_box(0, 0, 0.0, d, w, h, gold, 'frame', 0.7), _box(d / 2 + 0.002, 0, 0.04, 0.004, w - 0.08, h - 0.08, (0.8, 0.9, 0.96, 1), 'glass', 1.0)]
PRIMITIVES = {'wardrobe': prim_wardrobe, 'piano': prim_piano, 'toy_chest': prim_toy_chest, 'glasses': prim_glasses, 'picture': prim_picture, 'mirror': prim_mirror}
KENNEY = {'refrigerator': ('kitchenFridgeLarge', '-z'), 'cabinet': ('bookcaseClosedDoors', '-z'), 'counter': ('kitchenCabinet', '-z'), 'sink': ('kitchenSink', '-z'), 'stove': ('kitchenStove', '-z'), 'trash_can': ('trashcan', '-z'), 'chair': ('chair', '-z'), 'potted_plant': ('pottedPlant', '-z'), 'tv_stand': ('cabinetTelevision', '-z'), 'tv': ('televisionModern', '-z'), 'sofa': ('loungeSofa', '-z'), 'coffee_table': ('tableCoffee', '-z'), 'armchair': ('loungeChair', '-z'), 'stool': ('stoolBar', '-z'), 'bookshelf': ('bookcaseOpen', '-z'), 'lamp': ('lampRoundFloor', '-z'), 'rug': ('rugRectangle', '-z'), 'bed': ('bedDouble', '-z'), 'nightstand': ('cabinetBedDrawer', '-z'), 'desk': ('desk', '-z'), 'round_table': ('tableRound', '-z'), 'bench': ('benchCushion', '-z'), 'console_table': ('sideTableDrawers', '-z'), 'shoe_rack': ('bookcaseOpenLow', '-z')}

def item_parts(item: Item, spec: dict=None) -> list:
    spec = spec or {}
    size = (item.depth, item.width, item.height)
    if 'model' not in spec and item.cat in PRIMITIVES:
        fn = PRIMITIVES[item.cat]
        if item.cat == 'picture':
            return fn(*size, theme=int(spec.get('theme', sum(map(ord, item.id)))))
        return fn(*size)
    model, front = KENNEY.get(item.cat, (None, '-z'))
    model, front = (spec.get('model', model), spec.get('front', front))
    if model is None:
        raise KeyError(f'no model for category {item.cat!r}')
    return kenney_parts(model, size, front)

@dataclass
class FurnitureNames:
    body: dict = field(default_factory=dict)
    visual: dict = field(default_factory=dict)
    collider: dict = field(default_factory=dict)

def yaw_quat(yaw: float) -> list:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

def _material_for(spec, cache: dict, rgba: tuple, shiny: float) -> str:
    key = tuple((round(float(c), 3) for c in rgba)) + (round(shiny, 2),)
    if key not in cache:
        name = f'fmat_{len(cache)}'
        spec.add_material(name=name, rgba=list(key[:4]), specular=min(1.0, 0.1 + 0.5 * shiny), shininess=min(1.0, 0.1 + 0.6 * shiny))
        cache[key] = name
    return cache[key]

def add_item(spec, wb, it: Item, ispec: dict, mats: dict, names: FurnitureNames, body_name: str=None):
    parts = item_parts(it, ispec)
    bname = body_name or f'f_{it.id}'
    body = wb.add_body(name=bname, pos=[it.x, it.y, it.z0], quat=yaw_quat(it.yaw))
    names.body[it.id] = bname
    names.visual[it.id] = []
    for k, p in enumerate(parts):
        gname = f'{bname}_v{k}'
        common = dict(name=gname, material=_material_for(spec, mats, p.rgba, p.shiny), contype=0, conaffinity=0, mass=0.0)
        if p.kind == 'mesh':
            mname = f'{bname}_m{k}'
            spec.add_mesh(name=mname, uservert=p.verts.reshape(-1).tolist(), userface=np.arange(len(p.verts), dtype=np.int32).tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            body.add_geom(type=G.mjGEOM_MESH, meshname=mname, **common)
        elif p.kind == 'box':
            dx, dy, dz = p.size
            body.add_geom(type=G.mjGEOM_BOX, size=[dx / 2, dy / 2, dz / 2], pos=[p.pos[0], p.pos[1], p.pos[2] + dz / 2], **common)
        elif p.kind == 'cyl':
            r, hh = p.size
            body.add_geom(type=G.mjGEOM_CYLINDER, size=[r, hh / 2, 0.0], pos=[p.pos[0], p.pos[1], p.pos[2] + hh / 2], **common)
        else:
            body.add_geom(type=G.mjGEOM_SPHERE, size=[p.size[0], 0, 0], pos=list(p.pos), **common)
        names.visual[it.id].append(gname)
    if it.blocks:
        cname = f'{bname}_col'
        if it.shape == 'round':
            body.add_geom(name=cname, type=G.mjGEOM_CYLINDER, size=[it.depth / 2, it.height / 2, 0.0], pos=[0, 0, it.height / 2], group=COLLIDER_GROUP, mass=0.0, rgba=[1, 0, 0, 0.3])
        else:
            body.add_geom(name=cname, type=G.mjGEOM_BOX, size=[it.depth / 2, it.width / 2, it.height / 2], pos=[0, 0, it.height / 2], group=COLLIDER_GROUP, mass=0.0, rgba=[1, 0, 0, 0.3])
        names.collider[it.id] = cname

def add_furniture(spec, wb, layout, skip_ids=('table_0',)) -> FurnitureNames:
    names = FurnitureNames()
    mats = {}
    specs = {s['id']: s for s in layout.specs}
    for it in layout.items:
        if it.id not in skip_ids:
            add_item(spec, wb, it, specs.get(it.id), mats, names)
    return names
