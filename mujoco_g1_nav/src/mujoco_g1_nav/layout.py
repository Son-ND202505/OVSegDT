# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import itertools
import math
from dataclasses import dataclass, field
import numpy as np
from .scene import OBJECT_KEYS, Scene, yaw_quat
ROLE_NAMES = ('center', 'corner', 'long_edge', 'short_edge', 'free')

@dataclass
class Layout:
    seed: int
    mode: str
    objects: dict = field(default_factory=dict)
    role_of: dict = field(default_factory=dict)
    mirror: tuple = (1, 1)
    perm: tuple = None

def all_slot_permutations():
    return list(itertools.permutations(range(5)))

def _free_point(rng, cfg, taken):
    cx, cy = cfg['table']['center_xy']
    sx, sy = cfg['table']['size_xy']
    margin, spacing = (cfg['free_slot']['edge_margin_m'], cfg['free_slot']['min_spacing_m'])
    for _ in range(2000):
        p = (cx + rng.uniform(-sx / 2 + margin, sx / 2 - margin), cy + rng.uniform(-sy / 2 + margin, sy / 2 - margin))
        if all((math.dist(p, q) >= spacing for q in taken)):
            return p
    raise RuntimeError('could not place the free object')
MIRRORS = {'n': (1, 1), 'x': (-1, 1), 'y': (1, -1), 'xy': (-1, -1)}

def make_layout(cfg: dict, seed: int, mode: str='slots', perm=None, mirror=None) -> Layout:
    rng = np.random.default_rng(seed)
    cx, cy = cfg['table']['center_xy']
    lay = Layout(seed=seed, mode=mode)
    if mode == 'slots':
        if perm is None:
            perm = all_slot_permutations()[int(rng.integers(120))]
        if mirror is None:
            mirror = (int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))) if cfg.get('mirror_slots') else (1, 1)
        perm, (fx, fy) = (tuple((int(i) for i in perm)), mirror)
        lay.mirror, lay.perm = ((fx, fy), perm)
        jit = cfg['slot_jitter_m']
        fixed = []
        for role in ROLE_NAMES[:4]:
            dx, dy = cfg['slots'][role]
            fixed.append((cx + fx * dx + rng.uniform(-jit, jit), cy + fy * dy + rng.uniform(-jit, jit)))
        points = fixed + [_free_point(rng, cfg, fixed)]
        for key, role_idx in zip(OBJECT_KEYS, perm):
            x, y = points[role_idx]
            lay.objects[key] = (x, y, float(rng.uniform(-math.pi, math.pi)))
            lay.role_of[key] = ROLE_NAMES[role_idx]
    elif mode == 'random':
        placed = []
        margin, spacing = (cfg['random_layout']['edge_margin_m'], cfg['random_layout']['min_spacing_m'])
        sx, sy = cfg['table']['size_xy']
        for key in OBJECT_KEYS:
            for _ in range(2000):
                p = (cx + rng.uniform(-sx / 2 + margin, sx / 2 - margin), cy + rng.uniform(-sy / 2 + margin, sy / 2 - margin))
                if all((math.dist(p, q) >= spacing for q in placed)):
                    break
            else:
                raise RuntimeError('could not place objects without overlap')
            placed.append(p)
            lay.objects[key] = (p[0], p[1], float(rng.uniform(-math.pi, math.pi)))
    else:
        raise ValueError(mode)
    return lay

def apply_layout(scene: Scene, layout: Layout):
    m = scene.model
    for key, (x, y, yaw) in layout.objects.items():
        b = scene.obj_body[key]
        m.body_pos[b] = [x, y, scene.table_top_z]
        m.body_quat[b] = yaw_quat(yaw)

def start_pose(cfg: dict, start_id: int):
    s = cfg['starts'][start_id]
    return (tuple(s['xy']), math.radians(s['yaw_deg']))

def catalog_index(perm) -> int:
    return all_slot_permutations().index(tuple(perm))

def describe_perm(perm) -> str:
    return ' '.join((f'{k}={ROLE_NAMES[r]}' for k, r in zip(OBJECT_KEYS, perm)))

def parse_layout_spec(spec: str) -> dict:
    idx, _, mir = spec.partition(':')
    return dict(perm=all_slot_permutations()[int(idx)], mirror=MIRRORS[mir or 'n'])

def balanced_playlist(k: int) -> list:
    if k not in (5, 10, 20, 40):
        raise ValueError('balanced playlist size must be 5, 10, 20 or 40')
    shifts = [tuple(((i + s) % 5 for i in range(5))) for s in range(5)]
    reverse = [tuple(((s - i) % 5 for i in range(5))) for s in range(5)]
    perms = shifts if k in (5, 20) else shifts + reverse
    mirrors = [MIRRORS['n']] if k in (5, 10) else [MIRRORS[m] for m in ('n', 'x', 'y', 'xy')]
    return [dict(perm=p, mirror=m) for p in perms for m in mirrors]

def layout_key(layout: Layout) -> tuple:
    perm = layout.perm if layout.perm is not None else tuple((ROLE_NAMES.index(layout.role_of[k]) for k in OBJECT_KEYS))
    return (tuple(perm), tuple(layout.mirror))
