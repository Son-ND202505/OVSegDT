# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass, field
import mujoco
import numpy as np
G = mujoco.mjtGeom
HIDDEN_GROUP = 5

@dataclass
class Skin:
    name: str
    host: str
    parts: list
    prompts: tuple = ()
    height_m: float = 0.0
    note: str = ''

def _q_y(deg):
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, math.sin(a), 0.0)
SKINS = {'apple': Skin('apple', 'orange', [dict(type=G.mjGEOM_SPHERE, size=(0.04, 0, 0), pos=(0, 0, 0.04), rgba=(0.78, 0.08, 0.08, 1))], ('apple',), 0.08, 'ball, red'), 'banana': Skin('banana', 'orange', [dict(type=G.mjGEOM_CAPSULE, size=(0.018, 0.075, 0), pos=(0, 0, 0.019), quat=_q_y(90), rgba=(0.95, 0.8, 0.1, 1))], ('banana',), 0.04, 'capsule 18 cm, yellow'), 'bottle': Skin('bottle', 'coca', [dict(type=G.mjGEOM_CYLINDER, size=(0.03, 0.08, 0), pos=(0, 0, 0.08), rgba=(0.1, 0.45, 0.16, 1)), dict(type=G.mjGEOM_CYLINDER, size=(0.012, 0.05, 0), pos=(0, 0, 0.21), rgba=(0.07, 0.3, 0.11, 1))], ('bottle', 'glass bottle'), 0.26, 'cylinder + neck, green'), 'book': Skin('book', 'basket', [dict(type=G.mjGEOM_BOX, size=(0.095, 0.068, 0.002), pos=(0, 0, 0.002), rgba=(0.7, 0.1, 0.12, 1)), dict(type=G.mjGEOM_BOX, size=(0.09, 0.064, 0.011), pos=(0, 0, 0.015), rgba=(0.93, 0.9, 0.8, 1)), dict(type=G.mjGEOM_BOX, size=(0.095, 0.068, 0.002), pos=(0, 0, 0.028), rgba=(0.7, 0.1, 0.12, 1))] + [None] * 5, ('book', 'notebook'), 0.03, 'three boxes, red cover')}

def _rbound(gtype, size):
    s = np.asarray(size, float)
    if gtype == G.mjGEOM_SPHERE:
        return float(s[0])
    if gtype == G.mjGEOM_CAPSULE:
        return float(s[0] + s[1])
    if gtype == G.mjGEOM_CYLINDER:
        return float(math.hypot(s[0], s[1]))
    return float(np.linalg.norm(s))
_SAVED = {}

def apply_skin(scene, name: str):
    sk = SKINS[name]
    m = scene.model
    ids = list(scene.obj_geoms[sk.host])
    if len(sk.parts) != len(ids):
        raise ValueError(f'skin {name!r} has {len(sk.parts)} parts but the host {sk.host!r} has {len(ids)} geoms')
    if any((m.geom_type[g] == G.mjGEOM_MESH for g in ids)):
        raise ValueError('a mesh host cannot be re-shaped')
    if sk.host in _SAVED:
        restore(scene)
    names = ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_rgba', 'geom_matid', 'geom_group', 'geom_rbound')
    _SAVED[sk.host] = (ids, {n: getattr(m, n)[ids].copy() for n in names}, name)
    for g, part in zip(ids, sk.parts):
        m.geom_matid[g] = -1
        if part is None:
            m.geom_group[g] = HIDDEN_GROUP
            m.geom_rgba[g] = (0, 0, 0, 0)
            m.geom_size[g] = (0.0001, 0.0001, 0.0001)
            m.geom_rbound[g] = 0.0001
            continue
        m.geom_type[g] = part['type']
        m.geom_size[g] = part['size']
        m.geom_pos[g] = part['pos']
        m.geom_quat[g] = part.get('quat', (1.0, 0.0, 0.0, 0.0))
        m.geom_rgba[g] = part['rgba']
        m.geom_group[g] = 0
        m.geom_rbound[g] = _rbound(part['type'], part['size'])
    return sk

def restore(scene, host: str=None):
    m = scene.model
    for h in [host] if host else list(_SAVED):
        if h not in _SAVED:
            continue
        ids, saved, _ = _SAVED.pop(h)
        for n, v in saved.items():
            getattr(m, n)[ids] = v

def active() -> dict:
    return {h: v[2] for h, v in _SAVED.items()}
