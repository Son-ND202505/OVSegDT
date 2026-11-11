# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import colorsys
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .config import PKG_ROOT
from .furniture import Layout
from .furniture_assets import yaw_quat
from .house_check import check_house, sample_valid_layout
POOL_DIR = PKG_ROOT / 'outputs' / 'house_variants'
NO_OVERRIDE = (0.5, 0.5, 0.5, 1.0)
COLOUR_P = {'table': 0.0, 'tv': 0.0, 'glasses': 0.0, 'mirror': 0.0, 'potted_plant': 0.3, 'picture': 0.5, 'default': 0.7}

@dataclass
class Variant:
    index: int
    layout: Layout
    stand_xy: dict
    seed: int = None
    stats: dict = field(default_factory=dict)

@dataclass
class StandReport:
    stand_xy: dict

def _stats(rep, attempts: int=1) -> dict:
    s = rep.stats
    return dict(attempts=attempts, circulation_m=round(float(s.get('circulation_aisle_m', float('nan'))), 3), min_item_aisle_m=round(float(s.get('min_item_aisle_m', float('nan'))), 3), n_narrow=int(s.get('n_narrow', 0)))

def build_pool(layout: Layout, n: int, seed: int=0, sigma: float=0.3, tries: int=40, cfg: dict=None, progress=None) -> list:
    rep0 = check_house(layout.building, layout, cfg)
    if not rep0.ok:
        raise RuntimeError(f'the base layout is not valid: {rep0.failures()[:3]}')
    pool = [Variant(0, layout, {k: np.asarray(v, dtype=np.float32) for k, v in rep0.stand_xy.items()}, None, _stats(rep0))]
    rng = np.random.default_rng(seed)
    for k in range(1, n + 1):
        cand, rep, attempts = sample_valid_layout(layout, rng, tries=tries, sigma=sigma, cfg=cfg)
        pool.append(Variant(k, cand, {i: np.asarray(v, dtype=np.float32) for i, v in rep.stand_xy.items()}, seed, _stats(rep, attempts)))
        if progress:
            progress(k, n)
    return pool

def save_pool(pool: list, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lay = pool[0].layout
    blob = dict(building=lay.building.name, name=lay.name, rooms=lay.rooms, source=lay.source, variants=[dict(index=v.index, specs=v.layout.specs, stand_xy=v.stand_xy, seed=v.seed, stats=v.stats) for v in pool])
    with open(path, 'wb') as f:
        pickle.dump(blob, f, protocol=4)
    return path

def load_pool(path, building) -> list:
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    if blob['building'] != building.name:
        raise ValueError(f"pool built for {blob['building']!r}, not {building.name!r}")
    return [Variant(v['index'], Layout(building, v['specs'], name=blob['name'], rooms=blob['rooms'], source=blob['source']), v['stand_xy'], v['seed'], v['stats']) for v in blob['variants']]

def base_variant(env) -> Variant:
    rep = check_house(env.building, env.layout, env.cfg)
    return Variant(0, env.layout, {k: np.asarray(v, dtype=np.float32) for k, v in rep.stand_xy.items()}, None, _stats(rep))

def _snapshot_colours(ctx) -> dict:
    cache = getattr(ctx, '_base_rgba', None)
    if cache is None:
        m = ctx.scene.model
        cache = {}
        for iid, geoms in ctx.house.visual_geoms.items():
            if iid == 'table_0':
                continue
            for g in geoms:
                mat = int(m.geom_matid[g])
                cache[int(g)] = np.array(m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g], dtype=float)
        ctx._base_rgba = cache
    return cache

def reset_colours(ctx):
    m = ctx.scene.model
    for g in _snapshot_colours(ctx):
        m.geom_rgba[g] = NO_OVERRIDE

def randomize_colours(ctx, rng, hue_sat: tuple=(0.7, 1.15), hue_val: tuple=(0.85, 1.15)) -> int:
    base = _snapshot_colours(ctx)
    m = ctx.scene.model
    reset_colours(ctx)
    n = 0
    for it in ctx.house.layout.items:
        if it.id == 'table_0':
            continue
        p = COLOUR_P.get(it.cat, COLOUR_P['default'])
        if p <= 0 or rng.random() >= p:
            continue
        dh, ds, dv = (rng.random(), rng.uniform(*hue_sat), rng.uniform(*hue_val))
        for g in ctx.house.visual_geoms.get(it.id, []):
            r, gg, b, a = base[int(g)]
            h, s, v = colorsys.rgb_to_hsv(r, gg, b)
            r2, g2, b2 = colorsys.hsv_to_rgb((h + dh) % 1.0, min(1.0, s * ds), min(1.0, v * dv))
            m.geom_rgba[g] = [r2, g2, b2, a]
        n += 1
    return n

def move_furniture(ctx, layout: Layout):
    h, m = (ctx.house, ctx.scene.model)
    for it in layout.items:
        bid = h.body_id.get(it.id)
        if bid is None or (it.fixed and it.cat == 'table'):
            continue
        m.body_pos[bid] = [it.x, it.y, it.z0]
        m.body_quat[bid] = yaw_quat(it.yaw)
    h.layout = layout

class HouseSession:

    def __init__(self, env=None, pool: list=None, cache_size: int=4, ctx=None):
        from .envs import load_house_env, make_env_context
        self.env = env or load_house_env('home25')
        self.ctx = ctx or make_env_context(self.env)
        self.pool = pool or [base_variant(self.env)]
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.variant = None
        self.world = None
        from .scenario import TABLE_MAP_RES_M
        self._res = TABLE_MAP_RES_M
        scene = self.ctx.scene
        scene.place_table(tuple((float(v) for v in self.env.cfg['table']['center_xy'])))
        self.ctx.cfg = scene.cfg_at_table()
        self.set_variant(0)

    def _maps(self, v: Variant):
        from .house_eval import CategoryWorld
        from .house_map import HouseMap
        if v.index not in self._cache:
            ctx = self.ctx
            nav = HouseMap(ctx.building, ctx.cfg, v.layout, res=self._res)
            static = HouseMap(ctx.building, ctx.cfg, v.layout, with_table=False)
            self._cache[v.index] = dict(navmap=nav, static=static, world=None)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(v.index)
        return self._cache[v.index]

    def set_variant(self, k: int):
        from .house_eval import CategoryWorld
        v = self.pool[k]
        ctx = self.ctx
        move_furniture(ctx, v.layout)
        e = self._maps(v)
        ctx.navmap, ctx.navmap_static = (e['navmap'], e['static'])
        if e['world'] is None:
            e['world'] = CategoryWorld(ctx, StandReport(v.stand_xy))
        self.variant, self.world = (v, e['world'])
        import mujoco
        mujoco.mj_forward(ctx.scene.model, ctx.sim)
        return v
