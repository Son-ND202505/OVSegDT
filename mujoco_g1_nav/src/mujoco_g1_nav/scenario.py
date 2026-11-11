# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import json
import math
import os
from dataclasses import asdict, dataclass, field
import numpy as np
from .config import PKG_ROOT
from .layout import MIRRORS, apply_layout, make_layout
from .scene import OBJECT_KEYS, set_ego_pitch
from .world_map import WorldMap, _static
TABLE_MAP_RES_M = float(os.environ.get('MUJOCO_G1_TABLE_MAP_RES', 0.1))
TABLE_OFFSET_M = 1.0
START_WALL_CLEAR_M = 0.8
NEAR_TABLE_M = (1.2, 4.0)
SCENARIOS = ('S1a', 'S1b')

@dataclass
class Scenario:
    building: str
    seed: int
    scenario: str
    table_room: str
    table_xy: tuple
    perm: tuple
    mirror: tuple
    target: str
    start_xy: tuple
    start_yaw: float
    start_kind: str
    session_id: str = ''
    query_index: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['table_xy'], d['perm'], d['mirror'], d['start_xy'] = (list(self.table_xy), list(self.perm), list(self.mirror), list(self.start_xy))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'Scenario':
        d = dict(d)
        for k in ('table_xy', 'perm', 'mirror', 'start_xy'):
            d[k] = tuple(d[k])
        return cls(**d)

class ScenarioSampler:

    def __init__(self, building, cfg, wall_clear: float=START_WALL_CLEAR_M, layout=None):
        self.b, self.cfg = (building, cfg)
        xs, ys, X, Y, space, wall_d = _static(building, 0.1)
        ok = space & (wall_d >= wall_clear)
        if layout is not None:
            from .house_map import furniture_distance
            ok &= furniture_distance(layout, X, Y, 0.1) >= wall_clear
        reg = np.full(X.shape, '', dtype=object)
        for r in building.regions.values():
            m = ok & (X >= r.rect.x0) & (X <= r.rect.x1) & (Y >= r.rect.y0) & (Y <= r.rect.y1)
            reg[m] = r.id
        self.cells = {r.id: np.stack([X[reg == r.id], Y[reg == r.id]], 1) for r in building.regions.values()}
        with open(PKG_ROOT / 'configs' / 'splits.json') as f:
            self.splits = json.load(f)

    def sample(self, seed: int, scenario: str='S1b', table_rooms=None, target: str=None, perm_split: str='train', corridor_frac: float=0.4, table_room: str=None, perm=None, mirror=None, near_table: bool=False) -> Scenario:
        if scenario not in SCENARIOS:
            raise ValueError(f'scenario must be one of {SCENARIOS}')
        rng = np.random.default_rng(seed)
        b = self.b
        if table_room is None:
            table_room = b.origin if scenario == 'S1a' else str(rng.choice(list(table_rooms) if table_rooms else b.room_ids))
        if table_room == b.origin and scenario == 'S1a':
            table_xy = tuple((float(v) for v in self.cfg['table']['center_xy']))
        else:
            cx, cy = b.regions[table_room].center
            table_xy = (round(cx + float(rng.uniform(-TABLE_OFFSET_M, TABLE_OFFSET_M)), 3), round(cy + float(rng.uniform(-TABLE_OFFSET_M, TABLE_OFFSET_M)), 3))
        target = target or str(rng.choice(OBJECT_KEYS))
        if perm is None:
            pool = self.splits['train_perms' if perm_split == 'train' else 'heldout_perms']
            perm = tuple((int(i) for i in pool[int(rng.integers(len(pool)))]))
        if mirror is None:
            mirror = MIRRORS[str(rng.choice(['n', 'x', 'y', 'xy']))]
        if near_table:
            hx, hy = (self.cfg['table']['size_xy'][0] / 2, self.cfg['table']['size_xy'][1] / 2)
            pts = self.cells[table_room]
            d = np.hypot(np.maximum(np.abs(pts[:, 0] - table_xy[0]) - hx, 0.0), np.maximum(np.abs(pts[:, 1] - table_xy[1]) - hy, 0.0))
            pts = pts[(d >= NEAR_TABLE_M[0]) & (d <= NEAR_TABLE_M[1])]
            k = int(rng.integers(len(pts)))
            return Scenario(building=b.name, seed=int(seed), scenario=scenario, table_room=table_room, table_xy=table_xy, perm=tuple(perm), mirror=tuple(mirror), target=target, start_xy=(float(pts[k][0]), float(pts[k][1])), start_yaw=float(rng.uniform(-math.pi, math.pi)), start_kind='near:' + table_room)
        rooms_without_table = [r for r in b.room_ids if r != table_room]
        open_ids = [r for r in b.open_ids if len(self.cells[r])]
        room_ids = [r for r in rooms_without_table if len(self.cells[r])]
        use_open = bool(open_ids) and (not room_ids or rng.random() < corridor_frac)
        group = open_ids if use_open else room_ids
        pts = np.concatenate([self.cells[r] for r in group])
        owner = np.concatenate([[r] * len(self.cells[r]) for r in group])
        k = int(rng.integers(len(pts)))
        region = str(owner[k])
        kind = ('open:' if b.regions[region].kind != 'room' else 'room:') + region
        return Scenario(building=b.name, seed=int(seed), scenario=scenario, table_room=table_room, table_xy=table_xy, perm=tuple(perm), mirror=tuple(mirror), target=target, start_xy=(float(pts[k][0]), float(pts[k][1])), start_yaw=float(rng.uniform(-math.pi, math.pi)), start_kind=kind)

def apply_scenario(ctx, scn: Scenario):
    scene = ctx.scene
    scene.place_table(scn.table_xy)
    ctx.cfg = scene.cfg_at_table()
    layout = make_layout(ctx.cfg, scn.seed, 'slots', perm=scn.perm, mirror=scn.mirror)
    apply_layout(scene, layout)
    house = getattr(ctx, 'house', None)
    if house is not None:
        from .house_map import HouseMap
        ctx.navmap = HouseMap(ctx.building, ctx.cfg, house.layout, res=TABLE_MAP_RES_M)
    else:
        ctx.navmap = WorldMap(ctx.building, ctx.cfg, res=TABLE_MAP_RES_M)
    ctx.scenario = scn
    return layout

def building_episode_setup(ctx, scn: Scenario):
    from .goal import valid_approach_poses
    layout = apply_scenario(ctx, scn)
    set_ego_pitch(ctx.scene, ctx.cfg['camera']['pitch_deg'])
    approach = valid_approach_poses(ctx.cfg, ctx.scene, ctx.cam, ctx.probe, ctx.loco.q0, layout, scn.target, ctx.navmap)
    field_ = ctx.navmap.field([(x, y) for x, y, _ in approach]) if approach else None
    shortest = ctx.navmap.path_length(field_, scn.start_xy[0], scn.start_xy[1]) if approach else math.nan
    ctx.last_setup = dict(layout=layout, approach=approach, field=field_, shortest=shortest)
    return (layout, scn.start_xy, scn.start_yaw, approach, shortest)
