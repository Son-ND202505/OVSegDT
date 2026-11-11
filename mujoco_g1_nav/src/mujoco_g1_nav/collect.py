# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass
import mujoco
import numpy as np
from .actions import ActionExecutor
from .goal import mask_stats
from .layout import start_pose
from .locomotion import G1Locomotion
from .nav import NavMap, rect_distance
from .scene import OBJECT_KEYS, Scene, build_scene
from .sensors import EgoCamera
ACTION_NAMES = {0: 'stop', 1: 'move_forward', 2: 'turn_left', 3: 'turn_right'}

def resolve_start(cfg, start, rng=None):
    if start == 'random':
        xy, yaw = random_start(cfg, rng)
        return (xy, yaw, 'rand')
    if isinstance(start, str) and start.startswith('manual:'):
        name = start.split(':', 1)[1]
        s = next((m for m in cfg['manual_starts'] if m['name'] == name))
        return (tuple(s['xy']), math.radians(s['yaw_deg']), name)
    xy, yaw = start_pose(cfg, int(start))
    return (xy, yaw, f's{int(start)}')

def random_start(cfg, rng):
    half = cfg['room']['size_xy'][0] / 2 - 0.8
    while True:
        x, y = rng.uniform(-half, half, 2)
        if float(rect_distance(x, y, cfg)) >= 1.6:
            return ((float(x), float(y)), float(rng.uniform(-math.pi, math.pi)))

@dataclass
class Context:
    cfg: dict
    scene: Scene
    sim: mujoco.MjData
    probe: mujoco.MjData
    loco: G1Locomotion
    cam: EgoCamera
    navmap: NavMap
    executor: ActionExecutor
    geom_code: np.ndarray
    robot_geoms: np.ndarray
    env_geoms: np.ndarray
    collisions: list

def make_context(cfg) -> Context:
    scene = build_scene(cfg)
    m = scene.model
    sim, probe = (mujoco.MjData(m), mujoco.MjData(m))
    loco = G1Locomotion(m, sim, cfg)
    cam = EgoCamera(scene)
    geom_code = np.zeros(m.ngeom + 1, dtype=np.uint8)
    for k, key in enumerate(OBJECT_KEYS):
        geom_code[scene.obj_geoms[key]] = k + 1
    robot_geoms = np.nonzero(m.body_rootid[m.geom_bodyid] == scene.pelvis_id)[0]
    wall_ids = [m.geom(n).id for n in ('wall_n', 'wall_s', 'wall_e', 'wall_w')]
    env_geoms = np.array(scene.table_geoms + wall_ids)
    ctx = Context(cfg, scene, sim, probe, loco, cam, NavMap(cfg), None, geom_code, robot_geoms, env_geoms, [0])

    def count_contacts():
        d = ctx.sim
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = (c.geom1, c.geom2)
            if a in ctx.robot_geoms and b in ctx.env_geoms or (b in ctx.robot_geoms and a in ctx.env_geoms):
                ctx.collisions[0] += 1
                break
    ctx.executor = ActionExecutor(loco, cfg, on_step=count_contacts)
    return ctx

def instance_map(ctx: Context, seg_ids: np.ndarray) -> np.ndarray:
    return ctx.geom_code[np.where(seg_ids < 0, len(ctx.geom_code) - 1, seg_ids)]

@dataclass
class Obs:
    rgb: np.ndarray
    inst: np.ndarray
    ms: dict
    x: float
    y: float
    yaw: float
    px_all: dict

def observe(ctx: Context, target: str) -> Obs:
    rgb = ctx.cam.rgb(ctx.sim)
    inst = instance_map(ctx, ctx.cam.segmentation(ctx.sim))
    ms = mask_stats(inst == OBJECT_KEYS.index(target) + 1)
    x, y = ctx.loco.xy()
    return Obs(rgb, inst, ms, float(x), float(y), ctx.loco.yaw(), {key: int((inst == i + 1).sum()) for i, key in enumerate(OBJECT_KEYS)})

def rounded(d: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
