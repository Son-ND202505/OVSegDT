# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import os
from dataclasses import dataclass
import mujoco
import numpy as np
from .actions import ActionExecutor
from .building import Building
from .building_scene import BuildingEgoCamera, build_building_scene
from .collect import Context
from .locomotion import G1Locomotion
from .scene import OBJECT_KEYS
from .world_map import WorldMap

@dataclass
class BuildingContext(Context):
    building: Building = None
    navmap_static: WorldMap = None
    scenario: object = None
    house: object = None

def make_building_context(cfg: dict, building: Building, scene=None, extra_env_geoms=()) -> BuildingContext:
    scene = scene if scene is not None else build_building_scene(cfg, building)
    m = scene.model
    sim, probe = (mujoco.MjData(m), mujoco.MjData(m))
    loco = G1Locomotion(m, sim, cfg)
    cam = BuildingEgoCamera(scene, exact_ids=getattr(scene, 'house', None) is not None or os.environ.get('MUJOCO_G1_EXACT_IDS') == '1')
    geom_code = np.zeros(m.ngeom + 1, dtype=np.uint8)
    for k, key in enumerate(OBJECT_KEYS):
        geom_code[scene.obj_geoms[key]] = k + 1
    robot_geoms = np.nonzero(m.body_rootid[m.geom_bodyid] == scene.pelvis_id)[0]
    env_geoms = np.array(scene.table_geoms + scene.wall_geoms + [int(g) for g in extra_env_geoms])
    house = getattr(scene, 'house', None)
    if house is not None:
        from .house_map import HouseMap
        navmap, navmap_static = (HouseMap(building, cfg, house.layout), HouseMap(building, cfg, house.layout, with_table=False))
    else:
        navmap, navmap_static = (WorldMap(building, cfg), WorldMap(building, cfg, with_table=False))
    ctx = BuildingContext(cfg, scene, sim, probe, loco, cam, navmap, None, geom_code, robot_geoms, env_geoms, [0], building=building, navmap_static=navmap_static, house=house)

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

def make_house_context(cfg: dict, building: Building, layout) -> BuildingContext:
    from .house_scene import build_house_scene
    scene = build_house_scene(cfg, building, layout)
    return make_building_context(cfg, building, scene=scene, extra_env_geoms=scene.house.all_colliders)
