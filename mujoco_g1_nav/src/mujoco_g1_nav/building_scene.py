# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import copy
import math
import mujoco
import numpy as np
from .building import Building
from .config import PKG_ROOT, resolve
from .scene import CEILING_GROUP, OBJECT_KEYS, Scene, _add_object_geoms, _calibrate_camera, _material, pitch_quat, vfov_from_hfov, G
from .sensors import EgoCamera
from .textures import ensure_textures
MAX_LAMPS_ON = 7
OWN_REGION_BONUS_M = 4.0
CORRIDOR_LAMP_SPACING_M = 3.5

class Lamp:
    __slots__ = ('light_id', 'xy', 'region')

    def __init__(self, light_id, xy, region):
        self.light_id, self.xy, self.region = (light_id, xy, region)

class BuildingScene(Scene):

    def __init__(self, base: Scene, building: Building, table_body: int, wall_geoms: list, lamps: list, table_xy):
        self.__dict__.update(base.__dict__)
        self.building, self.table_body, self.wall_geoms, self.lamps, self.table_xy = (building, table_body, wall_geoms, lamps, tuple(table_xy))
        self._lamp_xy = np.array([l.xy for l in lamps])
        self._lamp_region = np.array([l.region for l in lamps])
        self._lit_at = None

    def place_table(self, xy) -> tuple:
        self.model.body_pos[self.table_body] = [xy[0], xy[1], 0.0]
        self.model.cam_pos[self.zoom_cam_id][:2] = xy
        self.table_xy = (float(xy[0]), float(xy[1]))
        return self.table_xy

    def cfg_at_table(self) -> dict:
        cfg = copy.deepcopy(self.cfg)
        cfg['table']['center_xy'] = list(self.table_xy)
        return cfg

    def update_lighting(self, xy, force: bool=False):
        xy = np.asarray(xy, dtype=float)[:2]
        if not force and self._lit_at is not None and (np.hypot(*xy - self._lit_at) < 0.05):
            return
        region = self.building.region_at(*xy) or ''
        if region in self.building.doors:
            region = self.building.doors[region].room
        d = np.hypot(*(self._lamp_xy - xy).T) - np.where(self._lamp_region == region, OWN_REGION_BONUS_M, 0.0)
        on = np.argsort(d, kind='stable')[:MAX_LAMPS_ON]
        active = np.zeros(len(self.lamps), dtype=bool)
        active[on] = True
        for lamp, a in zip(self.lamps, active):
            self.model.light_active[lamp.light_id] = int(a)
        self._lit_at = xy.copy()

class BuildingEgoCamera(EgoCamera):

    def __init__(self, scene, exact_ids: bool=False):
        c = scene.cfg['camera']
        self.scene, self.w, self.h, self.size = (scene, c['width'], c['height'], c['policy_input_size'])
        self.renderer = mujoco.Renderer(scene.model, self.h, self.w)
        quality = scene.model.vis.quality
        samples = int(quality.offsamples)
        if exact_ids:
            quality.offsamples = 0
        try:
            self.seg_renderer = mujoco.Renderer(scene.model, self.h, self.w)
        finally:
            quality.offsamples = samples
        self.seg_renderer.enable_segmentation_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[CEILING_GROUP] = 1
        self.overview_opt = mujoco.MjvOption()
        self.third_cam, self.display_cam = (mujoco.MjvCamera(), mujoco.MjvCamera())
        for cam in (self.third_cam, self.display_cam):
            cam.type, cam.trackbodyid = (mujoco.mjtCamera.mjCAMERA_TRACKING, scene.pelvis_id)
            cam.distance, cam.elevation, cam.azimuth = (5.5, -58.0, 0.0)
        self._lazy = {}

    def _get(self, name, h, w, depth=False):
        if name not in self._lazy:
            r = mujoco.Renderer(self.scene.model, h, w)
            if depth:
                r.enable_depth_rendering()
            self._lazy[name] = r
        return self._lazy[name]
    depth_renderer = property(lambda self: self._get('depth', self.h, self.w, depth=True))
    overview = property(lambda self: self._get('overview', 720, 720))
    third = property(lambda self: self._get('third', 480, 640))
    display = property(lambda self: self._get('display', 480, 640))

    def close(self):
        for r in (self.renderer, self.seg_renderer, *self._lazy.values()):
            r.close()

    def _lit(self, data):
        self.scene.update_lighting(data.qpos[:2])

    def rgb(self, data):
        self._lit(data)
        return super().rgb(data)

    def overview_rgb(self, data, zoom: bool=False):
        self._lit(data)
        return super().overview_rgb(data, zoom)

    def third_rgb(self, data, yaw_rad):
        self._lit(data)
        return super().third_rgb(data, yaw_rad)

    def display_rgb(self, data):
        self._lit(data)
        return super().display_rgb(data)

def _lamp_positions(region) -> list:
    r = region.rect
    cx, cy = r.center
    if region.is_room:
        return [(cx, cy)] + [(cx + sx * r.w / 4, cy + sy * r.h / 4) for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    along_x = r.w >= r.h
    length = r.w if along_x else r.h
    n = max(1, round(length / CORRIDOR_LAMP_SPACING_M))
    ts = [(k + 0.5) * length / n for k in range(n)]
    return [(r.x0 + t, cy) if along_x else (cx, r.y0 + t) for t in ts]

def build_building_scene(cfg: dict, building: Building, extra=None) -> BuildingScene:
    tex = ensure_textures(PKG_ROOT / 'assets' / 'textures')
    robot_xml = resolve(cfg['paths']['robot_xml'])
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.meshdir = str(robot_xml.parent / 'meshes')
    spec.option.timestep = cfg['locomotion']['sim_dt']
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 1280
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 4096
    for name, path in tex.items():
        spec.add_texture(name=f'tex_{name}', type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
    gso = resolve(cfg['paths']['gso_dir']) / 'Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White'
    spec.add_texture(name='tex_cup', type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(gso / 'texture.png'))
    spec.add_mesh(name='cup_mesh', file=str(gso / 'model.obj'))
    _material(spec, 'mat_floor', 'tex_floor', [1, 1], True, specular=0.1, shininess=0.1, reflectance=0.02)
    _material(spec, 'mat_wall', rgba=[0.86, 0.85, 0.82, 1], specular=0.0)
    _material(spec, 'mat_ceiling', rgba=[0.95, 0.95, 0.95, 1], specular=0.0, emission=0.55)
    _material(spec, 'mat_wood', 'tex_wood', [1, 1], False, specular=0.15, shininess=0.2)
    _material(spec, 'mat_coca', 'tex_coca', specular=0.7, shininess=0.7)
    _material(spec, 'mat_pepsi', 'tex_pepsi', specular=0.7, shininess=0.7)
    _material(spec, 'mat_orange', 'tex_orange', specular=0.3, shininess=0.4)
    _material(spec, 'mat_basket', 'tex_basket', specular=0.1, shininess=0.1)
    _material(spec, 'mat_cup', 'tex_cup', specular=0.5, shininess=0.5)
    wb = spec.worldbody
    table = cfg['table']
    wh = cfg['room']['wall_height']
    fp = building.footprint
    floor_mat = {}
    for r in building.regions.values():
        if r.floor_rgb is None or r.id == building.origin:
            floor_mat[r.id] = 'mat_floor'
        else:
            floor_mat[r.id] = f'mat_floor_{r.id}'
            _material(spec, floor_mat[r.id], 'tex_floor', [1, 1], True, rgba=[*r.floor_rgb, 1], specular=0.1, shininess=0.1, reflectance=0.0)
    for r in building.regions.values():
        cx, cy = r.rect.center
        kw = {} if r.id == building.origin else dict(contype=0, conaffinity=0)
        wb.add_geom(name='floor' if r.id == building.origin else f'floor_{r.id}', type=G.mjGEOM_PLANE, size=[r.rect.w / 2, r.rect.h / 2, 0.1], pos=[cx, cy, 0.0], material=floor_mat[r.id], **kw)
    for d in building.doors.values():
        cx, cy = d.rect.center
        wb.add_geom(name=f'floor_{d.id}', type=G.mjGEOM_PLANE, size=[d.rect.w / 2, d.rect.h / 2, 0.1], pos=[cx, cy, 0.0], material=floor_mat[d.to], contype=0, conaffinity=0)
    wall_mat = {None: 'mat_wall'}
    wall_names = []
    for k, w in enumerate(building.walls):
        if w.rgb not in wall_mat:
            wall_mat[w.rgb] = f'mat_wall_{len(wall_mat)}'
            _material(spec, wall_mat[w.rgb], rgba=[*w.rgb, 1], specular=0.0)
        cx, cy = w.rect.center
        wb.add_geom(name=f'wall_{k:03d}', type=G.mjGEOM_BOX, size=[w.rect.w / 2, w.rect.h / 2, wh / 2], pos=[cx, cy, wh / 2], material=wall_mat[w.rgb])
        wall_names.append(f'wall_{k:03d}')
    fx, fy = fp.center
    wb.add_geom(name='ceiling', type=G.mjGEOM_BOX, size=[fp.w / 2, fp.h / 2, 0.05], pos=[fx, fy, wh + 0.05], material='mat_ceiling', contype=0, conaffinity=0, group=CEILING_GROUP)
    cx0, cy0 = table['center_xy']
    sx, sy = (table['size_xy'][0] / 2, table['size_xy'][1] / 2)
    top, th = (table['top_z'], table['thickness'])
    table_geoms = ['table_top']
    tb = wb.add_body(name='table', pos=[cx0, cy0, 0.0])
    tb.add_geom(name='table_top', type=G.mjGEOM_BOX, size=[sx, sy, th / 2], pos=[0, 0, top - th / 2], material='mat_wood')
    for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        tb.add_geom(name=f'table_leg{i}', type=G.mjGEOM_BOX, size=[0.035, 0.035, (top - th) / 2], pos=[dx * (sx - 0.06), dy * (sy - 0.06), (top - th) / 2], material='mat_wood')
        table_geoms.append(f'table_leg{i}')
    lamp_specs = []
    for r in building.regions.values():
        for k, (lx, ly) in enumerate(_lamp_positions(r)):
            centre = r.is_room and k == 0
            name = f'lamp_{r.id}_{k}'
            if centre:
                wb.add_light(name=name, pos=[lx, ly, wh - 0.05], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], specular=[0.1, 0.1, 0.1], castshadow=True)
            else:
                wb.add_light(name=name, pos=[lx, ly, wh - 0.05], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05], castshadow=False)
            lamp_specs.append((name, r.id, (lx, ly)))
    for key in OBJECT_KEYS:
        body = wb.add_body(name=f'obj_{key}', pos=[cx0, cy0, top])
        _add_object_geoms(spec, cfg, body, key, tex, gso)
    c = cfg['camera']
    torso = spec.body('torso_link')
    cam = torso.add_camera(name=c['name'], pos=[c['forward_offset_m'], 0.0, 0.45], fovy=vfov_from_hfov(c['hfov_deg'], c['width'], c['height']))
    cam.quat = pitch_quat(c['pitch_deg'])
    z_over = max(9.5, 0.5 * max(fp.w, fp.h) / math.tan(math.radians(30.0)) * 1.05)
    wb.add_camera(name='overview', pos=[fx, fy, z_over], fovy=60.0)
    wb.add_camera(name='topdown_zoom', pos=[cx0, cy0, 5.5], fovy=40.0)
    if extra is not None:
        extra(spec, wb)
    model = spec.compile()
    base = Scene(cfg=cfg, model=model, ego_cam_id=model.camera(c['name']).id, overview_cam_id=model.camera('overview').id, zoom_cam_id=model.camera('topdown_zoom').id, torso_id=model.body('torso_link').id, pelvis_id=model.body('pelvis').id)
    for key in OBJECT_KEYS:
        base.obj_body[key] = model.body(f'obj_{key}').id
        ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == base.obj_body[key]]
        base.obj_geoms[key] = ids
        for g in ids:
            base.geom_to_obj[g] = key
    base.table_geoms = [model.geom(n).id for n in table_geoms]
    lamps = [Lamp(model.light(n).id, xy, region) for n, region, xy in lamp_specs]
    scene = BuildingScene(base, building, model.body('table').id, [model.geom(n).id for n in wall_names], lamps, (cx0, cy0))
    scene.update_lighting((cx0, cy0), force=True)
    return _calibrate_camera(scene)
