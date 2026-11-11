# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from dataclasses import dataclass, field
from pathlib import Path
import mujoco
import numpy as np
from .config import PKG_ROOT, resolve
from .textures import ensure_textures
G = mujoco.mjtGeom
TEX_RGB = mujoco.mjtTextureRole.mjTEXROLE_RGB
OBJECT_KEYS = ('coca', 'pepsi', 'cup', 'orange', 'basket')
CEILING_GROUP = 3

def vfov_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) * height / width))

def yaw_quat(yaw_rad: float) -> np.ndarray:
    return np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])

def pitch_quat(pitch_deg: float) -> np.ndarray:
    pitch = math.radians(pitch_deg)
    fwd = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    up = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    right = np.cross(fwd, up)
    R = np.stack([right, up, -fwd], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat

def set_ego_pitch(scene, pitch_deg: float):
    scene.model.cam_quat[scene.ego_cam_id] = pitch_quat(pitch_deg)

@dataclass
class Scene:
    cfg: dict
    model: mujoco.MjModel
    ego_cam_id: int
    overview_cam_id: int
    zoom_cam_id: int
    torso_id: int
    pelvis_id: int
    obj_body: dict = field(default_factory=dict)
    obj_geoms: dict = field(default_factory=dict)
    geom_to_obj: dict = field(default_factory=dict)
    table_geoms: list = field(default_factory=list)
    camera_calibration: dict = field(default_factory=dict)

    @property
    def table_top_z(self) -> float:
        return self.cfg['table']['top_z']

def _material(spec, name, tex=None, texrepeat=None, texuniform=False, rgba=None, specular=None, shininess=None, reflectance=None, emission=None):
    m = spec.add_material(name=name)
    if tex is not None:
        m.textures[TEX_RGB] = tex
    if texrepeat is not None:
        m.texrepeat = texrepeat
    m.texuniform = texuniform
    if rgba is not None:
        m.rgba = rgba
    if specular is not None:
        m.specular = specular
    if shininess is not None:
        m.shininess = shininess
    if reflectance is not None:
        m.reflectance = reflectance
    if emission is not None:
        m.emission = emission
    return m

def _add_object_geoms(spec, cfg, body, key, tex, gso_dir):
    kw = dict(contype=0, conaffinity=0)
    if key in ('coca', 'pepsi'):
        body.add_geom(name=f'obj_{key}_g0', type=G.mjGEOM_CYLINDER, size=[0.033, 0.06, 0], pos=[0, 0, 0.06], material=f'mat_{key}', **kw)
        body.add_geom(name=f'obj_{key}_g1', type=G.mjGEOM_CYLINDER, size=[0.03, 0.0015, 0], pos=[0, 0, 0.1205], rgba=[0.78, 0.78, 0.8, 1], **kw)
    elif key == 'orange':
        body.add_geom(name='obj_orange_g0', type=G.mjGEOM_SPHERE, size=[0.038, 0, 0], pos=[0, 0, 0.037], material='mat_orange', **kw)
    elif key == 'basket':
        L, W, H, t = (0.16, 0.11, 0.06, 0.006)
        parts = [((L / 2, W / 2, t / 2), (0, 0, t / 2)), ((L / 2, t / 2, H / 2), (0, W / 2 - t / 2, H / 2)), ((L / 2, t / 2, H / 2), (0, -W / 2 + t / 2, H / 2)), ((t / 2, W / 2, H / 2), (L / 2 - t / 2, 0, H / 2)), ((t / 2, W / 2, H / 2), (-L / 2 + t / 2, 0, H / 2)), ((0.004, 0.004, 0.04), (L / 2 - 0.004, 0, H + 0.03)), ((0.004, 0.004, 0.04), (-L / 2 + 0.004, 0, H + 0.03)), ((L / 2, 0.005, 0.004), (0, 0, H + 0.07))]
        for i, (size, pos) in enumerate(parts):
            body.add_geom(name=f'obj_basket_g{i}', type=G.mjGEOM_BOX, size=list(size), pos=list(pos), material='mat_basket', **kw)
    elif key == 'cup':
        body.add_geom(name='obj_cup_g0', type=G.mjGEOM_MESH, meshname='cup_mesh', material='mat_cup', **kw)
    else:
        raise KeyError(key)

def build_scene(cfg: dict) -> Scene:
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
    room, table = (cfg['room'], cfg['table'])
    rx, ry = (room['size_xy'][0] / 2, room['size_xy'][1] / 2)
    wh, wt = (room['wall_height'], 0.1)
    wb.add_geom(name='floor', type=G.mjGEOM_PLANE, size=[rx, ry, 0.1], material='mat_floor')
    for nm, pos, size in [('wall_n', [0, ry + wt / 2, wh / 2], [rx + wt, wt / 2, wh / 2]), ('wall_s', [0, -ry - wt / 2, wh / 2], [rx + wt, wt / 2, wh / 2]), ('wall_e', [rx + wt / 2, 0, wh / 2], [wt / 2, ry, wh / 2]), ('wall_w', [-rx - wt / 2, 0, wh / 2], [wt / 2, ry, wh / 2])]:
        wb.add_geom(name=nm, type=G.mjGEOM_BOX, size=size, pos=pos, material='mat_wall')
    wb.add_geom(name='ceiling', type=G.mjGEOM_BOX, size=[rx + wt, ry + wt, 0.05], pos=[0, 0, wh + 0.05], material='mat_ceiling', contype=0, conaffinity=0, group=CEILING_GROUP)
    cx, cy = table['center_xy']
    sx, sy = (table['size_xy'][0] / 2, table['size_xy'][1] / 2)
    top, th = (table['top_z'], table['thickness'])
    table_geoms = ['table_top']
    wb.add_geom(name='table_top', type=G.mjGEOM_BOX, size=[sx, sy, th / 2], pos=[cx, cy, top - th / 2], material='mat_wood')
    for i, (dx, dy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        wb.add_geom(name=f'table_leg{i}', type=G.mjGEOM_BOX, size=[0.035, 0.035, (top - th) / 2], pos=[cx + dx * (sx - 0.06), cy + dy * (sy - 0.06), (top - th) / 2], material='mat_wood')
        table_geoms.append(f'table_leg{i}')
    wb.add_light(name='lamp_table', pos=[0.0, 0.0, wh - 0.05], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], specular=[0.1, 0.1, 0.1], castshadow=True)
    for i, (lx, ly) in enumerate([(-2.5, -2.5), (-2.5, 2.5), (2.5, -2.5), (2.5, 2.5)]):
        wb.add_light(name=f'lamp_{i}', pos=[lx, ly, wh - 0.05], dir=[0, 0, -1], diffuse=[0.35, 0.35, 0.35], specular=[0.05, 0.05, 0.05], castshadow=False)
    for key in OBJECT_KEYS:
        body = wb.add_body(name=f'obj_{key}', pos=[cx, cy, top])
        _add_object_geoms(spec, cfg, body, key, tex, gso)
    c = cfg['camera']
    torso = spec.body('torso_link')
    cam = torso.add_camera(name=c['name'], pos=[c['forward_offset_m'], 0.0, 0.45], fovy=vfov_from_hfov(c['hfov_deg'], c['width'], c['height']))
    cam.quat = pitch_quat(c['pitch_deg'])
    wb.add_camera(name='overview', pos=[0.0, 0.0, 9.5], fovy=60.0)
    wb.add_camera(name='topdown_zoom', pos=[0.0, 0.0, 5.5], fovy=40.0)
    model = spec.compile()
    scene = Scene(cfg=cfg, model=model, ego_cam_id=model.camera(c['name']).id, overview_cam_id=model.camera('overview').id, zoom_cam_id=model.camera('topdown_zoom').id, torso_id=model.body('torso_link').id, pelvis_id=model.body('pelvis').id)
    for key in OBJECT_KEYS:
        scene.obj_body[key] = model.body(f'obj_{key}').id
        ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == scene.obj_body[key]]
        scene.obj_geoms[key] = ids
        for g in ids:
            scene.geom_to_obj[g] = key
    scene.table_geoms = [model.geom(n).id for n in table_geoms]
    return _calibrate_camera(scene)

def _calibrate_camera(scene: Scene) -> Scene:
    from .locomotion import G1Locomotion
    m = scene.model
    d = mujoco.MjData(m)
    loco = G1Locomotion(m, d, scene.cfg)
    loco.reset(xy=(-3.0, 0.0), yaw=0.0)
    for _ in range(100):
        loco.step((0.0, 0.0, 0.0))
    cam_z = float(d.cam_xpos[scene.ego_cam_id][2])
    dz = scene.cfg['camera']['height_m'] - cam_z
    m.cam_pos[scene.ego_cam_id][2] += dz
    scene.camera_calibration = {'stand_height_before': cam_z, 'shift_m': dz}
    return scene
