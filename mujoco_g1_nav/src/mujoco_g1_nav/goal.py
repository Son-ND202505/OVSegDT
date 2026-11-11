# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import mujoco
import numpy as np
from .nav import NavMap, rect_distance, wrap
from .scene import Scene, yaw_quat
STAND_Z = 0.784

def object_xy(layout, key):
    x, y, _ = layout.objects[key]
    return (x, y)

def bearing_to(x, y, tx, ty):
    return math.atan2(ty - y, tx - x)

def mask_stats(mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return dict(px=0, bbox=None, fully_visible=False)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    touches = ys.min() <= 1 or ys.max() >= h - 2 or xs.min() <= 1 or (xs.max() >= w - 2)
    return dict(px=n, bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], fully_visible=not touches)

def stop_metrics(cfg, layout, key, x, y, yaw, ms: dict) -> dict:
    tx, ty = object_xy(layout, key)
    return dict(clearance=float(rect_distance(x, y, cfg)), yaw_err_deg=math.degrees(abs(wrap(yaw - bearing_to(x, y, tx, ty)))), obj_dist=math.hypot(tx - x, ty - y), px=ms['px'], fully_visible=ms['fully_visible'])

def is_success(cfg, m: dict) -> bool:
    c = cfg['success']
    lo, hi = c['clearance_range_m']
    return lo <= m['clearance'] <= hi and m['yaw_err_deg'] <= c['yaw_tol_deg'] and (m['px'] >= c['min_px']) and m['fully_visible'] and (m['obj_dist'] <= c['max_object_dist_m'])

def is_success_close(cfg, m: dict) -> bool:
    c = cfg['success_close']
    lo, hi = c['clearance_range_m']
    return lo <= m['clearance'] <= hi and m['yaw_err_deg'] <= c['yaw_tol_deg'] and (m['px'] >= c['min_px']) and m['fully_visible'] and (m['obj_dist'] <= c['max_object_dist_m'])

def explain_success(cfg, m: dict) -> list:
    c = cfg['success']
    lo, hi = c['clearance_range_m']
    out = []
    if not lo <= m['clearance'] <= hi:
        out.append(f"cách bàn {m['clearance']:.2f} m (cần {lo}-{hi})")
    if m['yaw_err_deg'] > c['yaw_tol_deg']:
        out.append(f"lệch hướng {m['yaw_err_deg']:.0f}° (cần ≤ {c['yaw_tol_deg']})")
    if m['px'] < c['min_px']:
        out.append(f"vật chỉ {m['px']} px (cần ≥ {c['min_px']})")
    elif not m['fully_visible']:
        out.append('vật bị cắt mép khung hình')
    if m['obj_dist'] > c['max_object_dist_m']:
        out.append(f"cách vật {m['obj_dist']:.2f} m (cần ≤ {c['max_object_dist_m']})")
    return out

def expert_may_stop(cfg, m: dict) -> bool:
    e, c = (cfg['expert'], cfg['success'])
    return m['clearance'] <= e['stop_clearance_max_m'] and m['clearance'] >= e['stop_clearance_min_m'] and (m['yaw_err_deg'] <= e['stop_yaw_tol_deg']) and (m['px'] >= c['min_px']) and m['fully_visible'] and (m['obj_dist'] <= e['approach_max_obj_dist_m'] + 0.1)

def pose_probe(scene: Scene, probe: mujoco.MjData, q0: np.ndarray, x, y, yaw):
    probe.qpos[:] = 0.0
    probe.qpos[0:2] = (x, y)
    probe.qpos[2] = STAND_Z
    probe.qpos[3:7] = yaw_quat(yaw)
    probe.qpos[7:] = q0
    mujoco.mj_forward(scene.model, probe)

def offset_ring(cfg, clearance, step=0.2):
    cx, cy = cfg['table']['center_xy']
    hx, hy = (cfg['table']['size_xy'][0] / 2, cfg['table']['size_xy'][1] / 2)
    pts = []
    for sx in (-1, 1):
        for y in np.arange(-hy, hy + 1e-09, step):
            pts.append((cx + sx * (hx + clearance), cy + y))
    for sy in (-1, 1):
        for x in np.arange(-hx, hx + 1e-09, step):
            pts.append((cx + x, cy + sy * (hy + clearance)))
    for sx in (-1, 1):
        for sy in (-1, 1):
            for a in np.linspace(0.0, math.pi / 2, max(2, int(clearance * math.pi / 2 / step)))[1:-1]:
                pts.append((cx + sx * (hx + clearance * math.cos(a)), cy + sy * (hy + clearance * math.sin(a))))
    return pts

def valid_approach_poses(cfg, scene, cam, probe, q0, layout, key, navmap: NavMap, offsets_deg=(-15, 0, 15)):
    e = cfg['expert']
    tx, ty = object_xy(layout, key)
    out = []
    for x, y in offset_ring(cfg, e['goal_clearance_m']):
        if not navmap.is_free(x, y) or math.hypot(tx - x, ty - y) > e['approach_max_obj_dist_m']:
            continue
        aim = bearing_to(x, y, tx, ty)
        ok = True
        for off in offsets_deg:
            pose_probe(scene, probe, q0, x, y, aim + math.radians(off))
            ms = mask_stats(cam.object_mask(cam.segmentation(probe), key))
            if ms['px'] < e['goal_min_px'] or not ms['fully_visible']:
                ok = False
                break
        if ok:
            out.append((x, y, aim))
    return out
