# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
DEFAULTS = dict(phi_deg=30.0, corner_m=0.15)

def table_edges(cfg) -> list:
    cx, cy = cfg['table']['center_xy']
    hx, hy = (cfg['table']['size_xy'][0] / 2, cfg['table']['size_xy'][1] / 2)
    return [('+x', (1.0, 0.0), cx + hx), ('-x', (-1.0, 0.0), -(cx - hx)), ('+y', (0.0, 1.0), cy + hy), ('-y', (0.0, -1.0), -(cy - hy))]

def frontality(cfg, obj_xy, robot_xy) -> dict:
    p = {**DEFAULTS, **cfg.get('success_front', {})}
    ox, oy = obj_xy
    vx, vy = (robot_xy[0] - ox, robot_xy[1] - oy)
    rows = []
    for name, (nx, ny), off in table_edges(cfg):
        d = off - (ox * nx + oy * ny)
        dn = vx * nx + vy * ny
        lat = abs(-vx * ny + vy * nx)
        phi = math.degrees(math.atan2(lat, dn))
        rows.append((d, phi, lat, name))
    dmin = min((r[0] for r in rows))
    near = [r for r in rows if r[0] <= dmin + p['corner_m']]
    _, phi, lat, edge = min(near, key=lambda r: r[1])
    return dict(phi_deg=phi, lateral_m=lat, edge=edge, front=phi <= p['phi_deg'])

def is_success_front(cfg, m: dict, obj_xy, robot_xy) -> bool:
    from .goal import is_success_close
    return is_success_close(cfg, m) and frontality(cfg, obj_xy, robot_xy)['front']
