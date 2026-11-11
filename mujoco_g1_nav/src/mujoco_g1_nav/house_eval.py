# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from collections import Counter
from types import SimpleNamespace
import cv2
import numpy as np
from .goal import mask_stats
from .house_goal import MIN_START_PATH_M, category_lookup, goal_cells, goal_metrics, success_flags
from .scene import set_ego_pitch
ACTION_NAMES = ['stop', 'forward', 'left', 'right', 'look_up', 'look_down']
STUCK_STEPS, STUCK_MOVE_M = (100, 0.25)
ESCAPE_WINDOW, ESCAPE_HITS, ESCAPE_MOVE_M = (8, 6, 0.5)

class CategoryWorld:

    def __init__(self, ctx, report):
        self.ctx, self.layout, self.house = (ctx, ctx.house.layout, ctx.house)
        self.cats = [c for c in dict.fromkeys((it.cat for it in self.layout.items))]
        self.report = report
        self.lut = {c: category_lookup(self.house, self.layout, c, ctx.scene.model.ngeom) for c in self.cats}
        self.goals = {c: goal_cells(report, self.layout, c) for c in self.cats}
        self.fields = {}

    def field(self, cat):
        if cat not in self.fields:
            self.fields[cat] = self.ctx.navmap.field(self.goals[cat]) if self.goals[cat] else None
        return self.fields[cat]

    def path_length(self, cat, x, y) -> float:
        f = self.field(cat)
        return float(self.ctx.navmap.path_length(f, x, y)) if f is not None else math.nan

    def sample_start(self, cat: str, rng, cells: np.ndarray, tries: int=200):
        for _ in range(tries):
            x, y = (float(v) for v in cells[int(rng.integers(len(cells)))])
            sh = self.path_length(cat, x, y)
            if math.isfinite(sh) and sh >= MIN_START_PATH_M:
                return (x, y, float(rng.uniform(-math.pi, math.pi)), sh)
        raise RuntimeError(f'no start for {cat}')

def _escape_plan(ctx, x: float, y: float, yaw: float) -> list:
    from .nav import rect_distance
    nav, step, turn = (ctx.navmap, float(ctx.cfg['navigation']['forward_step_m']), math.radians(2 * float(ctx.cfg['navigation']['turn_step_deg'])))
    room = {}
    for act, sgn in ((2, 1), (3, -1)):
        px, py = (x + step * math.cos(yaw + sgn * turn), y + step * math.sin(yaw + sgn * turn))
        room[act] = min(nav.wall_clearance(px, py), float(rect_distance(px, py, ctx.cfg)))
    side = max(room, key=room.get)
    return [side, side, 1, 1]

def category_mask(world: CategoryWorld, seg_ids: np.ndarray, cat: str) -> np.ndarray:
    lut = world.lut[cat]
    return lut[np.where(seg_ids < 0, len(lut) - 1, seg_ids)]

def run_category_episode(world: CategoryWorld, client, embed, phrase: str, cat: str, start, mode: str='gt', max_steps: int=450, deterministic: bool=True, seed: int=0, stop_thresh=None, on_step=None, controller=None, handover_reset: bool=False, escape: bool=False) -> dict:
    ctx, layout = (world.ctx, world.layout)
    x0, y0, yaw0, shortest = start
    set_ego_pitch(ctx.scene, ctx.cfg['camera']['pitch_deg'])
    ctx.loco.reset((x0, y0), yaw0)
    ctx.executor.settle()
    ctx.collisions[0] = 0
    client.reset(embed, phrase, cat, mode, [0, 1, 2, 3], deterministic, seed, **{} if stop_thresh is None else dict(stop_thresh=stop_thresh))
    hist, path, called_stop, fell, collisions = (Counter(), 0.0, False, False, 0)
    last, trace, mask_px = (ctx.loco.xy(), [], [])
    reach_loc_step = reach_step = None
    stop_prob_max = 0.0
    min_d = math.inf
    stuck, recent = (False, [])
    blk, escape_acts, n_escapes = ([], [], 0)
    for t in range(max_steps):
        rgb = ctx.cam.rgb(ctx.sim)
        cm = category_mask(world, ctx.cam.segmentation(ctx.sim), cat)
        px = int(cm.sum())
        x, y = (float(v) for v in ctx.loco.xy())
        yaw = ctx.loco.yaw()
        m = goal_metrics(layout, cat, x, y, yaw, px)
        min_d = min(min_d, m['d'])
        if reach_loc_step is None and m['loc']:
            reach_loc_step = t
        if reach_step is None and m['loose']:
            reach_step = t
        gt = cv2.resize(cm.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) if mode == 'gt' else None
        forced = None
        if controller is not None:
            forced = controller(SimpleNamespace(rgb=rgb, x=x, y=y, yaw=yaw, ms=mask_stats(cm)), t)
            if handover_reset and controller.handover_step == t:
                client.reset(embed, phrase, cat, mode, [0, 1, 2, 3], deterministic, seed, **{} if stop_thresh is None else dict(stop_thresh=stop_thresh))
        if forced is None and escape_acts:
            forced = escape_acts.pop(0)
        idx, info = client.step(rgb, gt)
        if forced is not None:
            if forced != idx:
                client.override(int(forced))
            idx = int(forced)
        if on_step is not None:
            if hasattr(on_step, 'driver'):
                on_step.driver = 'explorer' if forced is not None else 'policy'
            on_step(ctx, t, idx, info, m)
        if controller is None or controller.handover_step is not None:
            stop_prob_max = max(stop_prob_max, info['probs'][0])
        hist[ACTION_NAMES[idx]] += 1
        mask_px.append(info['mask_px'])
        trace.append((round(x, 3), round(y, 3)))
        if idx == 0:
            called_stop = True
            break
        if idx in (4, 5):
            continue
        res = ctx.executor.execute(idx)
        hit = int(ctx.collisions[0] > 0)
        collisions += hit
        ctx.collisions[0] = 0
        nx, ny = ctx.loco.xy()
        path += math.hypot(nx - last[0], ny - last[1])
        last = (nx, ny)
        recent.append((nx, ny, hit))
        if escape and forced is None:
            blk.append((nx, ny, hit))
            blk[:] = blk[-ESCAPE_WINDOW:]
            if not escape_acts and len(blk) == ESCAPE_WINDOW and (sum((b[2] for b in blk)) >= ESCAPE_HITS) and (math.hypot(blk[0][0] - nx, blk[0][1] - ny) < ESCAPE_MOVE_M):
                escape_acts, n_escapes = (_escape_plan(ctx, nx, ny, ctx.loco.yaw()), n_escapes + 1)
                blk.clear()
        if len(recent) > STUCK_STEPS:
            recent.pop(0)
            if math.hypot(recent[0][0] - nx, recent[0][1] - ny) < STUCK_MOVE_M and sum((r[2] for r in recent)) >= 0.8 * STUCK_STEPS:
                stuck = True
                break
        if res.fell:
            fell = True
            break
    rgb = ctx.cam.rgb(ctx.sim)
    px = int(category_mask(world, ctx.cam.segmentation(ctx.sim), cat).sum())
    x, y = (float(v) for v in ctx.loco.xy())
    m = goal_metrics(layout, cat, x, y, ctx.loco.yaw(), px)
    fl = success_flags(m, called_stop and (not fell))
    sh = max(shortest, 1e-06)
    ratio = sh / max(path, sh)
    dtg = world.path_length(cat, x, y)
    near = layout.by_id[m['item']]
    return dict(valid=True, seed=seed, target=cat, variant='g1', mode=mode, steps=len(trace), called_stop=called_stop, fell=fell, collision_steps=collisions, path_m=round(path, 3), shortest_m=round(shortest, 3), **fl, reached=reach_step is not None, reached_loc=reach_loc_step is not None, reach_step=reach_step, min_obj_dist=round(min_d, 3), spl=float(fl['success']) * ratio, spl_front=float(fl['success_front']) * ratio, soft_spl=round(max(0.0, 1 - (dtg if math.isfinite(dtg) else sh) / sh) * ratio, 4), dtg=round(dtg, 3) if math.isfinite(dtg) else None, called_item=m['item'], stuck=stuck, escapes=n_escapes, stop_prob_max=round(stop_prob_max, 4), handover_step=None if controller is None else controller.handover_step, final=dict(obj_dist=round(m['d'], 3), yaw_err_deg=round(m['yaw_err'], 1), px=m['px'], front=bool(m['front']), item=m['item']), start=[round(x0, 3), round(y0, 3), round(yaw0, 3)], start_kind=('room:' if ctx.building.room_at(x0, y0) else 'open:') + (ctx.building.room_at(x0, y0) or ctx.building.region_at(x0, y0) or '-'), table_room=near.region, scenario='cat', rooms_visited=sorted({ctx.building.room_at(*p) or '-' for p in trace} - {'-'}), actions=dict(hist), trace=trace[::max(1, len(trace) // 40)], mask_px_mean=float(np.mean([p for p in mask_px if p >= 0])) if any((p >= 0 for p in mask_px)) else None)
