# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
from collections import Counter
import cv2
import numpy as np
from .collect import Context, observe, resolve_start
from .goal import bearing_to, is_success, is_success_close, mask_stats, offset_ring, pose_probe, stop_metrics, valid_approach_poses
from .layout import apply_layout, make_layout
from .scene import OBJECT_KEYS, set_ego_pitch
EVAL_PHRASE = {'coca': 'coca cola can', 'pepsi': 'pepsi can', 'cup': 'cup', 'orange': 'orange', 'basket': 'basket'}
DILATIONS = [0, 5, 9, 15, 25, 41, 61, 91]
DISTANCES = [1.2, 0.9, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12]
ACTION_NAMES = ['stop', 'forward', 'left', 'right', 'look_up', 'look_down']

def success_flags(cfg, m, called_stop, near_goal_m):
    lo, hi = cfg['success']['clearance_range_m']
    loc = called_stop and m['obj_dist'] <= cfg['success']['max_object_dist_m'] and (lo <= m['clearance'] <= hi)
    return dict(success=bool(called_stop and is_success(cfg, m)), success_close=bool(called_stop and is_success_close(cfg, m)), success_loc=bool(loc), at_table=bool(called_stop and m['clearance'] <= hi), near_goal=bool(called_stop and near_goal_m <= 0.5))

def episode_setup(ctx: Context, seed, start_id, target, perm=None, mirror=None, layout_mode='slots', start_override=None):
    cfg = ctx.cfg
    layout = make_layout(cfg, seed, layout_mode, perm=perm, mirror=mirror)
    apply_layout(ctx.scene, layout)
    if start_override is not None:
        xy, yaw = start_override
    else:
        xy, yaw, _ = resolve_start(cfg, start_id, np.random.default_rng(seed + 10000))
    set_ego_pitch(ctx.scene, cfg['camera']['pitch_deg'])
    approach = valid_approach_poses(cfg, ctx.scene, ctx.cam, ctx.probe, ctx.loco.q0, layout, target, ctx.navmap)
    field = ctx.navmap.field([(x, y) for x, y, _ in approach]) if approach else None
    shortest = ctx.navmap.path_length(field, xy[0], xy[1]) if approach else math.nan
    return (layout, xy, yaw, approach, shortest)

def run_policy_episode(ctx: Context, client, embeds, seed, start_id, target, variant='g1', mode='gt', max_steps=150, deterministic=False, perm=None, mirror=None, stop_thresh=None, phrase=None, on_step=None, layout_mode='slots', start_override=None, setup=None, stop_min_px=None, controller_factory=None):
    layout, xy, yaw, approach, shortest = setup() if setup is not None else episode_setup(ctx, seed, start_id, target, perm, mirror, layout_mode, start_override)
    cfg = ctx.cfg
    if not approach:
        return dict(valid=False, reason='no_valid_approach_pose', seed=seed, start=start_id, target=target)
    native = variant == 'native'
    pitch = 0.0 if native else cfg['camera']['pitch_deg']
    set_ego_pitch(ctx.scene, pitch)
    ctx.loco.reset(xy, yaw)
    ctx.executor.settle()
    ctx.collisions[0] = 0
    client.reset(embeds[target], phrase or EVAL_PHRASE[target], target, mode, [0, 1, 2, 3, 4, 5] if native else [0, 1, 2, 3], deterministic, seed, stop_thresh=stop_thresh, **{} if stop_min_px is None else dict(stop_min_px=stop_min_px))
    controller = controller_factory(ctx, layout, approach) if controller_factory is not None else None
    k = OBJECT_KEYS.index(target) + 1
    hist, path, called_stop, fell, collisions = (Counter(), 0.0, False, False, 0)
    last, trace, mask_px = (ctx.loco.xy(), [], [])
    o = None
    reach_step = reach_loc_step = None
    min_obj_dist, stop_prob_max = (math.inf, 0.0)
    lo, hi = cfg['success']['clearance_range_m']
    for t in range(max_steps):
        o = observe(ctx, target)
        gt = cv2.resize((o.inst == k).astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST) if mode == 'gt' else None
        mm = stop_metrics(cfg, layout, target, o.x, o.y, o.yaw, o.ms)
        min_obj_dist = min(min_obj_dist, mm['obj_dist'])
        if reach_step is None and is_success(cfg, mm):
            reach_step = t
        if reach_loc_step is None and mm['obj_dist'] <= cfg['success']['max_object_dist_m'] and (lo <= mm['clearance'] <= hi):
            reach_loc_step = t
        idx, info = client.step(o.rgb, gt)
        if controller is not None:
            forced = controller(o, t)
            if forced is not None:
                if forced != idx:
                    client.override(int(forced))
                idx = int(forced)
        if on_step is not None:
            on_step(ctx, o, t, idx, info)
        if controller is None or controller.handover_step is not None:
            stop_prob_max = max(stop_prob_max, info['probs'][0])
        hist[ACTION_NAMES[idx]] += 1
        mask_px.append(info['mask_px'])
        trace.append((round(o.x, 3), round(o.y, 3)))
        if idx == 0:
            called_stop = True
            break
        if idx in (4, 5):
            pitch = float(np.clip(pitch + (30.0 if idx == 5 else -30.0), -90.0, 90.0))
            set_ego_pitch(ctx.scene, pitch)
            continue
        res = ctx.executor.execute(idx)
        collisions += int(ctx.collisions[0] > 0)
        ctx.collisions[0] = 0
        nx, ny = ctx.loco.xy()
        path += math.hypot(nx - last[0], ny - last[1])
        last = (nx, ny)
        if res.fell:
            fell = True
            break
    if not called_stop or fell:
        o = observe(ctx, target)
    m = stop_metrics(cfg, layout, target, o.x, o.y, o.yaw, o.ms)
    near = min((math.hypot(ax - o.x, ay - o.y) for ax, ay, _ in approach))
    fl = success_flags(cfg, m, called_stop and (not fell), near)
    spl = lambda s: float(s) * shortest / max(path, shortest) if shortest > 0 else float(s)
    out = dict(valid=True, seed=seed, start=start_id, target=target, variant=variant, mode=mode, steps=len(trace), called_stop=called_stop, fell=fell, collision_steps=collisions, path_m=round(path, 3), shortest_m=round(shortest, 3), **fl, reached=reach_step is not None, reached_loc=reach_loc_step is not None, reach_step=reach_step, min_obj_dist=round(min_obj_dist, 3), stop_prob_max=round(stop_prob_max, 4), spl=spl(fl['success']), spl_close=spl(fl['success_close']), spl_loc=spl(fl['success_loc']), final=dict(clearance=round(m['clearance'], 3), obj_dist=round(m['obj_dist'], 3), yaw_err_deg=round(m['yaw_err_deg'], 1), px=m['px']), actions=dict(hist), final_pitch=pitch, mask_px_mean=float(np.mean([p for p in mask_px if p >= 0])) if any((p >= 0 for p in mask_px)) else None, trace=trace[::max(1, len(trace) // 40)])
    if controller is not None:
        out['handover_step'] = controller.handover_step
    return out

class VideoRecorder:

    def __init__(self, path, goal='', fps=3, topdown=None, table=None, start=None):
        import imageio.v2 as imageio
        self.w = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
        self.goal, self.topdown, self.table, self.start, self.poses = (goal, topdown, table, start, [])

    def __call__(self, ctx, o, t, idx, info):
        third = ctx.cam.third_rgb(ctx.sim, o.yaw)
        ego = o.rgb
        tw = int(round(third.shape[1] * ego.shape[0] / third.shape[0] / 2)) * 2
        th = cv2.resize(third, (tw, ego.shape[0]))
        panels = [ego, th]
        if self.topdown is not None:
            self.poses.append((o.x, o.y))
            mp = self.topdown.frame(self.poses, (o.x, o.y, o.yaw), self.table, self.start)
            mw = int(round(mp.shape[1] * ego.shape[0] / mp.shape[0] / 2)) * 2
            panels.append(cv2.resize(mp, (mw, ego.shape[0])))
        frame = np.concatenate(panels, axis=1).copy()
        bar = np.zeros((34, frame.shape[1], 3), np.uint8)
        txt = f"goal: {self.goal}   t={t:3d}   action: {ACTION_NAMES[idx]:8s}   p(stop)={info['probs'][0]:.2f}"
        cv2.putText(bar, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        self.w.append_data(np.concatenate([bar, frame], axis=0))

    def close(self):
        self.w.close()

class RandomClient:

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, embed, phrase, key, mode, allowed, deterministic=False, seed=0):
        pass

    def step(self, rgb, gt224=None, force=None, peek=False):
        a = int(self.rng.choice([1, 2, 3], p=[0.5, 0.25, 0.25]))
        return (a, dict(probs=[0.0, *[float(a == j) for j in range(1, 6)]], mask_px=-1))

    def close(self):
        pass

def table_shortcut(ctx: Context, seed, start_id, target, perm=None, mirror=None, layout_mode='slots'):
    cfg = ctx.cfg
    layout, xy, yaw, approach, shortest = episode_setup(ctx, seed, start_id, target, perm, mirror, layout_mode)
    if not approach:
        return dict(valid=False)
    cx, cy = cfg['table']['center_xy']
    cands = [(x, y) for x, y in offset_ring(cfg, cfg['expert']['goal_clearance_m']) if ctx.navmap.is_free(x, y)]
    x, y = min(cands, key=lambda p: math.hypot(p[0] - xy[0], p[1] - xy[1]))
    pose_probe(ctx.scene, ctx.probe, ctx.loco.q0, x, y, bearing_to(x, y, cx, cy))
    ms = mask_stats(ctx.cam.object_mask(ctx.cam.segmentation(ctx.probe), target))
    m = stop_metrics(cfg, layout, target, x, y, bearing_to(x, y, cx, cy), ms)
    near = min((math.hypot(ax - x, ay - y) for ax, ay, _ in approach))
    fl = success_flags(cfg, m, True, near)
    return dict(valid=True, seed=seed, start=start_id, target=target, steps=0, called_stop=True, fell=False, collision_steps=0, path_m=0.0, shortest_m=round(shortest, 3), **fl, spl=float(fl['success']), spl_loc=float(fl['success_loc']))

def summarise(results):
    ok = [r for r in results if r.get('valid')]
    n = max(1, len(ok))
    mean = lambda key: sum((float(r[key]) for r in ok)) / n
    has = [r for r in ok if 'reached' in r]
    if has:
        out_extra = dict(reached=sum((r['reached'] for r in has)) / len(has), reached_loc=sum((r['reached_loc'] for r in has)) / len(has), min_obj_dist=sum((r['min_obj_dist'] for r in has)) / len(has), stop_prob_max=sum((r['stop_prob_max'] for r in has)) / len(has))
    else:
        out_extra = {}
    out = dict(n=len(ok), SR=mean('success'), SR_close=mean('success_close') if ok and 'success_close' in ok[0] else None, SR_loc=mean('success_loc'), **out_extra, near_goal=mean('near_goal'), at_table=mean('at_table'), called_stop=mean('called_stop'), SPL=mean('spl'), SPL_loc=mean('spl_loc'), steps=mean('steps'), collision_rate=mean('collision_steps') / max(1.0, mean('steps')) if ok else 0.0, falls=sum((r['fell'] for r in ok)))
    for grp in ('target', 'start'):
        out['by_' + grp] = {}
        for g in sorted({r[grp] for r in ok}, key=str):
            sub = [r for r in ok if r[grp] == g]
            out['by_' + grp][str(g)] = dict(n=len(sub), SR=sum((r['success'] for r in sub)) / len(sub), SR_loc=sum((r['success_loc'] for r in sub)) / len(sub), SR_close=sum((r.get('success_close', False) for r in sub)) / len(sub))
    return out
