# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import cv2
import numpy as np

def _layout_renderer(ctx, width: int=520):
    from .building_viz import TopDownRenderer
    return TopDownRenderer(ctx.building, width=width, layout=ctx.house.layout)

def _compose(ctx, rgb, mask, topdown, poses, pose, text, mark=None, explored=None, goal=None, fps_panel=True):
    ego = rgb.copy()
    if mask is not None and mask.any():
        ego[mask] = (0.55 * ego[mask] + 0.45 * np.array([40, 220, 80])).astype(np.uint8)
    third = ctx.cam.third_rgb(ctx.sim, pose[2])
    tw = int(round(third.shape[1] * ego.shape[0] / third.shape[0] / 2)) * 2
    panels = [ego, cv2.resize(third, (tw, ego.shape[0]))]
    mp = topdown.frame(poses, pose, None, mark, explored=explored, goal=goal)
    mw = int(round(mp.shape[1] * ego.shape[0] / mp.shape[0] / 2)) * 2
    panels.append(cv2.resize(mp, (mw, ego.shape[0])))
    frame = np.concatenate(panels, axis=1).copy()
    bar = np.zeros((34, frame.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], axis=0)

class StepVideo:
    NAMES = {0: 'STOP', 1: 'forward', 2: 'left', 3: 'right'}

    def __init__(self, path, ctx, cat: str, fps: int=4):
        import imageio.v2 as imageio
        self.w = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
        self.cat, self.poses, self.top = (cat, [], _layout_renderer(ctx))

    def __call__(self, ctx, o, t, action, info, teacher):
        self.poses.append((o.x, o.y))
        it = teacher.locked
        mark = (it.x, it.y) if it is not None else None
        d = info.get('obj_dist')
        text = f"goal: {self.cat}   t={t:3d}   {info.get('phase', ''):8s}  action {self.NAMES[int(action)]:7s}  d={('-' if d is None else f'{d:.2f}')}  px={o.ms['px']}"
        self.w.append_data(_compose(ctx, o.rgb, o.inst.astype(bool), self.top, self.poses, (o.x, o.y, o.yaw), text, mark, explored=teacher.fog.explored))

    def close(self):
        self.w.close()

class PolicyVideo:
    NAMES = ['STOP', 'forward', 'left', 'right', 'up', 'down']

    def __init__(self, path, ctx, cat: str, world, fps: int=4):
        import imageio.v2 as imageio
        self.w = imageio.get_writer(str(path), fps=fps, macro_block_size=None)
        self.cat, self.world, self.poses, self.top = (cat, world, [], _layout_renderer(ctx))
        self.driver = 'policy'

    def __call__(self, ctx, t, idx, info, m):
        from .house_eval import category_mask
        x, y = (float(v) for v in ctx.loco.xy())
        yaw = ctx.loco.yaw()
        self.poses.append((x, y))
        seg = ctx.cam.segmentation(ctx.sim)
        cm = category_mask(self.world, seg, self.cat)
        text = f"goal: {self.cat}   t={t:3d}   [{self.driver}] action {self.NAMES[idx]:7s}  p(stop)={info['probs'][0]:.2f}  d={m['d']:.2f}  px={m['px']}"
        self.w.append_data(_compose(ctx, ctx.cam.rgb(ctx.sim), cm, self.top, self.poses, (x, y, yaw), text))

    def close(self):
        self.w.close()
