# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle
OPEN_COLOURS = {'corridor': (0.9, 0.9, 0.92), 'spur': (0.86, 0.9, 0.95)}
WALL_COLOUR = (0.16, 0.16, 0.18)
DEFAULT_ROOM = (0.96, 0.95, 0.92)

def _rect(ax, r, **kw):
    ax.add_patch(Rectangle((r.x0, r.y0), r.w, r.h, **kw))

def plot_building(b, ax=None, table=None, robot=None, path=None, points=None, seen=None, labels=True, title=None, figsize=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(6.0, b.footprint.w / 3.2), max(4.0, b.footprint.h / 3.2)), dpi=110)
    for r in b.regions.values():
        c = r.floor_rgb if r.floor_rgb is not None else DEFAULT_ROOM if r.is_room else OPEN_COLOURS[r.kind]
        _rect(ax, r.rect, facecolor=c, edgecolor='none', zorder=1)
    for w in b.walls:
        _rect(ax, w.rect, facecolor=WALL_COLOUR, edgecolor='none', zorder=3)
    for d in b.doors.values():
        _rect(ax, d.rect, facecolor=(0.98, 0.72, 0.15), edgecolor='none', zorder=2)
    if seen is not None:
        xs, ys, mask = seen
        ax.imshow(np.where(mask, 1.0, np.nan), extent=(xs[0] - (xs[1] - xs[0]) / 2, xs[-1] + (xs[1] - xs[0]) / 2, ys[0] - (ys[1] - ys[0]) / 2, ys[-1] + (ys[1] - ys[0]) / 2), origin='lower', cmap='Greens', vmin=0, vmax=2, alpha=0.45, zorder=2, interpolation='nearest')
    if labels:
        for r in b.regions.values():
            ax.text(*r.center, r.id, ha='center', va='center', fontsize=13 if r.is_room else 9, color='0.25', zorder=5, fontweight='bold' if r.is_room else None)
    if table is not None:
        cx, cy, sx, sy = table
        _rect(ax, type('R', (), dict(x0=cx - sx / 2, y0=cy - sy / 2, w=sx, h=sy))(), facecolor=(0.55, 0.36, 0.18), edgecolor='k', linewidth=0.6, zorder=6)
    if path is not None and len(path) > 1:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], '-', color='tab:blue', linewidth=1.6, zorder=7)
    for name, (x, y) in (points or {}).items():
        ax.plot([x], [y], 'o', color='tab:red', markersize=4, zorder=8)
        ax.text(x + 0.3, y + 0.3, name, fontsize=7, color='tab:red', zorder=8)
    if robot is not None:
        x, y, yaw = robot
        ax.add_patch(FancyArrow(x, y, 0.9 * np.cos(yaw), 0.9 * np.sin(yaw), width=0.55, head_width=0.9, head_length=0.6, color='tab:red', zorder=9))
    fp = b.footprint
    ax.set_xlim(fp.x0 - 0.6, fp.x1 + 0.6)
    ax.set_ylim(fp.y0 - 0.6, fp.y1 + 0.6)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title or b.name, fontsize=10)
    ax.grid(True, color='0.8', linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return ax

class TopDownRenderer:
    MARGIN = 0.6

    def __init__(self, building, width: int=640, labels: bool=True, layout=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fp = building.footprint
        self.fp = fp
        self.x0, self.y1 = (fp.x0 - self.MARGIN, fp.y1 + self.MARGIN)
        fw, fh = (fp.w + 2 * self.MARGIN, fp.h + 2 * self.MARGIN)
        self.scale = width / fw
        self.w, self.h = (int(width), int(round(fh * self.scale)))
        fig = plt.figure(figsize=(self.w / 100, self.h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        if layout is not None:
            from .house_viz import plot_house
            plot_house(layout, ax=ax, labels=False, legend=False)
        else:
            plot_building(building, ax=ax, labels=labels)
        ax.set_xlim(fp.x0 - self.MARGIN, fp.x1 + self.MARGIN)
        ax.set_ylim(fp.y0 - self.MARGIN, fp.y1 + self.MARGIN)
        ax.axis('off')
        fig.canvas.draw()
        self.background = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)

    def px(self, x, y) -> tuple:
        return (int(round((x - self.x0) * self.scale)), int(round((self.y1 - y) * self.scale)))

    def frame(self, path=None, robot=None, table=None, mark=None, text: str='', explored=None, goal=None) -> np.ndarray:
        import cv2
        img = self.background.copy()
        if explored is not None:
            fp = self.fp
            (x0, y0), (x1, y1) = (self.px(fp.x0, fp.y1), self.px(fp.x1, fp.y0))
            m = cv2.resize(np.ascontiguousarray(explored[::-1]).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
            sub = img[y0:y1, x0:x1]
            sub[m] = (0.62 * sub[m] + 0.38 * np.array([60, 190, 90])).astype(np.uint8)
        if table is not None:
            cx, cy, sx, sy = table
            cv2.rectangle(img, self.px(cx - sx / 2, cy + sy / 2), self.px(cx + sx / 2, cy - sy / 2), (140, 92, 46), -1)
            cv2.rectangle(img, self.px(cx - sx / 2, cy + sy / 2), self.px(cx + sx / 2, cy - sy / 2), (20, 20, 20), 1)
        if path is not None and len(path) > 1:
            cv2.polylines(img, [np.array([self.px(x, y) for x, y in path], np.int32)], False, (31, 119, 180), 2, cv2.LINE_AA)
        if goal is not None:
            cv2.circle(img, self.px(*goal), int(0.5 * self.scale), (30, 60, 220), 2, cv2.LINE_AA)
        if mark is not None:
            cv2.circle(img, self.px(*mark), 5, (214, 39, 40), -1, cv2.LINE_AA)
        if robot is not None:
            x, y, yaw = robot
            p0, p1 = (self.px(x, y), self.px(x + 1.1 * np.cos(yaw), y + 1.1 * np.sin(yaw)))
            cv2.circle(img, p0, int(0.35 * self.scale), (214, 39, 40), -1, cv2.LINE_AA)
            cv2.arrowedLine(img, p0, p1, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        if text:
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        return img
