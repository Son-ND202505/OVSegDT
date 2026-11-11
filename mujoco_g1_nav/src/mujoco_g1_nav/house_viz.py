# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import math
import numpy as np
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from .building_viz import plot_building
from .furniture import CATEGORIES, Item, Layout
GROUP_COLOURS = {'seating': '#6fa8dc', 'table': '#d9a066', 'storage': '#a67c52', 'bed': '#b4a7d6', 'appliance': '#8fc1c4', 'electronics': '#555555', 'decor': '#c9c9c9'}
CAT_COLOURS = {'potted_plant': '#6aa84f', 'lamp': '#f1c232', 'trash_can': '#999999', 'glasses': '#76d7ea', 'rug': '#e6a8b8', 'picture': '#c2185b', 'mirror': '#7b1fa2', 'tv': '#222222', 'table': '#c07a3a', 'piano': '#333333'}
ROOM_TITLES = {'A': 'A  bếp (kitchen)', 'B': 'B  phòng khách (living room)', 'C': 'C  phòng ngủ 1 (bedroom 1)', 'D': 'D  phòng ngủ 2 (bedroom 2)'}
LEGEND = [('sofa, ghế, ghế đẩu, ghế dài', GROUP_COLOURS['seating']), ('bàn', GROUP_COLOURS['table']), ('tủ, kệ', GROUP_COLOURS['storage']), ('giường', GROUP_COLOURS['bed']), ('bếp, bồn rửa, tủ lạnh', GROUP_COLOURS['appliance']), ('tivi, đàn piano', GROUP_COLOURS['electronics']), ('cây', CAT_COLOURS['potted_plant']), ('đèn', CAT_COLOURS['lamp']), ('thảm (đi lên được)', CAT_COLOURS['rug']), ('tranh / gương (treo tường)', CAT_COLOURS['picture'])]
SHORT = {'refrigerator': 'fridge', 'potted_plant': 'plant', 'coffee_table': 'coffee tbl', 'console_table': 'console', 'round_table': 'round tbl', 'toy_chest': 'toy chest', 'bookshelf': 'bookshelf', 'nightstand': 'nightst.', 'trash_can': 'bin', 'armchair': 'armchair', 'tv_stand': 'tv stand', 'shoe_rack': 'shoes', 'wardrobe': 'wardrobe', 'glasses': 'glasses'}

def colour_of(item: Item) -> str:
    return CAT_COLOURS.get(item.cat) or GROUP_COLOURS.get(CATEGORIES[item.cat].group, '#bbbbbb')

def _draw_item(ax, it: Item, labels: bool, fontsize: float, z: int=6):
    c = colour_of(it)
    if it.kind == 'wall':
        k = it.corners()
        ax.plot([k[0, 0] + (k[3, 0] - k[0, 0]) * 0.0, k[3, 0]], [k[0, 1], k[3, 1]], color=c, linewidth=3.2, solid_capstyle='butt', zorder=z + 1)
        ax.plot([k[1, 0], k[2, 0]], [k[1, 1], k[2, 1]], color=c, linewidth=3.2, solid_capstyle='butt', zorder=z + 1)
        return
    alpha = 0.55 if it.kind == 'flat' else 1.0
    zo = {'flat': z - 2, 'floor': z, 'surface': z + 2}[it.kind]
    edge = 'none' if it.kind == 'flat' else '#222222'
    if it.shape == 'round':
        ax.add_patch(Circle((it.x, it.y), it.depth / 2, facecolor=c, edgecolor=edge, linewidth=0.7, alpha=alpha, zorder=zo))
    else:
        ax.add_patch(Polygon(it.corners(), closed=True, facecolor=c, edgecolor=edge, linewidth=0.7, alpha=alpha, zorder=zo))
    if it.kind != 'flat' and it.shape != 'round' and (CATEGORIES[it.cat].front_phi is not None):
        fx, fy = it.front
        a = (it.x + fx * it.depth / 2, it.y + fy * it.depth / 2)
        ax.plot([a[0], a[0] + fx * 0.14], [a[1], a[1] + fy * 0.14], color='#222222', linewidth=1.4, zorder=zo + 1, solid_capstyle='butt')
    if labels and it.kind != 'wall':
        txt = SHORT.get(it.cat, it.cat.replace('_', ' '))
        dark = c in ('#555555', '#222222', '#333333', '#c2185b', '#7b1fa2')
        x0, y0, x1, y1 = it.bbox()
        ax.text(it.x, it.y, txt, ha='center', va='center', fontsize=fontsize, color='white' if dark else '#111111', zorder=zo + 2, rotation=0 if x1 - x0 >= y1 - y0 else 90, clip_on=True)

def plot_house(layout: Layout, report=None, ax=None, labels: bool=True, xlim=None, ylim=None, fontsize: float=5.5, figsize=None, title: str=None, stand: bool=False, clearance: bool=False, ids: bool=False, legend: bool=True):
    import matplotlib.pyplot as plt
    b = layout.building
    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (13, 13), dpi=110)
    plot_building(b, ax=ax, labels=False, title=title or f'{layout.name}: {len(layout)} items')
    if clearance and report is not None:
        a = report.arrays
        clr = np.where(a['space'], np.minimum(a['clearance'], 2.0), np.nan)
        ext = (a['x0'], a['x0'] + clr.shape[1] * a['res'], a['y0'], a['y0'] + clr.shape[0] * a['res'])
        ax.imshow(2 * clr, extent=ext, origin='lower', cmap='RdYlGn', vmin=0.0, vmax=3.2, alpha=0.55, zorder=2, interpolation='nearest')
        ax.contour(np.linspace(ext[0], ext[1], clr.shape[1]), np.linspace(ext[2], ext[3], clr.shape[0]), np.where(np.isnan(clr), 0, 2 * clr), levels=[1.2], colors=['#b00020'], linewidths=0.8, zorder=3)
    order = {'flat': 0, 'floor': 1, 'surface': 2, 'wall': 3}
    for it in sorted(layout.items, key=lambda i: order[i.kind]):
        _draw_item(ax, it, labels, fontsize)
        if ids and it.kind != 'wall':
            ax.text(it.x, it.y - 0.18, it.id, ha='center', va='top', fontsize=fontsize - 1.5, color='#333333', zorder=12, clip_on=True)
    if stand and report is not None:
        for r in report.items:
            if r.stand is not None and r.kind in ('floor', 'surface'):
                x, y, yaw = r.stand
                ax.plot([x], [y], 'o', color='#1565c0', markersize=3, zorder=11)
                ax.plot([x, x + 0.35 * math.cos(yaw)], [y, y + 0.35 * math.sin(yaw)], '-', color='#1565c0', linewidth=0.9, zorder=11)
    for rid, reg in b.regions.items():
        r = reg.rect
        if reg.is_room:
            label = f'{rid}  {layout.rooms.get(rid, rid)}'
            ax.text(*reg.center, label, ha='center', va='center', fontsize=12, fontweight='bold', color='#8a8a8a', alpha=0.35, zorder=2, clip_on=True)
        elif rid != 'X':
            ax.text(*reg.center, f'hành lang {rid}', ha='center', va='center', fontsize=8, color='#555555', zorder=1, style='italic', clip_on=True)
        else:
            ax.text(*reg.center, 'ngã tư\n(để trống)', ha='center', va='center', fontsize=8, color='#555555', zorder=1, style='italic', clip_on=True)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor='#222222', label=n) for n, c in LEGEND], loc='upper center', bbox_to_anchor=(0.5, -0.045), ncol=5, fontsize=7.5, frameon=False)
    return ax

def plot_rooms(layout: Layout, report=None, path=None, margin: float=0.4, fontsize: float=7.5, **kw):
    import matplotlib.pyplot as plt
    room_ids = layout.building.room_ids
    nrows = (len(room_ids) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(17, 8.5 * nrows), dpi=100, squeeze=False)
    for k, rid in enumerate(room_ids):
        i, j = divmod(k, 2)
        r = layout.building.regions[rid].rect
        plot_house(layout, report, ax=axes[i, j], xlim=(r.x0 - margin, r.x1 + margin), ylim=(r.y0 - margin, r.y1 + margin), fontsize=fontsize, legend=False, title=f'{rid}  {layout.rooms.get(rid, rid)}', **kw)
    for k in range(len(room_ids), nrows * 2):
        axes.flat[k].axis('off')
    fig.legend(handles=[Patch(facecolor=c, edgecolor='#222222', label=n) for n, c in LEGEND], loc='lower center', ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path:
        fig.savefig(path)
        plt.close(fig)
    return fig
