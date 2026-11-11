# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from dataclasses import dataclass, field
import numpy as np
from .building import Building
from .building_scene import BuildingScene, build_building_scene
from .furniture import Layout
from .furniture_assets import add_furniture

@dataclass
class HouseInfo:
    layout: Layout
    body_id: dict = field(default_factory=dict)
    visual_geoms: dict = field(default_factory=dict)
    collider_geoms: dict = field(default_factory=dict)
    geom_item: np.ndarray = None

    def item_of_geom(self, g: int):
        k = int(self.geom_item[g]) if 0 <= g < len(self.geom_item) else 0
        return self.layout.items[k - 1] if k else None

    @property
    def all_colliders(self) -> list:
        out = []
        for v in self.collider_geoms.values():
            out += list(v) if isinstance(v, (list, tuple)) else [v]
        return out

def index_house(model, layout: Layout, names, table_body: int, table_geoms: list) -> HouseInfo:
    info = HouseInfo(layout)
    info.geom_item = np.zeros(model.ngeom + 1, dtype=np.int32)
    index = {it.id: k + 1 for k, it in enumerate(layout.items)}
    for iid, bname in names.body.items():
        info.body_id[iid] = model.body(bname).id
        info.visual_geoms[iid] = [model.geom(n).id for n in names.visual[iid]]
        for g in info.visual_geoms[iid]:
            info.geom_item[g] = index[iid]
        if iid in names.collider:
            info.collider_geoms[iid] = model.geom(names.collider[iid]).id
            info.geom_item[info.collider_geoms[iid]] = index[iid]
    for it in layout.items:
        if it.id not in names.body and it.fixed and (it.cat == 'table'):
            info.body_id[it.id] = table_body
            info.visual_geoms[it.id] = list(table_geoms)
            info.collider_geoms[it.id] = list(table_geoms)
            for g in table_geoms:
                info.geom_item[g] = index[it.id]
    return info

def build_house_scene(cfg: dict, building: Building, layout: Layout) -> BuildingScene:
    holder = {}

    def extra(spec, wb):
        bare = spec.copy().compile().stat
        for k in ('extent', 'center', 'meansize', 'meanmass', 'meaninertia'):
            v = getattr(bare, k)
            setattr(spec.stat, k, np.array(v, dtype=float) if np.ndim(v) else float(v))
        holder['names'] = add_furniture(spec, wb, layout)
    scene = build_building_scene(cfg, building, extra=extra)
    scene.house = index_house(scene.model, layout, holder['names'], scene.table_body, scene.table_geoms)
    return scene
