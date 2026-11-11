# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import numpy as np
START_CLEAR_M = 0.8
_SPLITS = {}

def choose_variant(seed: int, n_pool: int, p_base: float=0.15) -> int:
    rng = np.random.default_rng(int(seed) + 424242)
    if n_pool <= 1 or rng.random() < p_base:
        return 0
    return int(rng.integers(1, n_pool))

def prepare_episode(sess, seed: int, variant: int=None, colours: bool=True, perm_split: str='train', p_base: float=0.15, p_colour: float=0.8):
    import json
    from .config import PKG_ROOT
    from .house_variants import randomize_colours, reset_colours
    from .layout import MIRRORS, apply_layout, make_layout
    if not _SPLITS:
        with open(PKG_ROOT / 'configs' / 'splits.json') as f:
            _SPLITS.update(json.load(f))
    ctx = sess.ctx
    rng = np.random.default_rng(int(seed) + 737373)
    k = int(variant) if variant is not None else choose_variant(seed, len(sess.pool), p_base)
    sess.set_variant(k)
    n_col = randomize_colours(ctx, rng) if colours and rng.random() < p_colour else reset_colours(ctx) or 0
    pool = _SPLITS['train_perms' if perm_split == 'train' else 'heldout_perms']
    perm = tuple((int(i) for i in pool[int(rng.integers(len(pool)))]))
    mirror = MIRRORS[str(rng.choice(['n', 'x', 'y', 'xy']))]
    table_layout = make_layout(ctx.cfg, int(seed), 'slots', perm=perm, mirror=mirror)
    apply_layout(ctx.scene, table_layout)
    return (table_layout, dict(variant=k, recoloured=int(n_col), perm=list(perm), mirror=list(mirror)))

def start_cells(nav, clear: float=START_CLEAR_M) -> np.ndarray:
    ok = nav.space & (nav.wall_d >= clear) & (nav.furn_d >= clear) & (nav.clearance >= clear)
    return np.stack([nav.X[ok], nav.Y[ok]], 1)
