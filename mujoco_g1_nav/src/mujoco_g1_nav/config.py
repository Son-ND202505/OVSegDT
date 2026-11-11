# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import copy
import os
from pathlib import Path
import yaml
REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / 'mujoco_g1_nav'
DEFAULT_CONFIG = PKG_ROOT / 'configs' / 'default.yaml'

def load_config(path=None, profile=None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    profile = profile or os.environ.get('MUJOCO_G1_EXPERT_PROFILE') or 'classic'
    profiles = cfg.get('expert_profiles', {})
    if profile not in profiles:
        raise KeyError(f'unknown expert profile {profile!r}; available: {sorted(profiles)}')
    cfg['expert'] = {**copy.deepcopy(cfg['expert']), **copy.deepcopy(profiles[profile] or {})}
    cfg['expert_profile'] = profile
    return cfg

def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
