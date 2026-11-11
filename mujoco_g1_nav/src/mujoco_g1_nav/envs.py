# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from dataclasses import dataclass
from .building import Building, load_building
from .config import load_config
ENV_NAMES = ('single_room', 'building35', 'building45', 'building25', 'cross25')
DEFAULT_ENV = 'single_room'
STEP_BUDGETS = {'single_room': 150, 'building25': 400, 'cross25': 350, 'building35': 600, 'building45': 800}
FAST_ENV, TRAIN_ENVS, HELDOUT_ENV = ('cross25', ('cross25', 'building25', 'building35'), 'building45')
HOUSE_ENVS = {'home25': 450, 'l28': 850}

@dataclass
class Env:
    name: str
    cfg: dict
    building: Building = None
    step_budget: int = 150
    layout: object = None

    @property
    def is_building(self) -> bool:
        return self.building is not None

def add_env_argument(ap, default=DEFAULT_ENV):
    ap.add_argument('--env', default=default, choices=ENV_NAMES, help='single_room = the original 10 x 10 m room (default); cross25 / building25 / building35 / building45 = multi-room buildings (cross25 is the fastest)')

def load_env(name: str=DEFAULT_ENV, profile=None) -> Env:
    if name not in ENV_NAMES:
        raise KeyError(f'unknown environment {name!r}; choose from {ENV_NAMES}')
    cfg = load_config(profile=profile)
    return Env(name, cfg, None if name == 'single_room' else load_building(name), STEP_BUDGETS[name])

def load_house_env(name: str='home25', profile=None) -> Env:
    if name not in HOUSE_ENVS:
        raise KeyError(f'unknown house {name!r}; choose from {tuple(HOUSE_ENVS)}')
    from .furniture import enable_plus_catalogue, load_layout
    if name.endswith('_plus'):
        enable_plus_catalogue()
    building = load_building(name)
    return Env(name, load_config(profile=profile), building, HOUSE_ENVS[name], load_layout(name, building))

def make_env_context(env: Env):
    if env.layout is not None:
        from .building_context import make_house_context
        return make_house_context(env.cfg, env.building, env.layout)
    if not env.is_building:
        from .collect import make_context
        return make_context(env.cfg)
    from .building_context import make_building_context
    return make_building_context(env.cfg, env.building)
