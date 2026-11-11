# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from .furniture import CATEGORIES
EXTRA = {'table': ['kitchen table', 'the table'], 'refrigerator': ['the fridge', 'a refrigerator'], 'cabinet': ['a cabinet', 'storage cabinet'], 'counter': ['countertop', 'kitchen worktop'], 'sink': ['the sink', 'washbasin'], 'stove': ['the stove', 'gas stove'], 'trash_can': ['garbage can', 'waste bin'], 'chair': ['a chair', 'the chair', 'seat'], 'potted_plant': ['houseplant', 'a plant', 'indoor plant'], 'tv_stand': ['television stand', 'media console'], 'tv': ['the tv', 'a television'], 'sofa': ['the sofa', 'a couch'], 'coffee_table': ['the coffee table', 'low table'], 'glasses': ['glasses', 'drinking glass set'], 'armchair': ['lounge chair', 'easy chair', 'an armchair'], 'piano': ['the piano', 'upright piano'], 'stool': ['a stool', 'bar stool'], 'bookshelf': ['the bookshelf', 'shelf'], 'lamp': ['standing lamp', 'a lamp'], 'rug': ['the rug', 'a rug'], 'picture': ['a painting', 'wall picture'], 'bed': ['the bed', 'a bed'], 'nightstand': ['night table', 'bedside cabinet'], 'wardrobe': ['the wardrobe', 'a closet'], 'desk': ['writing desk', 'the desk'], 'mirror': ['the mirror', 'wall mirror'], 'toy_chest': ['the toy chest', 'a toy box'], 'round_table': ['circular table', 'the round table'], 'bench': ['a bench', 'the bench'], 'console_table': ['hall table', 'the console table'], 'shoe_rack': ['shoe stand', 'the shoe rack']}

def phrases_for(cat: str) -> list:
    out = []
    for p in list(CATEGORIES[cat].prompts) + EXTRA.get(cat, []):
        if p not in out:
            out.append(p)
    return out

def eval_phrase(cat: str) -> str:
    return CATEGORIES[cat].prompts[0]

def all_phrases() -> dict:
    table = {}
    for c in CATEGORIES:
        for p in phrases_for(c):
            if table.setdefault(p, c) != c:
                raise ValueError(f'phrase {p!r} belongs to both {table[p]!r} and {c!r}')
    return table
HELDOUT = {'table': 'dinner table', 'refrigerator': 'icebox', 'cabinet': 'storage unit', 'counter': 'worktop', 'sink': 'kitchen basin', 'stove': 'hob', 'trash_can': 'rubbish bin', 'chair': 'dining chair', 'potted_plant': 'flower pot', 'tv_stand': 'tv unit', 'tv': 'telly', 'sofa': 'settee', 'coffee_table': 'centre table', 'glasses': 'tumblers', 'armchair': 'recliner', 'piano': 'grand piano', 'stool': 'footstool', 'bookshelf': 'library shelf', 'lamp': 'floor light', 'rug': 'floor mat', 'picture': 'wall art', 'bed': 'sleeping bed', 'nightstand': 'bedside locker', 'wardrobe': 'clothes cabinet', 'desk': 'study table', 'mirror': 'looking glass', 'toy_chest': 'toy bin', 'round_table': 'round dining table', 'bench': 'wooden bench', 'console_table': 'entry table', 'shoe_rack': 'footwear rack'}

def heldout_phrase(cat: str) -> str:
    return HELDOUT[cat]
