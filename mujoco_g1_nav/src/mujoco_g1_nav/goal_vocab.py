# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
OBJECT_KEYS = ('coca', 'pepsi', 'cup', 'orange', 'basket')
COLOUR = {'red': 'coca', 'blue': 'pepsi', 'white': 'cup', 'yellow': 'basket'}
GENERIC_CAN = ('can', 'soda', 'drink')
STOP_WORDS = frozenset('a an the to of in on at for from with and or my me i you we please want wanna need like go get find take bring show walk run move reach head navigate come\nlook looking search locate visit approach stand sit near nearest next closest big small large little some any one ones something someone thing things object objects item items stuff\nthat this there here robot g1 now just can could would will it is are be into by over under up down left right front back side other another favourite favorite'.split())
FILLER = frozenset({'thing', 'things', 'one', 'ones', 'object', 'objects', 'item', 'items', 'stuff', 'something'})
ACCEPT_SIM, ACCEPT_MARGIN = (0.96, 0.0)
REJECT_SIM = 0.88
CHECK_SIM, CHECK_GAP = (0.89, 0.0)
TYPO_CUTOFF = 0.8
TYPO_SURE = 0.88
TYPO_LONG = 6

@dataclass
class Goal:
    key: str
    kind: str
    canonical: str
    phrases: list = field(default_factory=list)
    label: str = ''

@dataclass
class Resolution:
    goal: object = None
    method: str = ''
    confidence: float = 0.0
    candidates: list = field(default_factory=list)
    note: str = ''
    matched: str = ''
    repaired: str = ''
    uncovered: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.goal is not None

def norm(text: str) -> str:
    t = re.sub('[^a-z0-9 ]+', ' ', str(text).lower().replace("'", ''))
    return re.sub('\\s+', ' ', t).strip()

def words(text: str):
    return norm(text).split()
RECIPES = ('bare', 'photo', 'mix')
RECIPE = 'photo'

def photo_text(text: str) -> str:
    return 'a photo of a ' + re.sub('^(a|an|the) ', '', norm(text))

def recipe_inputs(text: str, recipe: str=None) -> list:
    recipe = recipe or RECIPE
    return {'bare': [text], 'photo': [photo_text(text)], 'mix': [text, photo_text(text)]}[recipe]

def recipe_vector(vectors, recipe: str=None) -> np.ndarray:
    v = np.asarray(vectors, np.float32)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-09)
    return v.sum(0)

def load_goals(cache_dir=None, extra_path=None, synonyms: bool=True, synonyms_path=None) -> dict:
    from .config import PKG_ROOT
    from .furniture import CATEGORIES
    from .house_phrases import eval_phrase, phrases_for
    from .policy_eval import EVAL_PHRASE
    cache = Path(cache_dir) if cache_dir else PKG_ROOT / 'datasets' / 'cache'
    goals = {}
    for cat in CATEGORIES:
        goals[cat] = Goal(cat, 'category', eval_phrase(cat), list(phrases_for(cat)), cat.replace('_', ' '))
    z = np.load(cache / 'goal_embeds.npz')
    ph = {k: [] for k in OBJECT_KEYS}
    for p, k in zip(z['phrases'], z['keys']):
        ph[str(k)].append(str(p))
    names = {'coca': 'coca can (red)', 'pepsi': 'pepsi can (blue)', 'cup': 'cup (white mug)', 'orange': 'orange', 'basket': 'basket (yellow)'}
    for k in OBJECT_KEYS:
        goals[k] = Goal(k, 'object', EVAL_PHRASE[k], sorted(set(ph[k])), names[k])
    for g in load_extra(extra_path):
        goals[g.key] = g
    if synonyms:
        for key, phr in load_synonyms(synonyms_path).items():
            if key in goals:
                goals[key].phrases = list(dict.fromkeys(goals[key].phrases + list(phr)))
    return goals

def load_synonyms(path=None) -> dict:
    import json
    from .config import PKG_ROOT
    p = Path(path) if path else PKG_ROOT / 'configs' / 'goal_synonyms.json'
    if not p.exists():
        return {}
    with open(p) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith('_')}

def load_extra(path=None) -> list:
    import json
    from .config import PKG_ROOT
    p = Path(path) if path else PKG_ROOT / 'configs' / 'extra_goals.json'
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        spec = json.load(f)
    for key, d in spec.items():
        out.append(Goal(key, d.get('kind', 'object'), d.get('canonical', key.replace('_', ' ')), list(d.get('phrases') or [d.get('canonical', key)]), d.get('label', key.replace('_', ' '))))
    return out

class Lexicon:

    def __init__(self, goals: dict):
        self.goals = goals
        self.phrase_key = {}
        for g in goals.values():
            for p in g.phrases:
                n = norm(p)
                if not n:
                    continue
                if self.phrase_key.setdefault(n, g.key) != g.key:
                    raise ValueError(f'phrase {n!r} belongs to both {self.phrase_key[n]!r} and {g.key!r}')
        self.compact = {n.replace(' ', ''): n for n in self.phrase_key if ' ' in n}
        self.phrase_list = sorted(self.phrase_key, key=lambda s: (-len(s), s))
        self.known = {w for n in self.phrase_key for w in n.split()} | STOP_WORDS
        self.vocab = sorted({w for n in self.phrase_key for w in n.split() if w not in STOP_WORDS and len(w) >= 3})

    def singular(self, w: str) -> str:
        if w in self.known or len(w) < 4:
            return w
        if w.endswith('es') and w[:-2] in self.known:
            return w[:-2]
        if w.endswith('s') and w[:-1] in self.known:
            return w[:-1]
        return w

    def prep(self, text: str) -> str:
        w = [self.singular(x) for x in words(text)]
        out, i = ([], 0)
        while i < len(w):
            if i + 1 < len(w) and w[i] + w[i + 1] in self.phrase_key and (' ' not in w[i] + w[i + 1]):
                out.append(w[i] + w[i + 1])
                i += 2
                continue
            out.append(self.compact.get(w[i], w[i]))
            i += 1
        return ' '.join(out)

    def find(self, text: str):
        t = ' ' + text + ' '
        spans = []
        for p in self.phrase_list:
            for m in re.finditer('(?<= )' + re.escape(p) + '(?= )', t):
                spans.append((m.start(), m.end(), p))
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        taken, out = ([], [])
        for a, b, p in spans:
            if any((a < tb and ta < b for ta, tb in taken)):
                continue
            taken.append((a, b))
            out.append((a, p))
        return [(p, self.phrase_key[p]) for a, p in sorted(out)]

    def repair(self, text: str):
        out, fixes = ([], [])
        for w in text.split():
            if w in self.known or len(w) < 4:
                out.append(w)
                continue
            m = difflib.get_close_matches(w, self.vocab, n=1, cutoff=TYPO_CUTOFF)
            if m:
                out.append(m[0])
                fixes.append((w, m[0], difflib.SequenceMatcher(None, w, m[0]).ratio()))
            else:
                out.append(w)
        return (self.prep(' '.join(out)), fixes)

class GoalResolver:

    def __init__(self, goals: dict=None, embed_fn=None, phrase_embeds=None, accept_sim=ACCEPT_SIM, accept_margin=ACCEPT_MARGIN, reject_sim=REJECT_SIM, check_sim=CHECK_SIM, check_gap=CHECK_GAP):
        self.goals = goals or load_goals()
        self.lex = Lexicon(self.goals)
        self.embed_fn = embed_fn
        self.accept_sim, self.accept_margin, self.reject_sim, self.check_sim, self.check_gap = (accept_sim, accept_margin, reject_sim, check_sim, check_gap)
        self.pe_phrases = self.pe_keys = self.pe = None
        self._lexical_cache = {}
        if phrase_embeds is not None:
            self.set_phrase_embeds(*phrase_embeds)

    def set_phrase_embeds(self, phrases, keys, emb):
        e = np.asarray(emb, np.float32)
        self.pe_phrases, self.pe_keys = ([str(p) for p in phrases], [str(k) for k in keys])
        self.pe = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-09)

    def lexicon_phrases(self):
        return list(self.lex.phrase_key.items())

    def ensure_phrase_embeds(self, cache_path, embed_batch_fn, recipe: str=None):
        recipe = recipe or RECIPE
        cache_path = Path(cache_path)
        have, old_keys = ({}, {})
        if cache_path.exists():
            z = np.load(cache_path)
            if str(z['recipe']) == recipe:
                have = {str(p): e for p, e in zip(z['phrases'], z['embeds'])}
                old_keys = {str(p): str(k) for p, k in zip(z['phrases'], z['keys'])}
        pk = self.lexicon_phrases()
        todo = [p for p, _ in pk if p not in have]
        if todo:
            rows = [recipe_inputs(p, recipe) for p in todo]
            flat = embed_batch_fn([x for r in rows for x in r])
            i = 0
            for p, r in zip(todo, rows):
                have[p] = recipe_vector(flat[i:i + len(r)], recipe)
                i += len(r)
        self.set_phrase_embeds([p for p, _ in pk], [k for _, k in pk], np.stack([have[p] for p, _ in pk]))
        if todo:
            keys = {**old_keys, **dict(pk)}
            names = sorted(have)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, phrases=np.array(names), keys=np.array([keys.get(n, '') for n in names]), embeds=np.stack([have[n] for n in names]).astype(np.float32), recipe=np.array(recipe))
        return len(todo)

    def embed_lexicon(self, embed_batch_fn, recipe: str=None):
        pk = self.lexicon_phrases()
        rows = [recipe_inputs(p, recipe) for p, _ in pk]
        flat = embed_batch_fn([x for r in rows for x in r])
        out, i = ([], 0)
        for r in rows:
            out.append(recipe_vector(flat[i:i + len(r)], recipe))
            i += len(r)
        return ([p for p, _ in pk], [k for _, k in pk], np.stack(out))

    def _hits(self, prepared: str, method: str, repaired: str=''):
        hits = self.lex.find(prepared)
        keys = list(dict.fromkeys((k for _, k in hits)))
        if not keys:
            return None
        matched = ', '.join((p for p, _ in hits))
        if len(keys) > 1:
            return Resolution(None, 'ambiguous', 0.0, [(self.goals[k], 1.0) for k in keys], f"the text names several goals: {', '.join((self.goals[k].label for k in keys))}", matched, repaired)
        covered = {w for p, _ in hits for w in p.split()}
        unc = [w for w in prepared.split() if w not in STOP_WORDS and w not in covered]
        return Resolution(self.goals[keys[0]], method, 1.0 if method == 'keyword' else 0.95, [(self.goals[keys[0]], 1.0)], '', matched, repaired, unc)

    def _colour_or_generic(self, prepared: str):
        w = prepared.split()
        if [x for x in w if x not in STOP_WORDS and x not in COLOUR and (x not in GENERIC_CAN)]:
            return None
        cols = list(dict.fromkeys((COLOUR[c] for c in w if c in COLOUR)))
        if set(w) & set(GENERIC_CAN):
            cands = [c for c in cols if c in ('coca', 'pepsi')] or ['coca', 'pepsi']
            if len(cands) == 1:
                return Resolution(self.goals[cands[0]], 'keyword', 0.9, [(self.goals[cands[0]], 0.9)], '', 'can + colour')
            return Resolution(None, 'ambiguous', 0.0, [(self.goals['coca'], 0.0), (self.goals['pepsi'], 0.0)], 'which can: coca (red) or pepsi (blue)?')
        if len(cols) == 1:
            return Resolution(self.goals[cols[0]], 'keyword', 0.85, [(self.goals[cols[0]], 0.85)], '', 'colour word')
        if len(cols) > 1:
            return Resolution(None, 'ambiguous', 0.0, [(self.goals[c], 0.0) for c in cols], 'the text names several colours')
        return None

    def _lexical(self, text: str):
        if text in self._lexical_cache:
            return self._lexical_cache[text]
        prepared = self.lex.prep(text)
        r = self._hits(prepared, 'keyword')
        if r is None or (r.ok and r.uncovered):
            repaired, fixes = self.lex.repair(prepared)
            r2 = self._hits(repaired, 'typo', repaired) if fixes else None
            if r2 is not None and (r is None or (r2.ok and len(r2.uncovered) < len(r.uncovered))):
                r2.note = 'repaired ' + ', '.join((f"'{a}' -> '{b}'" for a, b, _ in fixes))
                weak = [a for a, b, q in fixes if q < TYPO_SURE and len(a) < TYPO_LONG]
                if r2.ok and weak:
                    r2 = Resolution(None, 'ambiguous', 0.8, [(r2.goal, 0.8)], f"did you mean '{r2.matched}'? ({r2.note})", r2.matched, r2.repaired, r2.uncovered)
                r = r2
            if r is None:
                r = self._colour_or_generic(repaired if fixes else prepared)
        self._lexical_cache[text] = r
        return r

    def meaning_scores(self, text: str):
        if self.embed_fn is None or self.pe is None:
            return None
        v = np.asarray(self.embed_fn(text), np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-09)
        best = {}
        for k, s in zip(self.pe_keys, self.pe @ v):
            if s > best.get(k, -1.0):
                best[k] = float(s)
        return sorted(best.items(), key=lambda kv: -kv[1])

    def resolve(self, text: str) -> Resolution:
        t = norm(text)
        if not t:
            return Resolution(None, 'unknown', 0.0, [], 'empty text')
        r = self._lexical(t)
        if r is not None:
            if r.ok and r.uncovered and (self.embed_fn is not None) and (self.pe is not None):
                ranked = self.meaning_scores(text)
                own = dict(ranked).get(r.goal.key, -1.0)
                top_key, top = ranked[0]
                if top_key != r.goal.key and top >= self.check_sim and (top - own >= self.check_gap):
                    return Resolution(None, 'ambiguous', top, [(self.goals[top_key], top), (r.goal, own)], f"'{r.matched}' matches {r.goal.label}, but the whole text ({' '.join(r.uncovered)} ...) means {self.goals[top_key].label} more ({top:.2f} vs {own:.2f})", r.matched, r.repaired, r.uncovered)
            return r
        ranked = self.meaning_scores(text)
        if ranked is None:
            return Resolution(None, 'unknown', 0.0, [], 'no embedder for the meaning layer')
        cands = [(self.goals[k], s) for k, s in ranked[:3]]
        (g1, s1), s2 = (cands[0], cands[1][1] if len(cands) > 1 else -1.0)
        if s1 >= self.accept_sim and s1 - s2 >= self.accept_margin:
            return Resolution(g1, 'meaning', s1, cands, f'similarity {s1:.2f} (runner-up {s2:.2f})')
        if s1 < self.reject_sim:
            return Resolution(None, 'unknown', s1, cands, f'best similarity {s1:.2f} < {self.reject_sim:.2f}: not one of the goals')
        return Resolution(None, 'ambiguous', s1, cands, f'too close to call: {s1:.2f} vs {s2:.2f}')
