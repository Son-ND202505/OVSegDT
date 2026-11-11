# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import glob
import pickle
import numpy as np
from .config import REPO_ROOT
_MODEL = {}

def _load():
    if not _MODEL:
        import torch
        from open_clip import create_model_from_pretrained
        from tokenizers import Tokenizer
        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-256')
        model.eval()
        tk = Tokenizer.from_file(glob.glob(str(__import__('pathlib').Path.home() / '.cache/huggingface/hub/models--timm--ViT-B-16-SigLIP-256/snapshots/*/tokenizer.json'))[0])
        tk.enable_truncation(64)
        tk.enable_padding(length=64, pad_id=1, pad_token='</s>')
        _MODEL.update(torch=torch, model=model, tk=tk)
    return _MODEL

def encode_goals(phrases) -> np.ndarray:
    from open_clip.tokenizer import canonicalize_text
    m = _load()
    ids = m['torch'].tensor([e.ids for e in m['tk'].encode_batch([canonicalize_text(p) for p in phrases])])
    with m['torch'].no_grad():
        return m['model'].encode_text(ids).float().numpy()

def existing_cache() -> dict:
    with open(REPO_ROOT / 'data' / 'text_embeddings' / 'siglip.pkl', 'rb') as f:
        return pickle.load(f)

def release_text_model():
    import gc
    _MODEL.clear()
    gc.collect()
