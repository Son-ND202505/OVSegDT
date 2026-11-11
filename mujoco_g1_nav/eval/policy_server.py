# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
"""OVSegDT policy server for the MuJoCo demo. Run in the `ovsegdt` conda env (Python 3.8) from the repo root; play_house.py starts it.
Requests over a local TCP socket (4-byte big-endian header length | JSON header | raw payload):
  reset: {"goal_embed": [768 floats], "phrase": str, "key": str, "mode": "gt", ...}
  step : {"h": int, "w": int, "has_gt": bool} + RGB bytes [+ 224*224 target mask bytes]  ->  {"idx": action id, "name": str, "probs": [6 floats], "mask_px": int}
Action ids: 0 stop, 1 move_forward, 2 turn_left, 3 turn_right, 4 look_up, 5 look_down."""
import argparse
import os
import json
import socket
import struct
import sys
import time
import cv2
import numpy as np
import torch
sys.path.insert(0, '.')
from ovon.deploy.inference import OVSegDTPolicy
NAMES = ['stop', 'move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down']
TUNED = {'coca': ('tin can', 'red'), 'pepsi': ('tin can', 'blue'), 'cup': ('mug', None), 'orange': ('ball', None), 'basket': ('basket', None)}

def recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('client closed')
        buf += chunk
    return buf

def send(conn, header, payload=b''):
    h = json.dumps(dict(header, payload_len=len(payload))).encode()
    conn.sendall(struct.pack('>I', len(h)) + h + payload)

def receive(conn):
    n = struct.unpack('>I', recv_exact(conn, 4))[0]
    header = json.loads(recv_exact(conn, n).decode())
    return (header, recv_exact(conn, header.get('payload_len', 0)))

def colour_of(rgb, m):
    px = rgb[m].astype(float)
    if len(px) == 0:
        return 'none'
    r, g, b = px.mean(0)
    if r - max(g, b) > 40:
        return 'red'
    if b - max(r, g) > 25:
        return 'blue'
    return 'other'

class Detector:

    def __init__(self):
        from ultralytics import YOLOE
        self.model = YOLOE('yoloe-v8l-seg.pt').to('cuda')
        self.prompt = None

    def masks(self, image, prompt, imgsz, retina, conf):
        key = tuple(prompt) if isinstance(prompt, (list, tuple)) else (prompt,)
        if key != self.prompt:
            self.model.set_classes(list(key), self.model.get_text_pe(list(key)))
            self.prompt = key
        r = self.model.predict(image, imgsz=imgsz, conf=conf, retina_masks=retina, verbose=False)[0]
        if r.masks is None:
            return ([], [])
        return ([m > 0.5 for m in r.masks.data.cpu().numpy()], [float(c) for c in r.boxes.conf.cpu().numpy()])
RULES = {'coca': (('tin can', 'can', 'soda can'), 0.03, 'red', 0.2), 'pepsi': (('tin can', 'can', 'soda can'), 0.03, 'blue', 0.2), 'cup': (('mug', 'cup'), 0.2, 'white', 0.45), 'orange': (('ball', 'orange'), 0.2, 'orange', 0.3), 'basket': (('basket',), 0.05, 'yellow', 0.3)}

def colour_pixels(hsv, name):
    H, S, V = (hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int))
    if name == 'red':
        return ((H >= 170) | (H <= 6)) & (S > 60) & (V > 50)
    if name == 'blue':
        return (H >= 105) & (H <= 125) & (S > 60) & (V > 35)
    if name == 'orange':
        return (H >= 11) & (H <= 18) & (S > 170) & (V > 90)
    if name == 'yellow':
        return (H >= 19) & (H <= 29) & (S > 140) & (V > 80)
    return (S < 50) & (V > 110)

def detect_v2(det, rgb_full, key):
    prompts, min_conf, colour, min_frac = RULES[key]
    cm = colour_pixels(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2HSV), colour)
    ms, cs = det.masks(rgb_full, list(prompts), 640, True, min(min_conf, 0.05))
    return [m for m, c in zip(ms, cs) if c >= min_conf and m.sum() >= 40 and ((cm & m).sum() / m.sum() >= min_frac)]

class Episode:

    def __init__(self, policy):
        self.sess = policy.new_session()
        self.cfg = {}

    def start(self, header):
        self.sess.reset()
        self.cfg = header
        self.embed = torch.tensor(header['goal_embed'], dtype=torch.float32)
        torch.manual_seed(int(header.get('seed', 0)))

def make_mask(det, ep, rgb_full, gt224):
    mode = ep.cfg['mode']
    if mode == 'gt':
        return gt224.astype(np.float32)
    if mode == 'orig':
        return None
    out = np.zeros((224, 224), np.float32)
    if mode == 'prompt':
        img = cv2.resize(rgb_full, (224, 224), interpolation=cv2.INTER_LINEAR)
        ms, _ = det.masks(img, ep.cfg['phrase'], 224, False, 0.05)
        for m in ms:
            out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    if mode == 'tuned':
        for m in detect_v2(det, rgb_full, ep.cfg['key']):
            out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    if mode == 'cat':
        import house_perception as hp
        for m in hp.det_masks(det, rgb_full, hp.prompts_for(ep.cfg['key']), hp.DET_CONF):
            out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    if mode.startswith('cat2'):
        import house_perception as hp
        import siglip_verify as sv
        for m in sv.verified_masks(det, rgb_full, ep.cfg['key'], hp.prompts_for(ep.cfg['key']), hp.DET_CONF, mode.split(':', 1)[1] if ':' in mode else 'top2'):
            out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    if mode == 'open':
        import house_perception as hp
        for m in hp.det_masks(det, rgb_full, [w for w in str(ep.cfg['phrase']).split('|') if w], hp.DET_CONF):
            out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    if mode == 'tuned_v1':
        prompt, colour = TUNED[ep.cfg['key']]
        ms, _ = det.masks(rgb_full, prompt, 640, True, 0.05)
        for m in ms:
            if colour is None or colour_of(rgb_full, m) == colour:
                out = np.maximum(out, cv2.resize(m.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return out
    raise ValueError(mode)

@torch.inference_mode()
def policy_step(policy, ep, rgb224, mask224, force=None, commit=True):
    sess, dev = (ep.sess, policy.device)
    obs = {'rgb': torch.from_numpy(rgb224).unsqueeze(0), 'clip_objectgoal': ep.embed.unsqueeze(0), 'semantic': torch.zeros((1, 224, 224, 1), dtype=torch.int32), 'step_id': torch.tensor([[sess.step_count]], dtype=torch.int64)}
    if mask224 is not None:
        obs['segm_model'] = torch.from_numpy(mask224).reshape(1, 224, 224, 1)
    obs = {k: v.to(dev) for k, v in obs.items()}
    ac = policy.actor_critic
    features, hidden, aux = ac.net(obs, sess.rnn_hidden_states, sess.prev_actions, sess.prev_segm_masks, sess.not_done_masks)
    logits = ac.action_distribution(features).logits.clone()
    allowed = ep.cfg['allowed']
    for i in range(logits.shape[-1]):
        if i not in allowed:
            logits[:, i] = -1000000000.0
    probs = torch.softmax(logits, dim=-1)
    action = probs.argmax(-1) if ep.cfg['deterministic'] else torch.multinomial(probs, 1).squeeze(-1)
    tau = ep.cfg.get('stop_thresh')
    if tau is not None and 0 in allowed:
        rest = probs.clone()
        rest[:, 0] = 0.0
        action = torch.where(probs[:, 0] >= tau, torch.zeros_like(action), rest.argmax(-1) if ep.cfg['deterministic'] else torch.multinomial(rest, 1).squeeze(-1))
    min_px = ep.cfg.get('stop_min_px')
    if min_px and mask224 is not None and (0 in allowed) and (int((mask224 > 0.5).sum()) < min_px):
        rest = probs.clone()
        rest[:, 0] = 0.0
        action = torch.where(action == 0, rest.argmax(-1), action)
    if not commit:
        return (int(probs.argmax(-1).item()), probs[0].cpu().numpy().tolist())
    sess.rnn_hidden_states = hidden
    if force is not None:
        action = torch.tensor([int(force)], device=dev)
    sess.prev_actions = action.view(1, 1)
    sess.prev_segm_masks = aux['prev_segm_masks']
    sess.not_done_masks = torch.ones(1, 1, device=dev, dtype=torch.bool)
    sess.step_count += 1
    return (int(action.item()), probs[0].cpu().numpy().tolist())

def serve(args):
    os.environ.setdefault('OVSEGDT_TEXT_CACHE', args.text_cache)
    policy = OVSegDTPolicy(exp_config_path=args.exp_config, ckpt_path=args.ckpt, overrides=[f'habitat_baselines.eval_ckpt_path_dir={args.ckpt}', f'habitat_baselines.rl.policy.segmentation_source={args.vocab}', 'habitat.dataset.split=val_unseen', f'habitat.task.lab_sensors.clip_objectgoal_sensor.cache={args.text_cache}'])
    det = Detector()
    srv = socket.create_server(('127.0.0.1', args.port))
    print(f'READY port={args.port} ckpt={args.ckpt} vocab={args.vocab} actions={policy.action_names}', flush=True)
    ep = Episode(policy)
    while True:
        conn, _ = srv.accept()
        try:
            while True:
                head, payload = receive(conn)
                op = head['op']
                if op == 'hello':
                    send(conn, {'ok': True, 'actions': NAMES})
                elif op == 'reset':
                    ep.start(head)
                    send(conn, {'ok': True})
                elif op == 'step':
                    h, w = (head['h'], head['w'])
                    rgb = np.frombuffer(payload[:h * w * 3], np.uint8).reshape(h, w, 3)
                    gt = np.frombuffer(payload[h * w * 3:], np.uint8).reshape(224, 224).astype(bool) if head.get('has_gt') else None
                    rgb224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                    mask = make_mask(det, ep, rgb, gt)
                    idx, probs = policy_step(policy, ep, rgb224, mask, head.get('force'), not head.get('peek', False))
                    send(conn, {'idx': idx, 'name': NAMES[idx], 'probs': probs, 'mask_px': -1 if mask is None else int((mask > 0.5).sum())})
                elif op == 'override':
                    ep.sess.prev_actions = torch.tensor([[int(head['action'])]], device=policy.device, dtype=torch.long)
                    send(conn, {'ok': True})
                elif op == 'quit':
                    send(conn, {'ok': True})
                    return
        except ConnectionError:
            print('client disconnected', flush=True)
        finally:
            conn.close()
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8060)
    ap.add_argument('--exp-config', default='config/experiments/transformer_rl_segm_loss-validation.yaml')
    ap.add_argument('--ckpt', default='checkpoints/ckpt.5.pth')
    ap.add_argument('--text-cache', default='mujoco_g1_nav/eval/siglip_text_embeddings.pkl')
    ap.add_argument('--vocab', default='yolo_train', help="YOLOE vocabulary used by the repo's own pipeline (mode 'orig')")
    serve(ap.parse_args())
