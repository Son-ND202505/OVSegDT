# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
import numpy as np
from .config import PKG_ROOT, REPO_ROOT
OVSEGDT_PYTHON = Path(os.environ.get('OVSEGDT_PYTHON', Path.home() / 'miniforge3' / 'envs' / 'ovsegdt' / 'bin' / 'python'))

def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('policy server closed the connection')
        buf += chunk
    return buf

def _die_with_parent():
    import ctypes
    import signal
    ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)

class PolicyClient:

    def __init__(self, port=8060, ckpt='checkpoints/ckpt.5.pth', vocab='yolo_train', log_path=None, spawn=True, ready_timeout=240):
        self.proc = None
        if spawn:
            log = open(log_path or PKG_ROOT / 'outputs' / 'policy_server.log', 'w')
            self.proc = subprocess.Popen([str(OVSEGDT_PYTHON), str(PKG_ROOT / 'eval' / 'policy_server.py'), '--port', str(port), '--ckpt', ckpt, '--vocab', vocab], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
            t0 = time.time()
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'policy server died, see {log.name}')
                if 'READY' in Path(log.name).read_text():
                    break
                if time.time() - t0 > ready_timeout:
                    raise TimeoutError('policy server did not become ready')
                time.sleep(1.0)
        self.sock = socket.create_connection(('127.0.0.1', port), timeout=120)

    def _call(self, header, payload=b''):
        h = json.dumps(dict(header, payload_len=len(payload))).encode()
        self.sock.sendall(struct.pack('>I', len(h)) + h + payload)
        n = struct.unpack('>I', _recv_exact(self.sock, 4))[0]
        return json.loads(_recv_exact(self.sock, n).decode())

    def reset(self, embed, phrase, key, mode, allowed, deterministic=False, seed=0, stop_thresh=None, stop_min_px=None):
        return self._call(dict(op='reset', goal_embed=[float(x) for x in embed], phrase=phrase, key=key, mode=mode, allowed=list(allowed), deterministic=bool(deterministic), seed=int(seed), stop_thresh=stop_thresh, stop_min_px=stop_min_px))

    def step(self, rgb_full, gt224=None, force=None, peek=False):
        payload = np.ascontiguousarray(rgb_full, dtype=np.uint8).tobytes()
        if gt224 is not None:
            payload += np.ascontiguousarray(gt224, dtype=np.uint8).tobytes()
        r = self._call(dict(op='step', h=int(rgb_full.shape[0]), w=int(rgb_full.shape[1]), has_gt=gt224 is not None, force=force, peek=bool(peek)), payload)
        return (r['idx'], r)

    def override(self, action):
        return self._call(dict(op='override', action=int(action)))

    def close(self):
        try:
            self._call(dict(op='quit'))
        except Exception:
            pass
        self.sock.close()
        if self.proc is not None:
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
