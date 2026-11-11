# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
from dataclasses import dataclass
import numpy as np
from .locomotion import G1Locomotion

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

@dataclass
class ExecResult:
    action: int
    name: str
    forward_m: float
    lateral_m: float
    dyaw_deg: float
    sim_time_s: float
    fell: bool
    timed_out: bool

class ActionExecutor:

    def __init__(self, loco: G1Locomotion, cfg: dict, on_step=None):
        self.loco = loco
        self.nav = cfg['navigation']
        self.lc = cfg['locomotion']
        self.ids = self.nav['action_ids']
        self.names = {v: k for k, v in self.ids.items()}
        self.on_step = on_step

    def _forward_cmd(self, remaining_m: float) -> float:
        lc = self.lc
        gain = lc.get('forward_gain', 0.0)
        if gain <= 0:
            return lc['forward_cmd_mps']
        return float(np.clip(gain * remaining_m, lc.get('forward_min_cmd_mps', 0.15), lc['forward_cmd_mps']))

    def fallen(self) -> bool:
        return bool(self.loco.d.qpos[2] < self.lc['fall_height_m'])

    def _step(self, cmd):
        self.loco.step(cmd)
        if self.on_step is not None:
            self.on_step()

    def settle(self):
        calm = 0
        for _ in range(int(self.lc['settle_max_s'] / self.loco.step_dt)):
            self._step((0.0, 0.0, 0.0))
            v, w = self.loco.speed()
            calm = calm + 1 if v < 0.04 and w < 0.08 else 0
            if calm >= 8 or self.fallen():
                break

    def execute(self, action_id: int) -> ExecResult:
        name = self.names[action_id]
        loco, lc = (self.loco, self.lc)
        p0, y0, t0 = (loco.xy(), loco.yaw(), loco.d.time)
        heading = np.array([np.cos(y0), np.sin(y0)])
        timed_out = False
        if name == 'move_forward':
            target = self.nav['forward_step_m'] - lc['forward_lead_m']
            while (loco.xy() - p0) @ heading < target and (not self.fallen()):
                if loco.d.time - t0 > lc['action_timeout_s']:
                    timed_out = True
                    break
                wz = float(np.clip(-lc.get('heading_hold_gain', 0.0) * wrap(loco.yaw() - y0), -0.4, 0.4))
                self._step((self._forward_cmd(target - (loco.xy() - p0) @ heading), 0.0, wz))
            self.settle()
        elif name in ('turn_left', 'turn_right'):
            sign = 1.0 if name == 'turn_left' else -1.0
            target = np.radians(self.nav['turn_step_deg'] - lc['turn_lead_deg'])
            while sign * wrap(loco.yaw() - y0) < target and (not self.fallen()):
                if loco.d.time - t0 > lc['action_timeout_s']:
                    timed_out = True
                    break
                self._step((0.0, 0.0, sign * lc['turn_cmd_radps']))
            self.settle()
        elif name == 'stop':
            self.settle()
        else:
            raise ValueError(f"action '{name}' is not executable on G1 (no head tilt); mask it in the policy")
        d = loco.xy() - p0
        return ExecResult(action=action_id, name=name, forward_m=float(d @ heading), lateral_m=float(d @ np.array([-heading[1], heading[0]])), dyaw_deg=float(np.degrees(wrap(loco.yaw() - y0))), sim_time_s=loco.d.time - t0, fell=self.fallen(), timed_out=timed_out)
