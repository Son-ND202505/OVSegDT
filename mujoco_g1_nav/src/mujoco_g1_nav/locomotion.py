# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import numpy as np
import os
import onnxruntime as ort
import mujoco
import yaml
from .config import resolve
from .scene import yaw_quat
GAIT_PERIOD_S = 0.6

class G1Locomotion:

    def __init__(self, model, data, cfg):
        loco_dir = resolve(cfg['paths']['locomotion_dir'])
        with open(loco_dir / 'params' / 'deploy.yaml') as f:
            p = yaml.safe_load(f)
        options = ort.SessionOptions()
        threads = int(os.environ.get('G1_ORT_THREADS', '0'))
        if threads > 0:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(loco_dir / 'exported' / 'policy.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.in_name = self.sess.get_inputs()[0].name
        self.m, self.d = (model, data)
        self.kp = np.array(p['stiffness'])
        self.kd = np.array(p['damping'])
        self.q0 = np.array(p['default_joint_pos'])
        self.action_scale = np.array(p['actions']['JointPositionAction']['scale'])
        self.step_dt = p['step_dt']
        self.sub = cfg['locomotion']['decimation']
        assert abs(model.opt.timestep * self.sub - self.step_dt) < 1e-09, 'sim_dt x decimation must equal the policy period'
        assert model.nq == 7 + len(self.q0) and model.nu == len(self.q0), 'unexpected robot layout'
        self.reset()

    def reset(self, xy=(0.0, 0.0), yaw=0.0, z=0.793):
        m, d = (self.m, self.d)
        mujoco.mj_resetData(m, d)
        d.qpos[0:2] = xy
        d.qpos[2] = z
        d.qpos[3:7] = yaw_quat(yaw)
        d.qpos[7:] = self.q0
        mujoco.mj_forward(m, d)
        self.last_action = np.zeros_like(self.q0, dtype=np.float32)
        self.phase = 0.0

    def _obs(self, cmd):
        d = self.d
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        gravity = R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        self.phase = (self.phase + self.step_dt / GAIT_PERIOD_S) % 1.0
        gait = np.array([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)]) if np.linalg.norm(cmd) >= 0.1 else np.zeros(2)
        obs = np.concatenate([d.qvel[3:6], gravity, cmd, gait, d.qpos[7:] - self.q0, d.qvel[6:], self.last_action])
        return obs.astype(np.float32)[None]

    def step(self, cmd):
        cmd = np.asarray(cmd, dtype=float)
        action = self.sess.run(None, {self.in_name: self._obs(cmd)})[0][0]
        self.last_action = action.astype(np.float32)
        q_des = self.q0 + self.action_scale * action
        d = self.d
        for _ in range(self.sub):
            d.ctrl[:] = self.kp * (q_des - d.qpos[7:]) - self.kd * d.qvel[6:]
            mujoco.mj_step(self.m, d)

    def xy(self):
        return self.d.qpos[:2].copy()

    def yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def speed(self):
        return (float(np.linalg.norm(self.d.qvel[:2])), float(abs(self.d.qvel[5])))
