# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import sys
import cv2
import mujoco
import numpy as np
from .scene import CEILING_GROUP, Scene
GEOM_TYPE = int(mujoco.mjtObj.mjOBJ_GEOM)

class EgoCamera:

    def __init__(self, scene: Scene):
        c = scene.cfg['camera']
        self.scene, self.w, self.h = (scene, c['width'], c['height'])
        self.size = c['policy_input_size']
        self.renderer = mujoco.Renderer(scene.model, self.h, self.w)
        self.seg_renderer = mujoco.Renderer(scene.model, self.h, self.w)
        self.seg_renderer.enable_segmentation_rendering()
        self.depth_renderer = mujoco.Renderer(scene.model, self.h, self.w)
        self.depth_renderer.enable_depth_rendering()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[CEILING_GROUP] = 1
        self.overview = mujoco.Renderer(scene.model, 720, 720)
        self.third = mujoco.Renderer(scene.model, 480, 640)
        self.third_cam = mujoco.MjvCamera()
        self.third_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.third_cam.trackbodyid = scene.pelvis_id
        self.third_cam.distance, self.third_cam.elevation = (3.2, -28.0)
        self.display = mujoco.Renderer(scene.model, 480, 640)
        self.display_cam = mujoco.MjvCamera()
        self.display_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.display_cam.trackbodyid = scene.pelvis_id
        self.display_cam.distance, self.display_cam.elevation, self.display_cam.azimuth = (3.2, -28.0, 0.0)
        self.overview_opt = mujoco.MjvOption()

    def close(self):
        for r in (self.renderer, self.seg_renderer, self.depth_renderer, self.overview, self.third, self.display):
            r.close()

    def rgb(self, data) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.scene.ego_cam_id, scene_option=self.opt)
        return self.renderer.render().copy()

    def depth(self, data) -> np.ndarray:
        self.depth_renderer.update_scene(data, camera=self.scene.ego_cam_id, scene_option=self.opt)
        return self.depth_renderer.render().copy()

    def segmentation(self, data):
        self.seg_renderer.update_scene(data, camera=self.scene.ego_cam_id, scene_option=self.opt)
        try:
            seg = self.seg_renderer.render().copy()
        except IndexError:
            self.seg_errors = getattr(self, 'seg_errors', 0) + 1
            print(f'[segmentation] unreadable id colours at robot ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}) yaw {2 * np.arctan2(data.qpos[6], data.qpos[3]):.2f} rad: retrying without reflections', file=sys.stderr, flush=True)
            flags = self.seg_renderer.scene.flags
            saved = flags.copy()
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            try:
                seg = self.seg_renderer.render().copy()
            except IndexError:
                seg = np.full((self.h, self.w, 2), -1, dtype=np.int32)
            finally:
                flags[:] = saved
        ids = np.where(seg[..., 1] == GEOM_TYPE, seg[..., 0], -1)
        return ids

    def object_mask(self, geom_ids_map: np.ndarray, key: str) -> np.ndarray:
        return np.isin(geom_ids_map, self.scene.obj_geoms[key])

    def policy_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

    def overview_rgb(self, data, zoom: bool=False) -> np.ndarray:
        cam_id = self.scene.zoom_cam_id if zoom else self.scene.overview_cam_id
        self.overview.update_scene(data, camera=cam_id, scene_option=self.overview_opt)
        return self.overview.render().copy()

    def third_rgb(self, data, yaw_rad: float) -> np.ndarray:
        self.third_cam.azimuth = float(np.degrees(yaw_rad))
        self.third.update_scene(data, camera=self.third_cam, scene_option=self.overview_opt)
        return self.third.render().copy()

    def display_rgb(self, data) -> np.ndarray:
        self.display.update_scene(data, camera=self.display_cam, scene_option=self.overview_opt)
        return self.display.render().copy()
