# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
import multiprocessing as mp
import queue
import time
import numpy as np

def fit_canvas(frame: np.ndarray, fw: int, fh: int, fill: int=0) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    s = min(fw / w, fh / h)
    nw, nh = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
    out = np.full((fh, fw, 3), fill, np.uint8)
    y0, x0 = ((fh - nh) // 2, (fw - nw) // 2)
    out[y0:y0 + nh, x0:x0 + nw] = img
    return out

def _window_main(q, ready, title: str, aspect: float, rel_height: float, selftest: bool):
    import glfw
    import mujoco
    try:
        if not glfw.init():
            ready.put('glfw.init failed')
            return
        try:
            vm = glfw.get_video_mode(glfw.get_primary_monitor())
            screen_h = int(vm.size.height) if vm else 1080
        except Exception:
            screen_h = 1080
        h = max(240, int(rel_height * screen_h))
        glfw.window_hint(glfw.FOCUSED, glfw.FALSE)
        glfw.window_hint(glfw.FOCUS_ON_SHOW, glfw.FALSE)
        win = glfw.create_window(int(round(h * aspect)), h, title, None, None)
        if not win:
            ready.put('glfw.create_window failed')
            glfw.terminate()
            return
        glfw.make_context_current(win)
        glfw.swap_interval(0)
        con = mujoco.MjrContext(mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>'), mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, con)
        ready.put('ok')
    except Exception as e:
        ready.put(f'{type(e).__name__}: {e}')
        return
    frame, dirty, size, tested = (None, True, (0, 0), False)
    while not glfw.window_should_close(win):
        glfw.poll_events()
        try:
            while True:
                m = q.get_nowait()
                if m is None:
                    glfw.set_window_should_close(win, True)
                    break
                frame, dirty = (m, True)
        except queue.Empty:
            pass
        fw, fh = glfw.get_framebuffer_size(win)
        if (fw, fh) != size:
            size, dirty = ((fw, fh), True)
        if dirty and fw > 0 and (fh > 0):
            canvas = fit_canvas(frame, fw, fh) if frame is not None else np.zeros((fh, fw, 3), np.uint8)
            flat = np.ascontiguousarray(canvas[::-1]).reshape(-1)
            mujoco.mjr_drawPixels(flat, None, mujoco.MjrRect(0, 0, fw, fh), con)
            if selftest and frame is not None and (not tested):
                back = np.zeros((fh, fw, 3), np.uint8)
                mujoco.mjr_readPixels(back, None, mujoco.MjrRect(0, 0, fw, fh), con)
                ready.put('selftest ok' if int(np.abs(back[::-1].astype(int) - canvas.astype(int)).max()) <= 2 else 'selftest failed')
                tested = True
            glfw.swap_buffers(win)
            dirty = False
        time.sleep(0.01)
    glfw.destroy_window(win)
    glfw.terminate()

class FpvWindow:

    def __init__(self, aspect: float=360 / 640, rel_height: float=0.6, title: str='G1 first-person camera', selftest: bool=False, timeout: float=30.0):
        ctx = mp.get_context('spawn')
        self.q, self.ready = (ctx.Queue(maxsize=2), ctx.Queue())
        self.proc = ctx.Process(target=_window_main, args=(self.q, self.ready, title, aspect, rel_height, selftest), daemon=True)
        self.proc.start()
        try:
            self.status = self.ready.get(timeout=timeout)
        except queue.Empty:
            self.status = 'timeout while opening the window'
        if self.status != 'ok':
            self.close()

    @property
    def alive(self) -> bool:
        return self.status == 'ok' and self.proc.is_alive()

    def show(self, rgb: np.ndarray) -> bool:
        if not self.alive:
            return False
        try:
            self.q.put_nowait(np.ascontiguousarray(rgb, dtype=np.uint8))
            return True
        except queue.Full:
            return False

    def messages(self) -> list:
        out = []
        try:
            while True:
                out.append(self.ready.get_nowait())
        except queue.Empty:
            return out

    def close(self):
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.terminate()
