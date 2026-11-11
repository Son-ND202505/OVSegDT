# Part of the MuJoCo G1 object-goal navigation release (Apache-2.0, see LICENSE in this folder).
"""MuJoCo G1 object-goal navigation demo with the OVSegDT policy (the network alone drives).   (MuJoCo env, run from the repo root)

  python mujoco_g1_nav/scripts/play_house.py                                  # house home25 (25 x 25 m), MuJoCo viewer + first-person window
  python mujoco_g1_nav/scripts/play_house.py --house l28                      # the L-shaped house
  python mujoco_g1_nav/scripts/play_house.py --no-viewer --cmd "bed" "sofa"   # headless, scripted goals (videos in mujoco_g1_nav/outputs/play_house/)

Type a goal ("bed", "fridge", "something to sit on", "coke" ...). The text is mapped to one of the goals of the house (keyword -> typo repair -> SigLIP meaning -> it asks back),
then the OVSegDT policy drives the robot from its own camera image. The target mask given to the policy is the simulator's ground-truth mask of the goal category.

Commands:  <goal text> | new | again | objects | where <goal> | text canonical|raw | colours on|off | speed X | video on|off | fpv on|off|mode window|overlay|mask on|off|size X|corner tl|tr|bl|br | help | quit"""
import argparse
import os
import sys
import time
from pathlib import Path
SKIN_WORDS = {'apple': ['apple', 'an apple', 'the apple', 'red apple'], 'banana': ['banana', 'a banana', 'the banana', 'yellow fruit'], 'bottle': ['bottle', 'a bottle', 'the bottle', 'green bottle', 'glass bottle'], 'book': ['book', 'a book', 'the book', 'notebook']}
FPV_HZ = 15
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MUJOCO_GL', 'egl')
HELP = __doc__.split('Commands:')[1].strip()

def _text_embed(text):
    from mujoco_g1_nav.goal_text import encode_goals
    return encode_goals([text])[0]

def _text_embed_batch(texts):
    from mujoco_g1_nav.goal_text import encode_goals
    return encode_goals(list(texts))

def _text_warm():
    _text_embed('warm up')
    return True

def keyword_category(text: str):
    import re
    from mujoco_g1_nav.house_phrases import all_phrases
    t = ' ' + re.sub('[^a-z0-9 ]+', ' ', text.lower()) + ' '
    t = re.sub('\\s+', ' ', t)
    best = None
    for p, c in all_phrases().items():
        if f' {p} ' in t and (best is None or len(p) > len(best[1])):
            best = (c, p)
    return best

class PlayHouse:

    def __init__(self, a):
        import multiprocessing as mp
        import mujoco
        import numpy as np
        from mujoco_g1_nav.envs import load_house_env
        from mujoco_g1_nav.house_variants import HouseSession, base_variant, load_pool
        from mujoco_g1_nav.policy_client import PolicyClient
        self.np, self.mujoco, self.a = (np, mujoco, a)
        self.env = load_house_env(getattr(a, 'house', 'home25'))
        pool = load_pool(a.pool, self.env.building) if a.pool else [base_variant(self.env)]
        self.text_pool = mp.get_context('spawn').Pool(1)
        self.text_pool.apply_async(_text_warm)
        self.sess = HouseSession(self.env, pool)
        self.ctx = self.sess.ctx
        self.mode, self.perception, self.colours, self.variant = ('hybrid' if not a.e2e else 'e2e', a.perception, a.colours, a.variant)
        self.speed, self.video, self.viewer = (a.speed, not a.no_video, None)
        self.fpv, self.fpv_corner, self.fpv_mask, self.fpv_mode, self.fpv_win = (not a.no_fpv, a.fpv_corner, a.fpv_mask, a.fpv_mode, None)
        self.fpv_size = a.fpv_size if a.fpv_size is not None else 0.6 if a.fpv_mode == 'window' else 0.45
        self.hud, self._fpv_t = ({}, 0.0)
        self.max_steps = a.max_steps or 2 * self.env.step_budget
        from mujoco_g1_nav.config import PKG_ROOT
        z = np.load(PKG_ROOT / 'datasets' / 'cache' / 'goal_embeds_house.npz')
        self.known = {str(p): e for p, e in zip(z['phrases'], z['embeds'])}
        self.cat_of = {str(p): str(k) for p, k in zip(z['phrases'], z['keys'])}
        zo = np.load(PKG_ROOT / 'datasets' / 'cache' / 'goal_embeds.npz')
        self.known.update({str(p): e for p, e in zip(zo['phrases'], zo['embeds'])})
        self.PKG_ROOT = PKG_ROOT
        from mujoco_g1_nav import goal_vocab as gv
        self.gv = gv
        goals = gv.load_goals()
        self.skin = None
        if getattr(a, 'skin', None):
            import mujoco_g1_nav.object_skins as sk
            self.skin = sk.SKINS[a.skin]
            sk.apply_skin(self.ctx.scene, a.skin)
            mujoco.mj_forward(self.ctx.scene.model, self.ctx.sim)
            old = goals[self.skin.host]
            goals[self.skin.host] = gv.Goal(self.skin.host, 'object', old.canonical if a.skin_text == 'proxy' else a.skin, list(SKIN_WORDS[a.skin]), f'{a.skin} (new object, the {self.skin.host} slot)')
            print(f"NEW OBJECT on the table: a {a.skin} instead of the {self.skin.host}; the policy is told '{goals[self.skin.host].canonical}'; type '{a.skin}'", flush=True)
        self.resolver = gv.GoalResolver(goals, embed_fn=self._meaning_vector)
        self.text_mode, self.drawn = (a.text_mode, {})
        if not a.no_viewer:
            self._open_viewer()
            if self.viewer is not None and self.fpv and (self.fpv_mode == 'window'):
                self._fpv_open()
        base = self.ctx.executor.on_step
        self._t_last = time.time()

        def on_step():
            if base is not None:
                base()
            if self.viewer is not None:
                self.ctx.scene.update_lighting(self.ctx.loco.xy())
                self._sync()
                self._fpv_draw()
                if self.speed > 0:
                    wait = self.ctx.loco.step_dt / self.speed - (time.time() - self._t_last)
                    if wait > 0:
                        time.sleep(wait)
                self._t_last = time.time()
        self.ctx.executor.on_step = on_step
        print('starting the policy server (~30 s)...', flush=True)
        self.client = PolicyClient(port=a.port, ckpt=a.ckpt, log_path=PKG_ROOT / 'outputs' / 'play_house_policy_server.log')
        n_new = self.resolver.ensure_phrase_embeds(PKG_ROOT / 'datasets' / 'cache' / 'goal_vocab_siglip.npz', lambda t: self.text_pool.apply(_text_embed_batch, (t,)))
        print(f'command interpreter: {len(self.resolver.goals)} goals, {len(self.resolver.lex.phrase_key)} phrases' + (f' ({n_new} new phrases embedded)' if n_new else ''), flush=True)
        self.seed = a.seed if a.seed is not None else int(np.random.default_rng().integers(1, 10 ** 6))
        self.start, self.cur, self.last = (None, None, None)
        self.new_scene(keep_seed=True)

    def _open_viewer(self):
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.ctx.scene.model, self.ctx.sim, show_left_ui=False, show_right_ui=False)
            with self.viewer.lock():
                cam = self.viewer.cam
                cam.type, cam.trackbodyid = (self.mujoco.mjtCamera.mjCAMERA_TRACKING, self.ctx.scene.pelvis_id)
                cam.distance, cam.elevation, cam.azimuth = (3.4, -28.0, 0.0)
        except Exception as e:
            self.viewer = None
            print(f'(cannot open the MuJoCo window: {type(e).__name__}: {e}) - running without it, videos are still saved', flush=True)

    def _sync(self):
        try:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.viewer = None
                self._fpv_shut()
                print('(the MuJoCo window was closed - continuing without it)', flush=True)
        except Exception:
            self.viewer = None
            self._fpv_shut()

    def _fpv_open(self):
        from mujoco_g1_nav.fpv_window import FpvWindow
        self.fpv_win = FpvWindow(rel_height=self.fpv_size if self.fpv_size else 0.6)
        if not self.fpv_win.alive:
            print(f'(cannot open the first-person window: {self.fpv_win.status}; falling back to the overlay inside the MuJoCo window)', flush=True)
            self.fpv_win, self.fpv_mode = (None, 'overlay')
            return False
        return True

    def _fpv_shut(self):
        if getattr(self, 'fpv_win', None) is not None:
            self.fpv_win.close()
        self.fpv_win = None

    def _fpv_draw(self, force=False):
        if self.viewer is None or not self.fpv:
            return
        now = time.time()
        if not force and now - self._fpv_t < 1.0 / FPV_HZ:
            return
        self._fpv_t = now
        try:
            from mujoco_g1_nav.fpv_overlay import ACCENT, hud_lines, panel_image, panel_rect
            window = self.fpv_mode == 'window' and self.fpv_win is not None
            if window and (not self.fpv_win.alive):
                self._fpv_shut()
                self.fpv = False
                print('\n(the first-person window was closed - type `fpv on` to open it again)', flush=True)
                return
            vp = None
            if not window:
                vp = self.viewer.viewport
                if vp is None or vp.width < 200 or vp.height < 200:
                    return
            cam = self.ctx.cam
            rgb = cam.rgb(self.ctx.sim)
            mask, cat = (None, self.hud.get('cat'))
            if self.fpv_mask and cat:
                from mujoco_g1_nav.house_eval import category_mask
                mask = category_mask(self.sess.world, cam.segmentation(self.ctx.sim), cat)
            title, lines, accent = ('ROBOT CAMERA' + (' + mask' if mask is not None else ''), hud_lines(self.hud), ACCENT.get(self.hud.get('verdict_kind'), ACCENT[None]))
            if window:
                self.fpv_win.show(panel_image(rgb, 450, 800, mask, title, lines, accent))
                return
            left, bottom, w, h = panel_rect(vp.width, vp.height, rgb.shape[1] / rgb.shape[0], self.fpv_size, self.fpv_corner, left0=vp.left, bottom0=vp.bottom)
            self.viewer.set_images((self.mujoco.MjrRect(left=left, bottom=bottom, width=w, height=h), panel_image(rgb, w, h, mask, title, lines, accent)))
        except Exception as e:
            self.fpv = False
            print(f'\n(first-person panel switched off: {type(e).__name__}: {e})', flush=True)

    def fpv_command(self, args):
        from mujoco_g1_nav.fpv_overlay import CORNERS
        usage = 'usage: fpv on|off | fpv mode window|overlay | fpv mask on|off | fpv size 0.1-0.95 | fpv corner tl|tr|bl|br (overlay)'
        try:
            if args[0] == 'on':
                self.fpv = True
                if self.viewer is not None and self.fpv_mode == 'window' and (self.fpv_win is None or not self.fpv_win.alive):
                    self._fpv_shut()
                    self._fpv_open()
            elif args[0] == 'off':
                self.fpv = False
                self._fpv_shut()
                if self.viewer is not None:
                    self.viewer.clear_images()
            elif args[0] == 'mode' and len(args) > 1 and (args[1] in ('window', 'overlay')):
                self._fpv_shut()
                if self.viewer is not None:
                    self.viewer.clear_images()
                self.fpv_mode = args[1]
                if self.viewer is not None and self.fpv and (self.fpv_mode == 'window'):
                    self._fpv_open()
            elif args[0] == 'mask' and len(args) > 1 and (args[1] in ('on', 'off')):
                self.fpv_mask = args[1] == 'on'
            elif args[0] == 'size' and len(args) > 1:
                self.fpv_size = float(args[1])
                if self.viewer is not None and self.fpv and (self.fpv_mode == 'window'):
                    self._fpv_shut()
                    self._fpv_open()
            elif args[0] == 'corner' and len(args) > 1 and (args[1] in CORNERS):
                self.fpv_corner = args[1]
            else:
                print(usage)
                return
        except ValueError:
            print(usage)
            return
        print(f"first-person view: {('on' if self.fpv else 'off')}, mode {self.fpv_mode}, mask tint {('on' if self.fpv_mask else 'off')}, size {self.fpv_size:g}" + (f', corner {self.fpv_corner}' if self.fpv_mode == 'overlay' else '') + ('' if self.viewer is not None else '   (no MuJoCo window open)'))
        self._fpv_draw(force=True)

    def new_scene(self, keep_seed=False):
        from mujoco_g1_nav.house_collect import prepare_episode, start_cells
        from mujoco_g1_nav.scene import set_ego_pitch
        np = self.np
        if not keep_seed:
            self.seed = int(np.random.default_rng().integers(1, 10 ** 6))
        self.table_layout, drawn = prepare_episode(self.sess, self.seed, variant=None if self.variant == 'random' else int(self.variant), colours=self.colours, p_base=0.0 if self.a.pool else 1.0, p_colour=1.0)
        self.drawn = drawn
        rng = np.random.default_rng(self.seed + 99)
        cells = start_cells(self.ctx.navmap)
        x, y = (float(v) for v in cells[int(rng.integers(len(cells)))])
        self.start = (x, y, float(rng.uniform(-np.pi, np.pi)))
        self.cur = self.start
        set_ego_pitch(self.ctx.scene, self.ctx.cfg['camera']['pitch_deg'])
        self.ctx.loco.reset((x, y), self.start[2])
        self.ctx.executor.settle()
        self.ctx.scene.update_lighting((x, y), force=True)
        self.hud = {}
        if self.viewer is not None:
            self._sync()
            self._fpv_draw(force=True)
        room = self.ctx.building.room_at(x, y) or self.ctx.building.region_at(x, y)
        print(f"\n=== new scene (seed {self.seed}) ===  layout variant {drawn['variant']}, {drawn['recoloured']} pieces recoloured; the robot starts in {room}: ({x:+.1f}, {y:+.1f}) m, heading {np.degrees(self.start[2]):+.0f} deg   [mode {self.mode}, perception {self.perception}]")

    def objects(self):
        lay = self.ctx.house.layout
        print('furniture categories in this house (instances): ' + ', '.join((f'{c} ({len(lay.of_cat(c))})' for c in lay.categories())))
        objs = [g for g in self.resolver.goals.values() if g.kind == 'object']
        print('objects on the kitchen table: ' + ', '.join((g.label for g in objs)))
        print(f'{len(self.resolver.goals)} goals in all; type any of their names or synonyms (configs/goal_synonyms.json lists the extra words)')

    def where(self, name):
        goals = self.resolver.goals
        r = self.resolver.resolve(name)
        g = r.goal if r.ok else goals.get(name)
        if g is None:
            print(f"no goal '{name}'")
            return
        if g.kind == 'object':
            pos = self.table_layout.objects.get(g.key)
            print(f'   {g.label}: on the kitchen table at ({pos[0]:+.2f}, {pos[1]:+.2f}) m' if pos else f'   {g.label}: not on the table in this scene')
            return
        its = self.ctx.house.layout.of_cat(g.key)
        if not its:
            print(f"no category '{name}'")
            return
        for it in its:
            print(f'   {it.id:16s} room {it.region}  at ({it.x:+.1f}, {it.y:+.1f})')

    def _meaning_vector(self, text):
        gv = self.gv
        v = self.text_pool.apply(_text_embed_batch, (gv.recipe_inputs(text),))
        return gv.recipe_vector(v)

    def interpret(self, text, depth=0):
        r = self.resolver.resolve(text)
        if r.ok:
            how = {'keyword': f"keyword '{r.matched}'", 'typo': r.note, 'meaning': f'meaning, {r.note}'}.get(r.method, r.method)
            print(f'understood: {r.goal.label}  [{how}]' + (f"  -> the policy is told '{r.goal.canonical}'" if self.text_mode == 'canonical' else ''))
            return r.goal
        opts = [g for g, _ in r.candidates[:3]]
        if r.method == 'unknown':
            print(f"'{text}' is not one of the {len(self.resolver.goals)} things I can walk to ({r.note}).")
            if opts:
                print('   nearest: ' + '  '.join((f'{i + 1}) {g.label}' for i, g in enumerate(opts))))
            print('   type the number of what you meant, 0 (or Enter) for none, or say it differently:')
        else:
            print(f'not sure what you mean ({r.note}).  ' + '  '.join((f'{i + 1}) {g.label}' for i, g in enumerate(opts))))
            print('   type the number of what you meant (Enter = 1), 0 for none, or say it differently:')
        try:
            ans = input('  > ').strip()
        except EOFError:
            return None
        if r.method == 'ambiguous' and opts and (ans == ''):
            ans = '1'
        if ans.lower() in ('', '0', 'none', 'no', 'n', 'cancel'):
            print('ok, nothing to do')
            return None
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            print(f'understood: {opts[int(ans) - 1].label}  [your choice]')
            return opts[int(ans) - 1]
        return self.interpret(ans, depth + 1) if depth < 3 else None

    def embed(self, text):
        return self.known[text] if text in self.known else self.text_pool.apply(_text_embed, (text,))

    def go(self, text):
        goal = self.interpret(text)
        if goal is None:
            return
        phrase = goal.canonical if self.text_mode == 'canonical' else text
        if goal.kind == 'category':
            self._go_category(goal.key, phrase)
        else:
            self._go_object(goal, phrase)

    def _go_category(self, cat, phrase):
        from mujoco_g1_nav.house_eval import run_category_episode
        from mujoco_g1_nav.house_video import PolicyVideo
        world = self.sess.world
        shortest = world.path_length(cat, self.cur[0], self.cur[1])
        vid = None
        if self.video:
            d = self.PKG_ROOT / 'outputs' / 'play_house'
            d.mkdir(parents=True, exist_ok=True)
            vid = PolicyVideo(d / f"{time.strftime('%H%M%S')}_seed{self.seed}_{cat}.mp4", self.ctx, cat, world)
        self.hud = dict(cat=cat, t=0, driver='explorer' if self.mode == 'hybrid' else 'policy', action='-', p_stop=0.0, d=shortest, px=0)

        def hook(ctx, t, idx, info, m):
            self.hud.update(t=t + 1, driver=getattr(hook, 'driver', 'policy'), action=idx, p_stop=float(info['probs'][0]), d=float(m['d']), px=int(info['mask_px']))
            if vid is not None:
                vid.driver = getattr(hook, 'driver', 'policy')
                vid(ctx, t, idx, info, m)
            ctl = holder_ctl.get('c')
            if ctl is not None and hasattr(ctl, 'note'):
                ctl.note(info)
            print(f"\r  step {t + 1:3d}: {['STOP', 'F', 'L', 'R', 'u', 'd'][idx]:4s} p(stop)={info['probs'][0]:.2f}  mask={info['mask_px']:6d}px  d={m['d']:.2f} m", end='', flush=True)
        hook.driver = 'policy'
        holder_ctl = {}
        if getattr(self.a, 'stop_gate_px', 0) > 0 and (not getattr(self.client, '_gated', False)):
            _reset, px_gate = (self.client.reset, int(self.a.stop_gate_px))
            self.client.reset = lambda *a, **k: _reset(*a, **{**k, 'stop_min_px': k.get('stop_min_px') or px_gate})
            self.client._gated = True
        controller = None
        mode = {'gt': 'gt', 'cat': 'cat', 'cat2': 'cat2:top2'}[self.perception]
        res = run_category_episode(world, self.client, self.embed(phrase), phrase, cat, (self.cur[0], self.cur[1], self.cur[2], shortest), mode, self.max_steps, True, self.seed, on_step=hook, controller=controller, escape=not getattr(self.a, 'no_escape', False))
        x, y = (float(v) for v in self.ctx.loco.xy())
        self.cur = (x, y, self.ctx.loco.yaw())
        if vid is not None:
            vid.close()
        print()
        self.last = res
        f = res['final']
        verdict = 'CORRECT - stopped in front of it' if res['success_front'] else 'NEAR - at the right place, not in front / a bit off' if res['success'] else 'NOT THERE' if not res['called_stop'] or not res['success_loc'] else 'stopped near it but outside the success region'
        print(f"result: {verdict}  | called stop: {('yes' if res['called_stop'] else 'no (out of steps)')} | {res['steps']} steps | path {res['path_m']:.1f} m (shortest {res['shortest_m']:.1f}) | contacts: {res['collision_steps']} steps" + (f" | hand-over at step {res['handover_step']}" if res.get('handover_step') is not None else '') + (f' | stall rescues: {len(controller.rescues)} (steps {controller.rescues[:6]}, {controller.rescue_total_steps} steps driven by the explorer)' if holder_ctl else ''))
        print(f"   nearest {f['item']}: {f['obj_dist']:.2f} m away, heading error {f['yaw_err_deg']:.0f} deg, {f['px']} px in the image, in its front sector: {('yes' if f['front'] else 'no')}")
        short = 'CORRECT: stopped in front' if res['success_front'] else 'NEAR: not in front / a bit off' if res['success'] else 'NOT THERE' if not res['called_stop'] or not res['success_loc'] else 'STOPPED OUTSIDE THE ZONE'
        self.hud.update(verdict=short, verdict_kind='correct' if res['success_front'] else 'near' if res['success'] else 'fail', final_d=f['obj_dist'])
        self._fpv_draw(force=True)

    def _go_object(self, goal, phrase):
        import math
        from mujoco_g1_nav.front import frontality
        from mujoco_g1_nav.goal import object_xy
        from mujoco_g1_nav.policy_eval import VideoRecorder, run_policy_episode
        from mujoco_g1_nav.scenario import Scenario, building_episode_setup
        ctx, key = (self.ctx, goal.key)
        if key not in ctx.scene.obj_body:
            print(f"'{goal.label}' is registered as a goal but this scene has no such object on the table")
            return
        mode, det_phrase = ('gt' if self.perception == 'gt' else 'tuned', phrase)
        x0, y0, yaw0 = self.cur
        scn = Scenario(building=self.env.building.name, seed=int(self.seed), scenario='S1a', table_room=self.env.building.origin, table_xy=tuple((float(v) for v in self.env.cfg['table']['center_xy'])), perm=tuple(self.drawn['perm']), mirror=tuple(self.drawn['mirror']), target=key, start_xy=(float(x0), float(y0)), start_yaw=float(yaw0), start_kind='play')
        holder, rec = ({}, None)
        if self.video:
            from mujoco_g1_nav.building_viz import TopDownRenderer
            d = self.PKG_ROOT / 'outputs' / 'play_house'
            d.mkdir(parents=True, exist_ok=True)
            sx, sy = self.env.cfg['table']['size_xy']
            rec = VideoRecorder(d / f"{time.strftime('%H%M%S')}_seed{self.seed}_{key}.mp4", phrase, topdown=TopDownRenderer(self.env.building, layout=self.env.layout), table=(*scn.table_xy, sx, sy), start=scn.start_xy)
        self.hud = dict(cat=goal.label, t=0, driver='explorer' if self.mode == 'hybrid' else 'policy', action='-', p_stop=0.0, d=math.nan, px=0)

        def hook(c, o, t, idx, info):
            ox, oy = object_xy(c.last_setup['layout'], key)
            ctrl = holder.get('ctrl')
            driver = 'explorer' if ctrl is not None and ctrl.handover_step is None else 'policy'
            self.hud.update(t=t + 1, driver=driver, action=idx, p_stop=float(info['probs'][0]), d=math.hypot(ox - o.x, oy - o.y), px=int(info['mask_px']))
            if rec is not None:
                rec(c, o, t, idx, info)
            print(f"\r  step {t + 1:3d}: {['STOP', 'F', 'L', 'R', 'u', 'd'][idx]:4s} p(stop)={info['probs'][0]:.2f}  mask={info['mask_px']:6d}px  d={self.hud['d']:.2f} m", end='', flush=True)
        try:
            r = run_policy_episode(ctx, self.client, {key: self.embed(phrase)}, self.seed, 'random', key, 'g1', mode, self.max_steps, True, phrase=det_phrase, on_step=hook, setup=lambda: building_episode_setup(ctx, scn), controller_factory=None)
        finally:
            if rec is not None:
                rec.close()
            self.sess.set_variant(self.sess.variant.index)
        print()
        if not r.get('valid'):
            print('no valid approach pose for this goal in this scene (type `new` for another scene)')
            return
        ls = ctx.last_setup
        x, y = (float(v) for v in ctx.loco.xy())
        fr = frontality(ctx.cfg, object_xy(ls['layout'], key), (x, y))
        r['success_front'] = bool(r['success_close'] and fr['front'])
        self.cur = (x, y, ctx.loco.yaw())
        self.last = r
        f = r['final']
        verdict = 'CORRECT - stopped in front of it' if r['success_front'] else 'NEAR - at the table, close but not in front / a bit off' if r['success'] else 'NOT THERE' if not r['called_stop'] or not r['success_loc'] else 'stopped near it but outside the success region'
        print(f"result: {verdict}  | called stop: {('yes' if r['called_stop'] else 'no (out of steps)')} | {r['steps']} steps | path {r['path_m']:.1f} m (shortest {r['shortest_m']:.1f}) | contacts: {r['collision_steps']} steps" + (f" | hand-over at step {r['handover_step']}" if r.get('handover_step') is not None else ''))
        print(f"   {goal.label}: {f['obj_dist']:.2f} m away, {f['clearance']:.2f} m from the table edge (success needs 0.28-0.65), heading error {f['yaw_err_deg']:.0f} deg, {f['px']} px in the image, in front of it: {('yes' if fr['front'] else 'no')}")
        short = 'CORRECT: stopped in front' if r['success_front'] else 'NEAR: not in front / a bit off' if r['success'] else 'NOT THERE' if not r['called_stop'] or not r['success_loc'] else 'STOPPED OUTSIDE THE ZONE'
        self.hud.update(verdict=short, verdict_kind='correct' if r['success_front'] else 'near' if r['success'] else 'fail', final_d=f['obj_dist'])
        self._fpv_draw(force=True)

    def close(self):
        self._fpv_shut()
        try:
            self.client.close()
        finally:
            self.ctx.cam.close()
            self.text_pool.terminate()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--house', default='home25', choices=['home25', 'l28'], help='home25: the 25 x 25 m house (default); l28: the L-shaped house')
    ap.add_argument('--ckpt', default='mujoco_g1_nav/checkpoints/RECOMMENDED_house.pth', help='policy checkpoint (mujoco_g1_nav/scripts/download_weights.sh)')
    ap.add_argument('--port', type=int, default=8076, help='port of the policy server this script starts')
    ap.add_argument('--stop-gate-px', type=int, default=0, help='no stop while the 224 x 224 target mask has fewer than N pixels (0 = off)')
    ap.add_argument('--no-escape', action='store_true', help='switch off the escape layer (it turns away when the robot pushes against furniture or a wall)')
    ap.add_argument('--colours', action='store_true', help='recolour the furniture')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--speed', type=float, default=2.0, help='viewer playback speed (1 = real time, 0 = as fast as possible)')
    ap.add_argument('--max-steps', type=int, default=None, help='step budget of an episode (default 900)')
    ap.add_argument('--no-viewer', action='store_true')
    ap.add_argument('--no-fpv', action='store_true', help='no first-person view')
    ap.add_argument('--fpv-mode', default='window', choices=['window', 'overlay'], help='window (default): the first-person view in its own window; overlay: drawn inside the MuJoCo viewer')
    ap.add_argument('--fpv-size', type=float, default=None, help='window: height as a fraction of the screen (default 0.6); overlay: of the MuJoCo window (default 0.45)')
    ap.add_argument('--fpv-corner', default='tr', choices=['tl', 'tr', 'bl', 'br'], help='overlay mode: corner of the panel')
    ap.add_argument('--fpv-mask', action='store_true', help='tint the target mask green in the first-person panel')
    ap.add_argument('--text-mode', default='canonical', choices=['canonical', 'raw'], help="what the policy is told: the goal's canonical phrase (default) or your own words")
    ap.add_argument('--no-video', action='store_true')
    ap.add_argument('--cmd', nargs='*', default=None, help='headless scripted goals (then exit)')
    ap.set_defaults(perception='gt', e2e=True, pool=None, variant='random', skin=None, skin_text='new', gate_m=0.0, peek_done=None, stall_rescue=False)
    a = ap.parse_args()
    if a.pool is None and a.variant == 'random':
        a.variant = 0
    p = PlayHouse(a)
    try:
        if a.cmd:
            for c in a.cmd:
                print(f'\n> {c}')
                if c.lower() == 'new':
                    p.new_scene()
                elif c.lower() == 'again':
                    p.cur = p.start
                    p.ctx.loco.reset((p.start[0], p.start[1]), p.start[2])
                    p.ctx.executor.settle()
                else:
                    p.go(c)
            return
        print('type a piece of furniture (see `help`)')
        while True:
            try:
                line = input('\n> ').strip()
            except EOFError:
                break
            if not line:
                continue
            w = line.split()
            cmd = w[0].lower()
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                print(HELP)
            elif cmd == 'new':
                p.new_scene()
            elif cmd == 'again':
                p.cur = p.start
                p.ctx.loco.reset((p.start[0], p.start[1]), p.start[2])
                p.ctx.executor.settle()
                p.hud = {}
                p._fpv_draw(force=True)
                print('back to the start pose (same scene); type a goal')
            elif cmd == 'objects':
                p.objects()
            elif cmd == 'where' and len(w) > 1:
                p.where(' '.join(w[1:]))
            elif cmd == 'colours' and len(w) > 1:
                p.colours = w[1] == 'on'
                p.new_scene(keep_seed=True)
            elif cmd == 'speed' and len(w) > 1:
                p.speed = float(w[1])
            elif cmd == 'video' and len(w) > 1:
                p.video = w[1] == 'on'
            elif cmd == 'fpv' and len(w) > 1:
                p.fpv_command(w[1:])
            elif cmd == 'text' and len(w) > 1 and (w[1] in ('canonical', 'raw')):
                p.text_mode = w[1]
                print(f"the policy is told {('the goal canonical phrase' if p.text_mode == 'canonical' else 'your own words')}")
            else:
                p.go(line)
    finally:
        p.close()
if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
