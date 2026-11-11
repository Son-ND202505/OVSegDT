"""Standalone OVSegDT policy inference, decoupled from the habitat-baselines
train/eval loop, for real-robot / WebSocket serving.

First-draft approach: spin up one real habitat.Env briefly just to harvest the
authoritative observation_space / action_space the checkpoint was trained
against (safer than hand-reconstructing gym spaces and risking a silent shape
mismatch), then build the policy against those spaces directly.
"""
import os
import pickle
import re

import cv2
import numpy as np
import torch

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

from ovon.config import HabitatConfigPlugin
from habitat.config.default_structured_configs import register_hydra_plugin
import ovon.trainers  # noqa: F401  (registers policies/trainers as a side effect)


def register_plugins():
    register_hydra_plugin(HabitatConfigPlugin)


class OVSegDTPolicy:
    """Loads the OVSegDT checkpoint once; each `Session` then calls `step()`
    repeatedly, carrying its own recurrent state (mirrors how lightnav-serve
    keeps one session per WebSocket connection)."""

    def __init__(
        self,
        exp_config_path: str,
        ckpt_path: str,
        overrides=None,
        device: str = None,
    ):
        register_plugins()
        overrides = list(overrides or [])
        self.config = habitat.get_config(exp_config_path, overrides)

        # One throwaway Env just to read off the exact obs/action spaces the
        # checkpoint expects -- avoids hand-guessing sensor shapes/dtypes or
        # the discrete action ordering.
        spaces = os.environ.get("OVSEGDT_SPACES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_spaces.pkl"))
        if os.path.exists(spaces):            # saved spaces (release_tools/make_policy_spaces.py): no Habitat scene dataset needed
            with open(spaces, "rb") as f:
                self.observation_space, self.action_space = pickle.load(f)
        else:
            env = habitat.Env(config=self.config)
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            env.close()
            del env

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        policy_cls = baseline_registry.get_policy(
            self.config.habitat_baselines.rl.policy.name
        )
        self.actor_critic = policy_cls.from_config(
            self.config, self.observation_space, self.action_space
        )
        self.actor_critic.to(self.device)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = {
            k[len("actor_critic.") :]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("actor_critic.")
        }
        self.actor_critic.load_state_dict(state_dict)
        print("[OVSegDTPolicy] checkpoint loaded")
        self.actor_critic.eval()

        self.hidden_size = self.config.habitat_baselines.rl.ppo.hidden_size
        self.num_recurrent_layers = self.actor_critic.num_recurrent_layers

        # Goal-text -> embedding cache (same file the ClipObjectGoalSensor
        # reads from during training/eval).
        from ovon.utils.utils import load_pickle

        self.goal_cache = load_pickle(self.config.habitat.task.lab_sensors.clip_objectgoal_sensor.cache)
        # sorted longest-first so e.g. "trash can" is tried before "can"
        self._goal_categories_by_length = sorted(
            self.goal_cache.keys(), key=len, reverse=True
        )

        # The policy outputs HabitatSimActions integer ids, exactly like the
        # eval trainer sends to env.step(). Gym/Dict key order is not the id
        # order (it can be alphabetical), so never use action_space.spaces.keys()
        # as an index-to-name table.
        action_ids_to_names = {
            HabitatSimActions[name]: name
            for name in self.action_space.spaces.keys()
            if HabitatSimActions.has_action(name)
        }
        max_action_id = max(action_ids_to_names)
        self.action_names = [
            action_ids_to_names.get(i, f"action_{i}")
            for i in range(max_action_id + 1)
        ]
        print(f"[OVSegDTPolicy] HabitatSimActions index order: {self.action_names}")

        # Placeholder shape for the "semantic" key: some code paths derive a
        # zeros_like() template from it even when segmentation_source is a
        # yolo_* variant (the actual mask comes from YOLOE on rgb instead).
        # Must be (RGB_SIZE, RGB_SIZE, 1), not the raw sensor's (640, 360, 1):
        # YOLOE runs on the resized 224x224 rgb (imgsz=224) and its output
        # mask gets written into this same-shaped tensor downstream.

    def new_session(self):
        return _Session(self)

    def resolve_goal_category(self, text: str) -> str:
        """Map free-form text (a bare category like "chair", or a sentence
        like "go to the trashcan on the right") to one of the 379 trained
        category strings. Exact match first; otherwise the longest known
        category that appears as a whole word/phrase in the text (so "trash
        can" is preferred over a shorter partial match like "can").

        This is a stopgap for a UI (mujoco_demo's, built for VLN sentences)
        that doesn't know OVSegDT only wants a bare noun -- not a substitute
        for real open-vocab encoding via a live SigLIP text pass, which isn't
        wired up yet.
        """
        text = text.strip()
        if text in self.goal_cache:
            return text
        lowered = text.lower()
        if lowered in self.goal_cache:
            return lowered
        for category in self._goal_categories_by_length:
            if re.search(rf"\b{re.escape(category)}\b", lowered):
                return category
        raise KeyError(
            f"no known category found in {text!r} "
            f"({len(self.goal_cache)} categories known, e.g. "
            f"{self._goal_categories_by_length[-5:]}) -- open-vocab live "
            f"text encoding is not wired up yet"
        )


class _Session:
    """Per-connection recurrent state, analogous to lightnav-serve's
    per-session frame history / ViT cache."""

    def __init__(self, policy: OVSegDTPolicy):
        self.policy = policy
        self.reset()

    def reset(self):
        p = self.policy
        self.rnn_hidden_states = torch.zeros(
            1, p.num_recurrent_layers, p.hidden_size, device=p.device
        )
        self.prev_actions = torch.zeros(1, 1, device=p.device, dtype=torch.long)
        self.prev_segm_masks = torch.zeros(1, 128, device=p.device, dtype=torch.float)
        self.not_done_masks = torch.zeros(1, 1, device=p.device, dtype=torch.bool)
        self.step_count = 0
        self.last_action_idx = None

    # Matches habitat_baselines' `resize` obs_transform default
    # (`rgb_image_size: int = 224`) -- required both to match training and
    # because YOLOE needs dims divisible by 32 (the raw 640x360 camera size
    # is not: 360 % 32 != 0).
    RGB_SIZE = 224

    @torch.inference_mode()
    def step(self, rgb_hwc_uint8: np.ndarray, goal_category: str, deterministic: bool = False):
        """One control step. `rgb_hwc_uint8` can be any camera resolution
        (resized here to match training). Returns (action_name:str, stop:bool).

        `deterministic=False` (default) matches training/eval rollout
        behaviour (stochastic action sampling) -- re-running the identical
        episode can therefore give a different trajectory/outcome each time.
        Pass `deterministic=True` for reproducible demo/visual runs (greedy
        argmax action instead of sampling)."""
        p = self.policy
        if rgb_hwc_uint8.shape[:2] != (self.RGB_SIZE, self.RGB_SIZE):
            rgb_hwc_uint8 = cv2.resize(
                rgb_hwc_uint8, (self.RGB_SIZE, self.RGB_SIZE), interpolation=cv2.INTER_LINEAR
            )
        resolved_category = p.resolve_goal_category(goal_category)
        goal_embed = torch.from_numpy(p.goal_cache[resolved_category]).float()

        obs = {
            "rgb": torch.from_numpy(rgb_hwc_uint8).unsqueeze(0),
            "clip_objectgoal": goal_embed.unsqueeze(0),
            # unused when segmentation_source is a yolo_* variant, but a
            # same-shape placeholder is still read for its dtype/shape.
            "semantic": torch.zeros(
                (1, self.RGB_SIZE, self.RGB_SIZE, 1), dtype=torch.int32
            ),
            # Presence of this key is what makes transformer_policy.py take
            # the "online inference" branch (rnn_build_seq_info=None -> {})
            # instead of assuming a batched training update.
            "step_id": torch.tensor([[self.step_count]], dtype=torch.int64),
        }
        obs = {k: v.to(p.device) for k, v in obs.items()}

        _, action, _, self.rnn_hidden_states, self.prev_segm_masks = p.actor_critic.act(
            obs,
            self.rnn_hidden_states,
            self.prev_actions,
            self.prev_segm_masks,
            self.not_done_masks,
            deterministic=deterministic,
        )
        self.prev_actions = action
        self.not_done_masks = torch.ones(1, 1, device=p.device, dtype=torch.bool)
        self.step_count += 1

        action_idx = int(action.item())
        self.last_action_idx = action_idx
        action_name = p.action_names[action_idx]
        return action_name, action_name == "stop"
