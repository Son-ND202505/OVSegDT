# MuJoCo G1 object-goal navigation with OVSegDT

A Unitree G1 humanoid walks through two furnished MuJoCo houses and finds what you ask for. You type a goal ("bed", "fridge", "something to sit on", "coke"),
the OVSegDT policy, fine-tuned for these houses, drives the robot from its own camera image and stops in front of the target.

| House | |
|---|---|
| `home25` | 25 x 25 m house, the default |
| `l28` | L-shaped house |

## Layout

```
OVSegDT/                        this repository
├── ovon/, config/, ...         OVSegDT (original code; ovon/deploy/inference.py can read saved observation spaces, so no Habitat scene data is needed)
├── mujoco_g1_nav/              this demo
│   ├── scripts/play_house.py   the demo
│   ├── eval/policy_server.py   the OVSegDT policy as a local server (started by the demo)
│   ├── src/mujoco_g1_nav/      houses, robot, sensors, goal interpreter, one episode
│   └── configs/, assets/       the two houses, furniture (Kenney), one scanned mug
└── third_party/                Unitree robot model and walking policy (step 1)
```

## 1. Clone

```bash
git clone https://github.com/Son-ND202505/OVSegDT.git && cd OVSegDT
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
git -C third_party/unitree_mujoco checkout 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git third_party/unitree_rl_mjlab
git -C third_party/unitree_rl_mjlab checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790
```

## 2. Two Python environments

The simulation and the policy run in different environments (the policy needs habitat-lab 0.2.3 on Python 3.8).

MuJoCo side:

```bash
conda create -n g1nav python=3.11 -y && conda activate g1nav
pip install -r mujoco_g1_nav/requirements.txt
```

Policy side: follow **Installation** in the original README below (Python 3.8, habitat-sim 0.2.3, habitat-lab and habitat-baselines from this repo, ultralytics).
The HM3D scenes and the OVON episodes are **not** needed for this demo. Then tell the demo where that Python is:

```bash
export OVSEGDT_PYTHON=/path/to/conda/envs/ovon/bin/python
```

At its first start the policy server lets ultralytics download the YOLOE files (`yoloe-v8l-seg.pt`, `mobileclip_blt.ts`, about 0.7 GB): the policy class loads them,
the demo itself feeds the policy the simulator's mask of the goal. A CUDA GPU is needed.

## 3. Weights

```bash
bash mujoco_g1_nav/scripts/download_weights.sh      # -> mujoco_g1_nav/checkpoints/RECOMMENDED_house.pth
```

The SigLIP text model (`timm/ViT-B-16-SigLIP-256`) is downloaded by open_clip at the first goal.

## 4. Run (from the repo root, environment g1nav)

```bash
python mujoco_g1_nav/scripts/play_house.py                         # home25, MuJoCo viewer + the robot's camera in a second window
python mujoco_g1_nav/scripts/play_house.py --house l28             # the L-shaped house
python mujoco_g1_nav/scripts/play_house.py --no-viewer --cmd "bed" "sofa"     # headless; videos in mujoco_g1_nav/outputs/play_house/
```

In the terminal: type a goal, or `objects` (what is in the house), `where <goal>`, `new` (new start), `again`, `help`, `quit`.
Goals of the same scene are chained: the next one starts where the robot stopped.

## Notes

- The policy alone drives (no map, no planner); it gets the simulator's ground-truth mask of the goal category, as a stand-in for a detector.
- It is a research demo: far or hidden targets are not always found.

## License

`mujoco_g1_nav/` is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The rest of this repository belongs to the OVSegDT authors.
If you use it, please cite OVSegDT (citation in the original README below).
