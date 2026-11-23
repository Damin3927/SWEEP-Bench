# SWEEP-Bench

[Project page](https://damin3927.github.io/SWEEP-Bench)

SWEEP-Bench is a simulation benchmark that measures how robot manipulation success changes with the speed of the
object. It has 80 conveyor-belt manipulation tasks in 16 skill categories, each evaluated at 13 belt speeds from
0.10 to 0.40 m/s. Objects are carried by contact with a visibly moving belt, every task has a static counterpart
with the belt stopped, and the randomized scene factors of an episode are the same at every speed, so results at
different speeds are paired. The benchmark is built on [ManiSkill 3](https://github.com/haosulab/ManiSkill) with a
Franka Panda arm beside the conveyor; trained policies are served over a websocket.

## Tasks

Each task has a language instruction, a success condition computed from simulator state, and a scripted expert that
knows the object's pose and velocity; on every task, the expert succeeds in at least 80% of episodes at 0.10, 0.20,
0.30 and 0.40 m/s. The gym id of a task is `Conveyor<Name>-v0`, and its static counterpart is `Static<Name>-v0`.
The categories are Place (17 tasks), Pick (14), Insert (10), Press (8), Rotate (6), Push (5), Pull (4), Hang (3),
Hold (3), Reach (2), Pour (2), Tool use (2), Open (1), Close (1), Screw (1) and Wipe (1).
`sweep_bench.tasks.naming.TASK_NAMING` lists every task and maps each gym id to its display name and category.

## Setup

Requirements: Linux with an NVIDIA GPU for parallel simulation and the policy servers, `uv`, and Python 3.12 (uv
installs it). The simulator also runs on macOS with Vulkan, with one environment at a time; see the
[ManiSkill notes](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html).

```bash
cd SWEEP-Bench
scripts/fetch_upstreams.sh   # clones the model code of each policy server at its pinned commit
uv sync                      # add --extra data to collect demonstrations or extract adapter features
source .venv/bin/activate
```

From git, `git clone --recursive https://github.com/Damin3927/SWEEP-Bench.git SWEEP-Bench` fetches the same model code as submodules
(`git submodule update --init --recursive` after a plain clone) instead of `scripts/fetch_upstreams.sh`. On Linux,
`uv sync` installs torch 2.11.0 built for CUDA 12.6, the build the demonstrations were collected with. Then download
the room backgrounds (the plain table needs none) and platform materials, and build the object cache from the
original sources (object meshes are not redistributed with this code):

```bash
python -m mani_skill.utils.download_asset -y ReplicaCAD
python -m mani_skill.utils.download_asset -y AI2THOR      # ArchitecTHOR rooms, about 14.5 GB
python -m mani_skill.utils.download_asset -y RoboCasa     # rooms and the RoboCasa objects, about 3.5 GB
python -m mani_skill.utils.download_asset -y ycb
python scripts/fetch_table_materials.py                   # CC0 platform materials, checked by sha256
python scripts/import_objects.py --source gso --limit 1100 --merge   # all ~1030 Google Scanned Objects
python scripts/import_objects.py --source ycb --limit 100 --merge    # all 78 YCB models ManiSkill downloads
python scripts/import_polyhaven.py --merge
python scripts/import_robocasa.py --merge
python scripts/split_merged_sets.py --apply
python scripts/reclassify_roles.py --apply
python scripts/rehull_collisions.py --ids-file data/rehull_convex_hull_ids.txt --apply
python scripts/merge_mesh_manifests.py --check-only       # which published objects this cache can build
```

`split_merged_sets.py`, `reclassify_roles.py` and `rehull_collisions.py` print the physics checks to run again on
the objects they change.

## Quick start

Tasks are `gymnasium` environments. To
evaluate a trained policy, start its server, then run one task
at one speed, or the sweep over all 80 tasks and 13 speeds:

```bash
python scripts/eval.py --task ConveyorHexSocket-v0 --policy openpi --port 8000 \
    --randomize --speed-range 0.25 0.25 --num-episodes 5
python scripts/eval_band_sweep.py --policy openpi --port 8000 \
    --families $(python -c "from sweep_bench.tasks.naming import TASK_NAMING; print(*TASK_NAMING)") \
    --speeds 0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.275,0.30,0.325,0.35,0.375,0.40 \
    --episodes 20 --seed 1000 --out runs/openpi_main
```

## Evaluation protocol

The main evaluation runs each policy on the 80 tasks at 13 speeds (0.10 to 0.40 m/s in steps of 0.025 m/s), with
20 episodes per task and speed, on the table platform and 185 backgrounds (the scripts' defaults). The random
draws of an episode depend only on the task and the episode index, so the same `--seed` gives every speed and every
policy the same episodes (except the starting angle of the turntable in
`ConveyorTurntablePick-v0`, which depends on the speed).
An episode succeeds when the task's success condition holds for the task's required number of consecutive control
steps, within 300 control steps (15 s). The simulation pauses while the policy computes its action.

## License

The code is released under the MIT license ([LICENSE](LICENSE)), except the git submodules under
`policy_servers/*/upstream` (their own licenses) and the files listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which keep the licenses of the projects they are adapted from: the
S-Lab License 1.0 (non-commercial) for the DynamicVLA files, and Apache-2.0 or the upstream's MIT license for the
others. The demonstrations are on the Hugging Face Hub as
[Damin3927/SWEEP-Bench](https://huggingface.co/datasets/Damin3927/SWEEP-Bench) under CC BY-NC 4.0, the terms of the ReplicaCAD assets the scenes use.

Third-party assets are downloaded from their own distributions and are not redistributed here:

- Backgrounds: ReplicaCAD (CC BY-NC 4.0), [RoboCasa](https://github.com/robocasa/robocasa) (CC BY 4.0) and
  ArchitecTHOR (Apache 2.0), the licenses of the original releases, through ManiSkill's asset downloader (its copies
  are tagged CC BY 4.0 for ArchitecTHOR and MIT for RoboCasa).
- Objects: Google Scanned Objects (CC BY 4.0, from
  [Gazebo Fuel](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)),
  YCB (CC BY 4.0), RoboCasa objects (CC BY 4.0) and [Poly Haven](https://polyhaven.com/) models (CC0); each imported
  object keeps its source, license and attribution in the local manifest. Procedural objects are generated in code.
- Platform materials: [Poly Haven](https://polyhaven.com/) and [ambientCG](https://ambientcg.com/) (CC0), listed
  with their sources in `data/table_materials.json`.
- Model code: the `policy_servers/*/upstream` submodules, under their authors' licenses.
