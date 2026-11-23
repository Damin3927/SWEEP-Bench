from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_fruit_drop.config import RandomizationConfig
from sweep_bench.tasks.conveyor_fruit_drop.scripted import ScriptedPlaceMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

FRUIT_MODEL_IDS = {
    "strawberry": "012_strawberry",
    "apple": "013_apple",
    "lemon": "014_lemon",
    "peach": "015_peach",
    "orange": "017_orange",
    "plum": "018_plum",
}

_PALM_LINK = {
    "panda": "panda_hand",
    "panda_wristcam": "panda_hand",
}


@bench_task("ConveyorFruitDrop-v0")
@register_env("ConveyorFruitDrop-v0", max_episode_steps=300)
class ConveyorFruitDropEnv(ScriptedPlaceMixin, ConveyorWorkspaceEnv):
    """Drop a held fruit into the red bowl riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    SUPPORTED_FRUITS = tuple(FRUIT_MODEL_IDS)

    _bowl_half_height: float = 0.0277
    _bowl_outer_radius: float = 0.0811
    _bowl_patch_radius: float = 0.04
    _fruit_patch_radius: float = 0.02

    _conveyor_half_width: float = 0.12

    RandomizationConfig = RandomizationConfig

    def __init__(
        self,
        *args,
        fruits: tuple[str, ...] = ("strawberry",),
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        if not fruits:
            raise ValueError("fruits must name at least one fruit")
        unknown = [f for f in fruits if f not in FRUIT_MODEL_IDS]
        if unknown:
            raise ValueError(f"Unknown fruits {unknown}; supported: {sorted(FRUIT_MODEL_IDS)}")
        self.fruits: tuple[str, ...] = tuple(fruits)
        self.default_prompt = (
            f"put the {self.fruits[0]} into the red bowl on the conveyor belt"
            if len(self.fruits) == 1
            else "put the fruit into the red bowl on the conveyor belt"
        )
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if (num_envs == 1 and len(self.fruits) > 1) else 0
        super().__init__(*args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs)

    def built_object_ids(self) -> tuple[str, ...]:
        return self.fruits

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        builder = actors.get_actor_builder(self.scene, id="ycb:024_bowl")
        builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._bowl_half_height + 0.001])
        self.bowl = builder.build(name="bowl")
        self.belt.register_rider(self.bowl, patch_radius=self._bowl_patch_radius)

        fruit_names = self._batched_episode_rng.choice(np.array(self.fruits), replace=True)
        self._fruit_names: list[str] = [str(name) for name in fruit_names]
        self._fruit_objs: list[Actor] = []
        for i, fruit in enumerate(self._fruit_names):
            fruit_builder = actors.get_actor_builder(self.scene, id=f"ycb:{FRUIT_MODEL_IDS[fruit]}")
            fruit_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
            fruit_builder.set_scene_idxs([i])
            obj = fruit_builder.build(name=f"{fruit}-{i}")
            self.remove_from_state_dict_registry(obj)
            self._fruit_objs.append(obj)
        self.fruit = Actor.merge(self._fruit_objs, name="fruit")
        self.add_to_state_dict_registry(self.fruit)
        self.belt.register_rider(self.fruit, patch_radius=self._fruit_patch_radius)

    @property
    def episode_fruits(self) -> list[str]:
        return list(self._fruit_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"put the {name} into the red bowl on the conveyor belt" for name in self._fruit_names]

    def _after_reconfigure(self, options: dict) -> None:
        half_widths = []
        half_heights = []
        for obj in self._fruit_objs:
            mesh = obj.get_first_collision_mesh()
            assert mesh is not None, f"fruit actor {obj.name} has no collision mesh"
            bounds = mesh.bounding_box.bounds
            half_widths.append(float(bounds[1, 0] - bounds[0, 0] + bounds[1, 1] - bounds[0, 1]) / 4.0)
            half_heights.append(float(bounds[1, 2] - bounds[0, 2]) / 2.0)
        self._fruit_half_width = common.to_tensor(half_widths, device=self.device)
        self._fruit_half_height = common.to_tensor(half_heights, device=self.device)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        half_w = self._fruit_half_width[env_idx].cpu().numpy()
        return 2.0 * (half_w - 0.001)[:, None]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        bowl_pos = self._sample_bowl_position(env_idx, b)
        bowl_q = torch.zeros((b, 4), device=device)
        bowl_q[:, 0] = 1.0
        self.bowl.set_pose(Pose.create_from_pq(bowl_pos, bowl_q))
        bowl_vel = torch.zeros((b, 3), device=device)
        bowl_vel[:, 1] = self.belt.velocity[env_idx]
        self.bowl.set_linear_velocity(bowl_vel)
        self.bowl.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pos = cast("Panda", self.agent).tcp.pose.p[env_idx]
        fruit_q = torch.zeros((b, 4), device=device)
        fruit_q[:, 0] = 1.0
        self.fruit.set_pose(Pose.create_from_pq(tcp_pos, fruit_q))
        self.fruit.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.fruit.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_drop_trackers(env_idx)

    def _sample_bowl_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - 0.10
        bowl_pos = torch.zeros((b, 3), device=device)
        bowl_pos[:, 0] = ox
        bowl_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        bowl_pos[:, 2] = deck_z + self._bowl_half_height + 0.001
        return bowl_pos

    _in_bowl_xy_threshold: float = 0.05
    _in_bowl_z_range: tuple[float, float] = (-0.025, 0.05)
    _released_distance_threshold: float = 0.08
    _drop_sustain_steps: int = 5
    _settled_rel_speed_threshold: float = 0.3

    _bowl_interior_floor_offset: float = -0.025

    _held_force_threshold: float = 0.05
    _bowl_touch_force_threshold: float = 0.5

    def _reset_drop_trackers(self, env_idx: torch.Tensor) -> None:
        device = self.device
        if not hasattr(self, "_fruit_held_prev"):
            self._fruit_held_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._fruit_let_go = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._drop_height = torch.zeros(self.num_envs, device=device)
            self._arm_touched_bowl = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._drop_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._fruit_held_prev[env_idx] = False
        self._fruit_let_go[env_idx] = False
        self._drop_height[env_idx] = 0.0
        self._arm_touched_bowl[env_idx] = False
        self._drop_sustain_counter[env_idx] = 0

    def _update_drop_trackers(self, fruit_pos: torch.Tensor, bowl_pos: torch.Tensor) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        f1 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger1_link, self.fruit), dim=-1)
        f2 = torch.linalg.norm(self.scene.get_pairwise_contact_forces(agent.finger2_link, self.fruit), dim=-1)
        held = (f1 > self._held_force_threshold) & (f2 > self._held_force_threshold)

        just_let_go = self._fruit_held_prev & ~held & ~self._fruit_let_go
        rest_z = bowl_pos[:, 2] + self._bowl_interior_floor_offset + self._fruit_half_height
        self._drop_height = torch.where(just_let_go, fruit_pos[:, 2] - rest_z, self._drop_height)
        self._fruit_let_go = self._fruit_let_go | just_let_go
        self._fruit_held_prev = held

        palm_name = _PALM_LINK.get(self.robot_uids)
        if palm_name is None:
            raise ValueError(
                f"fruit-drop has no gripper palm link for robot_uids={self.robot_uids!r}; known: {sorted(_PALM_LINK)}"
            )
        for link in (agent.finger1_link, agent.finger2_link, agent.robot.links_map[palm_name]):
            force = torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, self.bowl), dim=-1)
            self._arm_touched_bowl = self._arm_touched_bowl | (force > self._bowl_touch_force_threshold)

        return held

    def evaluate(self):
        fruit_pos = self.fruit.pose.p
        bowl_pos = self.bowl.pose.p
        tcp_pos = cast("Panda", self.agent).tcp.pose.p

        held = self._update_drop_trackers(fruit_pos, bowl_pos)

        rel = fruit_pos - bowl_pos
        xy_dist = torch.linalg.norm(rel[:, :2], dim=-1)
        lo, hi = self._in_bowl_z_range
        in_bowl_xy = xy_dist < self._in_bowl_xy_threshold
        in_bowl_z = (rel[:, 2] > lo) & (rel[:, 2] < hi)

        rel_speed = torch.linalg.norm(self.fruit.linear_velocity - self.bowl.linear_velocity, dim=-1)
        settled = rel_speed < self._settled_rel_speed_threshold
        released = (~held) & (torch.linalg.norm(tcp_pos - fruit_pos, dim=-1) > self._released_distance_threshold)

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        bowl_on_belt = rider_on_belt(
            pose=self.bowl.pose,
            linear_velocity=self.bowl.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._bowl_outer_radius,
            y_tol=self._conveyor_half_length + self._bowl_outer_radius,
            rest_height=self._bowl_half_height,
        )

        dropped = in_bowl_xy & in_bowl_z & settled & released & bowl_on_belt

        self._drop_sustain_counter = self._advance_sustain("fruit_drop", self._drop_sustain_counter, dropped)
        success = self._drop_sustain_counter >= self._drop_sustain_steps

        return {
            "success": success,
            "in_bowl_xy": in_bowl_xy,
            "in_bowl_z": in_bowl_z,
            "settled": settled,
            "released": released,
            "bowl_on_belt": bowl_on_belt,
            "drop_sustain": self._drop_sustain_counter,
            "fruit_to_bowl_xy": xy_dist,
            "drop_height": self._drop_height,
            "arm_touched_bowl": self._arm_touched_bowl,
        }

    def compute_dense_reward(self, obs, action, info):
        fruit_pos = self.fruit.pose.p
        bowl_pos = self.bowl.pose.p

        xy_dist = torch.linalg.norm(fruit_pos[:, :2] - bowl_pos[:, :2], dim=-1)
        carry_reward = 1 - torch.tanh(5 * xy_dist)

        in_bowl = (info["in_bowl_xy"] & info["in_bowl_z"]).float()
        return carry_reward + 2.0 * in_bowl

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
