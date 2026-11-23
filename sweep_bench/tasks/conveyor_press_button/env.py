from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_press_button.assets import (
    build_button_cap,
    build_button_housing,
)
from sweep_bench.tasks.conveyor_press_button.config import RandomizationConfig
from sweep_bench.tasks.conveyor_press_button.scripted import ScriptedPressMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorPressButton-v0")
@register_env("ConveyorPressButton-v0", max_episode_steps=300)
class ConveyorPressButtonEnv(ScriptedPressMixin, ConveyorWorkspaceEnv):
    """Press a push-button riding the conveyor belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _bore_half_width: float = 0.021
    _housing_half_width: float = 0.035
    _wall_height: float = 0.030
    _floor_half_thickness: float = 0.005

    _cap_half_width: float = 0.019
    _cap_half_height: float = 0.0175
    _cap_rest_gap: float = 0.008

    _cap_spring_k: float = 300.0
    _cap_spring_damping: float = 8.0
    _cap_force_max: float = 20.0

    _housing_patch_radius: float = 0.03

    _press_success_depth: float = 0.005
    _cap_bore_slack: float = 0.005
    _press_sustain_steps: int = 3

    default_prompt: str = "press the button on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    @property
    def _cap_rest_local_z(self) -> float:
        return self._cap_rest_gap + self._cap_half_height

    @property
    def _housing_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        housing_builder = build_button_housing(
            self.scene,
            bore_half_width=self._bore_half_width,
            outer_half_width=self._housing_half_width,
            wall_height=self._wall_height,
            floor_half_thickness=self._floor_half_thickness,
        )
        housing_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._housing_origin_above_deck])
        self.housing = housing_builder.build(name="button_housing")
        self.belt.register_rider(self.housing, patch_radius=self._housing_patch_radius)

        cap_builder = build_button_cap(
            self.scene,
            cap_half_width=self._cap_half_width,
            cap_half_height=self._cap_half_height,
        )
        cap_builder.initial_pose = sapien.Pose(
            p=[ox, oy, deck_z + self._housing_origin_above_deck + self._cap_rest_local_z]
        )
        self.cap = cap_builder.build(name="button_cap")

    def _after_reconfigure(self, options: dict) -> None:
        super()._after_reconfigure(options)
        com = np.stack([np.asarray(body.cmass_local_pose.p) for body in self.housing._bodies])
        self._housing_com_local = torch.as_tensor(com, dtype=torch.float32, device=self.device)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, self.cap]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        button_q = self._sample_button_orientation(env_idx, b)

        housing_pos = self._sample_housing_position(env_idx, b)
        self.housing.set_pose(Pose.create_from_pq(housing_pos, button_q))
        housing_vel = torch.zeros((b, 3), device=device)
        housing_vel[:, 1] = self.belt.velocity[env_idx]
        self.housing.set_linear_velocity(housing_vel)
        self.housing.set_angular_velocity(torch.zeros((b, 3), device=device))

        cap_pos = housing_pos.clone()
        cap_pos[:, 2] = housing_pos[:, 2] + self._cap_rest_local_z
        self.cap.set_pose(Pose.create_from_pq(cap_pos, button_q))
        cap_vel = torch.zeros((b, 3), device=device)
        cap_vel[:, 1] = self.belt.velocity[env_idx]
        self.cap.set_linear_velocity(cap_vel)
        self.cap.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_press_trackers(env_idx)
        self._cap_spring_ready = True

    def _sample_button_orientation(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        q[:, 0] = 1.0
        if self.randomization.button_yaw:
            yaw = self._episode_rng.uniform(-np.pi / 4, np.pi / 4, size=b).astype(np.float32)
            yaw_t = torch.as_tensor(yaw, device=device)
            q[:, 0] = torch.cos(yaw_t * 0.5)
            q[:, 3] = torch.sin(yaw_t * 0.5)
        return q

    def _sample_housing_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._housing_half_width - 0.04
        housing_pos = torch.zeros((b, 3), device=device)
        housing_pos[:, 0] = ox
        housing_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        housing_pos[:, 2] = deck_z + self._housing_origin_above_deck
        return housing_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_cap_return_force()

    def _housing_com(self) -> torch.Tensor:
        com_local = self._housing_com_local.expand(self.num_envs, 3)
        return (self.housing.pose * Pose.create_from_pq(com_local)).p

    def _apply_cap_return_force(self) -> None:
        if not getattr(self, "_cap_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        anchor_z = self.housing.pose.p[:, 2] + self._cap_rest_local_z
        cap_z = self.cap.pose.p[:, 2]
        cap_vz = self.cap.linear_velocity[:, 2]
        disp = anchor_z - cap_z
        fz = (self._cap_spring_k * disp - self._cap_spring_damping * cap_vz).clamp(0.0, self._cap_force_max)
        force = torch.zeros((self.num_envs, 3), device=device)
        force[:, 2] = fz
        reaction = -force
        lever = self.cap.pose.p - self._housing_com()
        reaction_torque = torch.linalg.cross(lever, reaction, dim=-1)
        if gpu:
            force_buf = self.scene.px.cuda_rigid_body_force.torch()
            torque_buf = self.scene.px.cuda_rigid_body_torque.torch()
            force_buf[self.cap._body_data_index, :3] = force
            force_buf[self.housing._body_data_index, :3] += reaction
            torque_buf[self.housing._body_data_index, :3] += reaction_torque
            self.scene.px.gpu_apply_rigid_dynamic_force()
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.cap._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))
            self.housing._bodies[0].add_force_torque(
                reaction.squeeze(0).cpu().numpy().astype(np.float32),
                reaction_torque.squeeze(0).cpu().numpy().astype(np.float32),
            )

    def _cap_press_depth(self) -> torch.Tensor:
        rel_z = self.cap.pose.p[:, 2] - self.housing.pose.p[:, 2]
        return (self._cap_rest_local_z - rel_z).clamp(0.0, self._cap_rest_gap)

    def _cap_in_bore(self) -> torch.Tensor:
        local = (self.housing.pose.inv() * Pose.create_from_pq(self.cap.pose.p)).p
        return (torch.abs(local[:, 0]) <= self._cap_bore_slack) & (torch.abs(local[:, 1]) <= self._cap_bore_slack)

    def _reset_press_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_press_sustain_counter"):
            self._press_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._press_sustain_counter[env_idx] = 0

    def evaluate(self):
        depth = self._cap_press_depth()
        in_bore = self._cap_in_bore()
        pressed = (depth > self._press_success_depth) & in_bore

        self._press_sustain_counter = self._advance_sustain("press_sustain", self._press_sustain_counter, pressed)
        held = self._press_sustain_counter >= self._press_sustain_steps

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        housing_on_belt = rider_on_belt(
            pose=self.housing.pose,
            linear_velocity=self.housing.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._housing_half_width,
            y_tol=self._conveyor_half_length + self._housing_half_width,
            rest_height=self._housing_origin_above_deck,
        )

        success = held & housing_on_belt

        return {
            "success": success,
            "pressed": pressed,
            "press_depth": depth,
            "cap_in_bore": in_bore,
            "press_sustain": self._press_sustain_counter,
            "housing_on_belt": housing_on_belt,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        cap_pos = self.cap.pose.p

        tcp_to_cap = torch.linalg.norm(tcp_pos - cap_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_cap)

        press_reward = (info["press_depth"] / self._press_success_depth).clamp(0.0, 1.0)

        return reach_reward + 2.0 * press_reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
