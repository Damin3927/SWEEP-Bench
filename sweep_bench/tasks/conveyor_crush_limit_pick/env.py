from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_crush_limit_pick.assets import (
    build_shell_core,
    build_shell_panel,
)
from sweep_bench.tasks.conveyor_crush_limit_pick.config import RandomizationConfig
from sweep_bench.tasks.conveyor_crush_limit_pick.scripted import ScriptedCrushLimitPickMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorCrushLimitPick-v0")
@register_env("ConveyorCrushLimitPick-v0", max_episode_steps=300)
class ConveyorCrushLimitPickEnv(ScriptedCrushLimitPickMixin, ConveyorWorkspaceEnv):
    """Pick a crushable shell off the belt without squeezing it flat."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _core_half_x: float = 0.020
    _core_half_y: float = 0.035
    _core_half_z: float = 0.030
    _core_wall: float = 0.005
    _web_half_x: float = 0.005
    _core_density: float = 900.0

    _panel_half_thickness: float = 0.0085
    _panel_density: float = 1000.0

    _carrier_travel_frac: float = 0.5

    _panel_omega_dt_max: float = 0.80

    _panel_guide_clearance: float = 0.0015
    _panel_travel: float = 0.010
    _crush_depth: float = 0.0095

    _panel_stiffness: float = 250.0
    _panel_damping_ratio: float = 0.4
    _panel_force_max: float = 20.0

    _shell_patch_radius: float = 0.018

    _contact_force_threshold: float = 1e-3
    _grasp_max_angle_deg: float = 85.0
    _hold_dist: float = 0.06
    _lift_clearance: float = 0.15
    _carry_rel_speed: float = 0.03
    _pick_sustain_steps: int = 5

    default_prompt: str = "pick up the crushable shell from the conveyor belt without squashing it"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(self, *args, panel_stiffness: float | None = None, **kwargs):
        if panel_stiffness is not None:
            if not math.isfinite(panel_stiffness) or panel_stiffness <= 0.0:
                raise ValueError(f"panel_stiffness must be a positive finite number (got {panel_stiffness})")
            self._panel_stiffness = float(panel_stiffness)
        super().__init__(*args, **kwargs)

    @property
    def _pocket_half_y(self) -> float:
        return self._core_half_y - self._core_wall

    @property
    def _pocket_half_z(self) -> float:
        return self._core_half_z - self._core_wall

    @property
    def _panel_half_y(self) -> float:
        return self._pocket_half_y - self._panel_guide_clearance

    @property
    def _panel_half_z(self) -> float:
        return self._pocket_half_z - self._panel_guide_clearance

    @property
    def _panel_rest_local_x(self) -> float:
        return self._web_half_x + self._panel_travel + self._panel_half_thickness

    @property
    def _grip_half_width(self) -> float:
        return self._panel_rest_local_x + self._panel_half_thickness

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        core_builder = build_shell_core(
            self.scene,
            half_x=self._core_half_x,
            half_y=self._core_half_y,
            half_z=self._core_half_z,
            wall_thickness=self._core_wall,
            web_half_x=self._web_half_x,
            density=self._core_density,
        )
        core_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._core_half_z])
        self.core = core_builder.build(name="shell_core")
        self.belt.register_rider(self.core, patch_radius=self._shell_patch_radius)

        self.panels: list[Actor] = []
        for name, sign in (("shell_panel_minus", -1.0), ("shell_panel_plus", 1.0)):
            builder = build_shell_panel(
                self.scene,
                half_thickness=self._panel_half_thickness,
                half_y=self._panel_half_y,
                half_z=self._panel_half_z,
                density=self._panel_density,
            )
            builder.initial_pose = sapien.Pose(p=[ox + sign * self._panel_rest_local_x, oy, deck_z + self._core_half_z])
            self.panels.append(builder.build(name=name))
        self._panel_sign: tuple[float, ...] = (-1.0, 1.0)

    @property
    def _task_actors(self) -> list[Actor]:
        return [*self.belt.riders, *self.panels]

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        core_pos = self._sample_shell_position(env_idx, b)
        core_q = self._sample_shell_quat(b)
        self.core.set_pose(Pose.create_from_pq(core_pos, core_q))
        vel = torch.zeros((b, 3), device=device)
        vel[:, 1] = self.belt.velocity[env_idx]
        self.core.set_linear_velocity(vel)
        self.core.set_angular_velocity(torch.zeros((b, 3), device=device))

        rot = quaternion_to_matrix(core_q)
        for i, panel in enumerate(self.panels):
            local = torch.zeros((b, 3), device=device)
            local[:, 0] = self._panel_sign[i] * self._panel_rest_local_x
            panel_pos = core_pos + torch.bmm(rot, local.unsqueeze(-1)).squeeze(-1)
            panel.set_pose(Pose.create_from_pq(panel_pos, core_q))
            panel.set_linear_velocity(vel)
            panel.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_pick_trackers(env_idx)
        self._reset_spring(env_idx)

    def _sample_shell_quat(self, b: int) -> torch.Tensor:
        device = self.device
        q = torch.zeros((b, 4), device=device)
        if not self.randomization.shell_yaw:
            q[:, 0] = 1.0
            return q
        lo, hi = self.randomization.yaw_range
        yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)
        return q

    def _sample_shell_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        device = self.device
        start_y = self._conveyor_half_length - self._core_half_y - 0.04
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.shell_lateral_offset:
            lo, hi = self.randomization.lateral_range
            dx = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = pos[:, 0] + torch.as_tensor(dx, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = oz + self._conveyor_deck_height + self._core_half_z + 0.001
        return pos

    def _reset_spring(self, env_idx: torch.Tensor) -> None:
        device = self.device
        if not hasattr(self, "_panel_damping"):
            self._panel_damping = torch.zeros(self.num_envs, device=device)
        panel_mass = self.panels[0].mass.to(device)[env_idx]
        omega_dt = float(torch.sqrt(self._panel_stiffness / panel_mass.min()).item()) / self.sim_freq
        if omega_dt >= self._panel_omega_dt_max:
            raise ValueError(
                f"panel_stiffness={self._panel_stiffness} N/m gives omega*dt={omega_dt:.3f} at "
                f"{self.sim_freq} Hz with a {float(panel_mass.min().item()) * 1000:.1f} g panel, at or past the "
                f"measured stability edge of {self._panel_omega_dt_max}; lower the stiffness "
                f"(the shipped 250 N/m reads 0.741) or raise the simulation rate."
            )
        self._panel_damping[env_idx] = 2.0 * self._panel_damping_ratio * torch.sqrt(self._panel_stiffness * panel_mass)
        self._spring_ready = True

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_panel_spring_force()
        self._latch_crush()

    def _apply_panel_spring_force(self) -> None:
        if not getattr(self, "_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        rot = self._core_rotation()
        x_axis = rot[:, :, 0]
        core_v = self.core.linear_velocity
        force_buf = self.scene.px.cuda_rigid_body_force.torch() if gpu else None
        torque_buf = self.scene.px.cuda_rigid_body_torque.torch() if gpu else None
        reaction = torch.zeros((self.num_envs, 3), device=device)
        reaction_torque = torch.zeros((self.num_envs, 3), device=device)
        for i, panel in enumerate(self.panels):
            local_x = self._panel_local_x(panel, rot)
            rest_x = self._panel_sign[i] * self._panel_rest_local_x
            v_rel = ((panel.linear_velocity - core_v) * x_axis).sum(dim=-1)
            f = (-self._panel_stiffness * (local_x - rest_x) - self._panel_damping * v_rel).clamp(
                -self._panel_force_max, self._panel_force_max
            )
            force = f.unsqueeze(-1) * x_axis
            reaction = reaction - force
            r = panel.pose.p - self.core.pose.p
            reaction_torque = reaction_torque + torch.cross(r, -force, dim=-1)
            if force_buf is not None:
                force_buf[panel._body_data_index, :3] = force
            else:
                panel._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))
        if force_buf is not None:
            force_buf[self.core._body_data_index, :3] += reaction
            assert torque_buf is not None
            torque_buf[self.core._body_data_index, :3] = reaction_torque
            self.scene.px.gpu_apply_rigid_dynamic_force()
            self.scene.px.gpu_apply_rigid_dynamic_torque()
        else:
            self.core._bodies[0].add_force_torque(
                reaction.squeeze(0).cpu().numpy().astype(np.float32),
                reaction_torque.squeeze(0).cpu().numpy().astype(np.float32),
            )

    def _core_rotation(self) -> torch.Tensor:
        return quaternion_to_matrix(self.core.pose.q)

    def _panel_local_x(self, panel: Actor, rot: torch.Tensor) -> torch.Tensor:
        rel = panel.pose.p - self.core.pose.p
        return (rel * rot[:, :, 0]).sum(dim=-1)

    def _panel_compression(self) -> torch.Tensor:
        rot = self._core_rotation()
        cols = []
        for i, panel in enumerate(self.panels):
            sign = self._panel_sign[i]
            cols.append(self._panel_rest_local_x - sign * self._panel_local_x(panel, rot))
        return torch.stack(cols, dim=-1)

    def _reset_pick_trackers(self, env_idx: torch.Tensor) -> None:
        device = self.device
        if not hasattr(self, "_pick_sustain_counter"):
            self._pick_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self._shell_crushed = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._crush_peak = torch.zeros(self.num_envs, device=device)
            self._prev_core_pos = torch.zeros((self.num_envs, 3), device=device)
            self._prev_tcp_pos = torch.zeros((self.num_envs, 3), device=device)
            self._pre_grip_carried = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._was_gripping = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self._caught_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._pick_sustain_counter[env_idx] = 0
        self._shell_crushed[env_idx] = False
        self._crush_peak[env_idx] = 0.0
        self._prev_core_pos[env_idx] = self.core.pose.p[env_idx]
        self._prev_tcp_pos[env_idx] = cast("Panda", self.agent).tcp.pose.p[env_idx]
        self._pre_grip_carried[env_idx] = True
        self._was_gripping[env_idx] = False
        self._caught_moving[env_idx] = True

    def _latch_crush(self) -> None:
        if not getattr(self, "_spring_ready", False):
            return
        compression = self._panel_compression().amax(dim=-1)
        self._crush_peak = torch.maximum(self._crush_peak, compression)
        self._shell_crushed = self._shell_crushed | (compression >= self._crush_depth)

    def _fingers_on_opposite_panels(self) -> torch.Tensor:
        agent = cast("Panda", self.agent)
        directions = (
            agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1],
            -agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1],
        )
        f: list[list[torch.Tensor]] = []
        for link, direction in zip((agent.finger1_link, agent.finger2_link), directions, strict=True):
            row = []
            for panel in self.panels:
                force = self.scene.get_pairwise_contact_forces(link, panel)
                loaded = torch.linalg.norm(force, dim=-1) > self._contact_force_threshold
                angle = torch.rad2deg(common.compute_angle_between(direction, force))
                row.append(loaded & (angle <= self._grasp_max_angle_deg))
            f.append(row)
        return (f[0][0] & f[1][1]) | (f[0][1] & f[1][0])

    def evaluate(self):
        oz = self.workspace_offset[2]
        agent = cast("Panda", self.agent)
        self._latch_crush()
        core_pos = self.core.pose.p
        tcp_pos = agent.tcp.pose.p

        gripped = self._fingers_on_opposite_panels()
        at_hand = torch.linalg.norm(core_pos - tcp_pos, dim=-1) < self._hold_dist
        shell_held = gripped & at_hand

        deck_z = oz + self._conveyor_deck_height
        lift_height = core_pos[:, 2] - deck_z
        shell_lifted = lift_height >= self._lift_clearance

        advanced = self._eval_advanced("crush_limit_carry")
        rel_vel = ((core_pos - tcp_pos) - (self._prev_core_pos - self._prev_tcp_pos)) * self.control_freq
        self._prev_core_pos = torch.where(advanced[:, None], core_pos, self._prev_core_pos)
        self._prev_tcp_pos = torch.where(advanced[:, None], tcp_pos, self._prev_tcp_pos)
        carried = torch.linalg.norm(rel_vel, dim=-1) < self._carry_rel_speed

        intact = ~self._shell_crushed

        pre_grip = advanced & ~shell_held
        carried_by_belt = self.belt_carried(self.core, self._carrier_travel_frac) & self.riding_on_deck(
            self.core, self._core_half_z
        )
        self._pre_grip_carried = torch.where(pre_grip, carried_by_belt, self._pre_grip_carried)
        first_grip = shell_held & ~self._was_gripping
        new_attempt = first_grip & ~shell_lifted
        regrasp = first_grip & shell_lifted
        self._caught_moving = torch.where(new_attempt, self._pre_grip_carried, self._caught_moving)
        self._caught_moving = torch.where(regrasp, self._caught_moving & self._pre_grip_carried, self._caught_moving)
        self._was_gripping = torch.where(advanced, shell_held, self._was_gripping)

        holding = shell_held & shell_lifted & carried & intact & self._caught_moving
        self._pick_sustain_counter = self._advance_sustain(
            "crush_limit_pick_sustain", self._pick_sustain_counter, holding
        )
        success = self._pick_sustain_counter >= self._pick_sustain_steps

        compression = self._panel_compression()
        return {
            "success": success,
            "shell_held": shell_held,
            "shell_gripped": gripped,
            "shell_at_hand": at_hand,
            "shell_lifted": shell_lifted,
            "shell_lift_height": lift_height,
            "shell_carried": carried,
            "shell_intact": intact,
            "caught_moving": self._caught_moving,
            "panel_compression": compression.amax(dim=-1),
            "panel_peak_compression": self._crush_peak,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        core_pos = self.core.pose.p

        reach_reward = 1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - core_pos, dim=-1))

        lift_reward = (info["shell_lift_height"] / self._lift_clearance).clamp(0.0, 1.0)
        earned = (info["shell_held"] & info["shell_intact"] & info["shell_carried"] & info["caught_moving"]).float()

        return reach_reward + 2.0 * lift_reward * earned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
