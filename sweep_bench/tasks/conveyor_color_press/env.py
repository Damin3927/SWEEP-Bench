from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_color_press.assets import (
    CAP_PALETTE,
    build_button_cap,
    build_button_panel,
)
from sweep_bench.tasks.conveyor_color_press.config import RandomizationConfig
from sweep_bench.tasks.conveyor_color_press.scripted import ScriptedColorPressMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda


@bench_task("ConveyorColorPress-v0")
@register_env("ConveyorColorPress-v0", max_episode_steps=300)
class ConveyorColorPressEnv(ScriptedColorPressMixin, ConveyorWorkspaceEnv):
    """Press the colour named by the instruction on a multi-button moving panel."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _num_bores: int = 3
    _bore_pitch: float = 0.045
    _bore_half_width: float = 0.016
    _panel_wall_thickness: float = 0.010
    _wall_height: float = 0.030
    _floor_half_thickness: float = 0.005

    _cap_half_width: float = 0.014
    _cap_half_height: float = 0.0175
    _cap_rest_gap: float = 0.008

    _cap_spring_k: float = 300.0
    _cap_spring_damping: float = 8.0
    _cap_force_max: float = 20.0
    _cap_stiffness_range: tuple[float, float] = (0.8, 1.25)

    _panel_yaw_max: float = 0.35

    _panel_patch_radius: float = 0.026

    _press_success_depth: float = 0.006
    _decoy_max_depth: float = 0.004
    _press_sustain_steps: int = 5
    _cap_bore_slack: float = 0.005
    _on_belt_x_tol: float = 0.05
    _on_belt_y_tol: float = 0.85

    default_prompt: str = "press the named colored button on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        reconfiguration_freq: int | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if (num_envs == 1 and len(CAP_PALETTE) > 1) else 0
        super().__init__(*args, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **kwargs)

    @property
    def _cap_rest_local_z(self) -> float:
        return self._cap_rest_gap + self._cap_half_height

    @property
    def _panel_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    @property
    def _panel_half_x(self) -> float:
        return 0.5 * (self._num_bores - 1) * self._bore_pitch + self._bore_half_width + self._panel_wall_thickness

    @property
    def _panel_half_y(self) -> float:
        return self._bore_half_width + self._panel_wall_thickness

    def _bore_local_x(self, bore: torch.Tensor) -> torch.Tensor:
        return (bore.to(torch.float32) - 0.5 * (self._num_bores - 1)) * self._bore_pitch

    @property
    def episode_objects(self) -> list[str]:
        return list(self._target_color_names)

    @property
    def episode_prompts(self) -> list[str]:
        return [f"press the {name} button on the conveyor belt" for name in self._target_color_names]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        panel_builder = build_button_panel(
            self.scene,
            bore_half_width=self._bore_half_width,
            bore_pitch=self._bore_pitch,
            num_bores=self._num_bores,
            wall_thickness=self._panel_wall_thickness,
            wall_height=self._wall_height,
            floor_half_thickness=self._floor_half_thickness,
        )
        panel_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._panel_origin_above_deck])
        self.panel = panel_builder.build(name="button_panel")
        self.belt.register_rider(self.panel, patch_radius=self._panel_patch_radius)

        perm = np.asarray(self._batched_episode_rng.permutation(len(CAP_PALETTE)))[:, : self._num_bores]
        self._cap_color_names: list[list[str]] = [[CAP_PALETTE[j][0] for j in row] for row in perm]
        self._target_color_names: list[str] = [row[0] for row in self._cap_color_names]

        self.caps: list[Actor] = []
        cap_z = deck_z + self._panel_origin_above_deck + self._cap_rest_local_z
        for k in range(self._num_bores):
            objs: list[Actor] = []
            for i in range(self.num_envs):
                name, color = CAP_PALETTE[perm[i, k]]
                cap_builder = build_button_cap(
                    self.scene,
                    cap_half_width=self._cap_half_width,
                    cap_half_height=self._cap_half_height,
                    color=color,
                )
                cap_builder.initial_pose = sapien.Pose(p=[ox, oy, cap_z])
                cap_builder.set_scene_idxs([i])
                obj = cap_builder.build(name=f"cap{k}_{name}-{i}")
                self.remove_from_state_dict_registry(obj)
                objs.append(obj)
            merged = Actor.merge(objs, name=f"button_cap_{k}")
            self.add_to_state_dict_registry(merged)
            self.caps.append(merged)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        yaw = self._sample_panel_yaw(env_idx, b)
        panel_q = torch.zeros((b, 4), device=device)
        panel_q[:, 0] = torch.cos(yaw * 0.5)
        panel_q[:, 3] = torch.sin(yaw * 0.5)

        panel_pos = self._sample_panel_position(env_idx, b)
        self.panel.set_pose(Pose.create_from_pq(panel_pos, panel_q))
        ride_vel = torch.zeros((b, 3), device=device)
        ride_vel[:, 1] = self.belt.velocity[env_idx]
        self.panel.set_linear_velocity(ride_vel)
        self.panel.set_angular_velocity(torch.zeros((b, 3), device=device))

        bores = self._sample_bore_assignment(b)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        for k, cap in enumerate(self.caps):
            local_x = self._bore_local_x(bores[:, k])
            cap_pos = torch.zeros((b, 3), device=device)
            cap_pos[:, 0] = panel_pos[:, 0] + local_x * cos_yaw
            cap_pos[:, 1] = panel_pos[:, 1] + local_x * sin_yaw
            cap_pos[:, 2] = panel_pos[:, 2] + self._cap_rest_local_z
            cap.set_pose(Pose.create_from_pq(cap_pos, panel_q))
            cap.set_linear_velocity(ride_vel)
            cap.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_press_trackers(env_idx)
        self._target_bore[env_idx] = bores[:, 0]
        self._cap_bore_index[env_idx] = bores
        self._cap_spring_k_env[env_idx] = self._sample_cap_stiffness(b)
        self._cap_spring_ready = True

    def _sample_panel_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.panel_yaw:
            return torch.zeros(b, device=device)
        yaw = self._episode_rng.uniform(-self._panel_yaw_max, self._panel_yaw_max, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=device)

    def _sample_bore_assignment(self, b: int) -> torch.Tensor:
        perm = np.asarray(self._episode_rng.permutation(self._num_bores))
        assert perm.shape == (b, self._num_bores), perm.shape
        return torch.as_tensor(perm, dtype=torch.long, device=self.device)

    def _sample_cap_stiffness(self, b: int) -> torch.Tensor:
        if not self.randomization.cap_stiffness:
            return torch.full((b,), self._cap_spring_k, device=self.device)
        lo, hi = self._cap_stiffness_range
        scale = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(scale * self._cap_spring_k, device=self.device)

    def _sample_panel_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._panel_half_y - 0.04
        panel_pos = torch.zeros((b, 3), device=device)
        panel_pos[:, 0] = ox
        panel_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        panel_pos[:, 2] = deck_z + self._panel_origin_above_deck
        return panel_pos

    def _before_simulation_step(self) -> None:
        super()._before_simulation_step()
        self._apply_cap_return_force()

    def _apply_cap_return_force(self) -> None:
        if not getattr(self, "_cap_spring_ready", False):
            return
        device = self.device
        gpu = self.scene.gpu_sim_enabled
        if gpu:
            self.scene.px.gpu_fetch_rigid_dynamic_data()
        anchor_z = self.panel.pose.p[:, 2] + self._cap_rest_local_z
        force_buf = self.scene.px.cuda_rigid_body_force.torch() if gpu else None
        for cap in self.caps:
            disp = anchor_z - cap.pose.p[:, 2]
            fz = (self._cap_spring_k_env * disp - self._cap_spring_damping * cap.linear_velocity[:, 2]).clamp(
                0.0, self._cap_force_max
            )
            force = torch.zeros((self.num_envs, 3), device=device)
            force[:, 2] = fz
            if force_buf is not None:
                force_buf[cap._body_data_index, :3] = force
            else:
                cap._bodies[0].add_force_torque(force.squeeze(0).cpu().numpy(), np.zeros(3, dtype=np.float32))
        if gpu:
            self.scene.px.gpu_apply_rigid_dynamic_force()

    def _cap_press_depth(self, cap: Actor) -> torch.Tensor:
        rel_z = cap.pose.p[:, 2] - self.panel.pose.p[:, 2]
        return (self._cap_rest_local_z - rel_z).clamp(0.0, self._cap_rest_gap)

    def _cap_in_bore(self, k: int) -> torch.Tensor:
        local = (self.panel.pose.inv() * Pose.create_from_pq(self.caps[k].pose.p)).p
        dx = local[:, 0] - self._bore_local_x(self._cap_bore_index[:, k])
        return (torch.abs(dx) <= self._cap_bore_slack) & (torch.abs(local[:, 1]) <= self._cap_bore_slack)

    def _reset_press_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_press_sustain_counter"):
            b, device = self.num_envs, self.device
            self._press_sustain_counter = torch.zeros(b, dtype=torch.long, device=device)
            self._decoys_clean = torch.ones(b, dtype=torch.bool, device=device)
            self._target_bore = torch.zeros(b, dtype=torch.long, device=device)
            self._cap_bore_index = torch.zeros((b, self._num_bores), dtype=torch.long, device=device)
            self._cap_spring_k_env = torch.full((b,), self._cap_spring_k, device=device)
        self._press_sustain_counter[env_idx] = 0
        self._decoys_clean[env_idx] = True

    def evaluate(self):
        depth = self._cap_press_depth(self.caps[0])
        in_bore = self._cap_in_bore(0)
        pressed = (depth > self._press_success_depth) & in_bore

        self._press_sustain_counter = self._advance_sustain("press_sustain", self._press_sustain_counter, pressed)
        held = self._press_sustain_counter >= self._press_sustain_steps

        decoy_depth = torch.stack([self._cap_press_depth(cap) for cap in self.caps[1:]], dim=-1)
        disturbed = (decoy_depth >= self._decoy_max_depth).any(dim=-1)
        self._decoys_clean = self._decoys_clean & ~disturbed

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        panel_on_belt = rider_on_belt(
            pose=self.panel.pose,
            linear_velocity=self.panel.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._on_belt_x_tol,
            y_tol=self._on_belt_y_tol,
            rest_height=self._panel_origin_above_deck,
        )

        success = held & self._decoys_clean & panel_on_belt

        return {
            "success": success,
            "pressed": pressed,
            "press_depth": depth,
            "cap_in_bore": in_bore,
            "press_sustain": self._press_sustain_counter,
            "decoy_depth": decoy_depth.amax(dim=-1),
            "decoys_clean": self._decoys_clean,
            "panel_on_belt": panel_on_belt,
            "target_bore": self._target_bore,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        target_pos = self.caps[0].pose.p

        tcp_to_cap = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_cap)

        press_reward = (info["press_depth"] / self._press_success_depth).clamp(0.0, 1.0)

        return (reach_reward + 2.0 * press_reward) * info["decoys_clean"].to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3.0
