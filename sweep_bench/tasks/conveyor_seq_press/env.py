from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import carried_speed
from sweep_bench.tasks.conveyor_seq_press.assets import (
    CAP_PALETTE,
    build_button_cap,
    build_clock_body,
)
from sweep_bench.tasks.conveyor_seq_press.config import RandomizationConfig
from sweep_bench.tasks.conveyor_seq_press.scripted import ScriptedSeqPressMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.utils.structs.actor import Actor


@bench_task("ConveyorSeqPress-v0")
@register_env("ConveyorSeqPress-v0", max_episode_steps=300)
class ConveyorSeqPressEnv(ScriptedSeqPressMixin, ConveyorWorkspaceEnv):
    """Press the two-button panel's buttons in the COMMANDED ORDER."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _num_bores: int = 2
    _bore_pitch: float = 0.045
    _bore_half_width: float = 0.020
    _body_wall_thickness: float = 0.012
    _wall_height: float = 0.030
    _floor_half_thickness: float = 0.005

    _cap_half_width: float = 0.018
    _cap_half_height: float = 0.0175
    _cap_rest_gap: float = 0.008

    _cap_spring_k: float = 300.0
    _cap_spring_damping: float = 8.0
    _cap_force_max: float = 20.0
    _cap_stiffness_range: tuple[float, float] = (0.8, 1.25)

    _body_yaw_max: float = 0.21

    _body_patch_radius: float = 0.032

    _press_success_depth: float = 0.006
    _press_sustain_steps: int = 3
    _cap_bore_slack: float = 0.005
    _lane_x_tol: float = 0.04
    _lane_y_tol: float = 0.85
    _lane_z_tol: float = 0.006
    _min_travel_frac: float = 0.5
    _travel_eps: float = 0.01

    default_prompt: str = "press the two panel buttons in the commanded order"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        press_order: str | None = None,
        **kwargs,
    ) -> None:
        names = [name for name, _ in CAP_PALETTE]
        if press_order is not None and press_order not in names:
            raise ValueError(f"press_order={press_order!r} must be one of {names} (or None to re-draw per episode)")
        self._pinned_first_cap: int | None = None if press_order is None else names.index(press_order)
        super().__init__(*args, **kwargs)

    @property
    def _cap_rest_local_z(self) -> float:
        return self._cap_rest_gap + self._cap_half_height

    @property
    def _body_origin_above_deck(self) -> float:
        return 2.0 * self._floor_half_thickness

    @property
    def _body_half_x(self) -> float:
        return self._bore_half_width + self._body_wall_thickness

    @property
    def _body_half_y(self) -> float:
        return 0.5 * self._bore_pitch + self._bore_half_width + self._body_wall_thickness

    def _bore_local_y(self, bore: int) -> float:
        return (bore - 0.5 * (self._num_bores - 1)) * self._bore_pitch

    @property
    def episode_objects(self) -> list[str]:
        return [CAP_PALETTE[int(i)][0] for i in self._first_cap.tolist()]

    @property
    def episode_prompts(self) -> list[str]:
        return [
            f"press the {CAP_PALETTE[int(a)][0]} button, then the {CAP_PALETTE[int(b)][0]} button"
            for a, b in zip(self._first_cap.tolist(), self._second_cap.tolist(), strict=True)
        ]

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        body_builder = build_clock_body(
            self.scene,
            bore_half_width=self._bore_half_width,
            bore_pitch=self._bore_pitch,
            wall_thickness=self._body_wall_thickness,
            wall_height=self._wall_height,
            floor_half_thickness=self._floor_half_thickness,
        )
        body_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._body_origin_above_deck])
        self.body = body_builder.build(name="clock_body")
        self.belt.register_rider(self.body, patch_radius=self._body_patch_radius)

        self.caps: list[Actor] = []
        cap_z = deck_z + self._body_origin_above_deck + self._cap_rest_local_z
        for k in range(self._num_bores):
            name, color = CAP_PALETTE[k]
            cap_builder = build_button_cap(
                self.scene,
                cap_half_width=self._cap_half_width,
                cap_half_height=self._cap_half_height,
                color=color,
            )
            cap_builder.initial_pose = sapien.Pose(p=[ox, oy + self._bore_local_y(k), cap_z])
            self.caps.append(cap_builder.build(name=f"button_cap_{name}"))

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        yaw = self._sample_body_yaw(env_idx, b)
        body_q = torch.zeros((b, 4), device=device)
        body_q[:, 0] = torch.cos(yaw * 0.5)
        body_q[:, 3] = torch.sin(yaw * 0.5)

        body_pos = self._sample_body_position(env_idx, b)
        self.body.set_pose(Pose.create_from_pq(body_pos, body_q))
        ride_vel = torch.zeros((b, 3), device=device)
        ride_vel[:, 1] = self.belt.velocity[env_idx]
        self.body.set_linear_velocity(ride_vel)
        self.body.set_angular_velocity(torch.zeros((b, 3), device=device))

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        for k, cap in enumerate(self.caps):
            local_y = self._bore_local_y(k)
            cap_pos = torch.zeros((b, 3), device=device)
            cap_pos[:, 0] = body_pos[:, 0] - local_y * sin_yaw
            cap_pos[:, 1] = body_pos[:, 1] + local_y * cos_yaw
            cap_pos[:, 2] = body_pos[:, 2] + self._cap_rest_local_z
            cap.set_pose(Pose.create_from_pq(cap_pos, body_q))
            cap.set_linear_velocity(ride_vel)
            cap.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_press_trackers(env_idx)
        self._first_cap[env_idx] = self._sample_commanded_order(b)
        self._second_cap[env_idx] = self._num_bores - 1 - self._first_cap[env_idx]
        self._cap_spring_k_env[env_idx] = self._sample_cap_stiffness(b)
        self._cap_spring_ready = True

    def _sample_commanded_order(self, b: int) -> torch.Tensor:
        if self._pinned_first_cap is not None:
            return torch.full((b,), self._pinned_first_cap, dtype=torch.long, device=self.device)
        draw = self._episode_rng.randint(0, self._num_bores, size=b)
        return torch.as_tensor(np.asarray(draw), dtype=torch.long, device=self.device)

    def _sample_body_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        device = self.device
        if not self.randomization.body_yaw:
            return torch.zeros(b, device=device)
        yaw = self._episode_rng.uniform(-self._body_yaw_max, self._body_yaw_max, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=device)

    def _sample_cap_stiffness(self, b: int) -> torch.Tensor:
        if not self.randomization.cap_stiffness:
            return torch.full((b,), self._cap_spring_k, device=self.device)
        lo, hi = self._cap_stiffness_range
        scale = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(scale * self._cap_spring_k, device=self.device)

    def _sample_body_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = self._conveyor_half_length - self._body_half_y - 0.04
        body_pos = torch.zeros((b, 3), device=device)
        body_pos[:, 0] = ox
        body_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        body_pos[:, 2] = deck_z + self._body_origin_above_deck
        return body_pos

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
        anchor_z = self.body.pose.p[:, 2] + self._cap_rest_local_z
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

    def _cap_press_depths(self) -> torch.Tensor:
        body_z = self.body.pose.p[:, 2]
        return torch.stack(
            [
                (self._cap_rest_local_z - (cap.pose.p[:, 2] - body_z)).clamp(0.0, self._cap_rest_gap)
                for cap in self.caps
            ],
            dim=-1,
        )

    def _caps_in_bore(self) -> torch.Tensor:
        body_pose = self.body.pose
        cols = []
        for k, cap in enumerate(self.caps):
            local = (body_pose.inv() * Pose.create_from_pq(cap.pose.p)).p
            dy = local[:, 1] - self._bore_local_y(k)
            cols.append((torch.abs(local[:, 0]) <= self._cap_bore_slack) & (torch.abs(dy) <= self._cap_bore_slack))
        return torch.stack(cols, dim=-1)

    def _clock_in_lane(self) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        rest_z = oz + self._conveyor_deck_height + self._body_origin_above_deck
        p = self.body.pose.p
        return (
            (torch.abs(p[:, 2] - rest_z) <= self._lane_z_tol)
            & (torch.abs(p[:, 0] - ox) <= self._lane_x_tol)
            & (torch.abs(p[:, 1] - oy) <= self._lane_y_tol)
        )

    def _clock_travelling(self) -> torch.Tensor:
        v_along = self.body.linear_velocity[:, 1] * self.belt.direction_sign
        return v_along >= carried_speed(torch.abs(self.belt.velocity), self._min_travel_frac) - self._travel_eps

    def _reset_press_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_first_sustain"):
            b, device = self.num_envs, self.device
            self._first_sustain = torch.zeros(b, dtype=torch.long, device=device)
            self._second_sustain = torch.zeros(b, dtype=torch.long, device=device)
            self._first_latched = torch.zeros(b, dtype=torch.bool, device=device)
            self._second_latched = torch.zeros(b, dtype=torch.bool, device=device)
            self._violated = torch.zeros(b, dtype=torch.bool, device=device)
            self._first_cap = torch.zeros(b, dtype=torch.long, device=device)
            self._second_cap = torch.ones(b, dtype=torch.long, device=device)
            self._cap_spring_k_env = torch.full((b,), self._cap_spring_k, device=device)
        self._first_sustain[env_idx] = 0
        self._second_sustain[env_idx] = 0
        self._first_latched[env_idx] = False
        self._second_latched[env_idx] = False
        self._violated[env_idx] = False

    def evaluate(self):
        depths = self._cap_press_depths()
        down = (depths >= self._press_success_depth) & self._caps_in_bore()
        first_depth = depths.gather(1, self._first_cap.unsqueeze(-1)).squeeze(-1)
        second_depth = depths.gather(1, self._second_cap.unsqueeze(-1)).squeeze(-1)
        first_down = down.gather(1, self._first_cap.unsqueeze(-1)).squeeze(-1)
        second_down = down.gather(1, self._second_cap.unsqueeze(-1)).squeeze(-1)

        in_lane = self._clock_in_lane()
        travelling = self._clock_travelling()
        guards = in_lane & travelling

        self._first_sustain = self._advance_sustain("seq_first", self._first_sustain, first_down)
        self._second_sustain = self._advance_sustain("seq_second", self._second_sustain, second_down)

        self._violated = self._violated | (second_down & ~self._first_latched) | ((first_down | second_down) & ~in_lane)

        self._second_latched = self._second_latched | ((self._second_sustain >= self._press_sustain_steps) & guards)
        self._first_latched = self._first_latched | ((self._first_sustain >= self._press_sustain_steps) & guards)

        success = self._second_latched & ~self._violated

        return {
            "success": success,
            "press_depth_first": first_depth,
            "press_depth_second": second_depth,
            "first_latched": self._first_latched,
            "second_latched": self._second_latched,
            "order_violated": self._violated,
            "clock_in_lane": in_lane,
            "clock_travelling": travelling,
            "first_cap": self._first_cap,
        }

    def compute_dense_reward(self, obs, action, info):
        agent = cast("Panda", self.agent)
        tcp_pos = agent.tcp.pose.p
        caps_p = torch.stack([cap.pose.p for cap in self.caps], dim=1)
        stage = torch.where(info["first_latched"], self._second_cap, self._first_cap)
        target_pos = caps_p.gather(1, stage.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

        tcp_to_cap = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
        reach_reward = 1 - torch.tanh(5 * tcp_to_cap)

        stage_depth = torch.where(info["first_latched"], info["press_depth_second"], info["press_depth_first"])
        press_reward = (stage_depth / self._press_success_depth).clamp(0.0, 1.0)
        first_reward = info["first_latched"].to(torch.float32)

        return (reach_reward + press_reward + 2.0 * first_reward) * (~info["order_violated"]).to(torch.float32)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
