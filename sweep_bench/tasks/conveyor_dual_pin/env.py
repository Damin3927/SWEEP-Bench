from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_dual_pin.assets import (
    build_pin_base,
    build_two_bore_bracket,
    grip_bar_half_y,
)
from sweep_bench.tasks.conveyor_dual_pin.config import RandomizationConfig
from sweep_bench.tasks.conveyor_dual_pin.scripted import ScriptedDualPinMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_GRIP_TO_BRACKET_Q = (0.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)


@bench_task("ConveyorDualPin-v0")
@register_env("ConveyorDualPin-v0", max_episode_steps=300)
class ConveyorDualPinEnv(ScriptedDualPinMixin, ConveyorWorkspaceEnv):
    """Register a held two-bore bracket onto BOTH pins of a base riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _base_half_x: float = 0.012
    _base_half_y: float = 0.050
    _base_half_height: float = 0.008

    _pin_radius: float = 0.005
    _pin_shank_height: float = 0.018
    _pin_cone_height: float = 0.010
    _pin_tip_radius: float = 0.0015

    _bracket_pad_half_extent: float = 0.022
    _bracket_half_z: float = 0.014
    _bore_chamfer: float = 0.008

    _base_patch_radius: float = 0.012

    _base_plate_density: float = 2500.0
    _bracket_density: float = 500.0

    _base_spawn_max_dist: float = 0.60

    _seat_depth_tol: float = 0.003
    _released_distance_threshold: float = 0.08
    _settled_rel_speed_threshold: float = 0.3
    _seat_sustain_steps: int = 4

    default_prompt: str = "fit the bracket onto both pins on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        bore_half_width: float = 0.009,
        pin_pitch: float = 0.036,
        **kwargs,
    ) -> None:
        if not self._pin_radius < bore_half_width:
            raise ValueError(f"bore_half_width={bore_half_width} must exceed the pin radius {self._pin_radius}")
        if not bore_half_width + self._bore_chamfer < self._bracket_pad_half_extent:
            raise ValueError(
                f"bore mouth {bore_half_width + self._bore_chamfer} does not fit the "
                f"{self._bracket_pad_half_extent} m bracket pad"
            )
        self.bore_half_width: float = bore_half_width
        self.pin_pitch: float = pin_pitch
        if pin_pitch / 2.0 + self._bracket_pad_half_extent > self._base_half_y:
            raise ValueError(
                f"pin_pitch={pin_pitch} makes the bracket ({pin_pitch / 2.0 + self._bracket_pad_half_extent} m "
                f"half-length) overhang the {self._base_half_y} m base plate"
            )

        super().__init__(*args, **kwargs)

    @property
    def _bore_clear_half_width(self) -> float:
        return self.bore_half_width - self._pin_radius

    @property
    def _bracket_bar_half_y(self) -> float:
        return grip_bar_half_y(
            pitch=self.pin_pitch,
            bore_half_width=self.bore_half_width,
            chamfer=self._bore_chamfer,
        )

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        base_builder = build_pin_base(
            self.scene,
            plate_half_x=self._base_half_x,
            plate_half_y=self._base_half_y,
            plate_half_height=self._base_half_height,
            pin_radius=self._pin_radius,
            shank_height=self._pin_shank_height,
            cone_height=self._pin_cone_height,
            tip_radius=self._pin_tip_radius,
            pitch=self.pin_pitch,
            plate_density=self._base_plate_density,
        )
        base_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._base_half_height + 0.001])
        self.base = base_builder.build(name="pin_base")
        self.belt.register_rider(self.base, patch_radius=self._base_patch_radius)

        bracket_builder = build_two_bore_bracket(
            self.scene,
            bore_half_width=self.bore_half_width,
            pad_half_extent=self._bracket_pad_half_extent,
            half_thickness=self._bracket_half_z,
            pitch=self.pin_pitch,
            chamfer=self._bore_chamfer,
            bar_half_y=self._bracket_bar_half_y,
            density=self._bracket_density,
        )
        bracket_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.bracket = bracket_builder.build(name="two_bore_bracket")

        plus = torch.zeros((self.num_envs, 3), device=self.device)
        plus[:, 1] = self.pin_pitch / 2.0
        self._bore_offset_plus = Pose.create_from_pq(p=plus)
        self._bore_offset_minus = Pose.create_from_pq(p=-plus)
        underside = torch.zeros((self.num_envs, 3), device=self.device)
        underside[:, 2] = -self._bracket_half_z
        self._bracket_underside_offset = Pose.create_from_pq(p=underside)

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._bracket_pad_half_extent - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        yaw = self._sample_base_yaw(env_idx, b)
        if not hasattr(self, "_scripted_base_yaw"):
            self._scripted_base_yaw = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
        self._scripted_base_yaw[env_idx] = yaw
        base_q = torch.zeros((b, 4), device=device)
        base_q[:, 0] = torch.cos(0.5 * yaw)
        base_q[:, 3] = torch.sin(0.5 * yaw)

        base_pos = self._sample_base_position(env_idx, b)
        self.base.set_pose(Pose.create_from_pq(base_pos, base_q))
        base_vel = torch.zeros((b, 3), device=device)
        base_vel[:, 1] = self.belt.velocity[env_idx]
        self.base.set_linear_velocity(base_vel)
        self.base.set_angular_velocity(torch.zeros((b, 3), device=device))

        from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        grip_to_bracket = torch.tensor(_GRIP_TO_BRACKET_Q, device=device, dtype=torch.float32).expand(b, 4)
        bracket_q = quaternion_multiply(tcp_pose.q[env_idx], grip_to_bracket)
        self.bracket.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], bracket_q))
        self.bracket.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.bracket.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx)

    def _sample_base_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.pin_yaw:
            return torch.zeros(b, device=self.device, dtype=torch.float32)
        lo, hi = self.randomization.yaw_range
        yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=self.device)

    def _sample_base_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._base_half_y - 0.04,
            self._base_spawn_max_dist,
        )
        base_pos = torch.zeros((b, 3), device=device)
        base_pos[:, 0] = ox
        if self.randomization.base_lateral:
            lo, hi = self.randomization.base_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            base_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        base_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        base_pos[:, 2] = deck_z + self._base_half_height + 0.001
        return base_pos

    def _bore_centres_in_base_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        rel = self.base.pose.inv() * self.bracket.pose
        return (rel * self._bore_offset_plus).p, (rel * self._bore_offset_minus).p

    def _bracket_underside_z_in_base_frame(self) -> torch.Tensor:
        rel = self.base.pose.inv() * self.bracket.pose
        return (rel * self._bracket_underside_offset).p[:, 2]

    def _reset_seat_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seat_sustain_counter[env_idx] = 0

    def evaluate(self):
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        half_pitch = self.pin_pitch / 2.0
        clear = self._bore_clear_half_width

        bore_plus, bore_minus = self._bore_centres_in_base_frame()

        def _linf_to_pin(bore: torch.Tensor, pin_y: float) -> torch.Tensor:
            return torch.maximum(bore[:, 0].abs(), (bore[:, 1] - pin_y).abs())

        direct = (_linf_to_pin(bore_plus, half_pitch) < clear) & (_linf_to_pin(bore_minus, -half_pitch) < clear)
        flipped = (_linf_to_pin(bore_plus, -half_pitch) < clear) & (_linf_to_pin(bore_minus, half_pitch) < clear)
        registered = direct | flipped

        underside_z = self._bracket_underside_z_in_base_frame()
        seat_error = (underside_z - self._base_half_height).abs()
        bottomed = seat_error < self._seat_depth_tol

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        released = torch.linalg.norm(tcp_pos - self.bracket.pose.p, dim=-1) > self._released_distance_threshold
        rel_speed = torch.linalg.norm(self.bracket.linear_velocity - self.base.linear_velocity, dim=-1)
        comoving = rel_speed < self._settled_rel_speed_threshold

        base_on_belt = rider_on_belt(
            pose=self.base.pose,
            linear_velocity=self.base.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._base_half_x,
            y_tol=self._conveyor_half_length + self._base_half_y,
            rest_height=self._base_half_height,
        )

        seated = registered & bottomed & released & comoving & base_on_belt
        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, seated)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "registered": registered,
            "bottomed": bottomed,
            "released": released,
            "comoving": comoving,
            "base_on_belt": base_on_belt,
            "bore_plus_offset": _linf_to_pin(bore_plus, half_pitch),
            "bore_minus_offset": _linf_to_pin(bore_minus, -half_pitch),
            "seat_error": seat_error,
            "seat_sustain": self._seat_sustain_counter,
        }

    def compute_dense_reward(self, obs, action, info):
        bore_plus, bore_minus = self._bore_centres_in_base_frame()
        half_pitch = self.pin_pitch / 2.0

        d_plus = torch.linalg.norm(bore_plus[:, :2] - torch.tensor([0.0, half_pitch], device=self.device), dim=-1)
        d_minus = torch.linalg.norm(bore_minus[:, :2] - torch.tensor([0.0, -half_pitch], device=self.device), dim=-1)
        worst = torch.maximum(d_plus, d_minus)
        align_reward = 1 - torch.tanh(20.0 * worst)

        underside_z = self._bracket_underside_z_in_base_frame()
        span = self._pin_shank_height + self._pin_cone_height
        depth = ((span - (underside_z - self._base_half_height)) / span).clamp(0.0, 1.0)
        registered = info["registered"].float()
        return align_reward + 2.0 * depth * registered + 2.0 * (info["bottomed"] & info["released"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
