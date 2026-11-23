from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.assets.procedural import build_can
from sweep_bench.tasks.conveyor_axle_seat.assets import build_vnotch_carrier
from sweep_bench.tasks.conveyor_axle_seat.config import RandomizationConfig
from sweep_bench.tasks.conveyor_axle_seat.scripted import ScriptedAxleSeatMixin
from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_ROD_COLOR = (0.20, 0.42, 0.85, 1.0)

_ROD_IN_HAND_Q = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


@bench_task("ConveyorAxleSeat-v0")
@register_env("ConveyorAxleSeat-v0", max_episode_steps=300)
class ConveyorAxleSeatEnv(ScriptedAxleSeatMixin, ConveyorWorkspaceEnv):
    """Seat a held rod into the two V-notched forks of a carrier riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _rod_radius: float = 0.0035
    _rod_half_length: float = 0.065

    _plate_half_x: float = 0.040
    _plate_half_y: float = 0.052
    _plate_thickness: float = 0.008
    _fork_half_width: float = 0.032
    _fork_thickness: float = 0.008
    _throat_rise: float = 0.016
    _throat_wall_height: float = 0.005
    _flare_depth: float = 0.020
    _end_stop_height: float = 0.004

    _carrier_patch_radius: float = 0.030

    _carrier_spawn_max_dist: float = 0.80

    _seat_x_tol: float = 0.0025
    _seat_z_tol: float = 0.003
    _released_distance_threshold: float = 0.08
    _comoving_rel_speed_threshold: float = 0.3
    _seat_sustain_steps: int = 5

    default_prompt: str = "seat the rod in the notched carrier on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        fork_pitch: float = 0.080,
        throat_width: float = 0.009,
        flare_width: float = 0.032,
        **kwargs,
    ) -> None:
        if not 2.0 * self._rod_radius < throat_width:
            raise ValueError(
                f"throat_width={throat_width} must exceed the rod diameter {2.0 * self._rod_radius} "
                "(otherwise the rod hangs on the slot walls instead of bottoming on the throat floor)"
            )
        if not self._rod_radius < self._throat_wall_height:
            raise ValueError(
                f"throat_wall_height={self._throat_wall_height} must exceed the rod radius {self._rod_radius}, "
                "or the seated rod's widest point rests on already-flared wall"
            )
        if not throat_width < flare_width:
            raise ValueError(f"throat_width={throat_width} must be smaller than flare_width={flare_width}")
        if not flare_width / 2.0 < self._fork_half_width:
            raise ValueError(f"flare_width={flare_width} must fit inside the fork (half-width {self._fork_half_width})")
        if fork_pitch / 2.0 >= self._rod_half_length:
            raise ValueError(f"fork_pitch={fork_pitch} needs a rod longer than {2.0 * self._rod_half_length} m")
        if fork_pitch / 2.0 + self._fork_thickness / 2.0 > self._plate_half_y:
            raise ValueError(f"fork_pitch={fork_pitch} overhangs the carrier plate (half-length {self._plate_half_y})")
        self.fork_pitch: float = fork_pitch
        self.throat_width: float = throat_width
        self.flare_width: float = flare_width

        super().__init__(*args, **kwargs)

    @property
    def _plate_top_local_z(self) -> float:
        return self._plate_thickness / 2.0

    @property
    def _seated_rod_local_z(self) -> float:
        return self._plate_top_local_z + self._throat_rise + self._rod_radius

    @property
    def _mouth_top_local_z(self) -> float:
        return self._plate_top_local_z + self._throat_rise + self._throat_wall_height + self._flare_depth

    @property
    def _fork_top_local_z(self) -> float:
        return self._mouth_top_local_z + self._end_stop_height

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        carrier_builder = build_vnotch_carrier(
            self.scene,
            plate_half_x=self._plate_half_x,
            plate_half_y=self._plate_half_y,
            plate_thickness=self._plate_thickness,
            fork_pitch=self.fork_pitch,
            fork_half_width=self._fork_half_width,
            fork_thickness=self._fork_thickness,
            throat_rise=self._throat_rise,
            throat_width=self.throat_width,
            throat_wall_height=self._throat_wall_height,
            flare_width=self.flare_width,
            flare_depth=self._flare_depth,
            end_stop_height=self._end_stop_height,
        )
        carrier_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._plate_thickness / 2.0 + 0.001])
        self.carrier = carrier_builder.build(name="vnotch_carrier")
        self.belt.register_rider(self.carrier, patch_radius=self._carrier_patch_radius)

        rod_builder = build_can(
            self.scene,
            radius=self._rod_radius,
            half_length=self._rod_half_length,
            color=_ROD_COLOR,
        )
        rod_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.rod = rod_builder.build(name="axle_rod")

        ends = torch.zeros((self.num_envs, 3), device=self.device)
        ends[:, 2] = self._rod_half_length
        self._rod_end_plus = Pose.create_from_pq(p=ends)
        self._rod_end_minus = Pose.create_from_pq(p=-ends)
        self._fork_planes = torch.tensor(
            [[-self.fork_pitch / 2.0, self.fork_pitch / 2.0]],
            dtype=torch.float32,
            device=self.device,
        )

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._rod_radius - self._scripted_hold_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        carrier_pos = self._sample_carrier_position(env_idx, b)
        carrier_q = self._sample_carrier_quat(env_idx, b)
        self.carrier.set_pose(Pose.create_from_pq(carrier_pos, carrier_q))
        carrier_vel = torch.zeros((b, 3), device=device)
        carrier_vel[:, 1] = self.belt.velocity[env_idx]
        self.carrier.set_linear_velocity(carrier_vel)
        self.carrier.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        in_hand = Pose.create_from_pq(q=torch.tensor(_ROD_IN_HAND_Q, device=device).expand(b, 4))
        self.rod.set_pose(Pose.create_from_pq(tcp_pose.p[env_idx], tcp_pose.q[env_idx]) * in_hand)
        self.rod.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.rod.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx)

    def _sample_carrier_quat(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        q = torch.zeros((b, 4), device=self.device)
        q[:, 0] = 1.0
        if self.randomization.carrier_yaw:
            lo, hi = self.randomization.carrier_yaw_range
            yaw = torch.as_tensor(self._episode_rng.uniform(lo, hi, size=b).astype(np.float32), device=self.device)
            q[:, 0] = torch.cos(yaw / 2.0)
            q[:, 3] = torch.sin(yaw / 2.0)
        return q

    def _sample_carrier_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._plate_half_y - 0.04,
            self._carrier_spawn_max_dist,
        )
        pos = torch.zeros((b, 3), device=device)
        pos[:, 0] = ox
        if self.randomization.carrier_lateral:
            lo, hi = self.randomization.carrier_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        pos[:, 2] = deck_z + self._plate_thickness / 2.0 + 0.001
        return pos

    def _rod_axis_samples(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv = self.carrier.pose.inv()
        p_plus = (inv * (self.rod.pose * self._rod_end_plus)).p
        p_minus = (inv * (self.rod.pose * self._rod_end_minus)).p
        centre = 0.5 * (p_plus + p_minus)
        axis = (p_plus - p_minus) / (2.0 * self._rod_half_length)
        dy = axis[:, 1]
        dy_safe = torch.where(dy >= 0.0, dy.clamp(min=1e-6), dy.clamp(max=-1e-6))
        t = (self._fork_planes - centre[:, 1:2]) / dy_safe.unsqueeze(-1)
        x = centre[:, 0:1] + t * axis[:, 0:1]
        z = centre[:, 2:3] + t * axis[:, 2:3]
        return x, z, t.abs() <= self._rod_half_length

    def _reset_seat_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seat_sustain_counter[env_idx] = 0

    def evaluate(self):
        x, z, on_rod = self._rod_axis_samples()
        z_err = z - self._seated_rod_local_z
        in_notch = on_rod & (x.abs() < self._seat_x_tol) & (z_err.abs() < self._seat_z_tol)
        seated = in_notch.all(dim=-1)

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        released = torch.linalg.norm(tcp_pos - self.rod.pose.p, dim=-1) > self._released_distance_threshold
        rel_speed = torch.linalg.norm(self.rod.linear_velocity - self.carrier.linear_velocity, dim=-1)
        comoving = rel_speed < self._comoving_rel_speed_threshold

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        carrier_on_belt = rider_on_belt(
            pose=self.carrier.pose,
            linear_velocity=self.carrier.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._plate_half_x,
            y_tol=self._conveyor_half_length + self._plate_half_y,
            rest_height=self._plate_thickness / 2.0,
        )

        holding = seated & released & comoving & carrier_on_belt
        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, holding)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "released": released,
            "comoving": comoving,
            "carrier_on_belt": carrier_on_belt,
            "notch_x_offset": x.abs().amax(dim=-1),
            "notch_z_offset": z_err.abs().amax(dim=-1),
        }

    def compute_dense_reward(self, obs, action, info):
        x, z, on_rod = self._rod_axis_samples()
        z_err = z - self._seated_rod_local_z

        lateral = x.abs().amax(dim=-1)
        align_reward = 1 - torch.tanh(20.0 * lateral)

        height = 1 - torch.tanh(20.0 * z_err.abs().amax(dim=-1))
        capturable = (on_rod & (x.abs() < self.flare_width / 2.0)).all(dim=-1).float()
        return align_reward + 2.0 * height * capturable + 2.0 * (info["seated"] & info["released"]).float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5.0
