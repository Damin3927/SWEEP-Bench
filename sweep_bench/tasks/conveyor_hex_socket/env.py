from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sweep_bench.tasks.conveyor_base.env import ConveyorWorkspaceEnv
from sweep_bench.tasks.conveyor_base.on_belt import rider_on_belt
from sweep_bench.tasks.conveyor_hex_socket.assets import (
    HEX_PITCH,
    build_hex_bit,
    build_hex_socket_plate,
    fold_hex,
)
from sweep_bench.tasks.conveyor_hex_socket.config import RandomizationConfig
from sweep_bench.tasks.conveyor_hex_socket.scripted import ScriptedHexSocketMixin
from sweep_bench.tasks.registry import bench_task

if TYPE_CHECKING:
    from mani_skill.agents.robots.panda.panda import Panda

_SOCKET_FLAT_NORMALS: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.5, 0.8660254037844387),
    (-0.5, 0.8660254037844387),
)


@bench_task("ConveyorHexSocket-v0")
@register_env("ConveyorHexSocket-v0", max_episode_steps=300)
class ConveyorHexSocketEnv(ScriptedHexSocketMixin, ConveyorWorkspaceEnv):
    """Clock a held hex bit into the blind hex socket of a plate riding the belt."""

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    _bit_half_length: float = 0.030
    _bit_grasp_offset: float = 0.020

    _socket_depth: float = 0.020
    _chamfer: float = 0.008
    _plate_half_extent: float = 0.032
    _floor_thickness: float = 0.005

    _plate_patch_radius: float = 0.025

    _plate_spawn_max_dist: float = 0.55

    _finger_below_tcp: float = 0.012

    _insert_depth: float = 0.012
    _yaw_tol: float = np.deg2rad(6.0)
    _settled_rel_speed_threshold: float = 0.3
    _held_distance_threshold: float = 0.05
    _seat_sustain_steps: int = 5

    default_prompt: str = "seat the hex bit into the hex socket on the conveyor belt"

    RandomizationConfig = RandomizationConfig

    randomization: RandomizationConfig

    def __init__(
        self,
        *args,
        socket_across_flats: float = 0.0155,
        bit_across_flats: float = 0.0105,
        mouth_across_flats: float = 0.030,
        **kwargs,
    ) -> None:
        if not bit_across_flats < socket_across_flats:
            raise ValueError(
                f"bit_across_flats={bit_across_flats} must be smaller than socket_across_flats={socket_across_flats}"
            )
        if not socket_across_flats < mouth_across_flats:
            raise ValueError(
                f"socket_across_flats={socket_across_flats} must be smaller than "
                f"mouth_across_flats={mouth_across_flats}"
            )
        self.socket_across_flats: float = socket_across_flats
        self.bit_across_flats: float = bit_across_flats
        self.mouth_across_flats: float = mouth_across_flats
        self._socket_half_af: float = socket_across_flats / 2.0
        self._bit_half_af: float = bit_across_flats / 2.0
        self._mouth_half_af: float = mouth_across_flats / 2.0
        self._plate_half_height: float = (self._socket_depth + self._floor_thickness) / 2.0

        seat_target = self._insert_depth + self._scripted_seat_depth_margin
        if self._socket_depth <= seat_target:
            raise ValueError(f"_socket_depth={self._socket_depth} must exceed the expert's seat target {seat_target} m")
        finger_above_top = self._bit_half_length + self._bit_grasp_offset - seat_target - self._finger_below_tcp
        if finger_above_top < self._chamfer:
            raise ValueError(
                f"bit too short to grip clear of the chamfer collar: fingertips would sit "
                f"{finger_above_top:.4f} m above the plate top at the seat, inside the "
                f"{self._chamfer} m collar"
            )
        if self._bit_half_af <= self._scripted_hold_init_margin:
            raise ValueError(
                f"bit_across_flats={bit_across_flats} is too thin for the "
                f"{self._scripted_hold_init_margin} m initial grip overlap"
            )

        super().__init__(*args, **kwargs)

    def _load_objects(self, options: dict) -> None:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height

        plate_builder = build_hex_socket_plate(
            self.scene,
            socket_across_flats=self.socket_across_flats,
            mouth_across_flats=self.mouth_across_flats,
            chamfer=self._chamfer,
            socket_depth=self._socket_depth,
            plate_half_extent=self._plate_half_extent,
            floor_thickness=self._floor_thickness,
        )
        plate_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + self._plate_half_height + 0.001])
        self.plate = plate_builder.build(name="hex_socket_plate")
        self.belt.register_rider(self.plate, patch_radius=self._plate_patch_radius)

        bit_builder = build_hex_bit(
            self.scene,
            across_flats=self.bit_across_flats,
            half_length=self._bit_half_length,
        )
        bit_builder.initial_pose = sapien.Pose(p=[ox, oy, deck_z + 0.5])
        self.bit = bit_builder.build(name="hex_bit")

        tip = torch.zeros((self.num_envs, 3), device=self.device)
        tip[:, 2] = self._bit_half_length
        self._bit_tip_offset = Pose.create_from_pq(p=tip)

    @property
    def _clear_lateral(self) -> float:
        return self._socket_half_af - self._bit_half_af

    def _inside_clear_hex(self, rel_xy: torch.Tensor) -> torch.Tensor:
        ok = torch.ones(rel_xy.shape[0], dtype=torch.bool, device=rel_xy.device)
        for nx, ny in _SOCKET_FLAT_NORMALS:
            ok = ok & ((rel_xy[:, 0] * nx + rel_xy[:, 1] * ny).abs() < self._clear_lateral)
        return ok

    def _rest_aperture(self, env_idx: torch.Tensor) -> np.ndarray:
        b = len(env_idx)
        return 2.0 * np.full((b, 1), self._bit_half_af - self._scripted_hold_init_margin, dtype=np.float32)

    def _initialize_objects(self, env_idx: torch.Tensor, b: int) -> None:
        device = self.device

        yaw = self._sample_socket_yaw(env_idx, b)
        if not hasattr(self, "_socket_yaw"):
            self._socket_yaw = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
            self._scripted_socket_yaw_cmd = torch.zeros(self.num_envs, device=device, dtype=torch.float32)
        self._socket_yaw[env_idx] = yaw
        self._scripted_socket_yaw_cmd[env_idx] = fold_hex(yaw)
        plate_q = torch.zeros((b, 4), device=device)
        plate_q[:, 0] = torch.cos(0.5 * yaw)
        plate_q[:, 3] = torch.sin(0.5 * yaw)

        plate_pos = self._sample_plate_position(env_idx, b)
        self.plate.set_pose(Pose.create_from_pq(plate_pos, plate_q))
        plate_vel = torch.zeros((b, 3), device=device)
        plate_vel[:, 1] = self.belt.velocity[env_idx]
        self.plate.set_linear_velocity(plate_vel)
        self.plate.set_angular_velocity(torch.zeros((b, 3), device=device))

        if self.scene.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        tcp_pose = cast("Panda", self.agent).tcp.pose
        bit_pos = tcp_pose.p[env_idx].clone()
        bit_pos[:, 2] = bit_pos[:, 2] - self._bit_grasp_offset
        self.bit.set_pose(Pose.create_from_pq(bit_pos, tcp_pose.q[env_idx]))
        self.bit.set_linear_velocity(torch.zeros((b, 3), device=device))
        self.bit.set_angular_velocity(torch.zeros((b, 3), device=device))

        self._reset_seat_trackers(env_idx)

    def _sample_socket_yaw(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        if not self.randomization.socket_yaw:
            return torch.zeros(b, device=self.device, dtype=torch.float32)
        lo, hi = self.randomization.yaw_range
        yaw = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
        return torch.as_tensor(yaw, device=self.device)

    def _sample_plate_position(self, env_idx: torch.Tensor, b: int) -> torch.Tensor:
        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        device = self.device
        start_y = min(
            self._conveyor_half_length - self._plate_half_extent - 0.04,
            self._plate_spawn_max_dist,
        )
        plate_pos = torch.zeros((b, 3), device=device)
        plate_pos[:, 0] = ox
        if self.randomization.plate_lateral:
            lo, hi = self.randomization.plate_lateral_range
            offsets = self._episode_rng.uniform(lo, hi, size=b).astype(np.float32)
            plate_pos[:, 0] = ox + torch.as_tensor(offsets, device=device)
        plate_pos[:, 1] = oy - start_y * self.belt.direction_sign[env_idx]
        plate_pos[:, 2] = deck_z + self._plate_half_height + 0.001
        return plate_pos

    def _bit_tip_in_plate_frame(self) -> torch.Tensor:
        tip_world = self.bit.pose * self._bit_tip_offset
        return (self.plate.pose.inv() * tip_world).p

    def _clocking_error(self) -> torch.Tensor:
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix

        rb = quaternion_to_matrix(self.bit.pose.q)
        rp = quaternion_to_matrix(self.plate.pose.q)
        n_bit = rb[:, :2, 1]
        n_socket = rp[:, :2, 0]
        n_bit = n_bit / torch.linalg.norm(n_bit, dim=-1, keepdim=True).clamp(min=1e-9)
        n_socket = n_socket / torch.linalg.norm(n_socket, dim=-1, keepdim=True).clamp(min=1e-9)
        cross = n_bit[:, 0] * n_socket[:, 1] - n_bit[:, 1] * n_socket[:, 0]
        dot = (n_bit * n_socket).sum(dim=-1)
        angle = torch.atan2(cross, dot)
        return fold_hex(angle).abs()

    def _reset_seat_trackers(self, env_idx: torch.Tensor) -> None:
        if not hasattr(self, "_seat_sustain_counter"):
            self._seat_sustain_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seat_sustain_counter[env_idx] = 0

    def evaluate(self):
        rel = self._bit_tip_in_plate_frame()
        h = self._plate_half_height

        depth = h - rel[:, 2]
        lateral = torch.linalg.norm(rel[:, :2], dim=-1)
        seated = self._inside_clear_hex(rel[:, :2]) & (depth > self._insert_depth) & (rel[:, 2] > -h - 0.01)

        yaw_err = self._clocking_error()
        clocked = yaw_err < self._yaw_tol

        tcp_pos = cast("Panda", self.agent).tcp.pose.p
        held = torch.linalg.norm(tcp_pos - self.bit.pose.p, dim=-1) < self._held_distance_threshold
        rel_speed = torch.linalg.norm(self.bit.linear_velocity - self.plate.linear_velocity, dim=-1)
        comoving = rel_speed < self._settled_rel_speed_threshold

        ox, oy, oz = self.workspace_offset
        deck_z = oz + self._conveyor_deck_height
        plate_on_belt = rider_on_belt(
            pose=self.plate.pose,
            linear_velocity=self.plate.linear_velocity,
            belt_velocity=self.belt.velocity,
            deck_z=deck_z,
            centre_x=ox,
            centre_y=oy,
            x_tol=self._conveyor_half_width + self._plate_half_extent,
            y_tol=self._conveyor_half_length + self._plate_half_extent,
            rest_height=self._plate_half_height,
        )

        holding = seated & clocked & held & comoving & plate_on_belt
        self._seat_sustain_counter = self._advance_sustain("seat_sustain", self._seat_sustain_counter, holding)
        success = self._seat_sustain_counter >= self._seat_sustain_steps

        return {
            "success": success,
            "seated": seated,
            "clocked": clocked,
            "clocking_error": yaw_err,
            "held": held,
            "comoving": comoving,
            "plate_on_belt": plate_on_belt,
            "seat_depth": depth,
            "tip_lateral_offset": lateral,
        }

    def compute_dense_reward(self, obs, action, info):
        rel = self._bit_tip_in_plate_frame()
        h = self._plate_half_height

        lateral = torch.linalg.norm(rel[:, :2], dim=-1)
        align_reward = 1 - torch.tanh(20.0 * lateral)

        clock_reward = (1.0 - info["clocking_error"] / (HEX_PITCH / 2.0)).clamp(0.0, 1.0)

        depth = (h - rel[:, 2]).clamp(0.0, self._insert_depth) / self._insert_depth
        aligned = self._inside_clear_hex(rel[:, :2]).float()
        return align_reward + clock_reward + 2.0 * depth * aligned

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 4.0
